import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss
from typing import List, Tuple, Dict, Optional

class ColbertModule(torch.nn.Module):
    """
    Base module for ColBERT losses, handling shared utilities and hyperparameters.

    Args:
        max_batch_size (int): Maximum batch size for pre-allocating index buffer.
        tau (float): Temperature for smooth-max approximation.
        norm_tol (float): Tolerance for score normalization bounds.
        filter_threshold (float): Ratio threshold for pos-aware negative filtering.
        filter_factor (float): Multiplicative factor to down-weight high negatives.
    """

    def __init__(
        self,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("idx_buffer", torch.arange(max_batch_size), persistent=False)
        self.tau = tau
        self.norm_tol = norm_tol
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor

    def _get_idx(self, batch_size: int, offset: int, device: torch.device):
        """
        Retrieve index and positive index tensors for in-batch losses.
        """
        idx = self.idx_buffer[:batch_size].to(device)
        return idx, idx + offset

    def _smooth_max(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute smooth max via log-sum-exp along a given dimension.
        """
        return self.tau * torch.logsumexp(scores / self.tau, dim=dim)

    def _apply_normalization(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores by query lengths and enforce bounds.

        Args:
            scores (Tensor): Unnormalized score matrix [B, C].
            lengths (Tensor): Query lengths [B].

        Returns:
            Tensor: Normalized scores.

        Raises:
            ValueError: If normalized scores exceed tolerance.
        """
        if scores.ndim == 2:
            normalized = scores / lengths.unsqueeze(1)
        else:
            normalized = scores / lengths

        mn, mx = torch.aminmax(normalized)
        if mn < -self.norm_tol or mx > 1 + self.norm_tol:
            print(
                f"Scores out of bounds after normalization: "
                f"min={mn.item():.4f}, max={mx.item():.4f}, tol={self.norm_tol}"
            )
        return normalized

    def _aggregate(
        self,
        scores_raw: torch.Tensor,
        use_smooth_max: bool,
        dim_max: int,
        dim_sum: int,
    ) -> torch.Tensor:
        """
        Aggregate token-level scores into document-level.

        Args:
            scores_raw (Tensor): Raw scores tensor.
            use_smooth_max (bool): Use smooth-max if True.
            dim_max (int): Dimension to perform max/logsumexp.
            dim_sum (int): Dimension to sum over after max.
        """
        if use_smooth_max:
            return self._smooth_max(scores_raw, dim=dim_max).sum(dim=dim_sum)
        return scores_raw.amax(dim=dim_max).sum(dim=dim_sum)

    def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor) -> None:
        """
        Down-weight negatives whose score exceeds a fraction of the positive score.

        Args:
            scores (Tensor): In-batch score matrix [B, B].
            pos_idx (Tensor): Positive indices for each query in batch.
        """
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)
        pos_scores = scores[idx, pos_idx]
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)
        mask = scores > thresh
        mask[idx, pos_idx] = False
        scores[mask] *= self.filter_factor


class ColbertLoss(ColbertModule):
    """
    InfoNCE loss for late interaction (ColBERT) without explicit negatives.

    Args:
        temperature (float): Scaling factor for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Compute ColBERT InfoNCE loss over a batch of queries and documents.

        Args:
            query_embeddings (Tensor): (batch_size, query_length, dim)
            doc_embeddings (Tensor): positive docs (batch_size, pos_doc_length, dim)
            offset (int): Offset for positive doc indices (multi-GPU).

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)
        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        return self.ce_loss(scores / self.temperature, pos_idx)


class ColbertNegativeCELoss(ColbertModule):
    """
    InfoNCE loss with explicit negative documents.

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term_weight (float): Add in-batch CE term (between 0 and 1).
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        self.ce_loss = CrossEntropyLoss()

        assert in_batch_term_weight >= 0, "in_batch_term_weight must be non-negative"
        assert in_batch_term_weight <= 1, "in_batch_term_weight must be less than 1"

        self.inner_loss = ColbertLoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): (batch_size, query_length, dim)
            doc_embeddings (Tensor): positive docs (batch_size, pos_doc_length, dim)
            neg_doc_embeddings (Tensor): negative docs (batch_size, num_negs, neg_doc_length, dim)
            offset (int): Positional offset for in-batch CE.

        Returns:
            Tensor: Scalar loss.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        pos_raw = torch.einsum(
            "bnd,bsd->bns", query_embeddings, doc_embeddings[offset : offset + neg_doc_embeddings.size(0)]
        )
        neg_raw = torch.einsum("bnd,blsd->blns", query_embeddings, neg_doc_embeddings)
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        neg_scores = self._aggregate(neg_raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        loss = F.softplus((neg_scores - pos_scores.unsqueeze(1)) / self.temperature).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight

        return loss


class ColbertPairwiseCELoss(ColbertModule):
    """
    Pairwise loss for ColBERT (no explicit negatives).

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Compute pairwise softplus loss over in-batch document pairs.

        Args:
            query_embeddings (Tensor): (batch_size, query_length, dim)
            doc_embeddings (Tensor): positive docs (batch_size, pos_doc_length, dim)
            offset (int): Positional offset for positives.

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        pos_scores = scores.diagonal(offset=offset)
        top2 = scores.topk(2, dim=1).values
        neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])

        return F.softplus((neg_scores - pos_scores) / self.temperature).mean()


class ColbertPairwiseNegativeCELoss(ColbertModule):
    """
    Pairwise loss with explicit negatives and optional in-batch term.

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term_weight (float): Add in-batch CE term (between 0 and 1).
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        assert in_batch_term_weight >= 0, "in_batch_term_weight must be non-negative"
        assert in_batch_term_weight <= 1, "in_batch_term_weight must be less than 1"
        self.inner_pairwise = ColbertPairwiseCELoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute pairwise softplus loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): (batch_size, query_length, dim)
            doc_embeddings (Tensor): positive docs (batch_size, pos_doc_length, dim)
            neg_doc_embeddings (Tensor): negative docs (batch_size, num_negs, neg_doc_length, dim)
            offset (int): Positional offset for positives.

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        pos_raw = torch.einsum(
            "bnd,bld->bnl", query_embeddings, doc_embeddings[offset : offset + query_embeddings.size(0)]
        )
        neg_raw = torch.einsum("bnd,bsld->bsnl", query_embeddings, neg_doc_embeddings)  # B x Nneg x Nq x Lneg
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        neg_scores = self._aggregate(neg_raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        loss = F.softplus((neg_scores - pos_scores.unsqueeze(1)) / self.temperature).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_pairwise(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight

        return loss


class ColbertSigmoidLoss(ColbertModule):
    """
    Sigmoid loss for ColBERT with explicit negatives.

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Compute sigmoid loss over positive and negative document pairs.

        Args:
            query_embeddings (Tensor): (batch_size, query_length, dim)
            doc_embeddings (Tensor): positive docs (batch_size, pos_doc_length, dim)

        Returns:
            Tensor: Scalar loss value.
        """

        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        # for each idx in pos_idx, the 2D index (idx, idx) → flat index = idx * B + idx
        # build a 1-D mask of length B*B with ones at those positions
        flat_pos = pos_idx * (batch_size + 1)
        pos_mask = -torch.ones(batch_size * batch_size, device=scores.device)
        pos_mask[flat_pos] = 1.0

        # flatten the scores to [B * B]
        scores = scores.view(-1) / self.temperature

        return F.softplus(scores * pos_mask).mean()


class MetaEmbedMatryoshkaLoss(ColbertModule):
    """
    Implements the Matryoshka Multi-Vector Retrieval (MMR) training objective 
    from the MetaEmbed paper.
    
    It applies a base ColBERT loss to multiple nested groups (prefixes) of the 
    query and document embeddings.
    
    Equation: L_final = sum(w_g * L_NCE(Q[:rq], D[:rc]))
    """
    
    def __init__(
        self,
        base_loss_fn: ColbertModule,
        matryoshka_groups: List[Tuple[int, int]],
        group_weights: Optional[List[float]] = None,
    ):
        """
        Args:
            base_loss_fn: An instance of ColbertLoss, ColbertNegativeCELoss, etc.
            matryoshka_groups: A list of tuples [(q_len1, d_len1), (q_len2, d_len2), ...].
                               Example: [(1, 1), (2, 4), (4, 8), (16, 64)]
            group_weights: Optional list of weights for each group. 
                           If None, defaults to 1.0 for all groups.
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.groups = matryoshka_groups
        
        if group_weights is None:
            self.group_weights = [1.0] * len(matryoshka_groups)
        else:
            assert len(group_weights) == len(matryoshka_groups), "Weights must match groups length"
            self.group_weights = group_weights

    def forward(
        self, 
        query_embeddings: torch.Tensor, 
        doc_embeddings: torch.Tensor, 
        neg_doc_embeddings: Optional[torch.Tensor] = None,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Computes the weighted sum of losses across all Matryoshka groups.
        
        NOTE: This assumes query_embeddings and doc_embeddings are SORTED by granularity.
        For Coconut/MetaEmbed:
            - query_embeddings should be the generated Meta Tokens [B, N_meta, D]
            - doc_embeddings should be the generated Meta Tokens (and/or patches) [B, N_doc, D]
        
        Args:
            query_embeddings: [Batch, N_q_total, Dim]
            doc_embeddings: [Batch, N_d_total, Dim]
            neg_doc_embeddings: [Batch, N_neg, N_d_neg_total, Dim] (Optional)
            offset: Offset for distributed training logic inside base loss.
        """
        total_loss = 0.0
        
        for idx, (r_q, r_c) in enumerate(self.groups):
            weight = self.group_weights[idx]
            
            # 1. Slicing (The Matryoshka Operation)
            # We take the first r_q tokens from query and r_c from doc.
            # Safety check to ensure we don't slice out of bounds
            cur_q_len = min(r_q, query_embeddings.shape[1])
            cur_d_len = min(r_c, doc_embeddings.shape[1])
            
            q_slice = query_embeddings[:, :cur_q_len, :]
            d_slice = doc_embeddings[:, :cur_d_len, :]
            
            # 2. Compute Base Loss for this group
            if neg_doc_embeddings is not None:
                # Handle Explicit Negatives if provided
                cur_neg_len = min(r_c, neg_doc_embeddings.shape[2])
                neg_slice = neg_doc_embeddings[:, :, :cur_neg_len, :]
                
                loss_group = self.base_loss_fn(q_slice, d_slice, neg_slice, offset=offset)
            else:
                # Standard In-batch Negative Loss
                loss_group = self.base_loss_fn(q_slice, d_slice, offset=offset)
            
            # 3. Aggregate
            total_loss += weight * loss_group

        return total_loss

# -------------------------------------------------------------------------
# 使用示例 (Usage Example)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. 定义基础 Loss (例如标准的 ColBERT In-batch NCE Loss)
    base_loss = ColbertLoss(temperature=0.02)
    
    # 2. 定义 MetaEmbed 风格的嵌套配置
    # 论文中推荐的配置示例: Query tokens vs Candidate tokens
    # (1, 1): 最粗粒度，类似 CLIP/DPR
    # (2, 4): 稍微细一点
    # (16, 64): 完整粒度
    mmr_config = [(1, 1), (2, 4), (4, 8), (8, 16), (16, 64)]
    
    # 3. 实例化 Wrapper
    criterion = MetaEmbedMatryoshkaLoss(
        base_loss_fn=base_loss,
        matryoshka_groups=mmr_config
    )
    
    # 4. 模拟数据 (假设从 ColQwen2_5_Coconut 出来的只有 Meta Tokens)
    B, Dim = 8, 128
    # 假设生成了 16 个 Query Meta Tokens
    Q_meta = torch.randn(B, 16, Dim, requires_grad=True) 
    # 假设生成了 64 个 Doc Meta Tokens (或者 Doc 侧保留了更多)
    D_meta = torch.randn(B, 64, Dim, requires_grad=True)
    
    # 5. 计算 Loss
    loss = criterion(Q_meta, D_meta)
    
    print(f"Total Matryoshka Loss: {loss.item()}")
    
    # 6. 反向传播验证
    loss.backward()
    print("Gradient computed successfully.")
    # Q_meta 的梯度流：Q_meta[:, 0] 会收到来自所有 5 个组的梯度叠加
    # Q_meta[:, 15] 只能收到最后一组 (16, 64) 的梯度