from typing import ClassVar

import torch
import copy
from torch import nn
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLModel

from typing import ClassVar, Optional, List, Union



class MetaQwen2_5(Qwen2_5_VLModel):  # noqa: N801
    """
    ColQwen2.5 with Coconut Latent Reasoning.
    
    Combines ColPali/ColQwen architecture with Coconut's continuous latent space reasoning.
    It autoregressively generates 'num_latent_tokens' using the previous hidden state 
    as the input embedding for the next step, serving as 'Meta Embeddings'.

    Args:
        config (Qwen2_5VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore text tokens. 
            NOTE: Latent tokens are ALWAYS preserved even if this is True.
        num_latent_tokens (int): Number of continuous latent tokens to generate. 
            Default is 64 (similar to MetaEmbed candidate side).
    """

    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(
        self, 
        config: Qwen2_5_VLConfig, 
        mask_non_image_embeddings: bool = False,
        num_latent_tokens: int = 64  # Default meta tokens count
    ):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.num_latent_tokens = num_latent_tokens
        
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        super().train()
        # 1. Handle Pixel Values (ColQwen logic)
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        # Prepare arguments for the base model
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)  # We control cache manually
        
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids", None)

        # ---------------------------------------------------------
        # Step 1: Initial Forward Pass (Image + Text)
        # ---------------------------------------------------------
        # We must use cache=True to enable efficient latent generation
        outputs = super().forward(
            *args, 
            **kwargs, 
            use_cache=True, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # (batch_size, seq_len, hidden_size)
        last_hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        # print("step 0: ", last_hidden_states.requires_grad)
        
        # ---------------------------------------------------------
        # Step 2: Autoregressive Latent Generation (Coconut Logic)
        # ---------------------------------------------------------
        latent_hidden_states_list = []
        
        if self.num_latent_tokens > 0:
            # Shape: (batch_size, 1, hidden_size)
            current_input_embeds = last_hidden_states[:, -1:, :]
            
            # Prepare extended attention mask and position ids for generation
            batch_size = current_input_embeds.shape[0]
            device = current_input_embeds.device
            
            # Current sequence length
            cur_len = input_ids.shape[1]

            for i in range(self.num_latent_tokens):
                # Update position IDs: [cur_len, cur_len+1, ...]
                next_position_ids = torch.full((batch_size, 1), cur_len + i, dtype=torch.long, device=device)

                next_attention_mask = torch.ones((batch_size, 1), device=device)
                
                if attention_mask is not None:
                    # Append 1 to the attention mask
                    attention_mask = torch.cat([attention_mask, next_attention_mask], dim=1)
                
                # --- COCONUT CORE ---
                # Use current_input_embeds (which is previous hidden state) as inputs_embeds
                step_outputs = super().forward(
                    inputs_embeds=current_input_embeds,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=next_position_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract outputs
                step_hidden_state = step_outputs.last_hidden_state # (batch, 1, hidden_size)
                # print(f"step {i+1}: ", step_hidden_state.requires_grad)
                past_key_values = step_outputs.past_key_values
                
                latent_hidden_states_list.append(step_hidden_state.requires_grad_(True))

                current_input_embeds = step_hidden_state
        # ---------------------------------------------------------
        # Step 3: Combine and Project
        # ---------------------------------------------------------
        if latent_hidden_states_list:
            latent_hidden_states = torch.cat(latent_hidden_states_list, dim=1)
            full_hidden_states = torch.cat([last_hidden_states, latent_hidden_states], dim=1)
        else:
            full_hidden_states = last_hidden_states

        proj = self.custom_text_proj(full_hidden_states)

        proj = proj / proj.norm(dim=-1, keepdim=True)
        
        # ---------------------------------------------------------
        # Step 4: Masking Logic (Preserve Images + Latents)
        # ---------------------------------------------------------
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_mask = (input_ids == self.config.image_token_id) # (batch, seq_len)
            if self.num_latent_tokens > 0:
                latent_mask = torch.ones(
                    (batch_size, self.num_latent_tokens), 
                    dtype=image_mask.dtype, 
                    device=device
                )
                full_mask = torch.cat([image_mask, latent_mask], dim=1)
            else:
                full_mask = image_mask
            proj = proj * full_mask.unsqueeze(-1)
            
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size

class ColQwen2_5(Qwen2_5_VLModel):  # noqa: N801
    """
    ColQwen2.5 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen2.5-VL backbone.

    Args:
        config (Qwen2.5VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2_5_VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)# (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
