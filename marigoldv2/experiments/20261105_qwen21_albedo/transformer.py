"""Qwen-Image 2.1's DiT for a text prompt followed by one target picture, with the
sequence layout built once per shape.

diffusers' forward rebuilds the layout (RoPE positions, image block ids, attention
segments) from GPU masks on every call, which makes the CPU wait for the GPU about
a dozen times per forward. For a text-only prompt and one target picture the layout
depends only on the prompt length and the latent size, so it is computed once and
reused; the joint sequence is then simply [text tokens, picture tokens]. Anything
else (condition pictures, padded prompts, KV caching, the flex processor) goes
through the stock forward.
"""

import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    _qwenimage21_prefix_segments,
)
from diffusers.utils.peft_utils import apply_lora_scale


class QwenImage21TextThenTarget(QwenImage21Transformer2DModel):
    """Pass ``img_mask`` on the CPU to take the cached path (a CUDA mask, or any
    layout other than text then one target picture, runs the stock forward)."""

    def _layout(self, text_len, shape, device):
        cache = self.__dict__.setdefault("_layouts", {})
        key = (text_len, tuple(shape), device)
        if key not in cache:
            image_pad_mask = torch.zeros(
                text_len + shape[1] * shape[2], dtype=torch.bool, device=device
            )
            image_pad_mask[text_len:] = True
            rotary_emb = self.pos_embed([shape], image_pad_mask, device=device)
            image_ids, target_mask = self.build_token_metadata(image_pad_mask, [shape])
            segments = _qwenimage21_prefix_segments(image_ids, text_len)
            cache[key] = (rotary_emb, target_mask, segments)
        return cache[key]

    def _takes_cached_path(self, encoder_hidden_states, img_shapes, img_mask, extra):
        if any(v is not None for v in extra) or img_mask.device.type != "cpu":
            return False
        if len(img_shapes[0]) != 1 or not all(
            isinstance(b.attn.processor, QwenImage21AttnProcessor)
            for b in self.transformer_blocks
        ):
            return False
        text_len = encoder_hidden_states.shape[1]
        _, height, width = img_shapes[0][0]
        return (
            img_mask.shape[1] == text_len + height * width // 4
            and not img_mask[:, :text_len].any()
            and bool(img_mask[:, text_len:].all())
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_shapes,
        img_mask,
        encoder_hidden_states_mask=None,
        attention_kwargs=None,
        kv_cache=None,
        kv_cache_mode=None,
        return_dict=True,
    ):
        extra = (encoder_hidden_states_mask, kv_cache, kv_cache_mode)
        if not self._takes_cached_path(
            encoder_hidden_states, img_shapes, img_mask, extra
        ):
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_shapes,
                img_mask.to(hidden_states.device),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                attention_kwargs=attention_kwargs,
                kv_cache=kv_cache,
                kv_cache_mode=kv_cache_mode,
                return_dict=return_dict,
            )

        rotary_emb, target_mask, segments = self._layout(
            encoder_hidden_states.shape[1], img_shapes[0][0], hidden_states.device
        )
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        joint = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        timestep = timestep.to(hidden_states.dtype)
        modulation_mask = None
        if self.config.causal_condition:
            # The text modulates from an extra t = 0 row, as in the stock forward.
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_mask
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        for block in self.transformer_blocks:
            args = (joint, modulation, rotary_emb, None, modulation_mask)
            args += (None, None, None, segments, None)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                joint = self._gradient_checkpointing_func(block, *args)
            else:
                joint = block(*args)

        output = self.proj_out(self.norm_out(joint, temb, modulation_mask))
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
