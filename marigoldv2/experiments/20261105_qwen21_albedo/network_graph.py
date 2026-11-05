import functools
import os

import torch
from torch.utils.checkpoint import checkpoint

from marigoldv2.core.registry import REGISTRY, get, register


def _latent_stats(vae, ref):
    """Per-channel mean and std of the Qwen-Image 2.1 latent space."""
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=ref.device, dtype=ref.dtype)
    std = torch.tensor(vae.config.latents_std, device=ref.device, dtype=ref.dtype)
    return mean.view(shape), std.view(shape)


@register("network_graph")
class QwenImage21Encode:
    """Encode an RGB image in [-1, 1] into normalized Qwen-Image 2.1 latents.

    The 2.1 VAE reads RGBA; the picture goes in opaque. The latent is the
    distribution's mode, as the Qwen-Image 2.1 pipeline encodes its images.
    """

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = kwargs.get("input_key", "rgb_norm")
        self.output_key = kwargs.get("output_key", "lat_encoding")

    @torch.no_grad()
    def __call__(self, batch):
        if self.input_key not in batch:
            return
        vae = get("network_components", "VAE")
        rgb = batch[self.input_key].to(device=vae.device, dtype=vae.dtype)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected {self.input_key} [B,3,H,W], got {rgb.shape}")
        rgba = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)
        out = vae.encode(rgba.unsqueeze(2)).latent_dist.mode()
        mean, std = _latent_stats(vae, out)
        batch["out"][self.output_key] = (out - mean) / std


def _decode_checkpointed(vae, z):
    """``vae.decode(z).sample`` for one frame, bit for bit, recomputing each decoder
    block in the backward pass: halves the memory of training the decoder."""
    dec = vae.decoder
    x = dec.conv_in(vae.post_quant_conv(z))
    x = checkpoint(dec.mid_block, x, use_reentrant=False)
    for up in dec.up_blocks:
        x = checkpoint(functools.partial(up, first_chunk=True), x, use_reentrant=False)
    x = dec.conv_out(dec.nonlinearity(dec.norm_out(x)))
    return torch.clamp(x, -1.0, 1.0)


@register("network_graph")
class QwenImage21Decode:
    """Decode normalized Qwen-Image 2.1 latents into RGB in [-1, 1] (alpha dropped)."""

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = kwargs.get("input_key", "lat_encoding")
        self.output_key = kwargs.get("output_key", "pixel_pred")
        self.checkpoint = bool(kwargs.get("checkpoint", False))

    def __call__(self, batch):
        if self.input_key not in batch["out"]:
            return
        vae = get("network_components", "VAE")
        latents = batch["out"][self.input_key]
        mean, std = _latent_stats(vae, latents)
        z = latents * std + mean
        if self.checkpoint and torch.is_grad_enabled() and z.shape[2] == 1:
            decoded = _decode_checkpointed(vae, z)
        else:
            decoded = vae.decode(z).sample
        batch["out"][self.output_key] = decoded[:, :3, 0]


@register("network_graph")
class QwenImage21Step:
    """One transformer pass of Qwen-Image 2.1 at a fixed timestep, Marigold V2 style.

    The picture's own latent is the target image at t = 0.499, conditioned on a
    precomputed text-only prompt (``<embed_dir>/{prefix}_prompt_{embeds,mask}.pt``);
    the prediction is the input minus the predicted velocity.
    """

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        embed_dir = REGISTRY["cfg"].paths.embed_dir
        prefix = kwargs.get("prefix", "qwen21_albedo")
        self.prompt_embeds_cpu = torch.load(
            os.path.join(embed_dir, f"{prefix}_prompt_embeds.pt"), map_location="cpu"
        ).contiguous()  # [1, L, 4096]
        mask = torch.load(
            os.path.join(embed_dir, f"{prefix}_prompt_mask.pt"), map_location="cpu"
        )
        self.prompt_mask_cpu = (mask > 0).contiguous()  # [1, L]
        self.unpadded = bool(self.prompt_mask_cpu.all())
        # Cast to bf16 like the hidden states: 0.499 runs as 0.498046875.
        self.timestep = float(kwargs.get("timestep", 0.499))

    def __call__(self, batch):
        diffuser = get("network_components", "Diffuser")
        lat = batch["out"]["lat_encoding"]
        if lat.ndim != 5 or lat.shape[2] != 1:
            raise ValueError(f"Expected lat_encoding [B,C,1,H,W], got {lat.shape}")
        B, C, _, H, W = lat.shape
        if H % 2 or W % 2:
            # Each image slot of the joint sequence stands for 2x2 latent tokens.
            raise ValueError(
                f"Latent size {H}x{W} must be even: use multiples of 32 px"
            )
        device = next(diffuser.parameters()).device
        dtype = torch.bfloat16

        embeds = self.prompt_embeds_cpu.to(device=device, dtype=dtype).expand(B, -1, -1)
        mask = self.prompt_mask_cpu.to(device=device).expand(B, -1)
        text_slots = torch.zeros_like(mask)  # a text-only prompt holds no image slots
        img_mask = torch.cat([text_slots, mask.new_ones(B, H * W // 4)], dim=1)
        packed = lat[:, :, 0].flatten(2).transpose(1, 2).to(dtype)  # [B, H*W, C]
        timestep = torch.full((B,), self.timestep, device=device, dtype=dtype)

        pred = diffuser(
            hidden_states=packed,
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=None if self.unpadded else mask,
            timestep=timestep,
            img_shapes=[[(1, H, W)]] * B,
            img_mask=img_mask,
            attention_kwargs=getattr(diffuser, "attention_kwargs", None) or {},
            return_dict=False,
        )[0][:, -H * W :]
        pred = pred.transpose(1, 2).reshape(B, C, 1, H, W)
        batch["out"]["lat_encoding"] = lat - pred.to(lat.dtype)
