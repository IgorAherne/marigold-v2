"""Qwen-Image 2.1 components, loaded the way the Qwen-Image-Edit-2509 ones are: the
DiT quantized to 4 bit with LoRA adapters, the VAE with an optionally trainable
decoder."""

import logging

import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

from marigoldv2.core.registry import REGISTRY, register


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register("network_components")
class LoadQwenImage21VAE:
    """The Qwen-Image 2.1 VAE (RGBA in and out, 16x spatial, 64 latent channels)."""

    def __init__(self, name="VAE", train_decoder=False, **kwargs):
        self.name = name
        self.train_decoder = bool(train_decoder)

    def __call__(self, _=None):
        cfg = REGISTRY["cfg"]
        vae = AutoencoderKLQwenImage21.from_pretrained(
            cfg.paths.ckpt_qwen_image21,
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        vae.requires_grad_(False)
        if self.train_decoder:
            for sub in ("post_quant_conv", "decoder"):
                getattr(vae, sub).requires_grad_(True)
            n_train = sum(p.numel() for p in vae.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in vae.parameters())
            logging.info(
                f"[Qwen-Image 2.1 VAE] decoder unfrozen; trainable params: {n_train} / {n_total}"
            )
        vae.to(_device(), dtype=torch.bfloat16)
        REGISTRY["network_components"][self.name] = vae


@register("network_components")
class LoadQwenImage21Transformer:
    """The Qwen-Image 2.1 DiT, quantized per ``cfg.optimization.quantization`` and
    with LoRA adapters per ``cfg.optimization.lora``; only the LoRA trains.

    ``skip_modules`` stay in bf16: the shared modulation, the timestep embedder and
    the input/output projections are small and every token passes through them.
    """

    def __init__(self, name="Diffuser", **kwargs):
        self.name = name

    def __call__(self, _=None):
        cfg = REGISTRY["cfg"]
        quant_cfg = cfg.optimization.get("quantization", {})
        level = str(quant_cfg.get("level", "4bit")).lower()
        skip_modules = list(quant_cfg.get("skip_modules", []))
        if level == "4bit":
            quantization_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip_modules,
            )
        elif level == "none":
            quantization_config = None
        else:
            raise ValueError(f"Unsupported quantization level '{level}': 4bit or none")

        kwargs = dict(
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            cfg.paths.ckpt_qwen_image21, **kwargs
        )
        if quantization_config is not None:
            transformer = prepare_model_for_kbit_training(
                transformer, use_gradient_checkpointing=False
            )
        else:
            transformer.to(_device())
        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()

        lora = cfg.optimization.lora
        transformer.add_adapter(
            LoraConfig(
                r=int(lora.rank),
                lora_alpha=int(lora.lora_alpha),
                lora_dropout=float(lora.lora_dropout),
                init_lora_weights=str(lora.init_lora_weights),
                target_modules=list(lora.target_modules),
            )
        )
        for name, p in transformer.named_parameters():
            if "lora_" in name:
                p.data = p.data.to(torch.bfloat16)
                p.requires_grad_(True)

        logging.info(
            f"[Qwen-Image 2.1] quantization={level}; trainable params: "
            f"{transformer.num_parameters(only_trainable=True)} / {transformer.num_parameters()}"
        )
        REGISTRY["network_components"][self.name] = transformer
