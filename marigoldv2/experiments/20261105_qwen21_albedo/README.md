# Albedo on Qwen-Image 2.1 (Spellcast3D)

The Marigold V2 albedo recipe (`20260803_qwen_albedo`) moved from
Qwen-Image-Edit-2509 to Qwen-Image-2.1-Turbo: a 7.1B single-stream DiT (32 blocks,
width 4096) with a Qwen3-VL text encoder and an RGBA VAE (16x, 64 channels).

What stays: one transformer pass with the picture's own latent as the target at
t = 0.499 (bf16 rounds it to 0.498046875), prediction = input - velocity, a rank-128
LoRA on the DiT held in 4-bit NF4, a fine-tuned VAE decoder, masked L1 + DINOv3
spatial loss, batch 1, 30k steps.

What differs:
- Data: Hypersim at 768x576 (indoor rooms) mixed 50/50 with Arb-Objaverse at
  512x512 (single objects, seven lightings each, over black like Spellcast3D's
  views). Arb-Objaverse stored both its pictures through Blender 3.3's Filmic view;
  `data.py` undoes it with `blender33_filmic_inverse.txt` (made by
  `scripts/arb_objaverse/make_filmic_table.py`), shows the lit picture either as
  rendered or as plain sRGB, and leaves unlit pixels out of the loss.
- The VAE reads RGBA: the picture goes in opaque; the latent is the mode, as the
  Qwen-Image 2.1 pipeline encodes. Picture sides must be multiples of 32.
- The prompt is a text-only embedding made with 2.1's own text encoder
  (`scripts/qwen21/make_prompt_embeds.py`); the training never loads the encoder.
- LoRA targets are 2.1's layers (attention q/k/v/out, the SwiGLU proj/gate/out, the
  shared modulation, text/image input and output norm); the shared modulation,
  timestep embedder and input/output projections stay in bf16.
- `QwenImage21Decode` takes `checkpoint: true` to recompute the decoder in the
  backward pass: bit-identical output, decoder training memory at 768x576 from
  about 12 to 6.5 GiB (measured), for cards under 32 GB.

## On a Linux pod (one 32 GB card with bf16, about 1 TB of disk)

```bash
git clone -b spellcast3d <this repository> marigold-v2 && cd marigold-v2
bash setup/setup_env.sh && conda activate marigold-v2   # or a venv: torch 2.10 cu128, then pip install -e .
hf auth login                                           # after accepting the DINOv3 terms on Hugging Face
python scripts/download_assets.py --qwen21-albedo       # Qwen-Image-2.1-Turbo, DINOv3, Hypersim RGB (127 GB)
bash scripts/hypersim_albedo/download_and_preprocess_hypersim_albedo.sh   # ~500 GB
python scripts/arb_objaverse/prepare_arb_objaverse.py   # 15 zips: ~185 GB down, ~120 GB kept
python scripts/qwen21/make_prompt_embeds.py
python marigoldv2/script/train/train.py \
  --config marigoldv2/experiments/20261105_qwen21_albedo/training_albedo_qwen21.yaml \
  --output_dir output/train_runs --no_wandb
python scripts/qwen21/infer_albedo.py --image_dir <pictures> --output_dir output/albedo \
  --checkpoint output/train_runs/training_albedo_qwen21/checkpoint/checkpoint-5000
```

Only Objaverse objects under CC-BY or CC0 are kept; `credits.csv` next to the
data lists each one's author and page for attribution.

Checkpoints land every 5k steps; `trainables.safetensors` holds the LoRA
(`Diffuser.*`) and the decoder (`VAE.decoder.*`, `VAE.post_quant_conv.*`).
