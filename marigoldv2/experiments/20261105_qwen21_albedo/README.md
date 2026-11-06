# Albedo on Qwen-Image 2.1 (Spellcast3D)

The Marigold V2 albedo recipe (`20260803_qwen_albedo`) moved from
Qwen-Image-Edit-2509 to Qwen-Image-2.1-Turbo: a 7.1B single-stream DiT (32 blocks,
width 4096) with a Qwen3-VL text encoder and an RGBA VAE (16x, 64 channels).

What stays: one transformer pass with the picture's own latent as the target at
t = 0.499 (bf16 rounds it to 0.498046875), prediction = input - velocity, a rank-128
LoRA on the DiT held in 4-bit NF4, a fine-tuned VAE decoder, masked L1 + DINOv3
spatial loss, batch 1, 30k steps.

What differs:
- Data: Hypersim at its native 1024x768 (indoor rooms; Marigold V2 halves it to
  768x576) mixed 50/50 with Arb-Objaverse at
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
- A training step waits for the GPU once: the trainer reads the loss terms and the
  gradient norm back in one copy, `transformer.py` builds the DiT's sequence layout
  once per picture size instead of from GPU masks on every forward (bit-identical),
  and constants are uploaded once.
- `QwenImage21Decode` recomputes the decoder in the backward pass (`checkpoint:
  true`): bit-identical output. At 1024x768 a step otherwise peaks at 28.6 GiB on a
  32 GB RTX 5090 (18.5 GiB with it, 2.06 vs 1.95 s a step; DINOv3 not included).
- `compile_blocks: true` runs each DiT block through torch.compile: about 20 % off a
  step, losses within bf16 rounding of eager. Each new picture size compiles once
  (~20 s). RTX 5090, without DINOv3: Hypersim 1024x768 1.67 s / 18.5 GiB; Arb-Objaverse
  512, 768, 1024 (drawn at random) 0.58 s / 10.5, 1.21 s / 15.5, 2.24 s / 22.5 GiB.

## On a Linux pod (one 32 GB card with bf16, about 1 TB of disk)

```bash
git clone -b spellcast3d <this repository> marigold-v2 && cd marigold-v2
bash setup/setup_env.sh && conda activate marigold-v2   # or a venv: torch 2.10 cu128, then pip install -e .
hf auth login                                           # after accepting the DINOv3 terms on Hugging Face
export DEPTH_DOWNLOAD_FILE_TIMEOUT=7200                 # a killed transfer starts over: give each 4.5 GB shard time
python scripts/download_assets.py --qwen21-albedo       # Qwen-Image-2.1-Turbo (32.5 GB), DINOv3, Hypersim RGB (127 GB)
bash scripts/hypersim_albedo/download_and_preprocess_hypersim_albedo.sh   # checks for 500 GB free, writes ~275 GB
python scripts/arb_objaverse/prepare_arb_objaverse.py   # 15 zips: ~150 GB down, ~96 GB kept (~6.3k objects)
python scripts/qwen21/make_prompt_embeds.py
python marigoldv2/script/train/train.py \
  --config marigoldv2/experiments/20261105_qwen21_albedo/training_albedo_qwen21.yaml \
  --output_dir output/train_runs --no_wandb
python scripts/qwen21/infer_albedo.py --image_dir <pictures> --output_dir output/albedo \
  --checkpoint output/train_runs/training_albedo_qwen21/checkpoint/checkpoint-5000
```

Only Objaverse objects under CC-BY or CC0 are kept; `credits.csv` next to the
data lists each one's author and page for attribution.

Checkpoints land every 5k steps (plus `checkpoint-best` and `checkpoint-latest` at
every 2k-step validation, each about 2.5 GB: `trainables.safetensors` with the LoRA
(`Diffuser.*`) and the decoder (`VAE.decoder.*`, `VAE.post_quant_conv.*`), the
AdamW8bit state, the scheduler, the samplers and the random states). The validation
is 400 pictures (200 Hypersim, 200 Arb-Objaverse: one HDRI and one point-light
picture per object), so budget a few minutes for each.

## First run (2026-11-06, one RTX 5090, 14 h)

- Validation PSNR at 2k / 10k / 20k / 30k steps: Hypersim 19.1 / 22.8 / 24.1 / 24.1,
  Arb-Objaverse 20.1 / 21.7 / 22.3 / 22.2. Flat after 20k, where the learning rate
  reaches its floor: 20k steps would do. Best mean MAE at 24k. The Hypersim
  validation frames are held out from scenes that are also trained on, so they are
  easier than Marigold V2's test-split numbers; Arb-Objaverse is split by object.
- Trained on Turbo, the add-on also runs on the full Qwen-Image 2.1
  (`--model_dir .../Qwen-Image-2.1`): the transformers differ by 0.24 % (the VAE is
  the same), and on 1024-px object views the albedo stays within 36-45 dB PSNR of
  Turbo's.
- Arb-Objaverse label noise, left in: about 1 % of objects look the same under all
  seven lights (unlit assets) and about 1.6 % have an albedo far darker than their
  renders. Its albedo is the base colour (metals keep theirs, per the `orm` maps).
