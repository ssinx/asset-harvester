# Post-Training (Rectified-Flow Fine-Tuning)

Asset Harvester supports post-training of the SparseViewDiT multiview diffusion model on
your own data. Fine-tuning uses the same rectified-flow objective the released checkpoint
was trained with, implemented on HuggingFace `diffusers` + `accelerate`:

- **Noising**: `x_t = (1 - σ_t)·x0 + σ_t·ε` with a `linear_flow` sigma schedule and an
  optional resolution-dependent timestep shift (`--flow_shift`, 1.0 at 512 px).
- **Timestep sampling**: logit-normal (SD3-style), one timestep per object shared by all
  of its views.
- **Target**: velocity `v = ε - x0`, plain MSE on target views (conditioning views are
  pasted back losslessly inside the model).
- **CFG**: the C-RADIO image prompt is replaced by a null (grey-image) embedding with
  probability `--cfg_dropout_prob`, and objects may draw 0 conditioning views
  (`--conditioning_min_n 0`), preserving classifier-free guidance after fine-tuning.

## Data

Training consumes the same sample directories that `ncore-parser` produces and
`run_inference.py` consumes: any folder tree whose leaves contain
`input_views/camera.json` plus `frame_*` / `mask_*` images. Samples need **at least 2
views** (≥1 conditioning + ≥1 target); single-view samples are skipped automatically.

```
my_dataset/
  <sample_id>/
    input_views/
      camera.json
      frame_00.jpeg  mask_00.png
      frame_01.jpeg  mask_01.png
      ...
```

At each step a random subset of views (between `--conditioning_min_n` and
`--conditioning_max_n`) is used as conditioning and the rest become denoising targets,
with the same preprocessing as inference (white-masked target backgrounds, Plucker rays,
relative cameras, optional symmetry augmentation).

### Dataset preparation

For post-training on a new domain (fleet, sensor rig, object category), we recommend
building a **mixed dataset** rather than fine-tuning on sparse real-world crops alone:

- **Synthetic data (SDG)**: include multi-view object samples rendered or exported in
  the `ncore-parser` layout above (posed images, masks, `camera.json`). In practice,
  mixing general Objaverse-style objects with your own **vehicle SDG** helps stabilize
  geometry, broaden viewpoint coverage, and reduce overfitting to the limited angles
  typical of AV logs.
- **Combining sources**: place all sample directories under one `--data_root`; training
  randomly draws objects from the union. Tune the SDG vs. real/self-distilled ratio for
  your domain.

### Self-distillation from AV logs

When your primary source is real driving data, a practical approach is to **expand and
curate** sparse observations before fine-tuning:

1. **NCore → cropped posed images** — parse clips with `ncore-parser` to obtain per-object
   crops. See
   [Step 1: NCore Parsing](end_to_end_example.md#step-1-ncore-parsing).
2. **MVD generation** — run multiview diffusion (`run_inference.py` / `run.sh`) on those
   crops to synthesize additional views around each object.
3. **Filter and pick pairs** — combine the original sparse views with generated views in
   per-object sample folders. Drop poor cases: heavy occlusion, broken masks, incoherent
   generations. Optionally score retained objects with the
   [benchmark](end_to_end_example.md#benchmark-evaluation) or manual review.


## Setup

Install the training extra on top of the standard setup:

```bash
pip install -e ".[multiview_diffusion,post_training]"
```

## Configuration

The source of truth for both the model architecture and the training
hyperparameters is the checked-in unified config
[`asset_harvester/multiview_diffusion/configs/sparseviewdit_512.yaml`](../asset_harvester/multiview_diffusion/configs/sparseviewdit_512.yaml):

- `model`: the released 512 px checkpoint's transformer architecture, shared by
  inference (`model_builder`) and training. When training starts from a
  diffusers-format directory (`--pretrained_model_path`), that directory's
  `config.json` takes precedence.
- `training`: default hyperparameters, loaded by `train.py` as argparse
  defaults.

Precedence (lowest to highest): packaged YAML → `--config /path/to/my.yaml`
(a customized copy; only the keys you change need to differ) → CLI flags.
Boolean options are negatable on the CLI (e.g. `--no-use_ema`) so a config
`true` can be switched off per run. Run-specific paths (`--data_root`,
`--output_dir`, `--checkpoint_path`, ...) are CLI-only by design.

Every run writes the fully resolved configuration to
`<output_dir>/training_config.yaml` for reproducibility.

## Launch

Single GPU:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    -m asset_harvester.multiview_diffusion.training.train \
    --data_root /path/to/my_dataset \
    --checkpoint_path checkpoints/AH_multiview_diffusion.safetensors \
    --output_dir outputs/post_training \
    --train_batch_size 1 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --max_train_steps 10000
```

Multi-GPU (single node):

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
    -m asset_harvester.multiview_diffusion.training.train ...
```

`scripts/run_post_training.sh` wraps the common case. The `mvdiff-train` console script
is equivalent to `python -m asset_harvester.multiview_diffusion.training.train` (prefer
`accelerate launch` so mixed precision and multi-GPU are handled for you).

Useful options (see `--help` for all):

| Option | Default | Notes |
| --- | --- | --- |
| `--config` | packaged YAML | Unified config supplying all defaults below |
| `--checkpoint_path` | — | Released `.safetensors` file to start from |
| `--pretrained_model_path` | — | Alternative: diffusers-format dir (e.g. a previous fine-tune) |
| `--train_batch_size` | 1 | Objects (not views) per GPU; an object contributes up to `--max_views` views |
| `--learning_rate` | 1e-5 | AdamW, constant schedule with warmup |
| `--max_grad_norm` | 0.1 | Matches pretraining |
| `--lora_rank` | 0 | > 0 trains a LoRA adapter (attention q/k/v/out) instead of all weights |
| `--use_ema` | off | Maintain an EMA copy used for validation and the final save |
| `--flow_shift` | 1.0 | Set 3–4 when fine-tuning at 1024 px |
| `--cfg_dropout_prob` | 0.1 | Null-prompt probability (matches pretraining) |
| `--resume_from_checkpoint` | — | Path or `latest` |
| `--report_to` | tensorboard | `tensorboard`, `wandb`, or `all` |

## Monitoring

Loss, learning rate, and gradient norm are logged to the tracker; every
`--validation_steps` the current model generates the target views of a few validation
objects and logs `[generated | ground-truth | conditioning]` grids (also saved under
`<output_dir>/validation/`).

## Checkpoints and inference

- Every `--checkpointing_steps` a full training state (`checkpoint-<step>/`) is saved for
  resuming: transformer in diffusers format plus optimizer/scheduler/RNG state.
- The final model is written to `<output_dir>/transformer/` (`config.json` +
  safetensors). LoRA runs write `<output_dir>/transformer_lora/` instead.
- Fine-tuned checkpoints plug directly into inference:

```bash
python run_inference.py \
    --diffusion_checkpoint outputs/post_training/transformer \
    --data_root /path/to/samples ...
```

### Using LoRA checkpoints at inference

LoRA runs produce an adapter, not full weights. Merge it into the base model first
(`W + lora_scale * (alpha / r) * B @ A`); the result is a plain diffusers-format
transformer the unchanged inference pipeline loads:

```bash
mvdiff-merge-lora \
    --base_checkpoint checkpoints/AH_multiview_diffusion.safetensors \
    --lora_dir outputs/post_training/transformer_lora \
    --output_dir outputs/post_training/transformer_merged

python run_inference.py \
    --diffusion_checkpoint outputs/post_training/transformer_merged \
    --data_root /path/to/samples ...
```

`--base_checkpoint` accepts either checkpoint format and must be the same weights the
adapter was trained on. `--lora_scale` (default 1.0) attenuates or amplifies the
adapter's effect. Adapters saved inside training states
(`checkpoint-<step>/transformer_lora`) merge the same way.

## Tips

- The released model was pretrained with the CAME optimizer at lr 1e-4; for fine-tuning,
  AdamW at 1e-5–2e-5 with `--max_grad_norm 0.1` is a stable starting point.
- For small datasets, raise `--dataset_repeats` so an "epoch" is long enough, and
  consider `--lora_rank 16` to limit drift from the base model.
- VRAM: full fine-tuning with one object per GPU fits
  comfortably in 80 GB; gradient checkpointing may be needed for small GPUs.
