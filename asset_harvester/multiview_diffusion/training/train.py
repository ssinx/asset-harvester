# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rectified-flow post-training for the SparseViewDiT multiview diffusion model.

Fine-tunes the released checkpoint (or a previously fine-tuned diffusers-format
checkpoint) on ncore-parser sample directories with the same objective the
model was pretrained with: logit-normal timestep sampling, linear-flow noising,
and a velocity MSE on target views.

Launch (single GPU):

    accelerate launch -m asset_harvester.multiview_diffusion.training.train \\
        --data_root /path/to/parsed_samples \\
        --checkpoint_path checkpoints/AH_multiview_diffusion.safetensors \\
        --output_dir outputs/post_training

Multi-GPU: ``accelerate launch --multi_gpu --num_processes 8 -m ...``.
"""

import argparse
import logging
import math
import os
import shutil

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderDC
from diffusers.optimization import get_scheduler
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file

from ..configs import load_unified_config
from ..models.sparseviewdit import SparseViewDiTTransformer2DModelNative
from ..pipelines import SparseViewDiTPipeline
from ..utils.convert_checkpoint import convert_sana_ms_to_diffusers
from ..utils.model_builder import get_c_radio
from .cond_processor import MultiviewCondProcessor, make_validation_grid
from .dataset import MultiviewObjectDataset, build_dataloader
from .flow_match import RectifiedFlowSchedule, broadcast_to_views, flow_matching_loss, velocity_target

logger = get_logger(__name__)


def parse_args(input_args=None):
    """Parse CLI args with defaults from the unified YAML config.

    Precedence (lowest to highest): packaged sparseviewdit_512.yaml ->
    ``--config`` file (may override a subset of training keys) -> CLI flags.
    Run-specific paths (--data_root, --output_dir, ...) are CLI-only.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=None,
        help="Unified YAML config (model + training defaults). Defaults to the packaged sparseviewdit_512.yaml",
    )
    pre_args, _ = pre.parse_known_args(input_args)

    base_config = load_unified_config()
    model_config = base_config["model"]
    training_defaults = dict(base_config["training"])
    if pre_args.config is not None:
        user_config = load_unified_config(pre_args.config)
        model_config = user_config["model"]
        training_defaults.update(user_config["training"])

    parser = argparse.ArgumentParser(description="Rectified-flow post-training for SparseViewDiT", parents=[pre])

    # Run-specific paths (CLI-only, never in the YAML config)
    parser.add_argument("--data_root", type=str, required=True, help="Root dir of ncore-parser samples")
    parser.add_argument("--val_data_root", type=str, default=None, help="Validation samples (default: data_root)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Released single-file checkpoint (AH_multiview_diffusion.safetensors)",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Diffusers-format transformer directory (e.g. a previous fine-tune output)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/post_training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help='Path or "latest"')

    # Everything below defaults from the unified config's `training` section.
    # Data / preprocessing
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--max_views", type=int)
    parser.add_argument("--conditioning_min_n", type=int, help="0 supplies unconditional samples for CFG")
    parser.add_argument("--conditioning_max_n", type=int)
    parser.add_argument("--symmetry_aug_p", type=float)
    parser.add_argument("--dataset_repeats", type=int, help="Virtual epoch multiplier for small datasets")
    parser.add_argument("--dataloader_num_workers", type=int)

    # Model options
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lora_rank", type=int, help="LoRA rank; 0 trains all parameters")
    parser.add_argument("--lora_alpha", type=int, help="Defaults to lora_rank")

    # Flow matching
    parser.add_argument("--num_train_timesteps", type=int)
    parser.add_argument("--flow_shift", type=float, help="1.0 at 512px; 3-4 at 1024px")
    parser.add_argument("--weighting_scheme", type=str, choices=["logit_normal", "uniform"])
    parser.add_argument("--logit_mean", type=float)
    parser.add_argument("--logit_std", type=float)
    parser.add_argument("--cfg_dropout_prob", type=float)

    # Optimization
    parser.add_argument("--train_batch_size", type=int, help="Objects (not views) per device")
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_scheduler", type=str)
    parser.add_argument("--lr_warmup_steps", type=int)
    parser.add_argument("--adam_beta1", type=float)
    parser.add_argument("--adam_beta2", type=float)
    parser.add_argument("--adam_weight_decay", type=float)
    parser.add_argument("--adam_epsilon", type=float)
    parser.add_argument("--use_8bit_adam", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ema_decay", type=float)

    # Checkpointing / logging / validation
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpointing_steps", type=int)
    parser.add_argument("--checkpoints_total_limit", type=int)
    parser.add_argument("--validation_steps", type=int)
    parser.add_argument("--num_validation_samples", type=int)
    parser.add_argument("--validation_inference_steps", type=int)
    parser.add_argument("--validation_guidance_scale", type=float)
    parser.add_argument("--report_to", type=str, choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--tracker_project_name", type=str)
    parser.add_argument("--log_interval", type=int)

    known_dests = {action.dest for action in parser._actions}
    unknown_keys = set(training_defaults) - known_dests
    if unknown_keys:
        parser.error(f"Unknown keys in the config's training section: {sorted(unknown_keys)}")
    parser.set_defaults(**training_defaults)

    args = parser.parse_args(input_args)

    if (args.checkpoint_path is None) == (args.pretrained_model_path is None):
        parser.error("Provide exactly one of --checkpoint_path or --pretrained_model_path")
    if args.use_ema and args.lora_rank > 0:
        parser.error("--use_ema is not supported together with LoRA (--lora_rank > 0)")
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    args.model_config = model_config
    return args


def load_transformer(args) -> SparseViewDiTTransformer2DModelNative:
    """Load initial weights from the release single-file format or a diffusers dir."""
    if args.pretrained_model_path is not None:
        logger.info(f"Loading diffusers-format transformer from {args.pretrained_model_path}")
        return SparseViewDiTTransformer2DModelNative.from_pretrained(args.pretrained_model_path)

    logger.info(f"Loading release checkpoint from {args.checkpoint_path}")
    transformer = SparseViewDiTTransformer2DModelNative(**args.model_config)
    state_dict = load_file(args.checkpoint_path, device="cpu")
    hidden_size = args.model_config["num_attention_heads"] * args.model_config["attention_head_dim"]
    state_dict = convert_sana_ms_to_diffusers(state_dict, hidden_size=hidden_size)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    logger.info(f"Checkpoint loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    return transformer


def add_lora_adapter(transformer, rank: int, alpha: int):
    from peft import LoraConfig

    transformer.requires_grad_(False)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)


@torch.no_grad()
def log_validation(args, accelerator, transformer, vae, image_encoder, image_processor, val_dataset, global_step):
    """Generate target views for validation objects and log grids of
    [generated | ground truth | conditioning] rows to the trackers."""
    logger.info(f"Running validation at step {global_step}")
    transformer = accelerator.unwrap_model(transformer)
    was_training = transformer.training
    transformer.eval()

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="scaled_linear",
        prediction_type="flow_prediction",
        flow_shift=args.flow_shift,
        use_flow_sigmas=True,
    )
    pipeline = SparseViewDiTPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        image_encoder=image_encoder,
        image_processor=image_processor,
    )
    pipeline.set_progress_bar_config(disable=True)

    grids = []
    for i in range(min(args.num_validation_samples, len(val_dataset))):
        data_dict = val_dataset[i]
        output = pipeline(
            data_dict=data_dict,
            num_inference_steps=args.validation_inference_steps,
            guidance_scale=args.validation_guidance_scale,
            flow_shift=args.flow_shift,
            output_type="pil",
        )
        generated = output["images"]

        def to_pil(t):
            from PIL import Image

            arr = ((t + 1.0) / 2.0 * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            return Image.fromarray(arr)

        n_target = data_dict.n_target
        gt = [to_pil(data_dict.x[j]) for j in range(n_target)]
        cond = [to_pil(data_dict.x[j]) for j in range(n_target, data_dict.x.shape[0])]
        row_len = max(len(generated), 1)
        grids.append(make_validation_grid(generated + gt + cond, views_per_row=row_len))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, grid in enumerate(grids):
                tracker.writer.add_image(f"validation/sample_{i}", grid, global_step, dataformats="HWC")
        elif tracker.name == "wandb":
            import wandb

            tracker.log(
                {"validation": [wandb.Image(g, caption=f"sample {i}") for i, g in enumerate(grids)]},
                step=global_step,
            )

    vis_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(vis_dir, exist_ok=True)
    from PIL import Image

    for i, grid in enumerate(grids):
        Image.fromarray(grid).save(os.path.join(vis_dir, f"step{global_step:07d}_sample{i}.jpg"), quality=92)

    if was_training:
        transformer.train()
    del pipeline
    torch.cuda.empty_cache()


def main(args=None):
    if args is None:
        args = parse_args()

    logging_dir = os.path.join(args.output_dir, "logs")
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Persist the fully resolved configuration (packaged defaults + --config
        # overrides + CLI flags) next to the checkpoints for reproducibility.
        resolved = {
            "model": dict(args.model_config),
            "training": {k: v for k, v in vars(args).items() if k not in ("model_config", "config")},
        }
        with open(os.path.join(args.output_dir, "training_config.yaml"), "w") as f:
            yaml.safe_dump(resolved, f, sort_keys=True)

    device = accelerator.device
    weight_dtype = torch.float32  # params stay fp32; autocast handles bf16/fp16 compute

    # ---------------- Models ----------------
    # VAE always in fp32 (Sana requirement); frozen.
    vae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    vae.requires_grad_(False)
    vae.to(device, dtype=torch.float32).eval()

    image_encoder, image_processor = get_c_radio(device=device)
    image_encoder.requires_grad_(False)

    transformer = load_transformer(args)
    if args.lora_rank > 0:
        add_lora_adapter(transformer, args.lora_rank, args.lora_alpha)
    else:
        transformer.requires_grad_(True)
    transformer.to(device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            transformer.parameters(),
            decay=args.ema_decay,
            model_cls=SparseViewDiTTransformer2DModelNative,
            model_config=transformer.config,
        )
        ema_model.to(device)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    # ---------------- Checkpoint hooks ----------------
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if ema_model is not None:
                ema_model.save_pretrained(os.path.join(output_dir, "transformer_ema"))
            for model in models:
                m = unwrap_model(model)
                if args.lora_rank > 0:
                    m.save_lora_adapter(os.path.join(output_dir, "transformer_lora"))
                else:
                    m.save_pretrained(os.path.join(output_dir, "transformer"))
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        if ema_model is not None:
            loaded = EMAModel.from_pretrained(
                os.path.join(input_dir, "transformer_ema"), model_cls=SparseViewDiTTransformer2DModelNative
            )
            ema_model.load_state_dict(loaded.state_dict())
            ema_model.to(device)
            del loaded
        for _ in range(len(models)):
            model = models.pop()
            m = unwrap_model(model)
            if args.lora_rank > 0:
                # Copy the checkpointed adapter into the existing LoRA parameters
                # in place (keys in the file are adapter-name-stripped:
                # "X.lora_A.weight" vs parameter "X.lora_A.default.weight").
                # In-place copy keeps the optimizer's parameter references valid.
                adapter_state = load_file(
                    os.path.join(input_dir, "transformer_lora", "pytorch_lora_weights.safetensors")
                )
                named_params = dict(m.named_parameters())
                for key, value in adapter_state.items():
                    param_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                        ".lora_B.weight", ".lora_B.default.weight"
                    )
                    named_params[param_key].data.copy_(value)
            else:
                loaded = SparseViewDiTTransformer2DModelNative.from_pretrained(input_dir, subfolder="transformer")
                m.register_to_config(**loaded.config)
                m.load_state_dict(loaded.state_dict())
                del loaded

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ---------------- Data ----------------
    train_dataset = MultiviewObjectDataset(
        data_root=args.data_root,
        resolution=args.resolution,
        max_views=args.max_views,
        conditioning_min_n=args.conditioning_min_n,
        conditioning_max_n=args.conditioning_max_n,
        symmetry_aug_p=args.symmetry_aug_p,
        repeats=args.dataset_repeats,
    )
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        seed=args.seed,
    )
    # Validation: deterministic conditioning (max available views), no augmentation.
    val_dataset = MultiviewObjectDataset(
        data_root=args.val_data_root or args.data_root,
        resolution=args.resolution,
        max_views=args.max_views,
        conditioning_mode="n",
        conditioning_min_n=1,
        conditioning_max_n=args.conditioning_max_n,
        symmetry_aug_p=0.0,
    )

    processor = MultiviewCondProcessor(
        vae, image_encoder, image_processor, weight_dtype=weight_dtype, cfg_dropout_prob=args.cfg_dropout_prob
    )
    schedule = RectifiedFlowSchedule(num_train_timesteps=args.num_train_timesteps, flow_shift=args.flow_shift)

    # ---------------- Optimizer ----------------
    params = [p for p in transformer.parameters() if p.requires_grad]
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items() if isinstance(v, (bool, int, float, str)) or v is None}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running post-training *****")
    logger.info(f"  Num samples = {len(train_dataset)} ({len(train_dataset.sample_dirs)} unique)")
    logger.info(f"  Total objects/optimizer-step = {total_batch_size}")
    logger.info(f"  Max optimizer steps = {args.max_train_steps}")
    logger.info(f"  Trainable params = {sum(p.numel() for p in params) / 1e6:.1f}M")

    # ---------------- Resume ----------------
    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            path = os.path.join(args.output_dir, sorted(dirs, key=lambda d: int(d.split("-")[1]))[-1]) if dirs else None
        else:
            path = args.resume_from_checkpoint
        if path is None or not os.path.isdir(path):
            logger.info("No checkpoint found; starting fresh")
        else:
            logger.info(f"Resuming from {path}")
            accelerator.load_state(path)
            global_step = int(os.path.basename(os.path.normpath(path)).split("-")[1])

    # ---------------- Training loop ----------------
    transformer.train()
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    first_epoch = global_step // max(num_update_steps_per_epoch, 1)
    done = False

    for epoch in range(first_epoch, 10**9):
        if done:
            break
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                inputs = processor(batch, device)

                num_objects = len(inputs["views_per_object"])
                t_obj = schedule.sample_timestep_indices(
                    num_objects, args.weighting_scheme, args.logit_mean, args.logit_std
                )
                t_views = broadcast_to_views(t_obj, inputs["views_per_object"])

                x0 = inputs["clean_latents"]
                noise = torch.randn_like(x0)
                x_t, model_ts = schedule.add_noise(x0, noise, t_views)

                model_out = transformer(
                    hidden_states=x_t,
                    encoder_hidden_states=inputs["prompt_embeds"],
                    timestep=model_ts.to(device),
                    camera_emb=inputs["camera_emb"],
                    relative_brightness=inputs["relative_brightness"],
                    rays=inputs["rays"],
                    cond_mask=inputs["cond_mask"],
                    clean_images=x0 * inputs["cond_mask"],
                    x_seq_len=inputs["views_per_object"],
                    return_dict=False,
                )[0]

                target = velocity_target(x0, noise)
                loss = flow_matching_loss(model_out, target, cond_mask=inputs["cond_mask"])

                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if ema_model is not None:
                    ema_model.step(transformer.parameters())

                if global_step % args.log_interval == 0:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    if grad_norm is not None:
                        logs["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                    accelerator.log(logs, step=global_step)
                    logger.info(f"step {global_step}: " + ", ".join(f"{k}={v:.5g}" for k, v in logs.items()))

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")),
                            key=lambda d: int(d.split("-")[1]),
                        )
                        for d in ckpts[: max(0, len(ckpts) - args.checkpoints_total_limit + 1)]:
                            logger.info(f"Removing old checkpoint {d}")
                            shutil.rmtree(os.path.join(args.output_dir, d))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    if ema_model is not None:
                        ema_model.store(transformer.parameters())
                        ema_model.copy_to(transformer.parameters())
                    log_validation(
                        args, accelerator, transformer, vae, image_encoder, image_processor, val_dataset, global_step
                    )
                    if ema_model is not None:
                        ema_model.restore(transformer.parameters())

                if global_step >= args.max_train_steps:
                    done = True
                    break

    # ---------------- Final save ----------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if ema_model is not None:
            ema_model.copy_to(transformer.parameters())
        if args.lora_rank > 0:
            transformer.save_lora_adapter(os.path.join(args.output_dir, "transformer_lora"))
        else:
            transformer.save_pretrained(os.path.join(args.output_dir, "transformer"))
        logger.info(f"Final model saved to {args.output_dir}")
        log_validation(args, accelerator, transformer, vae, image_encoder, image_processor, val_dataset, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
