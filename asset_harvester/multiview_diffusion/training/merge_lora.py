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
Merge a LoRA adapter from post-training into the base transformer weights.

Produces a plain diffusers-format transformer directory (config.json +
safetensors, no adapter modules) that the unchanged inference pipeline loads
directly:

    mvdiff-merge-lora \\
        --base_checkpoint checkpoints/AH_multiview_diffusion.safetensors \\
        --lora_dir outputs/post_training/transformer_lora \\
        --output_dir outputs/post_training/transformer_merged

    python run_inference.py --diffusion_checkpoint outputs/post_training/transformer_merged ...

The merged weight of every adapted module is ``W + lora_scale * (alpha / r) * B @ A``.
"""

import argparse
import os

import torch

from ..utils.model_builder import load_transformer_checkpoint

LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def merge_lora(
    base_checkpoint: str,
    lora_dir: str,
    output_dir: str,
    lora_scale: float = 1.0,
) -> int:
    """Merge the adapter at ``lora_dir`` into ``base_checkpoint``; returns the
    number of weight tensors the merge changed."""
    if not os.path.isfile(os.path.join(lora_dir, LORA_WEIGHT_NAME)):
        raise FileNotFoundError(f"No {LORA_WEIGHT_NAME} in {lora_dir}")

    print(f"Loading base transformer from {base_checkpoint}")
    transformer = load_transformer_checkpoint(base_checkpoint)
    base_state = {k: v.detach().clone() for k, v in transformer.state_dict().items()}

    print(f"Merging LoRA from {lora_dir} (scale={lora_scale})")
    # prefix=None: adapters saved by save_lora_adapter carry unprefixed keys;
    # the default prefix="transformer" silently matches nothing.
    transformer.load_lora_adapter(lora_dir, weight_name=LORA_WEIGHT_NAME, prefix=None)
    transformer.fuse_lora(lora_scale=lora_scale)
    transformer.unload_lora()

    merged_state = transformer.state_dict()
    leftover = [k for k in merged_state if "lora" in k.lower()]
    if leftover:
        raise RuntimeError(f"Adapter modules were not fully unloaded: {leftover[:5]}")
    if set(merged_state) != set(base_state):
        raise RuntimeError("Merged state dict keys differ from the base model's")

    changed = sum(1 for k in base_state if not torch.equal(base_state[k], merged_state[k]))
    if changed == 0:
        raise RuntimeError(
            "Merge changed no weights — the adapter does not match this base model (or was saved untrained)"
        )
    print(f"Merge changed {changed}/{len(base_state)} weight tensors")

    transformer.save_pretrained(output_dir)
    print(f"Merged transformer saved to {output_dir}")
    return changed


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into SparseViewDiT base weights")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        required=True,
        help="Base weights: released .safetensors file or a diffusers-format directory",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help=f"Directory containing {LORA_WEIGHT_NAME} (e.g. <run>/transformer_lora)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the merged transformer")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Strength multiplier for the adapter delta")
    return parser.parse_args(input_args)


def main(args=None):
    if args is None:
        args = parse_args()
    merge_lora(args.base_checkpoint, args.lora_dir, args.output_dir, args.lora_scale)


if __name__ == "__main__":
    main()
