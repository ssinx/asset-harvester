#!/usr/bin/env bash

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

# Rectified-flow post-training of the SparseViewDiT multiview diffusion model.
# Usage: bash scripts/run_post_training.sh <data_root> [output_dir] [num_gpus]
# See docs/post_training.md for details and all options.

set -euo pipefail

DATA_ROOT="${1:?Usage: run_post_training.sh <data_root> [output_dir] [num_gpus]}"
OUTPUT_DIR="${2:-outputs/post_training}"
NUM_GPUS="${3:-1}"

MULTI_GPU_FLAG=""
if [[ "$NUM_GPUS" -gt 1 ]]; then
    MULTI_GPU_FLAG="--multi_gpu"
fi

accelerate launch $MULTI_GPU_FLAG --num_processes "$NUM_GPUS" --mixed_precision bf16 \
    -m asset_harvester.multiview_diffusion.training.train \
    --data_root "$DATA_ROOT" \
    --checkpoint_path checkpoints/AH_multiview_diffusion.safetensors \
    --output_dir "$OUTPUT_DIR" \
    --train_batch_size 1 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --lr_warmup_steps 200 \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 3 \
    --validation_steps 250 \
    --resume_from_checkpoint latest \
    --report_to tensorboard
