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

"""CPU tests for merging LoRA adapters into base weights (merge_lora.py)."""

import pytest
import torch
from peft import LoraConfig

from asset_harvester.multiview_diffusion.models.sparseviewdit import SparseViewDiTTransformer2DModelNative
from asset_harvester.multiview_diffusion.training.merge_lora import merge_lora

_TINY_CONFIG = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=2,
    attention_head_dim=8,
    num_layers=1,
    num_cross_attention_heads=2,
    cross_attention_head_dim=8,
    cross_attention_dim=16,
    caption_channels=12,
    mlp_ratio=1.0,
    patch_size=1,
    sample_size=4,
    camera_emb=True,
    camera_emb_dim=17,
    brightness_emb=True,
    cond_on_rays=True,
    cond_on_mask=True,
)

_RANK, _ALPHA = 4, 8


@pytest.fixture
def base_and_adapter(tmp_path):
    """A tiny base model saved to disk plus a nonzero LoRA adapter for it."""
    torch.manual_seed(0)
    model = SparseViewDiTTransformer2DModelNative(**_TINY_CONFIG)
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)

    model.add_adapter(
        LoraConfig(
            r=_RANK,
            lora_alpha=_ALPHA,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    # lora_B is zero-initialized; randomize it so the merge is a real delta.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.1)
    lora_dir = tmp_path / "lora"
    model.save_lora_adapter(lora_dir)

    lora_params = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    return base_dir, lora_dir, lora_params


def test_merge_matches_manual_delta(base_and_adapter, tmp_path):
    base_dir, lora_dir, lora_params = base_and_adapter
    out_dir = tmp_path / "merged"
    changed = merge_lora(str(base_dir), str(lora_dir), str(out_dir), lora_scale=1.0)
    assert changed == len(lora_params) // 2  # one changed weight per (A, B) pair

    base = SparseViewDiTTransformer2DModelNative.from_pretrained(base_dir)
    merged = SparseViewDiTTransformer2DModelNative.from_pretrained(out_dir)

    # No adapter keys survive, and every adapted module carries W + (alpha/r) B A.
    assert all("lora" not in k for k in merged.state_dict())
    module = "transformer_blocks.0.attn1.to_q"
    a = lora_params[f"{module}.lora_A.default.weight"]
    b = lora_params[f"{module}.lora_B.default.weight"]
    expected = base.state_dict()[f"{module}.weight"] + (_ALPHA / _RANK) * (b @ a)
    torch.testing.assert_close(merged.state_dict()[f"{module}.weight"], expected, atol=1e-6, rtol=0)
    # Non-adapted weights are untouched.
    torch.testing.assert_close(
        merged.state_dict()["patch_embed.proj.weight"], base.state_dict()["patch_embed.proj.weight"]
    )


def test_merge_lora_scale(base_and_adapter, tmp_path):
    base_dir, lora_dir, lora_params = base_and_adapter
    merge_lora(str(base_dir), str(lora_dir), str(tmp_path / "half"), lora_scale=0.5)

    base = SparseViewDiTTransformer2DModelNative.from_pretrained(base_dir)
    half = SparseViewDiTTransformer2DModelNative.from_pretrained(tmp_path / "half")
    module = "transformer_blocks.0.attn1.to_v"
    a = lora_params[f"{module}.lora_A.default.weight"]
    b = lora_params[f"{module}.lora_B.default.weight"]
    expected = base.state_dict()[f"{module}.weight"] + 0.5 * (_ALPHA / _RANK) * (b @ a)
    torch.testing.assert_close(half.state_dict()[f"{module}.weight"], expected, atol=1e-6, rtol=0)


def test_merge_rejects_untrained_adapter(tmp_path):
    # A freshly-initialized adapter (lora_B == 0) merges to a zero delta, which
    # almost certainly means the wrong adapter was passed — must be an error.
    torch.manual_seed(0)
    model = SparseViewDiTTransformer2DModelNative(**_TINY_CONFIG)
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    model.add_adapter(LoraConfig(r=_RANK, lora_alpha=_ALPHA, target_modules=["to_q"]))
    lora_dir = tmp_path / "lora"
    model.save_lora_adapter(lora_dir)

    with pytest.raises(RuntimeError, match="changed no weights"):
        merge_lora(str(base_dir), str(lora_dir), str(tmp_path / "merged"))


def test_merge_missing_adapter_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_lora("unused", str(tmp_path), str(tmp_path / "out"))
