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

"""save_pretrained/from_pretrained round-trip for the multiview transformer.

Regression test: the constructor widens in_channels for ray/mask conditioning
before calling the base class, whose register_to_config used to persist the
widened value — making every from_pretrained widen the input a second time
(39 -> 46 channels) and fail to load the saved weights.
"""

import torch

from asset_harvester.multiview_diffusion.models.sparseviewdit import SparseViewDiTTransformer2DModelNative

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


def test_save_load_round_trip(tmp_path):
    model = SparseViewDiTTransformer2DModelNative(**_TINY_CONFIG)
    # Config must record the logical latent channels, not the widened width.
    assert model.config.in_channels == 8
    assert model.patch_embed.proj.weight.shape[1] == 8 + 6 + 1  # latents + rays + mask

    model.save_pretrained(tmp_path)
    reloaded = SparseViewDiTTransformer2DModelNative.from_pretrained(tmp_path)

    assert reloaded.config.in_channels == 8
    assert reloaded.patch_embed.proj.weight.shape == model.patch_embed.proj.weight.shape
    torch.testing.assert_close(reloaded.patch_embed.proj.weight, model.patch_embed.proj.weight)
    for (name_a, a), (_name_b, b) in zip(model.state_dict().items(), reloaded.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, msg=f"mismatch in {name_a}")
