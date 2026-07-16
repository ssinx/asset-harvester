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

"""GPU integration test: dataset -> conditioning -> noising -> forward -> loss -> backward.

Requires a CUDA device and the released checkpoint; set AH_CHECKPOINT to the
multiview diffusion safetensors file (default: <repo>/checkpoints/).
"""

import os

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_ROOT = os.path.join(_REPO_ROOT, "data_samples", "rectified_AV_objects")
_CHECKPOINT = os.environ.get(
    "AH_CHECKPOINT", os.path.join(_REPO_ROOT, "checkpoints", "AH_multiview_diffusion.safetensors")
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.isfile(_CHECKPOINT), reason="checkpoint not found"),
    pytest.mark.skipif(not os.path.isdir(_DATA_ROOT), reason="data_samples not present"),
]


@pytest.fixture(scope="module")
def models():
    from asset_harvester.multiview_diffusion.utils.model_builder import get_models

    vae, cradio, cradio_proc, transformer = get_models(_CHECKPOINT, device="cuda", dtype=torch.bfloat16)
    # Training setup: fp32 VAE, trainable transformer.
    vae = vae.to(torch.float32)
    transformer.train().requires_grad_(True)
    return vae, cradio, cradio_proc, transformer


def test_train_step_end_to_end(models):
    from asset_harvester.multiview_diffusion.training.cond_processor import MultiviewCondProcessor
    from asset_harvester.multiview_diffusion.training.dataset import MultiviewObjectDataset
    from asset_harvester.multiview_diffusion.training.flow_match import (
        RectifiedFlowSchedule,
        broadcast_to_views,
        flow_matching_loss,
        velocity_target,
    )

    vae, cradio, cradio_proc, transformer = models
    device = torch.device("cuda")

    torch.manual_seed(0)
    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, conditioning_min_n=1)
    objects = [ds[0], ds[1]]

    processor = MultiviewCondProcessor(vae, cradio, cradio_proc, weight_dtype=torch.bfloat16)
    batch = processor(objects, device)

    total_views = sum(batch["views_per_object"])
    assert batch["clean_latents"].shape == (total_views, 32, 16, 16)
    assert batch["rays"].shape == (total_views, 6, 16, 16)
    assert batch["cond_mask"].shape == (total_views, 32, 16, 16)
    assert batch["camera_emb"].shape == (total_views, 17)
    assert batch["prompt_embeds"].shape[0] == len(objects)

    schedule = RectifiedFlowSchedule(num_train_timesteps=1000, flow_shift=1.0)
    t_obj = schedule.sample_timestep_indices(len(objects), "logit_normal")
    t_views = broadcast_to_views(t_obj, batch["views_per_object"])

    x0 = batch["clean_latents"].float()
    noise = torch.randn_like(x0)
    x_t, model_ts = schedule.add_noise(x0, noise, t_views)

    out = transformer(
        hidden_states=x_t.to(torch.bfloat16),
        encoder_hidden_states=batch["prompt_embeds"],
        timestep=model_ts.to(device),
        camera_emb=batch["camera_emb"],
        relative_brightness=batch["relative_brightness"],
        rays=batch["rays"],
        cond_mask=batch["cond_mask"],
        clean_images=batch["clean_latents"],
        x_seq_len=batch["views_per_object"],
        return_dict=False,
    )[0]

    target = velocity_target(x0, noise)
    loss = flow_matching_loss(out, target, cond_mask=batch["cond_mask"])
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()
    grads = [p.grad for p in transformer.parameters() if p.grad is not None]
    assert len(grads) > 0
    total_norm = torch.norm(torch.stack([g.detach().float().norm() for g in grads]))
    assert torch.isfinite(total_norm)
    transformer.zero_grad(set_to_none=True)

    # A pretrained model should predict velocity far better than chance:
    # the loss must be well below the variance of the target (~2 for v = eps - x0).
    assert loss.item() < 1.5, f"pretrained model loss suspiciously high: {loss.item()}"


def test_null_prompt_and_cfg_dropout(models):
    from asset_harvester.multiview_diffusion.training.cond_processor import MultiviewCondProcessor
    from asset_harvester.multiview_diffusion.training.dataset import MultiviewObjectDataset

    vae, cradio, cradio_proc, _ = models
    device = torch.device("cuda")

    processor = MultiviewCondProcessor(vae, cradio, cradio_proc, weight_dtype=torch.bfloat16, cfg_dropout_prob=1.0)
    null = processor.null_image_prompt(device)
    assert null.shape[0] == 1 and null.ndim == 3

    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, conditioning_min_n=1)
    batch = processor([ds[0]], device)
    # With dropout probability 1.0 the prompt must be the null embedding.
    torch.testing.assert_close(batch["prompt_embeds"], null.to(batch["prompt_embeds"].dtype))
