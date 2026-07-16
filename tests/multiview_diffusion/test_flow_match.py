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

"""CPU unit tests for rectified-flow training utilities.

The reference values encode the internal training implementation
(linear_flow betas -> sigmas -> shift -> velocity MSE) so a regression here
means a departure from the schedule the released checkpoint was trained with.
"""

import numpy as np
import pytest
import torch

from asset_harvester.multiview_diffusion.training.flow_match import (
    RectifiedFlowSchedule,
    broadcast_to_views,
    flow_matching_loss,
    velocity_target,
)


class TestRectifiedFlowSchedule:
    def test_sigma_endpoints_no_shift(self):
        sched = RectifiedFlowSchedule(num_train_timesteps=1000, flow_shift=1.0)
        assert sched.sigmas[0].item() == pytest.approx(0.0, abs=1e-12)
        assert sched.sigmas[-1].item() == pytest.approx(0.999, abs=1e-12)
        assert torch.all(sched.sigmas[1:] > sched.sigmas[:-1])

    def test_matches_reference_linear_flow_schedule(self):
        # Reference: betas = linspace(1.0, 0.001, N); sigmas = 1 - betas.
        n = 1000
        betas = np.linspace(1.0, 0.001, n, dtype=np.float64)
        expected = 1.0 - betas
        sched = RectifiedFlowSchedule(num_train_timesteps=n, flow_shift=1.0)
        np.testing.assert_allclose(sched.sigmas.numpy(), expected, rtol=0, atol=1e-14)

    def test_shift_formula(self):
        n, shift = 1000, 3.0
        base = RectifiedFlowSchedule(num_train_timesteps=n, flow_shift=1.0).sigmas.numpy()
        shifted = RectifiedFlowSchedule(num_train_timesteps=n, flow_shift=shift).sigmas.numpy()
        expected = shift * base / (1.0 + (shift - 1.0) * base)
        np.testing.assert_allclose(shifted, expected, rtol=0, atol=1e-14)
        # Shift preserves the endpoints and pushes interior sigmas toward noise.
        assert shifted[0] == pytest.approx(0.0, abs=1e-12)
        assert shifted[500] > base[500]

    def test_model_timesteps_are_scaled_sigmas(self):
        sched = RectifiedFlowSchedule(num_train_timesteps=1000, flow_shift=1.0)
        np.testing.assert_allclose(sched.timesteps.numpy(), sched.sigmas.numpy() * 1000, atol=1e-12)

    def test_logit_normal_sampling_range_and_center(self):
        sched = RectifiedFlowSchedule()
        gen = torch.Generator().manual_seed(0)
        idx = sched.sample_timestep_indices(20000, "logit_normal", 0.0, 1.0, generator=gen)
        assert idx.dtype == torch.long
        assert idx.min() >= 0 and idx.max() <= 999
        # sigmoid of a standard normal is symmetric around 0.5.
        assert abs(idx.float().mean().item() - 500.0) < 10.0

    def test_uniform_sampling(self):
        sched = RectifiedFlowSchedule()
        gen = torch.Generator().manual_seed(0)
        idx = sched.sample_timestep_indices(10000, "uniform", generator=gen)
        assert idx.min() >= 0 and idx.max() <= 999

    def test_add_noise_interpolation(self):
        sched = RectifiedFlowSchedule(num_train_timesteps=1000, flow_shift=1.0)
        x0 = torch.randn(4, 32, 16, 16)
        noise = torch.randn_like(x0)
        # sigma[0] == 0 -> x_t == x0 exactly.
        t0 = torch.zeros(4, dtype=torch.long)
        x_t, ts = sched.add_noise(x0, noise, t0)
        torch.testing.assert_close(x_t, x0)
        torch.testing.assert_close(ts, torch.zeros(4))
        # Interior index: exact linear interpolation.
        t = torch.full((4,), 500, dtype=torch.long)
        sigma = sched.sigmas[500].item()
        x_t, ts = sched.add_noise(x0, noise, t)
        torch.testing.assert_close(x_t, (1 - sigma) * x0 + sigma * noise)
        assert ts[0].item() == pytest.approx(sigma * 1000, rel=1e-6)

    def test_add_noise_rejects_unbroadcast_indices(self):
        sched = RectifiedFlowSchedule()
        x0 = torch.randn(6, 32, 16, 16)
        with pytest.raises(ValueError, match="broadcast"):
            sched.add_noise(x0, torch.randn_like(x0), torch.zeros(2, dtype=torch.long))


def test_broadcast_to_views():
    per_object = torch.tensor([10, 20, 30])
    out = broadcast_to_views(per_object, [2, 1, 3])
    torch.testing.assert_close(out, torch.tensor([10, 10, 20, 30, 30, 30]))
    with pytest.raises(ValueError):
        broadcast_to_views(per_object, [1, 2])


def test_velocity_target():
    x0 = torch.randn(3, 4)
    noise = torch.randn(3, 4)
    torch.testing.assert_close(velocity_target(x0, noise), noise - x0)


class TestFlowMatchingLoss:
    def test_unmasked_equals_mse(self):
        pred = torch.randn(5, 32, 16, 16)
        target = torch.randn_like(pred)
        expected = torch.nn.functional.mse_loss(pred, target)
        torch.testing.assert_close(flow_matching_loss(pred, target), expected)

    def test_masked_gradients_match_reference_unmasked_loss(self):
        # Reference behaviour: mean over ALL views of the per-view MSE, where the
        # model output on conditioning views is a pasted-back constant (no grad).
        # Our masked loss must produce identical gradients on the raw output.
        torch.manual_seed(0)
        n_views, n_target = 6, 4
        raw = torch.randn(n_views, 32, 8, 8, requires_grad=True)
        clean = torch.randn(n_views, 32, 8, 8)
        target = torch.randn(n_views, 32, 8, 8)
        cond_mask = torch.cat([torch.zeros(n_target, 32, 8, 8), torch.ones(n_views - n_target, 32, 8, 8)])

        pasted = cond_mask * clean + (1 - cond_mask) * raw

        # Reference: plain mean_flat over everything, including cond-view residual.
        ref_loss = ((pasted - target) ** 2).mean(dim=(1, 2, 3)).mean()
        (ref_grad,) = torch.autograd.grad(ref_loss, raw, retain_graph=True)

        ours = flow_matching_loss(pasted, target, cond_mask=cond_mask)
        (our_grad,) = torch.autograd.grad(ours, raw)

        torch.testing.assert_close(our_grad, ref_grad)
        # And the masked value excludes the conditioning-view residual.
        target_only = ((pasted[:n_target] - target[:n_target]) ** 2).mean(dim=(1, 2, 3)).sum() / n_views
        torch.testing.assert_close(ours, target_only)

    def test_all_target_views_mask_is_noop(self):
        pred = torch.randn(4, 8, 4, 4)
        target = torch.randn_like(pred)
        cond_mask = torch.zeros_like(pred)
        torch.testing.assert_close(flow_matching_loss(pred, target, cond_mask), flow_matching_loss(pred, target))
