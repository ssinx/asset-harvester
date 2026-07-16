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
Rectified-flow (flow matching) training utilities for SparseViewDiT.

Implements the training-time noising process the released checkpoint was trained
with: a "linear_flow" sigma schedule with an optional resolution-dependent
timestep shift, logit-normal timestep sampling (SD3-style), linear interpolation
noising, and a velocity prediction target:

    sigma_t' = shift * sigma_t / (1 + (shift - 1) * sigma_t)
    x_t      = (1 - sigma_t') * x0 + sigma_t' * eps
    target   = eps - x0

The model consumes a continuous timestep ``sigma_t' * num_train_timesteps``,
matching the flow-sigma convention used by ``DPMSolverMultistepScheduler``
(``use_flow_sigmas=True``) at inference.
"""

import numpy as np
import torch


class RectifiedFlowSchedule:
    """Training-time sigma schedule for rectified-flow fine-tuning.

    The base schedule is ``sigma_i = 1 - beta_i`` with ``beta`` linearly spaced
    from 1.0 to 0.001 over ``num_train_timesteps`` steps, so ``sigma`` ascends
    from 0.0 to 0.999. ``flow_shift`` (> 1 for higher resolutions) warps sigmas
    toward the noisy end; 1.0 is the identity and the correct value at 512 px.
    """

    def __init__(self, num_train_timesteps: int = 1000, flow_shift: float = 1.0):
        if num_train_timesteps < 2:
            raise ValueError(f"num_train_timesteps must be >= 2, got {num_train_timesteps}")
        if flow_shift <= 0:
            raise ValueError(f"flow_shift must be positive, got {flow_shift}")
        self.num_train_timesteps = num_train_timesteps
        self.flow_shift = flow_shift

        scale = 1000.0 / num_train_timesteps
        betas = np.linspace(scale * 1.0, scale * 0.001, num_train_timesteps, dtype=np.float64)
        sigmas = 1.0 - betas
        sigmas = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)

        self.sigmas = torch.from_numpy(sigmas)
        # Continuous timestep fed to the model (flow-sigma convention).
        self.timesteps = self.sigmas * num_train_timesteps

    def sample_timestep_indices(
        self,
        batch_size: int,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample per-sample timestep indices in ``[0, num_train_timesteps)``.

        ``logit_normal`` draws ``u = sigmoid(N(logit_mean, logit_std))`` (SD3
        section 3.1) and maps it to an index; ``uniform`` draws indices directly.
        """
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu", generator=generator)
            u = torch.sigmoid(u)
            indices = (u * self.num_train_timesteps).long()
        elif weighting_scheme == "uniform":
            indices = torch.randint(0, self.num_train_timesteps, (batch_size,), generator=generator)
        else:
            raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")
        return indices.clamp_(max=self.num_train_timesteps - 1)

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timestep_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise clean latents: ``x_t = (1 - sigma) * x0 + sigma * eps``.

        ``timestep_indices`` has one entry per item in ``x0``'s batch dim.
        Returns ``(x_t, model_timesteps)`` where ``model_timesteps`` is the
        continuous timestep tensor to feed the transformer.
        """
        if timestep_indices.shape[0] != x0.shape[0]:
            raise ValueError(
                f"timestep_indices batch ({timestep_indices.shape[0]}) != x0 batch ({x0.shape[0]}); "
                "broadcast per-object indices across views first (see broadcast_to_views)"
            )
        sigmas = self.sigmas.to(device=x0.device, dtype=x0.dtype)[timestep_indices.to(x0.device)]
        sigmas = sigmas.reshape(-1, *([1] * (x0.ndim - 1)))
        x_t = (1.0 - sigmas) * x0 + sigmas * noise
        model_timesteps = self.timesteps.to(x0.device)[timestep_indices.to(x0.device)].to(x0.dtype)
        return x_t, model_timesteps


def broadcast_to_views(per_object: torch.Tensor, views_per_object: list[int]) -> torch.Tensor:
    """Repeat one value per object across that object's views.

    All views of an object share a timestep during training; this expands a
    ``(num_objects,)`` tensor to ``(sum(views_per_object),)``.
    """
    if per_object.shape[0] != len(views_per_object):
        raise ValueError(f"got {per_object.shape[0]} values for {len(views_per_object)} objects")
    repeats = torch.tensor(views_per_object, device=per_object.device)
    return torch.repeat_interleave(per_object, repeats, dim=0)


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Rectified-flow velocity target ``v = eps - x0`` (points from data to noise)."""
    return noise - x0


def flow_matching_loss(
    model_output: torch.Tensor, target: torch.Tensor, cond_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Velocity MSE averaged per view, then over all views.

    Conditioning views (``cond_mask == 1``) are excluded from the numerator but
    kept in the denominator: the model output on those views is the pasted-back
    clean latent, which carries no gradient, so this matches the gradients of an
    unmasked mean over all views while reporting a loss free of the constant
    conditioning-view residual.
    """
    per_view = (model_output.float() - target.float()) ** 2
    per_view = per_view.mean(dim=tuple(range(1, per_view.ndim)))
    if cond_mask is None:
        return per_view.mean()
    # cond_mask is constant within a view; reduce to one weight per view.
    target_weight = 1.0 - cond_mask.reshape(cond_mask.shape[0], -1).amax(dim=1).float()
    return (per_view * target_weight).sum() / per_view.shape[0]
