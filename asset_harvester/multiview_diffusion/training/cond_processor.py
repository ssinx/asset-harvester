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
Training-time conditioning processor for SparseViewDiT.

Turns a list of preproc'd objects (variable view counts) into the flattened
tensors the transformer's forward expects, mirroring the tensor assembly of
``SparseViewDiTPipeline.prepare_multiview_inputs`` at inference:

- VAE-encode all views (fp32 VAE, scaled by the DC-AE scaling factor)
- downsample Plucker rays to the latent resolution
- per-view conditioning mask (targets first: 0, conditioning views: 1)
- camera embedding = flattened relative c2w (16) + fov (1)
- C-RADIO image embedding of the conditioning views (max-pooled over views),
  replaced by the null (grey image) embedding with probability
  ``cfg_dropout_prob`` — and always when an object has 0 conditioning views —
  to support classifier-free guidance.
"""

import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize as torch_resize

_VAE_MINIBATCH_SIZE = 16
_DEFAULT_SCALING_FACTOR = 0.41407


class MultiviewCondProcessor:
    def __init__(
        self,
        vae,
        image_encoder,
        image_processor,
        weight_dtype: torch.dtype = torch.bfloat16,
        cfg_dropout_prob: float = 0.1,
    ):
        self.vae = vae
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.weight_dtype = weight_dtype
        self.cfg_dropout_prob = cfg_dropout_prob
        self._null_embed: torch.Tensor | None = None

        if getattr(vae.config, "scaling_factor", None):
            self.scaling_factor = vae.config.scaling_factor
        else:
            self.scaling_factor = _DEFAULT_SCALING_FACTOR

    @torch.no_grad()
    def _encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        """VAE-encode [N, 3, H, W] images in minibatches; returns scaled latents."""
        chunks = []
        for i in range(math.ceil(images.shape[0] / _VAE_MINIBATCH_SIZE)):
            batch = images[i * _VAE_MINIBATCH_SIZE : (i + 1) * _VAE_MINIBATCH_SIZE]
            z = self.vae.encode(batch.to(self.vae.device, dtype=self.vae.dtype))
            if hasattr(z, "latent"):
                z = z.latent
            chunks.append(z * self.scaling_factor)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _encode_image_prompt(self, cond_images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """C-RADIO embed conditioning images ([-1, 1] tensors), max-pooled over views."""
        pil_images = []
        for img in cond_images:
            img_np = ((img + 1.0) / 2.0 * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            pil_images.append(Image.fromarray(img_np))
        pixel_values = self.image_processor(images=pil_images, return_tensors="pt", do_resize=True).pixel_values
        _summary, features = self.image_encoder(pixel_values.to(device))
        return torch.amax(features, dim=0, keepdim=True)  # [1, tokens, dim]

    @torch.no_grad()
    def null_image_prompt(self, device: torch.device) -> torch.Tensor:
        """Embedding of a grey image — the CFG unconditional prompt (cached)."""
        if self._null_embed is None:
            grey = torch.zeros(1, 3, 512, 512)  # 0.0 in [-1, 1] == pixel value 128
            self._null_embed = self._encode_image_prompt(grey, device)
        return self._null_embed

    @torch.no_grad()
    def __call__(self, objects: list, device: torch.device, generator: torch.Generator | None = None) -> dict:
        """Encode a batch (list) of preproc'd objects into flattened model inputs.

        Returns a dict with per-view tensors concatenated across objects
        (``clean_latents``, ``rays``, ``cond_mask``, ``camera_emb``,
        ``relative_brightness``), per-object ``prompt_embeds`` and
        ``views_per_object``/``n_targets`` bookkeeping.
        """
        clean_latents, rays, cond_masks, camera_embs, brightness, prompts = [], [], [], [], [], []
        views_per_object, n_targets = [], []

        for obj in objects:
            n_views = obj.x.shape[0]
            n_target = obj.n_target
            n_cond = n_views - n_target

            z = self._encode_latents(obj.x)
            latent_size = z.shape[-1]
            ray = torch_resize(obj.plucker_image.to(device, dtype=torch.float32), latent_size)

            mask = torch.zeros(n_views, z.shape[1], latent_size, latent_size, device=device)
            mask[n_target:] = 1.0

            cam = torch.cat(
                [obj.c2w_relatives.reshape(-1, 16), obj.fovs.unsqueeze(-1)],
                dim=-1,
            ).to(device)

            bright = obj.relative_brightness.to(device)
            if bright.dim() == 1:
                bright = bright.unsqueeze(-1)

            drop = torch.rand((), generator=generator).item() < self.cfg_dropout_prob
            if n_cond == 0 or drop:
                prompt = self.null_image_prompt(device)
            else:
                prompt = self._encode_image_prompt(obj.x_white_background[n_target:], device)

            clean_latents.append(z.to(device))
            rays.append(ray)
            cond_masks.append(mask)
            camera_embs.append(cam)
            brightness.append(bright)
            prompts.append(prompt)
            views_per_object.append(n_views)
            n_targets.append(n_target)

        dt = self.weight_dtype
        return {
            "clean_latents": torch.cat(clean_latents, dim=0).to(dt),
            "rays": torch.cat(rays, dim=0).to(dt),
            "cond_mask": torch.cat(cond_masks, dim=0).to(dt),
            "camera_emb": torch.cat(camera_embs, dim=0).to(dt),
            "relative_brightness": torch.cat(brightness, dim=0).to(dt),
            "prompt_embeds": torch.cat(prompts, dim=0).to(dt),  # [num_objects, tokens, dim]
            "views_per_object": views_per_object,
            "n_targets": n_targets,
        }


def make_validation_grid(images: list, views_per_row: int) -> np.ndarray:
    """Stack a flat list of PIL images into an HWC uint8 grid, one object per row."""
    rows = []
    for i in range(0, len(images), views_per_row):
        row = np.concatenate([np.asarray(im) for im in images[i : i + views_per_row]], axis=1)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    return np.concatenate(rows, axis=0)
