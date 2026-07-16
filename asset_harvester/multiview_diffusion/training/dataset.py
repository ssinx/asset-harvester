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
Training dataset over ncore-parser sample directories.

Each item is one object: the multiview sample is loaded from disk and run
through the same ``preproc`` used at inference, but in training mode
(``eval_mode=False``): a random subset of views becomes conditioning
(``random_n`` between ``conditioning_min_n`` and ``conditioning_max_n``,
0 supplying the unconditional case for CFG) and the rest become denoising
targets with their real images kept as ground truth. Target views come first
in every per-view tensor, conditioning views after.

Objects have variable view counts, so batches are plain lists of per-object
AttrDicts (``collate_objects``); flattening into model inputs happens in
``MultiviewCondProcessor``.
"""

import os
import random
import warnings
from functools import partial

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from ..data.nre_preproc import preproc
from ..data.sample_loading import discover_sample_dirs, load_camera_metadata, load_multiview_sample


def _count_views(sample_dir: str) -> int:
    """Number of views in a sample, from camera.json alone (no image reads)."""
    try:
        cam_data = load_camera_metadata(os.path.join(sample_dir, "input_views"))
        return len(cam_data.get("frame_filenames", []))
    except Exception:
        return 0


class MultiviewObjectDataset(Dataset):
    """Map-style dataset: one ncore-parser sample directory per item.

    Defaults mirror the released checkpoint's training configuration
    (512 px, up to 16 views, 0-4 conditioning views, symmetry augmentation,
    white-masked target backgrounds).
    """

    def __init__(
        self,
        data_root: str | None = None,
        sample_dirs: list[str] | None = None,
        resolution: int = 512,
        max_views: int = 16,
        conditioning_mode: str = "random_n",
        conditioning_min_n: int = 0,
        conditioning_max_n: int = 4,
        symmetry_aug_p: float = 0.4,
        mask_out_background_target: bool | str = "white",
        mask_out_background_cond: bool | str = False,
        plucker_scene_scale: float = 30.0,
        repeats: int = 1,
        max_load_retries: int = 4,
    ):
        if (data_root is None) == (sample_dirs is None):
            raise ValueError("Provide exactly one of data_root or sample_dirs")
        candidates = sample_dirs if sample_dirs is not None else discover_sample_dirs(data_root)
        if not candidates:
            raise ValueError(f"No sample directories with input_views/camera.json found under {data_root}")
        # Training needs >= 2 views per object (>= 1 conditioning + >= 1 target).
        self.sample_dirs = [d for d in candidates if _count_views(d) >= 2]
        dropped = len(candidates) - len(self.sample_dirs)
        if dropped:
            warnings.warn(f"Dropped {dropped}/{len(candidates)} samples with fewer than 2 views", stacklevel=2)
        if not self.sample_dirs:
            raise ValueError("No usable samples: all have fewer than 2 views")
        self.repeats = repeats
        self.max_load_retries = max_load_retries

        image_transform = T.Compose(
            [
                T.Resize(resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self._preproc = partial(
            preproc,
            image_transform=image_transform,
            resolution=resolution,
            max_views=max_views,
            conditioning_mode=conditioning_mode,
            conditioning_min_n=conditioning_min_n,
            conditioning_max_n=conditioning_max_n,
            symmetry_aug_p=symmetry_aug_p,
            mask_out_background_target=mask_out_background_target,
            mask_out_background_cond=mask_out_background_cond,
            plucker_scene_scale=plucker_scene_scale,
            eval_mode=False,
        )

    def __len__(self) -> int:
        return len(self.sample_dirs) * self.repeats

    def __getitem__(self, index: int):
        idx = index % len(self.sample_dirs)
        for attempt in range(self.max_load_retries + 1):
            sample_dir = self.sample_dirs[idx]
            try:
                item = load_multiview_sample(sample_dir)
                return self._preproc(item)
            except Exception as e:  # corrupt sample: fall back to a random other one
                if attempt == self.max_load_retries or len(self.sample_dirs) == 1:
                    raise RuntimeError(f"Failed to load sample after retries: {sample_dir}") from e
                warnings.warn(f"Skipping unloadable sample {sample_dir}: {e}", stacklevel=2)
                idx = random.choice([i for i in range(len(self.sample_dirs)) if i != idx])
        raise AssertionError("unreachable")


def collate_objects(batch: list) -> list:
    """Objects have variable view counts; keep the batch as a list of AttrDicts."""
    return list(batch)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    """DataLoader over objects; ``batch_size`` counts objects, not views."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_objects,
        generator=generator,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
