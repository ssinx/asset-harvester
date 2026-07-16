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

"""CPU tests for the training dataset over the bundled data_samples."""

import os

import pytest
import torch

from asset_harvester.multiview_diffusion.data.sample_loading import discover_sample_dirs
from asset_harvester.multiview_diffusion.training.dataset import (
    MultiviewObjectDataset,
    build_dataloader,
    collate_objects,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_ROOT = os.path.join(_REPO_ROOT, "data_samples", "rectified_AV_objects")

pytestmark = pytest.mark.skipif(not os.path.isdir(_DATA_ROOT), reason="data_samples not present")


def test_discover_sample_dirs():
    dirs = discover_sample_dirs(_DATA_ROOT)
    assert len(dirs) > 0
    for d in dirs:
        assert os.path.isfile(os.path.join(d, "input_views", "camera.json"))


def test_dataset_item_shapes():
    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, resolution=512, conditioning_min_n=1)
    obj = ds[0]

    n_views = obj.x.shape[0]
    assert 2 <= n_views <= 16
    assert 1 <= obj.n_target < n_views

    assert obj.x.shape == (n_views, 3, 512, 512)
    assert obj.x_white_background.shape == (n_views, 3, 512, 512)
    assert obj.plucker_image.shape[0] == n_views and obj.plucker_image.shape[1] == 6
    assert obj.c2w_relatives.shape == (n_views, 4, 4)
    assert obj.fovs.shape == (n_views,)
    assert obj.relative_brightness.shape[0] == n_views
    # Images are normalized to [-1, 1].
    assert obj.x.min() >= -1.001 and obj.x.max() <= 1.001


def test_dataset_repeats_and_collate():
    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, repeats=3, conditioning_min_n=1)
    assert len(ds) == 3 * len(ds.sample_dirs)
    batch = collate_objects([ds[0], ds[1]])
    assert isinstance(batch, list) and len(batch) == 2


def test_dataloader_yields_object_lists():
    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, repeats=2, conditioning_min_n=1)
    dl = build_dataloader(ds, batch_size=2, num_workers=0, shuffle=True, seed=0)
    batch = next(iter(dl))
    assert isinstance(batch, list) and len(batch) == 2
    assert all(torch.is_tensor(obj.x) for obj in batch)


def test_single_view_samples_filtered():
    # Any sample with fewer than 2 views cannot be used for training (needs
    # >= 1 conditioning + >= 1 target) and must be dropped at init.
    from asset_harvester.multiview_diffusion.training.dataset import _count_views

    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, conditioning_min_n=1)
    assert all(_count_views(d) >= 2 for d in ds.sample_dirs)
    # Every remaining item must load without retries exhausting.
    for i in range(len(ds.sample_dirs)):
        obj = ds[i]
        assert obj.x.shape[0] >= 2


def test_zero_conditioning_views_possible():
    # With conditioning_min_n=0 some draws must produce fully unconditional objects.
    ds = MultiviewObjectDataset(data_root=_DATA_ROOT, conditioning_min_n=0, repeats=1)
    torch.manual_seed(0)
    saw_zero_cond = False
    for _ in range(30):
        obj = ds[0]
        if obj.x.shape[0] - obj.n_target == 0:
            saw_zero_cond = True
            break
    assert saw_zero_cond
