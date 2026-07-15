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
Loading of ncore-parser (or compatible) multiview sample directories into MVData.

A sample directory contains ``input_views/`` with ``camera.json``, ``frame_*``
images, and ``mask_*`` images. Used by both inference (run_inference.py) and
post-training (training/dataset.py).
"""

import json
import os

import numpy as np
from PIL import Image

from .nre_preproc import MVData


def load_camera_metadata(input_dir: str) -> dict:
    """Load canonical camera metadata from input_views/camera.json."""
    camera_path = os.path.join(input_dir, "camera.json")
    if not os.path.isfile(camera_path):
        raise FileNotFoundError(f"Missing camera.json in {input_dir}")

    with open(camera_path) as f:
        return json.load(f)


def get_camera_file_paths(input_dir: str, cam_data: dict) -> tuple[list[str], list[str]]:
    """Resolve ordered frame and mask file paths from camera metadata."""
    frame_paths = [os.path.join(input_dir, filename) for filename in cam_data.get("frame_filenames", [])]
    mask_paths = [os.path.join(input_dir, filename) for filename in cam_data.get("mask_filenames", [])]
    return frame_paths, mask_paths


def load_multiview_sample(sample_dir: str, allowed_indices: list[int] | None = None) -> MVData:
    """
    Load one sample from a ncore_parser (or compatible) output directory into an MVData instance
    for use with preproc. Uses input_views only (frames, masks, camera.json).
    """
    input_dir = os.path.join(sample_dir, "input_views")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input_views: {input_dir}")

    cam_data = load_camera_metadata(input_dir)
    frame_filenames_all = cam_data["frame_filenames"]
    frame_paths_all, mask_paths_all = get_camera_file_paths(input_dir, cam_data)
    cam_poses_raw_all = cam_data["normalized_cam_positions"]
    dists_list_all = cam_data["cam_dists"]
    fov_list_all = cam_data["cam_fovs"]
    lwh_list = cam_data["object_lwh"]

    if allowed_indices is None:
        allowed_indices = list(range(len(frame_filenames_all)))

    frame_filenames = [frame_filenames_all[i] for i in allowed_indices]
    frame_paths = [frame_paths_all[i] for i in allowed_indices]
    mask_paths = [mask_paths_all[i] for i in allowed_indices]
    cam_poses_raw = [cam_poses_raw_all[i] for i in allowed_indices]
    dists_list = [dists_list_all[i] for i in allowed_indices]
    fov_list = [fov_list_all[i] for i in allowed_indices]

    n = len(frame_filenames)
    if n == 0:
        raise ValueError(f"No input views in {sample_dir}")

    frames = []
    masks = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img))
    for path in mask_paths:
        mask_img = Image.open(path)
        mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        masks.append(mask)

    # cam_poses: (N, 3) camera positions
    cam_poses = np.array(cam_poses_raw, dtype=np.float64).reshape(n, 3)

    dists = np.array(dists_list, dtype=np.float64).reshape(n)
    fov = np.array(fov_list, dtype=np.float64).reshape(n)
    lwh = np.asarray(lwh_list, dtype=np.float64).reshape(3)

    metadata_path = os.path.join(sample_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        clip_id = meta.get("clip_id", os.path.basename(sample_dir))
    else:
        clip_id = os.path.basename(sample_dir)
    obj_id = os.path.basename(sample_dir)

    return MVData(
        clip_id=clip_id,
        obj_id=obj_id,
        frames=frames,
        cam_poses=cam_poses,
        dists=dists,
        fov=fov,
        npct="vehicle",
        lwh=lwh,
        masks=np.array(masks),
        auto_label=None,
    )


def discover_sample_dirs(data_root: str) -> list[str]:
    """Find all sample directories (containing input_views/camera.json) under data_root."""
    sample_dirs = []
    for dirpath, dirnames, _filenames in os.walk(data_root, followlinks=True):
        if os.path.isfile(os.path.join(dirpath, "input_views", "camera.json")):
            sample_dirs.append(dirpath)
            dirnames.clear()  # don't descend into a sample
    return sorted(sample_dirs)
