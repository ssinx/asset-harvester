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

"""Build a quality-filtered 3D Gaussian asset database from Waymo TFRecords.

The pipeline extracts tracked 2D object observations, obtains instance masks from a
sibling Grounded-SAM-2 checkout, runs Asset Harvester reconstruction, screens the
result, and stores only accepted assets in SQLite plus self-contained JSON metadata.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_GROUNDED_SAM_ROOT = Path(__file__).resolve().parents[2] / "Grounded-SAM-2"
CATEGORY_PROMPTS = {
    "vehicle": "vehicle.",
    "pedestrian": "person.",
    "cyclist": "cyclist.",
    "sign": "traffic sign.",
}
WAYMO_CATEGORY_NAMES = {
    "TYPE_VEHICLE": "vehicle",
    "TYPE_PEDESTRIAN": "pedestrian",
    "TYPE_CYCLIST": "cyclist",
    "TYPE_SIGN": "sign",
}


@dataclass
class ViewCandidate:
    """A crop generated from a single Waymo camera label."""

    camera: str
    timestamp_micros: int
    frame_path: str
    source_bbox: list[float]
    crop_bbox: list[int]
    source_size: list[int]
    sharpness: float
    brightness: float
    source_bbox_area: float
    score: float


@dataclass
class TrackCandidate:
    """All candidate views for one Waymo tracked object."""

    asset_id: str
    context_name: str
    track_id: str
    category: str
    views_by_camera: dict[str, list[ViewCandidate]] = field(default_factory=lambda: defaultdict(list))


class Rejection(Exception):
    """Expected rejection from a quality gate."""


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def asset_identifier(context_name: str, track_id: str) -> str:
    digest = hashlib.sha256(f"{context_name}\0{track_id}".encode("utf-8")).hexdigest()[:16]
    return f"{context_name[:48].replace('/', '_')}_{digest}"


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def image_metrics(image: Image.Image) -> dict[str, float]:
    """Return inexpensive image sharpness and exposure measures without OpenCV."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) < 3:
        return {"sharpness": 0.0, "brightness": float(gray.mean())}
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return {"sharpness": float(laplacian.var()), "brightness": float(gray.mean())}


def crop_for_label(image: Image.Image, box: Any, padding: float) -> tuple[Image.Image, list[int], list[float]]:
    """Crop a label box with context, returning crop coordinates and source box."""
    width, height = image.size
    box_width = max(float(box.length), 1.0)
    box_height = max(float(box.width), 1.0)
    center_x = float(box.center_x)
    center_y = float(box.center_y)
    x1 = max(0, int(math.floor(center_x - box_width * (0.5 + padding))))
    y1 = max(0, int(math.floor(center_y - box_height * (0.5 + padding))))
    x2 = min(width, int(math.ceil(center_x + box_width * (0.5 + padding))))
    y2 = min(height, int(math.ceil(center_y + box_height * (0.5 + padding))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        raise Rejection("degenerate_crop")
    source_box = [center_x - box_width / 2, center_y - box_height / 2, center_x + box_width / 2, center_y + box_height / 2]
    return image.crop((x1, y1, x2, y2)), [x1, y1, x2, y2], source_box


def source_box_is_truncated(source_box: list[float], image_size: tuple[int, int], margin: int = 1) -> bool:
    width, height = image_size
    return source_box[0] <= margin or source_box[1] <= margin or source_box[2] >= width - margin or source_box[3] >= height - margin


def select_spread_views(views: Iterable[ViewCandidate], max_views: int, min_time_separation_micros: int) -> list[ViewCandidate]:
    """Keep high-quality views while avoiding near-duplicate timestamps per camera."""
    selected: list[ViewCandidate] = []
    for view in sorted(views, key=lambda item: (-item.score, item.timestamp_micros)):
        if all(abs(view.timestamp_micros - other.timestamp_micros) >= min_time_separation_micros for other in selected):
            selected.append(view)
        if len(selected) >= max_views:
            break
    return selected


def mask_metrics(mask: np.ndarray) -> dict[str, float]:
    foreground = mask > 0
    height, width = foreground.shape
    area = int(foreground.sum())
    if area == 0:
        return {"area_ratio": 0.0, "border_ratio": 1.0, "central_ratio": 0.0}
    border = np.concatenate((foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))
    y1, y2 = height // 4, max(height // 4 + 1, height * 3 // 4)
    x1, x2 = width // 4, max(width // 4 + 1, width * 3 // 4)
    central = foreground[y1:y2, x1:x2]
    return {
        "area_ratio": area / float(height * width),
        "border_ratio": float(border.sum()) / area,
        "central_ratio": float(central.sum()) / area,
    }


def choose_grounded_sam_mask(mask_paths: Iterable[Path]) -> tuple[Path, dict[str, float]] | None:
    """Choose the Grounded-SAM2 mask most likely to be the centered Waymo object."""
    candidates: list[tuple[float, Path, dict[str, float]]] = []
    for mask_path in mask_paths:
        if not mask_path.is_file():
            continue
        with Image.open(mask_path) as mask_image:
            metrics = mask_metrics(np.asarray(mask_image.convert("L")))
        if metrics["area_ratio"] == 0:
            continue
        score = metrics["central_ratio"] + min(metrics["area_ratio"], 0.4)
        candidates.append((score, mask_path, metrics))
    if not candidates:
        return None
    _, path, metrics = max(candidates, key=lambda item: item[0])
    return path, metrics


def ply_vertex_count(ply_path: Path) -> int | None:
    """Read the vertex count from an ASCII or binary PLY header."""
    try:
        with ply_path.open("rb") as handle:
            header = handle.read(131072).decode("ascii", errors="replace")
    except OSError:
        return None
    for line in header.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            try:
                return int(fields[2])
            except ValueError:
                return None
        if line == "end_header":
            break
    return None


def foreground_fraction(image: Image.Image) -> float:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return float(np.any(array < 245, axis=2).mean())


def require_module(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as error:
        raise RuntimeError(f"Missing optional dependency '{module_name}'. {install_hint}") from error


class AssetDatabaseBuilder:
    """Coordinates Waymo extraction, segmentation, reconstruction, and database insertion."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_root).resolve()
        self.workspace = self.output_root / "workspace"
        self.inputs_root = self.workspace / "reconstruction_inputs"
        self.sam_root = self.workspace / "grounded_sam"
        self.reconstruction_root = self.output_root / "reconstructions"
        self.asset_root = self.output_root / "assets"
        self.database_path = self.output_root / "assets.sqlite"
        self.rejections_path = self.output_root / "rejections.jsonl"
        self.prepared_assets: dict[str, dict[str, Any]] = {}
        self.allowed_categories = set(parse_csv(args.categories))
        self.allowed_cameras = set(parse_csv(args.camera_names))
        self.processed_asset_count = 0
        for directory in (self.workspace, self.inputs_root, self.sam_root, self.reconstruction_root, self.asset_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(self.database_path)
        self._create_schema()

    def close(self) -> None:
        self.database.close()

    def _create_schema(self) -> None:
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                context_name TEXT NOT NULL,
                track_id TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                description_source TEXT NOT NULL,
                asset_dir TEXT NOT NULL,
                ply_path TEXT NOT NULL,
                preview_path TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                metadata_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_assets_category ON assets(category);
            CREATE INDEX IF NOT EXISTS idx_assets_context_track ON assets(context_name, track_id);
            CREATE TABLE IF NOT EXISTS asset_views (
                asset_id TEXT NOT NULL,
                view_index INTEGER NOT NULL,
                camera TEXT NOT NULL,
                timestamp_micros INTEGER NOT NULL,
                frame_path TEXT NOT NULL,
                mask_path TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                PRIMARY KEY(asset_id, view_index),
                FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
            );
            """
        )
        self.database.commit()

    def reject(self, asset_id: str, reason: str, **details: Any) -> None:
        payload = {"asset_id": asset_id, "reason": reason, "time": utc_now(), **details}
        append_jsonl(self.rejections_path, payload)
        print(f"Reject {asset_id}: {reason}")

    def existing_asset(self, asset_id: str) -> bool:
        row = self.database.execute("SELECT 1 FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return row is not None

    def scan_waymo(self) -> None:
        tensorflow = require_module(
            "tensorflow",
            "Install a Waymo-compatible TensorFlow build before running this command.",
        )
        dataset_pb2 = require_module(
            "waymo_open_dataset.dataset_pb2",
            "Install the Waymo Open Dataset package that matches your TensorFlow version.",
        )
        record_paths = self._find_record_paths()
        if not record_paths:
            raise RuntimeError(f"No TFRecord files found under {self.args.waymo_root}")
        for record_path in record_paths:
            if self.args.max_assets and self.processed_asset_count >= self.args.max_assets:
                break
            print(f"Scanning {record_path}")
            tracks: dict[str, TrackCandidate] = {}
            dataset = tensorflow.data.TFRecordDataset(str(record_path), compression_type=self.args.tfrecord_compression or "")
            for record in dataset:
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytes(record.numpy()))
                self._extract_frame_tracks(frame, record_path, tracks, dataset_pb2)
            self._prepare_context_tracks(tracks.values())

    def _find_record_paths(self) -> list[Path]:
        root = Path(self.args.waymo_root)
        if root.is_file():
            return [root]
        patterns = ("*.tfrecord", "*.tfrecord-*", "*.tfrecords", "*.record")
        return sorted(path for pattern in patterns for path in root.rglob(pattern))

    def _extract_frame_tracks(self, frame: Any, record_path: Path, tracks: dict[str, TrackCandidate], dataset_pb2: Any) -> None:
        image_by_camera = {image.name: image for image in frame.images}
        context_name = frame.context.name or record_path.stem
        for camera_labels in frame.camera_labels:
            camera_name = dataset_pb2.CameraName.Name(camera_labels.name).removeprefix("UNKNOWN_")
            if self.allowed_cameras and camera_name not in self.allowed_cameras:
                continue
            camera_image = image_by_camera.get(camera_labels.name)
            if camera_image is None:
                continue
            image = Image.open(io.BytesIO(camera_image.image)).convert("RGB")
            for label in camera_labels.labels:
                type_name = dataset_pb2.Label.Type.Name(label.type)
                category = WAYMO_CATEGORY_NAMES.get(type_name)
                if category not in self.allowed_categories or not label.id:
                    continue
                asset_id = asset_identifier(context_name, label.id)
                if self.args.resume and self.existing_asset(asset_id):
                    continue
                try:
                    crop, crop_box, source_box = crop_for_label(image, label.box, self.args.crop_padding)
                except Rejection:
                    continue
                source_area = float(label.box.length * label.box.width)
                metrics = image_metrics(crop)
                if source_area < self.args.min_source_bbox_area:
                    continue
                if metrics["sharpness"] < self.args.min_input_sharpness:
                    continue
                if not self.args.min_brightness <= metrics["brightness"] <= self.args.max_brightness:
                    continue
                if source_box_is_truncated(source_box, image.size):
                    continue
                if asset_id not in tracks:
                    tracks[asset_id] = TrackCandidate(asset_id, context_name, label.id, category)
                track = tracks[asset_id]
                camera_dir = self.workspace / "raw_crops" / asset_id / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)
                frame_path = camera_dir / f"frame_{int(frame.timestamp_micros):016d}.jpeg"
                crop.save(frame_path, quality=95)
                score = math.log1p(source_area) + math.log1p(metrics["sharpness"])
                track.views_by_camera[camera_name].append(
                    ViewCandidate(
                        camera=camera_name,
                        timestamp_micros=int(frame.timestamp_micros),
                        frame_path=str(frame_path),
                        source_bbox=source_box,
                        crop_bbox=crop_box,
                        source_size=list(image.size),
                        sharpness=metrics["sharpness"],
                        brightness=metrics["brightness"],
                        source_bbox_area=source_area,
                        score=score,
                    )
                )

    def _prepare_context_tracks(self, tracks: Iterable[TrackCandidate]) -> None:
        ranked_tracks = sorted(
            tracks,
            key=lambda track: max((view.score for views in track.views_by_camera.values() for view in views), default=0.0),
            reverse=True,
        )
        for track in ranked_tracks:
            if self.args.max_assets and self.processed_asset_count >= self.args.max_assets:
                return
            if self.args.resume and self.existing_asset(track.asset_id):
                continue
            try:
                metadata = self._prepare_track(track)
            except Rejection as error:
                self.reject(track.asset_id, str(error), context_name=track.context_name, track_id=track.track_id)
                continue
            self.prepared_assets[track.asset_id] = metadata
            self.processed_asset_count += 1

    def _prepare_track(self, track: TrackCandidate) -> dict[str, Any]:
        selected_by_camera: dict[str, list[ViewCandidate]] = {}
        for camera, views in track.views_by_camera.items():
            selected = select_spread_views(
                views,
                max_views=self.args.max_raw_views_per_camera,
                min_time_separation_micros=self.args.min_time_separation_micros,
            )
            if selected:
                selected_by_camera[camera] = selected
        if sum(map(len, selected_by_camera.values())) < self.args.min_views:
            raise Rejection("too_few_sharp_untruncated_views")

        segmented_views: list[dict[str, Any]] = []
        for camera, views in selected_by_camera.items():
            segmented_views.extend(self._segment_camera_views(track, camera, views))
        if len(segmented_views) < self.args.min_views:
            raise Rejection("too_few_valid_grounded_sam_masks")

        segmented_views.sort(key=lambda view: (-view["view"]["score"], view["view"]["timestamp_micros"]))
        selected_views = segmented_views[: self.args.max_views_per_asset]
        if len(selected_views) < self.args.min_views:
            raise Rejection("too_few_selected_views")

        input_dir = self.inputs_root / track.asset_id
        input_dir.mkdir(parents=True, exist_ok=True)
        final_views: list[dict[str, Any]] = []
        for index, item in enumerate(selected_views):
            frame_path = input_dir / f"frame_{index:03d}.jpeg"
            mask_path = input_dir / f"mask_{index:03d}.png"
            shutil.copy2(item["view"]["frame_path"], frame_path)
            shutil.copy2(item["mask_path"], mask_path)
            final_views.append(
                {
                    **item["view"],
                    "frame_path": str(frame_path),
                    "mask_path": str(mask_path),
                    "mask_quality": item["mask_quality"],
                }
            )

        metadata = {
            "asset_id": track.asset_id,
            "context_name": track.context_name,
            "track_id": track.track_id,
            "category": track.category,
            "prepared_at": utc_now(),
            "views": final_views,
        }
        json_dump(self.workspace / "candidates" / f"{track.asset_id}.json", metadata)
        return metadata

    def _segment_camera_views(self, track: TrackCandidate, camera: str, views: list[ViewCandidate]) -> list[dict[str, Any]]:
        image_dir = Path(views[0].frame_path).parent
        output_dir = self.sam_root / track.asset_id / camera
        propainter_dir = output_dir / "propainter_masks"
        if not (self.args.resume and propainter_dir.is_dir()):
            command = [
                self.args.grounded_sam_python,
                str(Path(self.args.grounded_sam_root) / "grounded_sam2_tracking_demo_with_continuous_id_plus.py"),
                "--image_dir",
                str(image_dir),
                "--output_dir",
                str(output_dir),
                "--text",
                CATEGORY_PROMPTS[track.category],
                "--step",
                str(max(1, self.args.grounded_sam_step)),
                "--box_threshold",
                str(self.args.grounded_sam_box_threshold),
                "--text_threshold",
                str(self.args.grounded_sam_text_threshold),
            ]
            print(f"Segmenting {track.asset_id}/{camera} ({len(views)} views)")
            completed = subprocess.run(command, cwd=self.args.grounded_sam_root, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                stderr = completed.stderr.strip().replace("\n", " ")[-500:]
                raise Rejection(f"grounded_sam_failed:{stderr or completed.returncode}")

        mask_directories = sorted(path for path in propainter_dir.glob("object_*") if path.is_dir())
        if not mask_directories:
            raise Rejection("grounded_sam_found_no_instances")
        ordered_views = sorted(views, key=lambda view: Path(view.frame_path).name)
        width = max(5, len(str(len(ordered_views) - 1)))
        accepted: list[dict[str, Any]] = []
        for index, view in enumerate(ordered_views):
            mask_name = f"{index:0{width}d}.png"
            result = choose_grounded_sam_mask(mask_dir / mask_name for mask_dir in mask_directories)
            if result is None:
                continue
            mask_path, metrics = result
            if not self.args.min_mask_area_ratio <= metrics["area_ratio"] <= self.args.max_mask_area_ratio:
                continue
            if metrics["central_ratio"] < self.args.min_mask_central_ratio:
                continue
            if metrics["border_ratio"] > self.args.max_mask_border_ratio:
                continue
            accepted.append({"view": asdict(view), "mask_path": str(mask_path), "mask_quality": metrics})
        return accepted

    def run_reconstruction(self) -> None:
        pending = [asset_id for asset_id in self.prepared_assets if not (self.reconstruction_root / asset_id / "gaussians.ply").is_file()]
        if not pending:
            return
        if self.args.skip_reconstruction:
            print("Skipping reconstruction as requested.")
            return
        required = {"diffusion_checkpoint": self.args.diffusion_checkpoint, "ahc_checkpoint": self.args.ahc_checkpoint, "lifting_checkpoint": self.args.lifting_checkpoint}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Reconstruction requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
        command = [
            self.args.inference_python,
            str(Path(__file__).resolve().parents[1] / "run_inference.py"),
            "--image_dir",
            str(self.inputs_root),
            "--output_dir",
            str(self.reconstruction_root),
            "--diffusion_checkpoint",
            self.args.diffusion_checkpoint,
            "--ahc_checkpoint",
            self.args.ahc_checkpoint,
            "--lifting_checkpoint",
            self.args.lifting_checkpoint,
            "--num_steps",
            str(self.args.num_steps),
            "--cfg_scale",
            str(self.args.cfg_scale),
            "--precision",
            self.args.precision,
        ]
        if self.args.offload_model_to_cpu:
            command.append("--offload_model_to_cpu")
        print(f"Reconstructing {len(pending)} prepared assets")
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, check=False)
        if completed.returncode != 0:
            print(f"Asset Harvester inference exited with {completed.returncode}; screening available outputs.", file=sys.stderr)

    def screen_and_index(self) -> tuple[int, int]:
        accepted = 0
        rejected = 0
        for asset_id, metadata in self.prepared_assets.items():
            try:
                reconstruction_quality, preview_path = self._assess_reconstruction(asset_id)
                description, description_source = self._describe_asset(metadata, preview_path)
                self._store_asset(metadata, reconstruction_quality, preview_path, description, description_source)
                accepted += 1
                print(f"Accepted {asset_id}")
            except Rejection as error:
                rejected += 1
                self.reject(asset_id, str(error), context_name=metadata["context_name"], track_id=metadata["track_id"])
                if not self.args.keep_rejected:
                    shutil.rmtree(self.reconstruction_root / asset_id, ignore_errors=True)
                    shutil.rmtree(self.inputs_root / asset_id, ignore_errors=True)
        return accepted, rejected

    def _assess_reconstruction(self, asset_id: str) -> tuple[dict[str, Any], Path]:
        reconstruction_dir = self.reconstruction_root / asset_id
        ply_path = reconstruction_dir / "gaussians.ply"
        vertex_count = ply_vertex_count(ply_path)
        if vertex_count is None:
            raise Rejection("missing_or_invalid_gaussian_ply")
        if not self.args.min_gaussian_vertices <= vertex_count <= self.args.max_gaussian_vertices:
            raise Rejection(f"gaussian_vertex_count_out_of_range:{vertex_count}")
        generated_paths = sorted((reconstruction_dir / "multiview").glob("*.png"))
        if len(generated_paths) < self.args.min_generated_views:
            raise Rejection("too_few_generated_views")
        generated_metrics = []
        for path in generated_paths:
            with Image.open(path) as image:
                metrics = image_metrics(image)
            generated_metrics.append(metrics)
        if float(np.median([item["sharpness"] for item in generated_metrics])) < self.args.min_generated_sharpness:
            raise Rejection("generated_views_too_blurry")
        if not all(self.args.min_brightness <= item["brightness"] <= self.args.max_brightness for item in generated_metrics):
            raise Rejection("generated_views_bad_exposure")

        orbit_frames = self._read_orbit_frames(reconstruction_dir / "3d_lifted.mp4")
        if len(orbit_frames) < self.args.min_orbit_frames:
            raise Rejection("missing_or_invalid_lifted_orbit_render")
        orbit_sharpness = [image_metrics(frame)["sharpness"] for frame in orbit_frames]
        orbit_foreground = [foreground_fraction(frame) for frame in orbit_frames]
        if float(np.median(orbit_sharpness)) < self.args.min_orbit_sharpness:
            raise Rejection("lifted_orbit_too_blurry")
        if not self.args.min_orbit_foreground_ratio <= float(np.median(orbit_foreground)) <= self.args.max_orbit_foreground_ratio:
            raise Rejection("lifted_orbit_invalid_object_coverage")

        preview_path = reconstruction_dir / "preview.png"
        orbit_frames[len(orbit_frames) // 2].save(preview_path)
        quality = {
            "gaussian_vertex_count": vertex_count,
            "generated_view_count": len(generated_paths),
            "generated_median_sharpness": float(np.median([item["sharpness"] for item in generated_metrics])),
            "orbit_frame_count": len(orbit_frames),
            "orbit_median_sharpness": float(np.median(orbit_sharpness)),
            "orbit_median_foreground_ratio": float(np.median(orbit_foreground)),
        }
        return quality, preview_path

    def _read_orbit_frames(self, video_path: Path) -> list[Image.Image]:
        try:
            import imageio.v3 as iio
        except ImportError as error:
            raise Rejection("imageio_required_to_validate_lifted_orbit") from error
        if not video_path.is_file():
            return []
        try:
            frames = list(iio.imiter(video_path, plugin="pyav"))
        except Exception:
            try:
                frames = list(iio.imiter(video_path))
            except Exception:
                return []
        if not frames:
            return []
        indices = np.linspace(0, len(frames) - 1, min(self.args.max_orbit_frames_to_check, len(frames)), dtype=int)
        return [Image.fromarray(np.asarray(frames[index])[..., :3]).convert("RGB") for index in indices]

    def _describe_asset(self, metadata: dict[str, Any], preview_path: Path) -> tuple[str, str]:
        category = metadata["category"]
        fallback = f"High-quality street-scene {category} 3D Gaussian asset."
        if self.args.vlm_provider == "heuristic":
            return fallback, "heuristic"
        prompt = (
            "Describe this isolated road-scene object for asset search in one concise sentence. "
            "State object category, visible color or appearance, and relevant type; do not mention image quality, "
            "camera, reconstruction, or uncertainty."
        )
        try:
            if self.args.vlm_provider == "command":
                return self._describe_with_command(preview_path, prompt), "vlm_command"
            if self.args.vlm_provider == "openai-compatible":
                return self._describe_with_openai_compatible(preview_path, prompt), "openai_compatible_vlm"
            if self.args.vlm_command:
                return self._describe_with_command(preview_path, prompt), "vlm_command"
            if self.args.vlm_url and self.args.vlm_model:
                return self._describe_with_openai_compatible(preview_path, prompt), "openai_compatible_vlm"
        except (OSError, RuntimeError, urllib.error.URLError, ValueError) as error:
            print(f"VLM description failed for {metadata['asset_id']}: {error}; using fallback.", file=sys.stderr)
        return fallback, "heuristic_fallback"

    def _describe_with_command(self, preview_path: Path, prompt: str) -> str:
        if not self.args.vlm_command:
            raise RuntimeError("--vlm-command is required with --vlm-provider command")
        command = [part.format(image=str(preview_path), prompt=prompt) for part in shlex.split(self.args.vlm_command)]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"command exited with {completed.returncode}")
        description = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
        if not description:
            raise RuntimeError("VLM command returned no description")
        return description[:500]

    def _describe_with_openai_compatible(self, preview_path: Path, prompt: str) -> str:
        if not self.args.vlm_url or not self.args.vlm_model:
            raise RuntimeError("--vlm-url and --vlm-model are required for an OpenAI-compatible VLM")
        api_key = os.environ.get(self.args.vlm_api_key_env, "")
        with preview_path.open("rb") as handle:
            image_data = base64.b64encode(handle.read()).decode("ascii")
        payload = {
            "model": self.args.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    ],
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        }
        request = urllib.request.Request(
            self.args.vlm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.args.vlm_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        description = str(content).strip()
        if not description:
            raise RuntimeError("OpenAI-compatible VLM returned no description")
        return description[:500]

    def _store_asset(
        self,
        metadata: dict[str, Any],
        reconstruction_quality: dict[str, Any],
        preview_path: Path,
        description: str,
        description_source: str,
    ) -> None:
        asset_id = metadata["asset_id"]
        source_dir = self.reconstruction_root / asset_id
        destination_dir = self.asset_root / asset_id
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.copytree(source_dir, destination_dir)
        destination_preview = destination_dir / preview_path.name
        destination_ply = destination_dir / "gaussians.ply"
        archived_views = []
        source_views_dir = destination_dir / "source_views"
        source_views_dir.mkdir(parents=True, exist_ok=True)
        for index, view in enumerate(metadata["views"]):
            frame_path = source_views_dir / f"frame_{index:03d}.jpeg"
            mask_path = source_views_dir / f"mask_{index:03d}.png"
            shutil.copy2(view["frame_path"], frame_path)
            shutil.copy2(view["mask_path"], mask_path)
            archived_views.append({**view, "frame_path": str(frame_path), "mask_path": str(mask_path)})

        asset_metadata = {
            **metadata,
            "views": archived_views,
            "description": description,
            "description_source": description_source,
            "reconstruction_quality": reconstruction_quality,
            "accepted_at": utc_now(),
            "asset_dir": str(destination_dir),
            "ply_path": str(destination_ply),
            "preview_path": str(destination_preview),
        }
        metadata_path = destination_dir / "asset.json"
        json_dump(metadata_path, asset_metadata)
        with self.database:
            self.database.execute("DELETE FROM asset_views WHERE asset_id = ?", (asset_id,))
            self.database.execute(
                """
                INSERT OR REPLACE INTO assets (
                    asset_id, context_name, track_id, category, description, description_source,
                    asset_dir, ply_path, preview_path, quality_json, metadata_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    metadata["context_name"],
                    metadata["track_id"],
                    metadata["category"],
                    description,
                    description_source,
                    str(destination_dir),
                    str(destination_ply),
                    str(destination_preview),
                    json.dumps(reconstruction_quality, sort_keys=True),
                    str(metadata_path),
                    asset_metadata["accepted_at"],
                ),
            )
            for index, view in enumerate(archived_views):
                self.database.execute(
                    """
                    INSERT INTO asset_views (asset_id, view_index, camera, timestamp_micros, frame_path, mask_path, quality_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        index,
                        view["camera"],
                        view["timestamp_micros"],
                        view["frame_path"],
                        view["mask_path"],
                        json.dumps({"sharpness": view["sharpness"], "brightness": view["brightness"], **view["mask_quality"]}),
                    ),
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waymo-root", required=True, help="A Waymo TFRecord file or directory containing TFRecords.")
    parser.add_argument("--output-root", required=True, help="Directory for SQLite index, assets, reconstructions, and workspace.")
    parser.add_argument("--tfrecord-compression", default="", help="TFRecord compression type, for example GZIP.")
    parser.add_argument("--categories", default="vehicle,pedestrian,cyclist", help="Comma-separated categories: vehicle,pedestrian,cyclist,sign.")
    parser.add_argument("--camera-names", default="FRONT,FRONT_LEFT,FRONT_RIGHT", help="Comma-separated Waymo camera enum names.")
    parser.add_argument("--max-assets", type=int, default=0, help="Maximum tracks to prepare; 0 processes all tracks.")
    parser.add_argument("--resume", action="store_true", help="Skip assets already accepted and reuse Grounded-SAM2 outputs.")
    parser.add_argument("--keep-rejected", action="store_true", help="Keep reconstruction and input files for rejected candidates.")
    parser.add_argument("--crop-padding", type=float, default=0.35, help="Extra crop margin as a fraction of the labelled box size.")
    parser.add_argument("--min-source-bbox-area", type=float, default=4500.0, help="Minimum Waymo 2D label area in source-image pixels.")
    parser.add_argument("--min-input-sharpness", type=float, default=55.0, help="Minimum crop Laplacian variance before segmentation.")
    parser.add_argument("--min-brightness", type=float, default=25.0, help="Minimum acceptable grayscale mean.")
    parser.add_argument("--max-brightness", type=float, default=235.0, help="Maximum acceptable grayscale mean.")
    parser.add_argument("--min-views", type=int, default=3, help="Minimum segmented views required for an asset.")
    parser.add_argument("--max-raw-views-per-camera", type=int, default=10, help="Maximum sharp candidate crops passed to SAM2 per camera.")
    parser.add_argument("--max-views-per-asset", type=int, default=8, help="Maximum masked views supplied to reconstruction.")
    parser.add_argument("--min-time-separation-micros", type=int, default=500000, help="Minimum time spacing between selected views in one camera.")
    parser.add_argument("--grounded-sam-root", default=str(DEFAULT_GROUNDED_SAM_ROOT), help="Path to the sibling Grounded-SAM-2 checkout.")
    parser.add_argument("--grounded-sam-python", default=sys.executable, help="Python executable for the Grounded-SAM-2 environment.")
    parser.add_argument("--grounded-sam-step", type=int, default=5, help="Grounding DINO interval for Grounded-SAM2.")
    parser.add_argument("--grounded-sam-box-threshold", type=float, default=0.3)
    parser.add_argument("--grounded-sam-text-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-mask-central-ratio", type=float, default=0.35)
    parser.add_argument("--max-mask-border-ratio", type=float, default=0.12)
    parser.add_argument("--skip-reconstruction", action="store_true", help="Only prepare masks and candidates; no assets are indexed.")
    parser.add_argument("--inference-python", default=sys.executable, help="Python executable for the Asset Harvester environment.")
    parser.add_argument("--diffusion-checkpoint", help="Asset Harvester multiview diffusion checkpoint.")
    parser.add_argument("--ahc-checkpoint", help="Asset Harvester camera estimator checkpoint.")
    parser.add_argument("--lifting-checkpoint", help="Asset Harvester TokenGS lifting checkpoint.")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="bf16")
    parser.add_argument("--offload-model-to-cpu", action="store_true")
    parser.add_argument("--min-gaussian-vertices", type=int, default=5000)
    parser.add_argument("--max-gaussian-vertices", type=int, default=3000000)
    parser.add_argument("--min-generated-views", type=int, default=8)
    parser.add_argument("--min-generated-sharpness", type=float, default=25.0)
    parser.add_argument("--min-orbit-frames", type=int, default=6)
    parser.add_argument("--max-orbit-frames-to-check", type=int, default=12)
    parser.add_argument("--min-orbit-sharpness", type=float, default=12.0)
    parser.add_argument("--min-orbit-foreground-ratio", type=float, default=0.02)
    parser.add_argument("--max-orbit-foreground-ratio", type=float, default=0.92)
    parser.add_argument("--vlm-provider", choices=("auto", "heuristic", "command", "openai-compatible"), default="auto")
    parser.add_argument("--vlm-command", help="Shell-style VLM command; use {image} and {prompt} placeholders. It must print one description.")
    parser.add_argument("--vlm-url", help="OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--vlm-model", help="Model name for --vlm-url.")
    parser.add_argument("--vlm-api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the optional VLM API key.")
    parser.add_argument("--vlm-timeout-seconds", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grounded_sam_root = Path(args.grounded_sam_root)
    if not grounded_sam_root.is_dir():
        raise SystemExit(f"Grounded-SAM-2 checkout does not exist: {grounded_sam_root}")
    builder = AssetDatabaseBuilder(args)
    try:
        builder.scan_waymo()
        builder.run_reconstruction()
        if args.skip_reconstruction:
            print(f"Prepared {len(builder.prepared_assets)} candidates. No assets were indexed because reconstruction was skipped.")
            return 0
        accepted, rejected = builder.screen_and_index()
        print(f"Completed: accepted={accepted}, rejected={rejected}, database={builder.database_path}")
        return 0
    finally:
        builder.close()


if __name__ == "__main__":
    raise SystemExit(main())
