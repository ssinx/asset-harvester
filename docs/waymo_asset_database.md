# Waymo Street-Object Asset Database

`asset_harvester.waymo_asset_database` scans Waymo Open Dataset TFRecords and keeps
only usable 3D Gaussian assets. It creates:

- `assets.sqlite`: searchable SQLite index of accepted assets and their input views;
- `assets/<asset_id>/`: self-contained Gaussian PLY, preview, generated views, and `asset.json`;
- `rejections.jsonl`: audit trail for every quality-gate failure;
- `workspace/`: crops, Grounded-SAM2 results, and resumable intermediate inputs.

The command deliberately does **not** put candidates into SQLite until every quality
check passes. Source crops, segmentation, reconstruction, and database assets remain
separated so rejected samples cannot accidentally be used downstream.

## Prerequisites

1. Install Asset Harvester and its inference dependencies as described in `README.md`.
2. Install a Waymo Open Dataset Python package compatible with the TensorFlow version
   in the environment used by `--inference-python`.
3. Set up the sibling checkout at `../Grounded-SAM-2`, including its SAM2 checkpoint.
   Its Python environment can differ from Asset Harvester's; pass it with
   `--grounded-sam-python`.
4. Download the Asset Harvester diffusion, AHC, and TokenGS checkpoints.

The script uses delayed imports for TensorFlow and Waymo, so `--help` works without
them. The actual scan fails early with an install hint if either is unavailable.

## Run

```bash
waymo-asset-db \
  --waymo-root /datasets/waymo/training \
  --output-root /datasets/waymo_asset_db \
  --grounded-sam-root ../Grounded-SAM-2 \
  --grounded-sam-python /envs/grounded-sam2/bin/python \
  --inference-python /envs/asset-harvester/bin/python \
  --diffusion-checkpoint checkpoints/AH_multiview_diffusion.safetensors \
  --ahc-checkpoint checkpoints/AH_camera_estimator.safetensors \
  --lifting-checkpoint checkpoints/AH_tokengs_lifting.safetensors \
  --categories vehicle,pedestrian,cyclist \
  --camera-names FRONT,FRONT_LEFT,FRONT_RIGHT \
  --offload-model-to-cpu
```

Use `python -m asset_harvester.waymo_asset_database` when the package has not been
installed as an editable project. Start with `--max-assets 10`; once intermediates are
present, use `--resume` to skip assets already accepted and reuse SAM2 outputs.

## Quality gates

The defaults are conservative and can be tuned for a specific Waymo split:

1. **Input views:** rejects small labels, blurred crops, under/over-exposed crops,
   and 2D boxes touching an image edge (a strong truncation/occlusion signal).
2. **Grounded-SAM2 masks:** keeps masks centered on the annotated crop and rejects
   tiny, implausibly large, edge-touching, or off-center instances. Grounded-SAM2 is
   run per `context + track + camera`, so its continuous IDs are never shared between
   separate Waymo objects.
3. **View diversity:** selects high-quality views with a minimum timestamp separation
   and requires at least three valid masks before reconstruction.
4. **3D reconstruction:** requires a valid PLY with a bounded Gaussian count,
   enough generated views, sharp/exposed generated views, and a decodable lifted
   orbit whose foreground coverage and sharpness are plausible.

These are automated screening heuristics, not a replacement for task-specific human
review. Retain `rejections.jsonl` and periodically inspect a random accepted/rejected
sample to tune thresholds. `--keep-rejected` preserves rejected artifacts for this
calibration; by default the script removes their reconstruction and reconstruction
input folders.

## Descriptions

By default `--vlm-provider auto` calls a VLM only when configured, otherwise it stores
a safe category-only fallback description and records the source in SQLite. To require
an external VLM, select one of these providers:

```bash
# Command must print one short description. {image} and {prompt} are substituted.
--vlm-provider command \
--vlm-command 'my_local_vlm --image {image} --prompt {prompt}'

# Any OpenAI-compatible chat-completions endpoint.
--vlm-provider openai-compatible \
--vlm-url http://localhost:8000/v1/chat/completions \
--vlm-model Qwen2.5-VL-7B-Instruct
```

For an authenticated endpoint, put the key in `OPENAI_API_KEY` or choose another
variable with `--vlm-api-key-env`. The description, source label, quality metrics,
source Waymo `context_name`, and `track_id` are all stored in `asset.json` and
`assets.sqlite`.

## Query examples

```sql
SELECT asset_id, category, description, ply_path
FROM assets
WHERE category = 'vehicle'
ORDER BY created_at DESC;

SELECT a.asset_id, a.description, v.camera, v.timestamp_micros
FROM assets AS a
JOIN asset_views AS v USING(asset_id)
WHERE a.description LIKE '%truck%';
```
