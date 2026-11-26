# Automated Muay Thai Data Factory v2

This is the operator and teammate handoff for converting fight footage into
reviewable PoseC3D and TriDet training data. The existing v1 project is not
overwritten; v2 defaults to `generated/action_annotations_v2`.

## What is automated

| Stage | Automatic output | Human responsibility |
|---|---|---|
| Ingest | 30 FPS proxy, reversible frame formula, manifest | Confirm source video and rights |
| Perception | YOLO fighter tracks and RTMPose COCO17 skeletons | Correct identity, box, and pose errors in CVAT |
| Proposal | Motion windows, anatomical limb/side estimate, optional local Qwen labels | Verify every event and boundary |
| Review | FiftyOne timeline package, CVAT label contract, browser fallback | Accept/reject/split and label reviewed ranges |
| Export | PoseC3D tensors, TriDet JSON/features, YOLO and RTMPose feedback candidates | Review feedback preannotations before retraining |
| Freeze | Source-grouped train/validation/test data and fingerprint | Approve dataset version |

Automatic output is never promoted directly to training ground truth.

## Label design

The canonical record stores technique, anatomical side, effector limb,
lead/rear role, target, outcome and contact surface separately. This prevents
hundreds of sparse classes while still producing labels such as:

```text
jab__right_hand
hook__left_hand
roundhouse_kick__left_leg
horizontal_elbow__right_elbow
```

Jab/cross identify the punch type. `anatomical_side` identifies the actual
hand. `stance_role` is independent and remains correct through stance switches.
Punch trajectory or line is not annotated in this version.

## Windows quick start

The commands work in PowerShell for preparation and in WSL2/Docker for GPU
inference. Check the host first:

```powershell
python mtdata.py doctor
docker version
wsl --version
```

Create one project or a project for every video in a folder:

```powershell
python mtdata.py ingest datasets/videos/sparring_1.mp4
python mtdata.py ingest datasets/videos
```

Run core inference on a project:

```powershell
python mtdata.py infer generated/action_annotations_v2/sparring_1 --device cuda:0 --tracker botsort.yaml
python mtdata.py status generated/action_annotations_v2/sparring_1
```

Core inference deliberately proposes high-recall pose-actionness intervals and
coarse limb/family evidence. Exact techniques require a learned branch. Diagnose
coverage and classifier health with:

```powershell
python mtdata.py diagnose-actions generated/action_annotations_v2/sparring_1
```

After a short `--max-frames` smoke test, a later full tracking run refreshes an
automatic fighter map to the full duration. A verified or manually edited map
is preserved and must be extended by its reviewer when incomplete.

The current tracker output contains both the raw tracker ID and a stitched
canonical `fighter_id` on each selected detection. RTMPose saves that canonical
identity with every skeleton and PoseC3D export uses it directly. This prevents
raw-ID fragmentation from swapping actor/opponent slots. To deliberately
replace an older verified or modified map, use `--refresh-fighter-map`; the old
map is saved as a timestamped backup:

```powershell
python mtdata.py infer PROJECT --device cuda:0 --refresh-fighter-map --overwrite
```

For a pilot tracker bakeoff, export candidate JSONL files from BoT-SORT,
DeepOCSORT/OccluBoost and the simple baseline, review a short gold JSONL, then
run:

```powershell
python mtdata.py benchmark-trackers --gold review/tracker_gold.jsonl `
  --candidate botsort=tracks/botsort.jsonl `
  --candidate deepocsort=tracks/deepocsort.jsonl `
  --candidate occluboost=tracks/occluboost.jsonl `
  --output qa/tracker_benchmark.json
```

The report selects by IDF1 with an ID-switch penalty and also records coverage
and mean IoU.

Optional SAM2 identity masks and independent MMPose-compatible RTMW teachers
can be attached without changing the core run:

```powershell
python mtdata.py infer PROJECT --profile full --device cuda:0 `
  --sam2-config models/sam2.1/sam2.1_hiera_l.yaml `
  --sam2-checkpoint models/sam2.1/sam2.1_hiera_large.pt `
  --pose-teacher rtmw models/rtmw/config.py models/rtmw/checkpoint.pth
```

Teacher disagreement is automatically included in RTMPose feedback sampling.
Sapiens output can use the same COCO17 NPZ contract when supplied through a
local adapter; it is not required for core execution.

For a short compatibility test, add `--max-frames 120`. Do not use that option
for a real dataset.

## WSL2 and Docker on an RTX 5090

Use a current NVIDIA Windows driver and Docker Desktop's WSL2 backend. Do not
install a second Linux display driver inside WSL. The supplied image uses a
CUDA 12.8 base and installs a cu128 PyTorch build.

From WSL at the repository directory:

```bash
docker compose -f docker/compose.yaml build perception
docker compose -f docker/compose.yaml run --rm perception python mtdata.py doctor
docker compose -f docker/compose.yaml run --rm perception \
  python mtdata.py infer generated/action_annotations_v2/sparring_1 --device cuda:0
```

The full target adds Transformers/Qwen and SAM2 dependencies. FiftyOne remains
isolated in the annotation target:

```bash
docker compose -f docker/compose.yaml build full
```

For action discovery without SAM2, use the smaller isolated action target:

```bash
docker compose -f docker/compose.yaml --profile action build action
docker compose -f docker/compose.yaml --profile action run --rm action \
  python mtdata.py doctor --profile action --require-gpu
```

The Compose action image is layered from the already-built
`muay-thai-data-factory-perception:latest` image through
`docker/Dockerfile.action`. It installs only the Qwen/Transformers additions
and current application code, so it does not rebuild MMCV, MMPose, YOLO, or
PyTorch. Build the perception service once first if that local image is not
present. Do not add `--pull` when building this local derivative image.

Large model weights remain mounted under `models/` and are not downloaded by
pipeline code.

## Local Qwen3-VL review assistance

Download a Qwen3-VL model separately to a local directory. The adapter passes
`local_files_only=True`, so it cannot silently upload video or download a
model. Run:

```powershell
python mtdata.py infer generated/action_annotations_v2/sparring_1 `
  --profile full --vlm-model models/Qwen3-VL-8B-Instruct --device cuda:0
```

The VLM sees three-second actor-centric composite clips at 8 FPS. The ordered
frames are passed through Qwen's native video input rather than as unrelated
still images. Every sampled frame must contain the requested canonical actor;
windows with less than 60% actor-crop coverage are rejected instead of being
classified against the wrong fighter. Hybrid mode
classifies every pose-actionness candidate and scans uncovered time at a
1.5-second hop. It can return zero or multiple actions per window. Every
response must be a validated action assessment or an explicit no-action
assessment. Actions include ordered boundaries, ranked techniques, factorized
attributes, confidence and concise observable evidence; no-action responses
require a confidence and reason. Exact prompts, structured JSON, raw
responses, parse failures, crop statistics, preview contact sheets, window
coverage and per-branch evidence are retained.

Candidate windows are grouped before Qwen inference. Pose motion is segmented
component-first, so several numerical peaks within one strike no longer create
identical intervals. A component is split only when separated peaks have a
sustained valley, preserving fast combinations. `pose_attributes` enriches the
pose event and is not counted as an independent second vote during fusion.

Pose-only mode is a candidate generator, not a technique classifier. Hybrid
mode writes confirmed model classifications to
`proposals/classified_actions.jsonl` and keeps unmatched pose motion in
`proposals/unresolved_motion.jsonl`. This distinction prevents ordinary guard,
footwork, or tracking noise from being forced into a punch/kick label. Human
review can reject unresolved motion or assign a technique when the video makes
it clear.

Run only the action stages, reusing existing tracks and poses:

```powershell
python mtdata.py actions generated/action_annotations_v2/sparring_1 `
  --mode hybrid --scan-mode hybrid `
  --vlm-model models/Qwen3-VL-8B-Instruct --device cuda:0
```

Optional Boxing6/PoseC3D/TriDet adapters can supply canonical proposal JSONL
through repeatable `--model-branch PATH` arguments. Model predictions never
become ground truth without review.

## Review with FiftyOne and CVAT

Prepare review artifacts without launching anything:

```powershell
python mtdata.py review generated/action_annotations_v2/sparring_1
```

This writes `review/review_manifest.json` and a CVAT label contract. With
FiftyOne installed:

```powershell
python mtdata.py review generated/action_annotations_v2/sparring_1 --launch
```

Edit proposal fields/statuses in `action_proposals`. Add temporal labels named
`reviewed` for exhaustively watched ranges, or `unreviewed`, `unusable`,
`camera_cut`, or `replay` for ignored ranges. When the FiftyOne session closes,
the pipeline writes `review/fiftyone_reviewed_events.jsonl`, which can be
passed directly to `mtdata sync-review`.
The persistent FiftyOne dataset resumes when its proposal fingerprint matches.
After intentionally regenerating proposals, preserve any needed export and
relaunch with `--reset-dataset` to start a new review.

Use FiftyOne for temporal intervals. Use the official local CVAT Docker
deployment for track, box and skeleton corrections, importing the YOLO or COCO
preannotations from `exports/yolo_feedback` and `exports/rtmpose_feedback`.
The included HTML annotator is an offline fallback and supports every v2 field.

After review, promote the JSONL only if it validates:

```powershell
python mtdata.py sync-review PROJECT reviewed_events.jsonl --overwrite
```

For TriDet training, every usable second must be covered by a verified
`review_range`; all camera cuts/replays/unusable intervals need `ignore`.
Verified actions may not overlap ignored time. The exporter subtracts ignored
time and creates separate actor streams for the remaining contiguous ranges,
so stock TriDet never learns a replay or camera cut as background.

## PoseC3D output

```powershell
python mtdata.py export PROJECT --allow-partial
```

`exports/posec3d/posec3d_annotations.pkl` contains:

- actor in person slot 0 and opponent in slot 1;
- keypoints `[2, 48, 17, 2]` and confidence `[2, 48, 17]`;
- technique class label;
- anatomical-side, stance-role, target-zone and outcome label IDs;
- masks that remove unknown/not-applicable attributes from auxiliary losses.

Canonical action intervals contain `start_sec`, `peak_sec`, `end_sec` and their
30 FPS proxy-frame equivalents. Context is stored separately and added by the
PoseC3D exporter; it is never written into TriDet ground-truth boundaries.
Probable duplicate verified events are rejected before export.

`augmentation_policy.json` defines safe training-time skeleton augmentation:
small temporal jitter/speed changes, scale/rotation/translation, keypoint noise,
and light joint dropout. Apply it only after source-video splitting and only to
the training split. Horizontal flipping is disabled by default because it can
corrupt anatomical-side and stance-role supervision unless those semantics are
updated in lockstep.

`factorized_head.py` supplies the masked PyTorch multi-task head contract. The
standard technique label remains compatible with an ordinary PoseC3D head.

## TriDet and VideoMAE output

`exports/tridet/annotations.json` contains one actor-conditioned stream per
fighter and contiguous trainable range, preserving simultaneous overlapping
attacks while excluding ignored footage. Segments are relative to each
stream; source-video seconds and source/proxy frames remain alongside them for
audit. Canonical labels are techniques; `flattened_annotations` provides an
optional combined baseline.

Extract dense, local VideoMAE/VideoMAEv2 features from an already downloaded
Hugging Face-compatible model:

```powershell
python mtdata.py features PROJECT --model models/videomae-v2 --device cuda --stride 4
```

Run `mtdata export` before feature extraction. Features are FP16 `[T,C]` arrays
under `exports/tridet/features`, sliced so every receptive-field clip lies
fully inside its trainable range. The manifest records model path, checksum,
shape, clip length, timestamps and feature stride for OpenTAD/TriDet config.

## YOLO and RTMPose feedback

The export command mines difficult frames from low detection confidence,
fighter count mismatch, overlapping boxes, low joint confidence and action
peaks. Outputs are explicitly marked `preannotation`:

- `exports/yolo_feedback`: images, preliminary YOLO labels, review manifest.
- `exports/rtmpose_feedback`: images, preliminary COCO17 JSON, review manifest.

Review these in CVAT before adding them to the existing grouped YOLO and
RTMPose datasets. Never feed preliminary labels directly into training.

## Freeze a multi-video dataset

Export every fully reviewed project, then freeze a version:

```powershell
python mtdata.py freeze generated/action_annotations_v2 `
  --output generated/frozen/muay_thai_actions_v1 --seed 42
```

The split unit is the source video, never an event clip or frame. The frozen
folder contains combined PoseC3D and TriDet annotations, copied
`tridet_features/<stream>.npy` arrays, timestamps, checksums, a split manifest,
dataset card and fingerprint. Freeze fails when a feature stream is missing.

## Training launchers

The launcher prints commands by default and only trains with `--execute`:

```powershell
python mtdata.py train yolo -- --dataset PATH --device 0
python mtdata.py train rtmpose -- --data-root PATH --device cuda
python mtdata.py train posec3d --config PATH_TO_MMACTION2_CONFIG
python mtdata.py train tridet --repo PATH_TO_OPENTAD --config PATH_TO_CONFIG
```

Add `--execute` after the model name when the printed command is correct.

## Acceptance and troubleshooting

```powershell
python mtdata.py evaluate PROJECT
```

The report measures proposal recall at tIoU 0.3, top-three technique recall,
visible-side accuracy, median peak error and annotation validity. Missing
optional dependencies are reported by `mtdata doctor`; the core pipeline does
not require Qwen, SAM2, BoxMOT, FiftyOne, Sapiens or RTMW to remain usable.

Use `pipeline_state.sqlite3`, `logs/`, and `mtdata status` to diagnose a failed
stage. A repaired stage resumes without repeating current predecessors.

`qa/action_diagnostics.json` now fails closed for incomplete fighter mapping,
truncated timelines, high exact-duplicate rates, or a hybrid Qwen run that
processed windows but produced no normalized events. In that last case inspect
`proposals/raw/qwen/summary.json`, individual task JSON, and
`proposals/raw/qwen/previews/` rather than treating an empty JSONL as success.
