# Muay Thai Data Factory v2

This package turns sparring videos into one canonical, factorized temporal
dataset and derives PoseC3D, TriDet, YOLO-feedback and RTMPose-feedback data.

The pipeline is local-first and resumable. `pipeline_state.sqlite3` records
stage signatures, status, commands and outputs. Re-running a current stage
skips it; changed inputs invalidate only the affected stage.

Quick start from the repository root:

```powershell
python mtdata.py doctor
python mtdata.py ingest datasets/videos/sparring_1.mp4
python mtdata.py infer generated/action_annotations_v2/sparring_1 --device cuda:0
python mtdata.py diagnose-actions generated/action_annotations_v2/sparring_1
python mtdata.py review generated/action_annotations_v2/sparring_1
```

`infer` always creates high-recall pose-actionness windows. It does not invent
fine-grained techniques. Run the action-only hybrid stage in the local Qwen
image to classify candidates and scan uncovered time without repeating YOLO or
RTMPose:

```powershell
python mtdata.py actions generated/action_annotations_v2/sparring_1 `
  --mode hybrid --vlm-model models/Qwen3-VL-8B-Instruct --device cuda:0
```

Inspect `qa/action_diagnostics.json` and `qa/action_timeline.html`. Automatic
labels remain proposals until a person verifies them.

`--mode pose` intentionally produces motion candidates only; it cannot name a
jab, kick, elbow, or defense. `--mode hybrid` runs the Qwen technique branch.
Hybrid output is split into `proposals/classified_actions.jsonl` and
`proposals/unresolved_motion.jsonl`. Only the first file contains automatic
technique labels. Unresolved motion stays visible for review and is never
forced into a guessed class.

Projects created by an earlier build should refresh the canonical identity map
once. The existing map is backed up before replacement, then RTMPose is rerun
so every pose row contains a canonical `fighter_id`:

```powershell
python mtdata.py infer generated/action_annotations_v2/sparring_1 `
  --device cuda:0 --refresh-fighter-map --overwrite
```

The action stage now uses component-first temporal segmentation (one interval
per motion component), ordered Qwen video input at 8 FPS, actor-crop coverage
checks, and cross-window duplicate fusion. Check the generated Qwen preview
contact sheets and `proposals/raw/qwen/summary.json` before review.

Open the generated `review/annotate.html` for the dependency-free reviewer, or
install FiftyOne and add `--launch`. After exporting reviewed JSONL:

```powershell
python mtdata.py sync-review generated/action_annotations_v2/sparring_1 reviewed_events.jsonl --overwrite
python mtdata.py export generated/action_annotations_v2/sparring_1 --allow-partial
python mtdata.py evaluate generated/action_annotations_v2/sparring_1
```

See `DATA_FACTORY_V2.md` in the repository root for setup, full-profile model
adapters, CVAT handoff, training and dataset-freeze instructions.
