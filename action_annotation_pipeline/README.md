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
python mtdata.py review generated/action_annotations_v2/sparring_1
```

Open the generated `review/annotate.html` for the dependency-free reviewer, or
install FiftyOne and add `--launch`. After exporting reviewed JSONL:

```powershell
python mtdata.py sync-review generated/action_annotations_v2/sparring_1 reviewed_events.jsonl --overwrite
python mtdata.py export generated/action_annotations_v2/sparring_1 --allow-partial
python mtdata.py evaluate generated/action_annotations_v2/sparring_1
```

See `DATA_FACTORY_V2.md` in the repository root for setup, full-profile model
adapters, CVAT handoff, training and dataset-freeze instructions.
