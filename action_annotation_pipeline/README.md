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
contact sheets and `proposals/raw/qwen/summary.json` before review. Every Qwen
window must now return one of two validated results:

- one or more actions with ordered boundaries, ranked technique, confidence,
  concise observable classification/boundary evidence; or
- an explicit no-action decision with confidence and an observable reason.

The complete audit ledger is
`proposals/branches/qwen_window_assessments.jsonl`; malformed or semantically
incomplete JSON is retried once and then recorded as a failed window. Hybrid
diagnostics fail when dense assessment coverage is incomplete. Raw prompts,
responses, crop coverage and normalized IDs remain in `proposals/raw/qwen/`.

Open the generated `review/annotate.html` for the dependency-free reviewer, or
install FiftyOne and add `--launch`. Closing the FiftyOne session writes
`review/fiftyone_reviewed_events.jsonl`; labels named `reviewed`, `unreviewed`,
`unusable`, `camera_cut`, or `replay` round-trip as review/ignore intervals.
The persistent review resumes when the proposal fingerprint is unchanged. If
you intentionally regenerated proposals, export/close the old review and pass
`--reset-dataset` once to replace it.
After exporting reviewed JSONL:

```powershell
python mtdata.py sync-review generated/action_annotations_v2/sparring_1 reviewed_events.jsonl --overwrite
python mtdata.py export generated/action_annotations_v2/sparring_1 --allow-partial
python mtdata.py evaluate generated/action_annotations_v2/sparring_1
```

For training exports, `exports/posec3d/posec3d_annotations.pkl` follows the
MMAction2 PoseDataset tensor contract (`[2,T,17,2]` keypoints and
`[2,T,17]` confidence). Exact reviewed event seconds and proxy frames are also
stored in each action sample and in `sample_metadata.jsonl`. TriDet annotations
contain stream FPS, second/frame segments and stable `label_id` values. Run
`mtdata features` before `mtdata freeze`; freezing now copies each
actor-conditioned `[T,C]` VideoMAE array into `tridet_features/` and rejects an
incomplete training package.

TriDet streams are split around verified/uncertain `ignore` intervals. Their
local annotations and VideoMAE clips therefore contain only reviewed,
trainable footage; source seconds and frames remain attached for audit.
Verified actions that overlap an ignored range are rejected during sync/export.

See `DATA_FACTORY_V2.md` in the repository root for setup, full-profile model
adapters, CVAT handoff, training and dataset-freeze instructions.
