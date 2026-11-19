# Combine all pose datasets and fine-tune RTMPose in Google Colab

This workflow scans `datasets/`, accepts supported pose datasets as folders or
ZIP files, converts them to one COCO-17 schema, removes unusable samples, makes
grouped train/validation/test splits, and fine-tunes RTMPose-S.

The canonical output is one dataset folder:

```text
generated/combined_rtmpose_coco/
├── images/                 # one deduplicated image store
├── annotations/
│   ├── all.json            # the complete, single COCO annotation database
│   ├── train.json          # training view of all.json
│   ├── val.json            # validation view
│   └── test.json           # final test view
├── previews/               # keypoint overlays for manual QA
└── dataset_report.json     # source-by-source cleaning audit
```

The current local dataset snapshot produces 5,972 clean images and 11,853 pose
instances. Counts are printed at runtime and may change when more data is added.

## 1. Upload the project and data

Push the code in this repository to GitHub. Dataset and generated-output paths
are already excluded by `.gitignore`, so do not add the large archives to Git.

In Google Drive, create:

```text
MyDrive/Muay_thai/datasets/
```

Upload the pose archives from the local `datasets/` folder to that Drive folder:

```text
mma-fighter-pose-estimation-dataset.zip
keypoint_mma_remove_no_point_data_coco.zip
MMA_Pose_need_cleaning_coco.zip
```

Other ZIPs can remain in the folder. The preparation script identifies pose
annotations and reports detection-only archives as skipped. For large files,
upload through the Google Drive web interface before starting Colab instead of
using the notebook upload widget.

## 2. Start Colab and clone the repository

Open a new Colab notebook, choose **Runtime > Change runtime type > T4 GPU**, and
run:

```python
from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi
```

Clone the repository to Colab's local SSD:

```python
GIT_URL = "https://github.com/abhicake2048/muay_thai_POC.git"
BRANCH = "main"
REPO = "/content/Muay_thai"

from pathlib import Path
if Path(REPO).is_dir():
    !git -C {REPO} pull --ff-only
else:
    !git clone --depth 1 --branch {BRANCH} {GIT_URL} {REPO}
%cd {REPO}
```

Copy the archives from Drive to the local runtime. Training and image loading
are much faster from `/content` than directly from mounted Drive:

```python
from pathlib import Path
import shutil

drive_data = Path('/content/drive/MyDrive/Muay_thai/datasets')
local_data = Path('/content/Muay_thai/datasets')
local_data.mkdir(parents=True, exist_ok=True)

archives = sorted(drive_data.glob('*.zip'))
assert archives, f'No ZIP files found in {drive_data}'
for source in archives:
    destination = local_data / source.name
    print(f'Copying {source.name} ({source.stat().st_size / 1024**2:.1f} MiB)')
    shutil.copy2(source, destination)

print(f'Copied {len(archives)} archive(s).')
```

## 3. Install the pinned MMPose environment

The isolated Python 3.11 environment avoids conflicts with Colab's preinstalled
packages:

```python
%cd /content/Muay_thai
!python -m pip install -q uv
!uv venv --python 3.11 --seed .venv-mmpose
!.venv-mmpose/bin/python install_mmpose_deps.py --torch cu121
```

Verify CUDA inside that environment:

```python
!.venv-mmpose/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Combine and clean every pose dataset

Run the merger against the entire local `datasets/` folder:

```python
DATA_ROOT = '/content/Muay_thai/generated/combined_rtmpose_coco'

!.venv-mmpose/bin/python prepare_combined_pose_dataset.py \
  --input /content/Muay_thai/datasets \
  --output {DATA_ROOT} \
  --min-keypoints 4 \
  --train-ratio 0.80 \
  --val-ratio 0.10 \
  --render-samples 24 \
  --seed 42 \
  --overwrite
```

Cleaning includes:

- corrupt/unreadable images;
- images with no usable pose annotation;
- YOLO box-only rows;
- malformed or out-of-frame keypoints;
- one-point `clinch` annotations, which are not a body pose;
- pose instances with fewer than four usable keypoints;
- exact duplicate images, using decoded pixels rather than filenames.

The split is assigned by inferred source-frame/video group. A group can appear
in only one partition, preventing augmented copies or adjacent video frames from
leaking between training and evaluation.

Inspect the audit and several overlays before training:

```python
import json
from pathlib import Path
from IPython.display import display
from PIL import Image

root = Path(DATA_ROOT)
report = json.loads((root / 'dataset_report.json').read_text())
print(json.dumps({
    'combined': report['combined'],
    'splits': report['splits'],
    'cleaning': report['cleaning'],
    'group_overlaps': report['group_overlaps'],
}, indent=2))

for path in sorted((root / 'previews').glob('*'))[:12]:
    display(Image.open(path).resize((420, 420)))
```

Do not continue if the skeleton overlays are systematically wrong. The
14-keypoint archive with duplicate limb names lost explicit left/right labels
during export; the converter reconstructs the sides from its skeleton-link
order. The assumption is documented in `prepare_combined_pose_dataset.py`.

## 5. Run a bounded smoke test

Check the full software path with a small subset before starting the long run:

```python
SMOKE_DIR = '/content/drive/MyDrive/Muay_thai/runs/rtmpose_combined_smoke'

!.venv-mmpose/bin/python train_rtmpose.py \
  --data-root {DATA_ROOT} \
  --annotation-dir {DATA_ROOT}/annotations \
  --epochs 1 \
  --batch-size 32 \
  --workers 2 \
  --device cuda \
  --amp \
  --train-samples 256 \
  --validation-samples 64 \
  --work-dir {SMOKE_DIR}
```

## 6. Fine-tune RTMPose-S

This is a safe starting point for a T4. Checkpoints and logs go directly to
Drive so an interrupted Colab runtime does not lose them:

```python
RUN_DIR = '/content/drive/MyDrive/Muay_thai/runs/rtmpose_s_combined'

!.venv-mmpose/bin/python train_rtmpose.py \
  --data-root {DATA_ROOT} \
  --annotation-dir {DATA_ROOT}/annotations \
  --epochs 100 \
  --batch-size 64 \
  --workers 2 \
  --device cuda \
  --amp \
  --no-flip-test \
  --val-interval 5 \
  --checkpoint-interval 5 \
  --test-after-training \
  --work-dir {RUN_DIR}
```

If CUDA runs out of memory, change `--batch-size 64` to `32`. On an L4 or A100,
try `128`. The trainer scales the learning rate with batch size.

The best checkpoint is written as:

```text
MyDrive/Muay_thai/runs/rtmpose_s_combined/best_coco_AP_epoch_XX.pth
```

## 7. Resume after a Colab disconnect

Mount Drive, clone the same code, copy the data, rebuild the deterministic
combined dataset with seed 42, and reinstall the environment. Then locate the
latest epoch checkpoint:

```python
from pathlib import Path

run_dir = Path('/content/drive/MyDrive/Muay_thai/runs/rtmpose_s_combined')
checkpoints = sorted(run_dir.glob('epoch_*.pth'), key=lambda p: p.stat().st_mtime)
assert checkpoints, 'No epoch checkpoint found'
RESUME_FROM = str(checkpoints[-1])
print(RESUME_FROM)
```

Run the same full command with the final target epoch count and add:

```text
--resume {RESUME_FROM}
```

## Optional: persist the prepared dataset

Rebuilding is deterministic, but this optional archive avoids repeating the
merge on a later runtime:

```python
%cd /content/Muay_thai/generated
!tar -czf /content/drive/MyDrive/Muay_thai/combined_rtmpose_coco.tar.gz combined_rtmpose_coco
```

Keep the active training copy under `/content`; using the archived copy directly
from mounted Drive will slow data loading.
