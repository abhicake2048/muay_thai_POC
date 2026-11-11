# Colab training: YOLO11-S detector + RTMPose-S

The two datasets have separate jobs:

| Stage | Dataset | Annotation used |
|---|---|---|
| YOLO11-S fighter detector | `mma-fighter-detection-dataset` | Five-field `fighter` boxes |
| YOLO11-S supplemental detector data | `punch-kick-player detection.v4i.yolov11` | Blue/Red Player boxes remapped to `fighter` |
| RTMPose-S pose estimator | `mma-fighter-pose-estimation-dataset` | COCO-17 keypoints converted from YOLO-Pose |

Do not train the detector from the pose labels. `train_yolo.py` rejects pose
rows, and `convert_to_coco.py` only reads the pose dataset.

## 1. Put code and data in Colab

Push the code to GitHub, but keep both datasets out of Git. Upload these two
archives to `MyDrive/Muay_thai/datasets/`:

```text
mma-fighter-detection-dataset.zip
mma-fighter-pose-estimation-dataset.zip
kick-punch-fighter-detection-dataset.zip
```

Start a T4 runtime and run:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/abhicake2048/muay_thai_POC.git Muay_thai
%cd /content/Muay_thai

!unzip -q /content/drive/MyDrive/Muay_thai/datasets/mma-fighter-detection-dataset.zip -d /content/Muay_thai
!unzip -q /content/drive/MyDrive/Muay_thai/datasets/mma-fighter-pose-estimation-dataset.zip -d /content/Muay_thai
!unzip -q /content/drive/MyDrive/Muay_thai/datasets/kick-punch-fighter-detection-dataset.zip -d /content/Muay_thai
```

Verify that the archives produced exactly these paths:

```python
from pathlib import Path

root = Path('/content/Muay_thai')
det = root / 'mma-fighter-detection-dataset'
pose = root / 'mma-fighter-pose-estimation-dataset'
supplemental = root / 'punch-kick-player detection.v4i.yolov11'

assert (det / 'data.yaml').is_file()
assert (det / 'train/images').is_dir()
assert (det / 'train/labels').is_dir()
assert (pose / 'data.yaml').is_file()
assert (pose / 'train/images').is_dir()
assert (pose / 'train/labels').is_dir()
assert (supplemental / 'data.yaml').is_file()
assert (supplemental / 'train/images').is_dir()
assert (supplemental / 'train/labels').is_dir()
print('All three datasets are in the correct location.')
```

Keep the datasets under `/content`, not `/content/drive`, during training. Drive
is used only for persistent checkpoints and outputs.

## 2. Train YOLO11-S on detection boxes

Use Colab's CUDA-enabled Python for Ultralytics:

```python
%cd /content/Muay_thai
!python -m pip install -q -r requirements.txt
!python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Validate and regroup the dataset first:

```python
!python train_yolo.py --prepare-only
```

The preparation script keeps the datasets separate, remaps `Blue Player` and
`Red Player` to class `0:fighter`, ignores punch/kick boxes, and excludes 69
supplemental images that have no player box. Do not copy the supplemental files
into the primary dataset manually.

Expected combined counts:

```text
train: 4,590 images, 9,119 boxes
val:     519 images, 1,036 boxes
test:    513 images, 1,026 boxes
```

All 516 usable supplemental images are added to training only. The main MMA
validation and test fights remain unchanged and leakage-free. If the extracted
supplemental folder has a different name, pass its path explicitly:

```python
!python train_yolo.py --prepare-only \
  --supplemental-dataset "/content/Muay_thai/YOUR_SUPPLEMENTAL_FOLDER"
```

Run a short smoke test:

```python
!python train_yolo.py \
  --epochs 1 \
  --fraction 0.10 \
  --batch 16 \
  --workers 2 \
  --device 0 \
  --cache ram \
  --project /content/drive/MyDrive/Muay_thai/yolo_smoke \
  --name yolo11s_detector_smoke \
  --export /content/drive/MyDrive/Muay_thai/yolo_smoke/best.pt
```

Run the full detector fine-tune. `--batch -1` lets Ultralytics choose a batch
that fits the T4:

```python
!python train_yolo.py \
  --epochs 50 \
  --imgsz 640 \
  --batch -1 \
  --workers 2 \
  --device 0 \
  --cache ram \
  --patience 15 \
  --project /content/drive/MyDrive/Muay_thai/yolo_fighter_detector \
  --name yolo11s_fighter_detection \
  --export /content/drive/MyDrive/Muay_thai/yolo_fighter_detector/best.pt
```

The detector to use later is:

```text
/content/drive/MyDrive/Muay_thai/yolo_fighter_detector/best.pt
```

## 3. Train RTMPose-S on pose keypoints

Create the separate pinned OpenMMLab environment:

```python
%cd /content/Muay_thai
!python -m pip install -q uv
!uv venv --python 3.11 --seed .venv-mmpose
!.venv-mmpose/bin/python install_mmpose_deps.py --torch cu121
```

Convert only the pose dataset:

```python
!.venv-mmpose/bin/python convert_to_coco.py \
  --dataset /content/Muay_thai/mma-fighter-pose-estimation-dataset
```

Run a bounded smoke test:

```python
!.venv-mmpose/bin/python train_rtmpose.py \
  --epochs 1 \
  --batch-size 128 \
  --workers 4 \
  --device cuda \
  --amp \
  --train-samples 256 \
  --validation-samples 64 \
  --work-dir /content/drive/MyDrive/Muay_thai/rtmpose_smoke
```

Run the full pose fine-tune:

```python
!.venv-mmpose/bin/python train_rtmpose.py \
  --epochs 50 \
  --batch-size 128 \
  --workers 4 \
  --device cuda \
  --amp \
  --no-flip-test \
  --val-interval 10 \
  --checkpoint-interval 10 \
  --work-dir /content/drive/MyDrive/Muay_thai/rtmpose_s_combat
```

Use batch 64 if batch 128 runs out of GPU memory. The best checkpoint is named
`best_coco_AP_epoch_XX.pth` in the RTMPose work directory.

## 4. Run the combined video pipeline

Select the best pose checkpoint:

```python
from pathlib import Path

pose_dir = Path('/content/drive/MyDrive/Muay_thai/rtmpose_s_combat')
POSE_CHECKPOINT = str(max(pose_dir.glob('best_coco_AP*.pth'), key=lambda p: p.stat().st_mtime))
YOLO_CHECKPOINT = '/content/drive/MyDrive/Muay_thai/yolo_fighter_detector/best.pt'
INPUT_VIDEO = '/content/drive/MyDrive/Muay_thai/custom_videos/fight.mp4'
OUTPUT_VIDEO = '/content/drive/MyDrive/Muay_thai/custom_videos/fight_skeletons.mp4'
print(POSE_CHECKPOINT)
```

Run detection followed by top-down pose inference:

```python
!.venv-mmpose/bin/python infer_pose_video.py \
  "{INPUT_VIDEO}" \
  "{OUTPUT_VIDEO}" \
  --detector "{YOLO_CHECKPOINT}" \
  --pose-checkpoint "{POSE_CHECKPOINT}" \
  --device cuda:0 \
  --det-conf 0.35 \
  --keypoint-conf 0.30 \
  --max-fighters 2
```

Preview it:

```python
from IPython.display import Video
Video(OUTPUT_VIDEO, embed=True, width=900)
```

The inference script uses the detector only for fighter boxes and RTMPose only
for keypoints. It does not use the old detector trained from pose annotations.
