# Step 2: RTMPose-S Fighter Pose Estimation

This stage fine-tunes RTMPose-S on the UFC/MMA COCO-17 annotations. BoxingVI is
not used here because the local copy contains skeleton sequences but no aligned
RGB images. It remains input for the PoseConv3D stage.

## Dataset conversion

From the repository root:

```powershell
python convert_to_coco.py
```

The converter creates leakage-free grouped splits under
`generated/rtmpose_coco/`, drops the 31 box-only rows, validates the expected
counts, and renders 20 overlays for inspection.

## Google Colab T4

1. Select **Runtime > Change runtime type > T4 GPU** and verify with
   `!nvidia-smi`.
2. Upload or clone this repository and place
   `mma-fighter-pose-estimation-dataset/` at the repository root.
3. Create the pinned Python 3.11 environment and install the matching CUDA
   PyTorch/MMCV wheels:

```python
%cd /content/Muay_thai
!pip install -q uv
!uv venv --python 3.11 --seed .venv-mmpose
!.venv-mmpose/bin/python install_mmpose_deps.py --torch cu121
```

`--seed` installs pip into the new environment. The dependency installer also
bootstraps pip automatically if an older command created an unseeded environment.
The `uv` warning about `--system` is harmless and is unrelated to MMPose.

4. Convert and visually inspect the annotations:

```python
!.venv-mmpose/bin/python convert_to_coco.py
```

5. Run a one-epoch smoke test:

```python
!.venv-mmpose/bin/python train_rtmpose.py \
  --epochs 1 \
  --batch-size 32 \
  --workers 2 \
  --device cuda \
  --work-dir /content/drive/MyDrive/Muay_thai/rtmpose_smoke
```

The scripts force Matplotlib's headless `Agg` backend. This prevents Colab's
inherited `module://matplotlib_inline.backend_inline` setting from breaking the
isolated MMPose environment.

6. Run the full fine-tune. Batch 64 is the intended T4 setting; use 32 if GPU
   memory is insufficient:

```python
!.venv-mmpose/bin/python train_rtmpose.py \
  --epochs 100 \
  --batch-size 64 \
  --workers 2 \
  --device cuda \
  --work-dir /content/drive/MyDrive/Muay_thai/rtmpose_s_combat
```

For higher T4 throughput, try batch 128 first. The trainer scales the learning
rate automatically. Disabling flip-test and validating every 10 epochs reduces
evaluation overhead while retaining best-checkpoint selection:

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
  --work-dir /content/drive/MyDrive/Muay_thai/rtmpose_s_combat_fast
```

Resume an interrupted run using the epoch checkpoint named in `last_checkpoint`:

```python
!.venv-mmpose/bin/python train_rtmpose.py \
  --epochs 100 \
  --batch-size 64 \
  --workers 2 \
  --device cuda \
  --work-dir /content/drive/MyDrive/Muay_thai/rtmpose_s_combat \
  --resume /content/drive/MyDrive/Muay_thai/rtmpose_s_combat/epoch_XX.pth
```

## Windows CPU environment

Use a separate environment from YOLO:

```powershell
py -3.11 -m venv .venv-mmpose
.\.venv-mmpose\Scripts\python.exe install_mmpose_deps.py --torch cpu
.\.venv-mmpose\Scripts\python.exe convert_to_coco.py
```

Use CPU only for conversion and a one-epoch compatibility test. Full training is
not practical on the current machine:

```powershell
.\.venv-mmpose\Scripts\python.exe train_rtmpose.py `
  --epochs 1 `
  --batch-size 2 `
  --workers 0 `
  --device cpu `
  --no-amp `
  --train-samples 8 `
  --skip-validation `
  --work-dir work_dirs/rtmpose_cpu_smoke
```

`--train-samples` and `--skip-validation` are smoke-test controls only. Omit
both for the full leakage-free train/validation run.

To exercise the COCO evaluator without evaluating all 1,035 validation poses on
CPU, replace `--skip-validation` with `--validation-samples 16`.

## End-to-end MP4 inference

Use an unseen 30–90 second clip. The output contains video only; the script does
not copy the source audio stream.

```powershell
python infer_pose_video.py input.mp4 annotated.mp4 `
  --detector weights/yolo_fighter_detector/best.pt `
  --pose-checkpoint work_dirs/rtmpose_s_combat/best_coco_AP_epoch_XX.pth `
  --device cuda:0
```

The script keeps the two strongest fighter detections per frame and draws the
17-keypoint skeleton. Persistent fighter tracking and strike classification are
deliberately deferred to the next stage.
