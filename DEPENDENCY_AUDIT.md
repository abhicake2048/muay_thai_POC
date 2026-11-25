# Dependency compatibility audit

Audited for Ubuntu 22.04, CPython 3.10, CUDA 12.8 and the RTX 5090 runtime.
The source-of-truth pins are the files under `docker/`; this document records
why the environment is split and how it is verified.

## Profiles

| Profile | Important exact versions | Purpose |
|---|---|---|
| base | Python 3.10, NumPy 1.26.4, PyTorch 2.7.1+cu128, TorchVision 0.22.1 | Shared CUDA runtime |
| perception | OpenCV 4.11.0.86, Ultralytics 8.4.218, BoxMOT 16.0.11, MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.2.0, MMPose 1.3.2, xtcocotools 1.14.3, pycocotools 2.0.11 | YOLO, tracking and RTMPose |
| action | perception plus Transformers 4.57.6, Accelerate 1.10.1, Qwen VL Utils 0.0.14 and PyAV 16.0.1 | Local Qwen3-VL action discovery without SAM2 |
| full | perception plus Transformers 4.57.6, Accelerate 1.10.1, Qwen VL Utils 0.0.14, PyAV 16.0.1 and an immutable SAM2 commit | VideoMAE, Qwen3-VL and SAM2 |
| annotation | OpenCV-headless 4.11.0.86 and FiftyOne 1.10.0 | Temporal review UI, isolated from perception |
| RTMPose training | NumPy 1.26.4, PyTorch 2.1.0, TorchVision 0.16.0, OpenMMLab versions above | Existing Colab/local training workflow |

## Conflicts removed

1. `mmcv-lite` does not contain `mmcv._ext`.  The perception image builds full
   MMCV 2.1.0 with its CPU C++ extension before MMPose is imported.  RTMPose
   inference itself still runs on CUDA through PyTorch.
2. BoxMOT 26.0.0 requires NumPy 2.2 or newer.  OpenMMLab and the existing
   compiled extensions use NumPy 1.26.4, so BoxMOT is pinned to 16.0.11.
3. Transformers 5.x and current Accelerate releases require Python 3.11.  The
   Qwen stack is pinned to the last audited Transformers 4.x/Python 3.10 set.
4. `opencv-python` and `opencv-python-headless` install the same `cv2` files.
   FiftyOne is therefore built in a separate annotation image.  Albumentations
   was removed from the RTMPose environment and replaced by the local
   `CombatBlurDropout` transform so pose training also contains one OpenCV
   distribution only.
5. The SAM2 Git dependency is pinned to commit
   `393ae336a752d26e68fb9a586e3d4ac14ff1e3c5`; a moving `main` branch can no
   longer change an otherwise identical rebuild.  Its optional CUDA extension
   and PEP 517 build isolation are disabled during installation, preventing a
   temporary build environment from downloading another PyTorch version.
   SAM2 inference continues to use CUDA through the pinned runtime PyTorch.
6. pip, setuptools, wheel and Cython are pinned so chumpy and MMCV legacy build
   behaviour cannot change underneath the runtime pins.

## Enforcement

Each Docker stage runs both `pip check` and
`action_annotation_pipeline.dependency_check`.  The latter performs real
imports of the binary extensions and the APIs used by the pipeline, rejects
duplicate OpenCV distributions and `mmcv-lite`, checks the PyTorch CUDA runtime,
and checks every direct version above.  A complete resolved snapshot is saved
inside each image:

```text
/opt/mtdata/perception-environment.lock.txt
/opt/mtdata/action-environment.lock.txt
/opt/mtdata/full-environment.lock.txt
/opt/mtdata/annotation-environment.lock.txt
```

The native setup script writes the same snapshot to
`.venv-gpu-<profile>/environment.lock.txt`.

## Build and verify

```bash
docker compose -f docker/compose.yaml --profile perception build --no-cache perception
docker compose -f docker/compose.yaml --profile perception run --rm perception \
  python mtdata.py doctor --profile perception --require-gpu

docker compose -f docker/compose.yaml --profile action build action
docker compose -f docker/compose.yaml --profile action run --rm action \
  python mtdata.py doctor --profile action --require-gpu

docker compose -f docker/compose.yaml --profile full build full
docker compose -f docker/compose.yaml --profile full run --rm full \
  python mtdata.py doctor --profile full --require-gpu

docker compose -f docker/compose.yaml --profile annotation build annotation
docker compose -f docker/compose.yaml --profile annotation run --rm annotation \
  python mtdata.py doctor --profile annotation
```

`docker/Dockerfile.action` starts from the tagged local perception image. This
keeps YOLO/RTMPose unchanged and prevents ordinary source updates from
rebuilding the expensive MMCV/MMPose layers.

The first perception build compiles MMCV and can take several minutes.  Use
`MMCV_MAX_JOBS=1` (the default) on memory-constrained systems.
