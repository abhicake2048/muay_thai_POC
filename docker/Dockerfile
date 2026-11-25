# MMPose 1.3.2 requires xtcocotools.  xtcocotools 1.14.3 publishes Linux
# wheels for CPython 3.10/3.11, but not CPython 3.12 (Ubuntu 24.04's
# default).  Use Ubuntu 22.04/Python 3.10 so pip can install the supported
# wheel instead of trying to compile the legacy extension from source.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    git ffmpeg libgl1 libglib2.0-0 build-essential ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV"

RUN python -m pip install --upgrade \
    pip "setuptools<81" wheel "Cython>=0.29.36,<3" "numpy>=1.26,<2"

RUN python -m pip install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
COPY pyproject.toml mtdata.py /opt/mtdata/
COPY action_annotation_pipeline /opt/mtdata/action_annotation_pipeline
RUN python -m pip install -e /opt/mtdata

FROM base AS perception
COPY docker/requirements-perception.txt /opt/mtdata/requirements-perception.txt
RUN python -m pip install \
    --only-binary=:all: "xtcocotools==1.14.3"
RUN python -m pip install -r /opt/mtdata/requirements-perception.txt
RUN python -m pip check && python -c \
    "import cv2, mmcv, mmengine, mmpose, torch, ultralytics, xtcocotools; print('perception imports OK'); print('torch', torch.__version__, 'CUDA wheel', torch.version.cuda); print('mmcv', mmcv.__version__, 'mmpose', mmpose.__version__)"

FROM perception AS full
COPY docker/requirements-full.txt /opt/mtdata/requirements-full.txt
RUN python -m pip install -r /opt/mtdata/requirements-full.txt

CMD ["python", "mtdata.py", "doctor"]
