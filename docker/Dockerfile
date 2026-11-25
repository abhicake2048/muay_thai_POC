FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

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
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
COPY pyproject.toml mtdata.py /opt/mtdata/
COPY action_annotation_pipeline /opt/mtdata/action_annotation_pipeline
RUN python -m pip install -e /opt/mtdata

FROM base AS perception
COPY docker/requirements-perception.txt /opt/mtdata/requirements-perception.txt
RUN python -m pip install --no-build-isolation \
    "xtcocotools==1.14.3"
RUN python -m pip install -r /opt/mtdata/requirements-perception.txt
RUN python -m pip check && python -c \
    "import cv2, mmcv, mmengine, mmpose, torch, ultralytics, xtcocotools; print('perception imports OK'); print('torch', torch.__version__, 'CUDA wheel', torch.version.cuda); print('mmcv', mmcv.__version__, 'mmpose', mmpose.__version__)"

FROM perception AS full
COPY docker/requirements-full.txt /opt/mtdata/requirements-full.txt
RUN python -m pip install -r /opt/mtdata/requirements-full.txt

CMD ["python", "mtdata.py", "doctor"]
