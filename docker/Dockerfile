FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    NO_ALBUMENTATIONS_UPDATE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-setuptools python3-venv python3-wheel \
    git ffmpeg libgl1 libglib2.0-0 build-essential ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --upgrade \
    pip "setuptools<81" wheel "Cython>=0.29.36,<3" "numpy>=1.26,<2"

RUN python3 -m pip install --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
COPY pyproject.toml mtdata.py /opt/mtdata/
COPY action_annotation_pipeline /opt/mtdata/action_annotation_pipeline
RUN python3 -m pip install --break-system-packages -e /opt/mtdata

FROM base AS perception
COPY docker/requirements-perception.txt /opt/mtdata/requirements-perception.txt
RUN python3 -m pip install --break-system-packages --no-build-isolation \
    "xtcocotools==1.14.3"
RUN python3 -m pip install --break-system-packages -r /opt/mtdata/requirements-perception.txt
RUN python3 -m pip check && python3 -c \
    "import cv2, mmcv, mmengine, mmpose, torch, ultralytics, xtcocotools; print('perception imports OK'); print('torch', torch.__version__, 'CUDA wheel', torch.version.cuda); print('mmcv', mmcv.__version__, 'mmpose', mmpose.__version__)"

FROM perception AS full
COPY docker/requirements-full.txt /opt/mtdata/requirements-full.txt
RUN python3 -m pip install --break-system-packages -r /opt/mtdata/requirements-full.txt

CMD ["python3", "mtdata.py", "doctor"]
