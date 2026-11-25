FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    NO_ALBUMENTATIONS_UPDATE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
COPY pyproject.toml mtdata.py /opt/mtdata/
COPY action_annotation_pipeline /opt/mtdata/action_annotation_pipeline
RUN python3 -m pip install --break-system-packages -e /opt/mtdata

FROM base AS perception
COPY docker/requirements-perception.txt /opt/mtdata/requirements-perception.txt
RUN python3 -m pip install --break-system-packages -r /opt/mtdata/requirements-perception.txt
RUN mim install "mmcv>=2.0.1"

FROM perception AS full
COPY docker/requirements-full.txt /opt/mtdata/requirements-full.txt
RUN python3 -m pip install --break-system-packages -r /opt/mtdata/requirements-full.txt

CMD ["python3", "mtdata.py", "doctor"]
