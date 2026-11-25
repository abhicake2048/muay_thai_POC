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

COPY docker/constraints.txt /opt/mtdata/constraints.txt

# Pin the build frontend as well as the runtime packages.  This avoids a new
# pip/setuptools release changing legacy chumpy/MMCV build behaviour.
RUN python -m pip install --upgrade \
    "pip==25.3" "setuptools==80.9.0" "wheel==0.45.1" \
    "Cython==0.29.37" "numpy==1.26.4"

RUN python -m pip install \
    torch==2.7.1 torchvision==0.22.1 \
    -c /opt/mtdata/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
COPY pyproject.toml mtdata.py /opt/mtdata/
COPY action_annotation_pipeline /opt/mtdata/action_annotation_pipeline
RUN python -m pip install --no-deps --no-build-isolation -e /opt/mtdata

FROM base AS perception
COPY docker/requirements-perception.txt /opt/mtdata/requirements-perception.txt
RUN python -m pip install \
    -c /opt/mtdata/constraints.txt \
    --only-binary=:all: "xtcocotools==1.14.3"
# chumpy 0.70's legacy setup.py imports pip internals.  Modern PEP 517 build
# isolation deliberately omits pip from the temporary build environment, so
# install it once without isolation before MMPose resolves its dependencies.
RUN python -m pip install \
    -c /opt/mtdata/constraints.txt \
    --no-build-isolation "chumpy==0.70"
# MMPose imports transformer heads eagerly, and those imports require the
# mmcv._ext module even though this RTMPose/CSPNeXt configuration does not call
# custom MMCV CUDA kernels.  Build the full MMCV package with CPU extension
# symbols; PyTorch still runs the RTMPose model itself on CUDA.
ARG MMCV_MAX_JOBS=1
RUN MMCV_WITH_OPS=1 MAX_JOBS=${MMCV_MAX_JOBS} \
    python -m pip install -c /opt/mtdata/constraints.txt \
    --no-build-isolation "mmcv==2.1.0"
RUN python -m pip install \
    -c /opt/mtdata/constraints.txt \
    -r /opt/mtdata/requirements-perception.txt
RUN python -m pip check \
    && python -m action_annotation_pipeline.dependency_check --profile perception \
    && python -m pip freeze --all > /opt/mtdata/perception-environment.lock.txt
CMD ["python", "mtdata.py", "doctor", "--profile", "perception"]

FROM perception AS action
COPY docker/requirements-action.txt /opt/mtdata/requirements-action.txt
RUN python -m pip install \
    -c /opt/mtdata/constraints.txt \
    -r /opt/mtdata/requirements-action.txt
RUN python -m pip check \
    && python -m action_annotation_pipeline.dependency_check --profile action \
    && python -m pip freeze --all > /opt/mtdata/action-environment.lock.txt
CMD ["python", "mtdata.py", "doctor", "--profile", "action"]

FROM perception AS full
COPY docker/requirements-full.txt /opt/mtdata/requirements-full.txt
RUN SAM2_BUILD_CUDA=0 python -m pip install \
    -c /opt/mtdata/constraints.txt \
    --no-build-isolation \
    -r /opt/mtdata/requirements-full.txt
RUN python -m pip check \
    && python -m action_annotation_pipeline.dependency_check --profile full \
    && python -m pip freeze --all > /opt/mtdata/full-environment.lock.txt
CMD ["python", "mtdata.py", "doctor", "--profile", "full"]

FROM base AS annotation
COPY docker/requirements-annotation.txt /opt/mtdata/requirements-annotation.txt
RUN python -m pip install \
    -c /opt/mtdata/constraints.txt \
    -r /opt/mtdata/requirements-annotation.txt
RUN python -m pip check \
    && python -m action_annotation_pipeline.dependency_check --profile annotation \
    && python -m pip freeze --all > /opt/mtdata/annotation-environment.lock.txt

CMD ["python", "mtdata.py", "doctor", "--profile", "annotation"]
