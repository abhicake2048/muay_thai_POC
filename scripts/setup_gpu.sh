#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-perception}"
if [[ "$PROFILE" != "perception" && "$PROFILE" != "full" && "$PROFILE" != "annotation" ]]; then
  echo "Usage: bash scripts/setup_gpu.sh [perception|full|annotation]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
# Separate environments prevent the annotation profile's headless OpenCV wheel
# from overwriting the perception profile's desktop OpenCV wheel.
VENV_DIR="${MTDATA_VENV:-$PROJECT_ROOT/.venv-gpu-$PROFILE}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This setup script supports Ubuntu/Debian under Linux or WSL2." >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "sudo is required to install the system packages." >&2
  exit 1
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
  python3.10 python3.10-dev python3.10-venv \
  git ffmpeg libgl1 libglib2.0-0 build-essential ninja-build pkg-config

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN was not found. Use Ubuntu 22.04 or set PYTHON_BIN to Python 3.10/3.11." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise SystemExit(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported by "
        "the pinned xtcocotools wheel; use Python 3.10 or 3.11."
    )
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade \
  "pip==25.3" "setuptools==80.9.0" "wheel==0.45.1" \
  "Cython==0.29.37" "numpy==1.26.4"

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  -c "$PROJECT_ROOT/docker/constraints.txt" \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --no-deps --no-build-isolation -e "$PROJECT_ROOT"

if [[ "$PROFILE" == "annotation" ]]; then
  python -m pip install \
    -c "$PROJECT_ROOT/docker/constraints.txt" \
    -r "$PROJECT_ROOT/docker/requirements-annotation.txt"
else
  python -m pip install \
    -c "$PROJECT_ROOT/docker/constraints.txt" \
    --only-binary=:all: "xtcocotools==1.14.3"
  python -m pip install \
    -c "$PROJECT_ROOT/docker/constraints.txt" \
    --no-build-isolation "chumpy==0.70"
  MMCV_WITH_OPS=1 MAX_JOBS="${MMCV_MAX_JOBS:-1}" \
    python -m pip install \
      -c "$PROJECT_ROOT/docker/constraints.txt" \
      --no-build-isolation "mmcv==2.1.0"
  python -m pip install \
    -c "$PROJECT_ROOT/docker/constraints.txt" \
    -r "$PROJECT_ROOT/docker/requirements-perception.txt"
fi

if [[ "$PROFILE" == "full" ]]; then
  # SAM2 can run without its optional post-processing CUDA extension.  Set
  # SAM2_BUILD_CUDA=1 before running this script if a matching CUDA toolkit
  # and nvcc are installed and that extension is desired.
  SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" \
    python -m pip install \
      -c "$PROJECT_ROOT/docker/constraints.txt" \
      --no-build-isolation \
      -r "$PROJECT_ROOT/docker/requirements-full.txt"
fi

python -m pip check
if [[ "$PROFILE" == "annotation" ]]; then
  python -m action_annotation_pipeline.dependency_check --profile annotation
  python "$PROJECT_ROOT/mtdata.py" doctor --profile annotation
else
  python -m action_annotation_pipeline.dependency_check \
    --profile "$PROFILE" --require-gpu
  python "$PROJECT_ROOT/mtdata.py" doctor \
    --profile "$PROFILE" --require-gpu
fi

python -m pip freeze --all > "$VENV_DIR/environment.lock.txt"

echo
echo "Setup completed. Activate it later with:"
echo "  source '$VENV_DIR/bin/activate'"
echo "Then run:"
echo "  python '$PROJECT_ROOT/mtdata.py' doctor --profile '$PROFILE'"
