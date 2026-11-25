#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-perception}"
if [[ "$PROFILE" != "perception" && "$PROFILE" != "full" ]]; then
  echo "Usage: bash scripts/setup_gpu.sh [perception|full]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${MTDATA_VENV:-$PROJECT_ROOT/.venv-gpu}"

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
  pip "setuptools<81" wheel "Cython>=0.29.36,<3" "numpy>=1.26,<2"

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e "$PROJECT_ROOT"
python -m pip install --only-binary=:all: "xtcocotools==1.14.3"
python -m pip install --no-build-isolation "chumpy==0.70"
python -m pip install -r "$PROJECT_ROOT/docker/requirements-perception.txt"

if [[ "$PROFILE" == "full" ]]; then
  # SAM2 can run without its optional post-processing CUDA extension.  Set
  # SAM2_BUILD_CUDA=1 before running this script if a matching CUDA toolkit
  # and nvcc are installed and that extension is desired.
  SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" \
    python -m pip install -r "$PROJECT_ROOT/docker/requirements-full.txt"
fi

python -m pip check
python - <<'PY'
import sys
import torch
import mmcv
import mmengine
import mmpose
import ultralytics
import xtcocotools

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"MMCV: {mmcv.__version__}")
print(f"MMPose: {mmpose.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible. Check the Windows NVIDIA driver and WSL GPU access.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python "$PROJECT_ROOT/mtdata.py" doctor

echo
echo "Setup completed. Activate it later with:"
echo "  source '$VENV_DIR/bin/activate'"
echo "Then run:"
echo "  python '$PROJECT_ROOT/mtdata.py' doctor"
