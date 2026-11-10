"""Install the pinned MMPose stack, including the legacy chumpy workaround."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch",
        choices=("cpu", "cu121"),
        default="cpu",
        help="Install CPU PyTorch or the CUDA 12.1 build used by Colab T4.",
    )
    return parser.parse_args()


def run(*arguments: str) -> None:
    command = [sys.executable, "-m", "pip", "install", *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command)


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Python 3.11 is required by the pinned OpenMMLab wheels; found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    requirements = PROJECT_ROOT / "requirements-mmpose.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    run("--upgrade", "pip", "setuptools<81", "wheel")
    run("numpy>=1.24,<2.0")
    run("chumpy==0.70", "--no-build-isolation")
    torch_index = (
        "https://download.pytorch.org/whl/cpu"
        if args.torch == "cpu"
        else "https://download.pytorch.org/whl/cu121"
    )
    run(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "--index-url",
        torch_index,
    )
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-lite"]
    )
    mmcv_index = (
        "https://download.openmmlab.com/mmcv/dist/"
        f"{'cpu' if args.torch == 'cpu' else 'cu121'}/torch2.1/index.html"
    )
    run("mmcv==2.1.0", "--find-links", mmcv_index)
    run("-r", str(requirements))
    subprocess.check_call([sys.executable, "-m", "pip", "check"])
    print("MMPose dependencies installed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
