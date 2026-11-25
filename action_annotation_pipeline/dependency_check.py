"""Strict dependency and binary-compatibility checks for mtdata environments."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import sys
from typing import Any


BASE_VERSIONS = {
    "pip": "25.3",
    "setuptools": "80.9.0",
    "wheel": "0.45.1",
    "Cython": "0.29.37",
    "numpy": "1.26.4",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
}

PERCEPTION_VERSIONS = {
    **BASE_VERSIONS,
    "opencv-python": "4.11.0.86",
    "Pillow": "11.3.0",
    "scipy": "1.15.3",
    "matplotlib": "3.10.8",
    "ultralytics": "8.4.218",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.2.0",
    "mmpose": "1.3.2",
    "chumpy": "0.70",
    "xtcocotools": "1.14.3",
    "boxmot": "16.0.11",
    "pandas": "2.3.3",
    "scikit-learn": "1.7.2",
    "pycocotools": "2.0.11",
    "regex": "2024.11.6",
}

ACTION_VERSIONS = {
    **PERCEPTION_VERSIONS,
    "transformers": "4.57.6",
    "accelerate": "1.10.1",
    "qwen-vl-utils": "0.0.14",
    "av": "16.0.1",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1",
    "safetensors": "0.6.2",
}

FULL_VERSIONS = {
    **ACTION_VERSIONS,
    "SAM-2": "1.0",
}

ANNOTATION_VERSIONS = {
    **BASE_VERSIONS,
    "opencv-python-headless": "4.11.0.86",
    "Pillow": "11.3.0",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "fiftyone": "1.10.0",
}

RTMPOSE_TRAINING_VERSIONS = {
    "pip": "25.3",
    "setuptools": "80.9.0",
    "wheel": "0.45.1",
    "Cython": "0.29.37",
    "numpy": "1.26.4",
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "opencv-python": "4.11.0.86",
    "ultralytics": "8.4.218",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.2.0",
    "mmpose": "1.3.2",
    "chumpy": "0.70",
    "xtcocotools": "1.14.3",
    "pycocotools": "2.0.11",
    "Pillow": "11.3.0",
    "scipy": "1.15.3",
    "matplotlib": "3.10.8",
}

EXPECTED = {
    "base": BASE_VERSIONS,
    "perception": PERCEPTION_VERSIONS,
    "action": ACTION_VERSIONS,
    "full": FULL_VERSIONS,
    "annotation": ANNOTATION_VERSIONS,
    "rtmpose-training": RTMPOSE_TRAINING_VERSIONS,
}

OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _public_version(value: str) -> str:
    """Ignore a wheel's local build tag, such as ``+cu128``."""
    return value.split("+", 1)[0]


def detect_profile() -> str:
    if _version("fiftyone") and not _version("mmpose"):
        return "annotation"
    if _version("SAM-2") and _version("transformers"):
        return "full"
    if _version("transformers") and _version("mmpose"):
        return "action"
    if _version("mmpose"):
        return "perception"
    return "base"


def _import(name: str, errors: list[str]) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # importing binary extensions may raise non-ImportError
        errors.append(f"import {name!r} failed: {type(exc).__name__}: {exc}")
        return None


def audit_environment(profile: str = "auto", require_gpu: bool = False) -> dict[str, Any]:
    if profile == "auto":
        profile = detect_profile()
    if profile not in EXPECTED:
        raise ValueError(f"Unknown dependency profile: {profile}")

    errors: list[str] = []
    installed: dict[str, str | None] = {}
    for distribution, expected in EXPECTED[profile].items():
        actual = _version(distribution)
        installed[distribution] = actual
        if actual is None:
            errors.append(f"missing distribution {distribution}=={expected}")
        elif _public_version(actual) != expected:
            errors.append(
                f"{distribution} version mismatch: expected {expected}, found {actual}"
            )

    opencv_installed = [name for name in OPENCV_DISTRIBUTIONS if _version(name)]
    required_opencv = (
        "opencv-python-headless" if profile == "annotation" else "opencv-python"
    )
    if profile != "base":
        if opencv_installed != [required_opencv]:
            errors.append(
                "exactly one OpenCV wheel must be installed; expected "
                f"{required_opencv}, found {opencv_installed or 'none'}"
            )

    if profile in {"perception", "action", "full", "rtmpose-training"}:
        mmcv_variants = [name for name in ("mmcv", "mmcv-lite") if _version(name)]
        if mmcv_variants != ["mmcv"]:
            errors.append(
                "full mmcv (and no mmcv-lite) is required; found "
                f"{mmcv_variants or 'none'}"
            )

    numpy_module = _import("numpy", errors)
    torch_module = _import("torch", errors)
    _import("torchvision", errors)
    cuda: dict[str, Any] = {}
    if torch_module is not None:
        cuda = {
            "wheel_runtime": torch_module.version.cuda,
            "available": bool(torch_module.cuda.is_available()),
            "device": (
                torch_module.cuda.get_device_name(0)
                if torch_module.cuda.is_available()
                else None
            ),
            "bf16_supported": (
                bool(torch_module.cuda.is_bf16_supported())
                if torch_module.cuda.is_available()
                else False
            ),
        }
        expected_cuda = (
            (None, "12.1") if profile == "rtmpose-training" else ("12.8",)
        )
        if torch_module.version.cuda not in expected_cuda:
            errors.append(
                "PyTorch CUDA runtime mismatch: expected one of "
                f"{expected_cuda}, found "
                f"{torch_module.version.cuda!r}"
            )
        if require_gpu and not torch_module.cuda.is_available():
            errors.append(
                "CUDA is not visible at runtime; check the NVIDIA driver, WSL2 GPU "
                "access, Docker Desktop GPU integration, and the --gpus flag"
            )
    if numpy_module is not None and numpy_module.__version__ != "1.26.4":
        errors.append(f"NumPy ABI mismatch: imported {numpy_module.__version__}")

    imported: list[str] = []
    if profile in {"perception", "action", "full", "rtmpose-training"}:
        perception_modules = [
            "cv2",
            "ultralytics",
            "mmengine",
            "mmcv",
            "mmcv._ext",
            "mmdet",
            "mmpose",
            "xtcocotools._mask",
            "pycocotools._mask",
        ]
        if profile != "rtmpose-training":
            perception_modules.append("boxmot")
        for module_name in perception_modules:
            if _import(module_name, errors) is not None:
                imported.append(module_name)
        try:
            from mmpose.apis import inference_topdown, init_model  # noqa: F401
            from ultralytics import YOLO  # noqa: F401

            imported.extend(("mmpose.apis", "ultralytics.YOLO"))
        except Exception as exc:
            errors.append(
                "required perception API import failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if profile in {"action", "full"}:
        for module_name in ("accelerate", "qwen_vl_utils", "av"):
            if _import(module_name, errors) is not None:
                imported.append(module_name)
        try:
            from transformers import (  # noqa: F401
                AutoImageProcessor,
                AutoModel,
                AutoModelForImageTextToText,
                AutoProcessor,
            )
            imported.append("transformers auto APIs")
        except Exception as exc:
            errors.append(
                f"required action-profile API import failed: {type(exc).__name__}: {exc}"
            )
        if profile == "full":
            try:
                from sam2.build_sam import build_sam2_video_predictor  # noqa: F401

                imported.append("sam2.build_sam")
            except Exception as exc:
                errors.append(f"required full-profile SAM2 import failed: {type(exc).__name__}: {exc}")

    if profile == "annotation":
        _import("cv2", errors)
        fiftyone_module = _import("fiftyone", errors)
        if fiftyone_module is not None:
            imported.append("fiftyone")
            if not hasattr(fiftyone_module, "TemporalDetection"):
                errors.append("FiftyOne does not expose TemporalDetection")

    return {
        "profile": profile,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "installed": installed,
        "opencv_distributions": opencv_installed,
        "imports": imported,
        "cuda": cuda,
        "sam2_build_cuda": os.environ.get("SAM2_BUILD_CUDA"),
        "ok": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "auto",
            "base",
            "perception",
            "action",
            "full",
            "annotation",
            "rtmpose-training",
        ),
        default="auto",
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = audit_environment(args.profile, args.require_gpu)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
