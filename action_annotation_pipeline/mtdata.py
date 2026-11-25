"""Single-command orchestration for the Muay Thai Data Factory v2."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import PIPELINE_ROOT, PROJECT_ROOT, PipelineError, read_json, read_jsonl, write_json, write_jsonl
from .schema import canonicalize_event
from .state import StageState, stage_signature
from .validate_events import validate
from .dependency_check import audit_environment


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower() or "video"


def _script(name: str) -> Path:
    return PIPELINE_ROOT / name


def _run(command: Sequence[str], log_path: Path | None = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", newline="\n") if log_path else None
    tail: list[str] = []
    try:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
            tail.append(line)
            if len(tail) > 80:
                tail.pop(0)
        return_code = process.wait()
    finally:
        if log_handle:
            log_handle.close()
    if return_code:
        raise PipelineError(f"Command failed ({return_code}): {' '.join(command)}\n{''.join(tail)[-4000:]}")


def _stage(project: Path, name: str, script: str, arguments: list[str], inputs: list[Path], outputs: list[Path], force: bool = False) -> None:
    state = StageState(project)
    script_path = _script(script)
    signature = stage_signature([script_path, *inputs], {"arguments": arguments, "version": "2.0"})
    if not force and state.is_current(name, signature, outputs):
        print(json.dumps({"stage": name, "status": "current", "outputs": [str(path) for path in outputs]}))
        return
    command = [sys.executable, str(script_path), *arguments]
    if (force or any(path.exists() for path in outputs)) and "--overwrite" not in command:
        command.append("--overwrite")
    with state.stage(name, signature, command=" ".join(command), outputs=outputs):
        _run(command, project / "logs" / f"{name}.log")


def _video_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise PipelineError(f"Unsupported video extension: {source.suffix}")
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    raise PipelineError(f"Input not found: {source}")


def command_ingest(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    videos = _video_paths(source)
    if not videos:
        raise PipelineError(f"No videos found below {source}")
    config_root = output_root / ".configs"
    config_root.mkdir(parents=True, exist_ok=True)
    registry_path = output_root / "dataset_registry.json"
    registry = read_json(registry_path) if registry_path.is_file() else {"schema_version": "2.0", "projects": []}
    known = {item["video_id"]: item for item in registry.get("projects", [])}
    results = []
    for video in videos:
        video_id = _slug(video.stem)
        project = output_root / video_id
        if (project / "manifest.json").is_file() and not args.overwrite:
            results.append({"video_id": video_id, "project": str(project), "status": "exists"})
            continue
        config = {
            "project_id": f"{video_id}_actions_v2", "video_id": video_id,
            "source_video": str(video), "output_dir": str(project),
            "class_schema": str(PIPELINE_ROOT / "classes.json"),
            "yolo_weights": str(args.yolo_weights.expanduser().resolve()),
            "rtmpose_config": str(args.rtmpose_config.expanduser().resolve()),
            "rtmpose_checkpoint": str(args.rtmpose_checkpoint.expanduser().resolve()),
            "proxy_fps": args.proxy_fps, "proxy_width": args.proxy_width,
            "subset": "unassigned", "expected_fighters": 2,
            "detector_confidence": args.detector_confidence,
            "keypoint_confidence": args.keypoint_confidence,
            "feature_stride_frames": 4,
            "notes": "Data Factory v2 project. Every action requires human verification."
        }
        config_path = config_root / f"{video_id}.json"
        write_json(config_path, config)
        command = [sys.executable, str(_script("prepare_project.py")), "--config", str(config_path)]
        if args.skip_proxy:
            command.append("--skip-proxy")
        if args.overwrite:
            command.append("--overwrite")
        _run(command, project / "logs" / "ingest.log" if project.exists() else None)
        state = StageState(project)
        signature = stage_signature([video, Path(config["class_schema"])], config)
        state.record("ingest", signature, "complete", command=" ".join(command), outputs=[project / "manifest.json", project / "video" / "timestamp_map.json"])
        known[video_id] = {"video_id": video_id, "project": str(project), "source_video": str(video), "source_group": video_id}
        results.append({"video_id": video_id, "project": str(project), "status": "created"})
    registry["projects"] = [known[key] for key in sorted(known)]
    write_json(registry_path, registry)
    print(json.dumps({"videos": len(videos), "results": results, "registry": str(registry_path)}, indent=2))
    return 0


def _run_action_stages(
    project: Path,
    *,
    mode: str,
    device: str,
    vlm_model: Path | None,
    vlm_limit: int,
    scan_mode: str,
    window_seconds: float,
    hop_seconds: float,
    model_branches: list[Path] | None,
    overwrite: bool,
) -> Path:
    manifest = read_json(project / "manifest.json")
    proxy = Path(manifest["paths"]["proxy_video"])
    tracks = project / "tracks" / "tracks.jsonl"
    fighter_map = project / "tracks" / "fighter_map.json"
    poses = project / "poses" / "poses.npz"
    actionness = project / "proposals" / "branches" / "pose_actionness.jsonl"
    actionness_report = project / "qa" / "pose_actionness_report.json"
    _stage(
        project,
        "pose_actionness",
        "propose_actions.py",
        ["--project", str(project)],
        [poses, fighter_map],
        [actionness, actionness_report],
        overwrite,
    )
    attributes = project / "proposals" / "branches" / "pose_attributes.jsonl"
    _stage(
        project,
        "pose_attributes",
        "infer_attributes.py",
        ["--project", str(project), "--input", str(actionness)],
        [poses, actionness, fighter_map],
        [attributes],
        overwrite,
    )
    branches = [actionness, attributes]
    if mode == "hybrid":
        if vlm_model is None:
            raise PipelineError("hybrid action mode requires --vlm-model")
        model_path = vlm_model.expanduser().resolve()
        qwen = project / "proposals" / "branches" / "qwen_actions.jsonl"
        qwen_summary = project / "proposals" / "raw" / "qwen" / "summary.json"
        vlm_args = [
            "--project", str(project), "--model", str(model_path), "--device", device,
            "--input", str(attributes), "--scan-mode", scan_mode,
            "--window-seconds", str(window_seconds), "--hop-seconds", str(hop_seconds),
        ]
        if vlm_limit:
            vlm_args.extend(["--limit", str(vlm_limit)])
        _stage(
            project,
            "qwen_actions",
            "vlm_qwen.py",
            vlm_args,
            [proxy, tracks, fighter_map, attributes, model_path],
            [qwen, qwen_summary],
            overwrite,
        )
        branches.append(qwen)
    for branch in model_branches or []:
        branch_path = branch.expanduser().resolve()
        if not branch_path.is_file():
            raise PipelineError(f"Model proposal branch not found: {branch_path}")
        branches.append(branch_path)
    fused = project / "proposals" / "fused_proposals.jsonl"
    _stage(
        project,
        "proposal_fusion",
        "fuse_proposals.py",
        ["--project", str(project), "--branches", *[str(path) for path in branches]],
        branches,
        [fused],
        overwrite,
    )
    diagnostics = project / "qa" / "action_diagnostics.json"
    timeline = project / "qa" / "action_timeline.html"
    _stage(
        project,
        "action_diagnostics",
        "action_diagnostics.py",
        ["--project", str(project)],
        [tracks, poses, fighter_map, fused, *branches],
        [diagnostics, timeline],
        overwrite,
    )
    return fused


def command_actions(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    fused = _run_action_stages(
        project,
        mode=args.mode,
        device=args.device,
        vlm_model=args.vlm_model,
        vlm_limit=args.vlm_limit,
        scan_mode=args.scan_mode,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        model_branches=args.model_branch,
        overwrite=args.overwrite,
    )
    print(json.dumps({"project": str(project), "mode": args.mode, "fused_proposals": str(fused), "diagnostics": str(project / 'qa' / 'action_diagnostics.json'), "human_review_required": True}, indent=2))
    return 0


def command_diagnose_actions(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    _run([sys.executable, str(_script("action_diagnostics.py")), "--project", str(project)], project / "logs" / "action_diagnostics.log")
    return 0


def command_infer(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    manifest = read_json(project / "manifest.json")
    proxy = Path(manifest["paths"]["proxy_video"])
    tracks = project / "tracks" / "tracks.jsonl"
    fighter_map = project / "tracks" / "fighter_map.json"
    poses = project / "poses" / "poses.npz"
    common = ["--project", str(project), "--device", args.device]
    frame_args = ["--max-frames", str(args.max_frames)] if args.max_frames else []
    _stage(project, "tracks", "extract_tracks.py", [*common, "--tracker", args.tracker, *frame_args], [proxy, Path(manifest["paths"]["yolo_weights"])], [tracks, fighter_map], args.overwrite)
    if args.sam2_config and args.sam2_checkpoint:
        sam2_output = project / "masks" / "sam2_masks.jsonl"
        sam2_args = ["--project", str(project), "--config", str(args.sam2_config.expanduser().resolve()), "--checkpoint", str(args.sam2_checkpoint.expanduser().resolve()), "--device", args.device, *frame_args]
        _stage(project, "sam2_masks", "sam2_masks.py", sam2_args, [proxy, tracks, fighter_map, args.sam2_config.expanduser().resolve(), args.sam2_checkpoint.expanduser().resolve()], [sam2_output], args.overwrite)
    _stage(project, "poses", "extract_poses.py", [*common, *frame_args], [proxy, tracks, Path(manifest["paths"]["rtmpose_checkpoint"])], [poses], args.overwrite)
    for teacher_name, teacher_config, teacher_checkpoint in args.pose_teacher or []:
        safe_name = _slug(teacher_name)
        teacher_output = project / "poses" / "teachers" / f"{safe_name}.npz"
        teacher_args = [*common, *frame_args, "--config", str(Path(teacher_config).expanduser().resolve()), "--checkpoint", str(Path(teacher_checkpoint).expanduser().resolve()), "--output", str(teacher_output), "--overlay", str(project / "qa" / f"poses_{safe_name}_overlay.mp4"), "--metadata", str(project / "poses" / "teachers" / f"{safe_name}_metadata.json")]
        _stage(project, f"pose_teacher_{safe_name}", "extract_poses.py", teacher_args, [proxy, tracks, Path(teacher_config).expanduser().resolve(), Path(teacher_checkpoint).expanduser().resolve()], [teacher_output], args.overwrite)
    action_mode = "hybrid" if args.profile == "full" and args.vlm_model else "pose"
    fused = _run_action_stages(
        project,
        mode=action_mode,
        device=args.device,
        vlm_model=args.vlm_model,
        vlm_limit=args.vlm_limit,
        scan_mode="hybrid",
        window_seconds=3.0,
        hop_seconds=1.5,
        model_branches=None,
        overwrite=args.overwrite,
    )
    print(json.dumps({"project": str(project), "profile": args.profile, "action_mode": action_mode, "fused_proposals": str(fused), "human_review_required": True}, indent=2))
    return 0


def command_review(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    command = [sys.executable, str(_script("review.py")), "--project", str(project), "--port", str(args.port)]
    if args.launch:
        command.append("--launch")
    _run(command, project / "logs" / "review.log")
    return 0


def command_sync_review(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    schema = read_json(project / "annotations" / "classes.json")
    manifest = read_json(project / "manifest.json")
    proxy_fps = float(manifest["proxy_video"]["fps"])
    source_fps = float(manifest["source_video"]["fps"])
    rows = []
    for raw in read_jsonl(args.input.expanduser().resolve()):
        row = canonicalize_event(raw, schema)
        row["video_id"] = manifest["video_id"]
        row["source_group"] = manifest["video_id"]
        for boundary in ("start", "peak", "end"):
            seconds = float(row[f"{boundary}_sec"])
            row[f"proxy_{boundary}_frame"] = int(round(seconds * proxy_fps))
            row[f"source_{boundary}_frame"] = int(round(seconds * source_fps))
            row[f"original_{boundary}_sec"] = seconds
        rows.append(row)
    candidate = project / "review" / "sync_candidate.jsonl"
    write_jsonl(candidate, rows)
    errors, warnings, report = validate(project, candidate, args.require_full_review, False)
    write_json(project / "qa" / "sync_review_validation.json", report)
    if errors:
        raise PipelineError(f"Review import has {len(errors)} validation errors; inspect {project / 'qa' / 'sync_review_validation.json'}")
    destination = project / "annotations" / "events.jsonl"
    if destination.exists() and destination.stat().st_size and not args.overwrite:
        raise PipelineError(f"Reviewed events already exist: {destination}; use --overwrite")
    shutil.copy2(candidate, destination)
    print(json.dumps({"events": len(rows), "output": str(destination), "warnings": warnings}, indent=2))
    return 0


def command_export(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    events = project / "annotations" / "events.jsonl"
    poses = project / "poses" / "poses.npz"
    pose_output = project / "exports" / "posec3d" / "posec3d_annotations.pkl"
    tri_output = project / "exports" / "tridet" / "annotations.json"
    _stage(project, "export_posec3d", "export_posec3d.py", ["--project", str(project), "--num-frames", "48"], [events, poses], [pose_output], args.overwrite)
    tri_args = ["--project", str(project)]
    if args.allow_partial:
        tri_args.append("--allow-partial")
    _stage(project, "export_tridet", "export_tridet.py", tri_args, [events], [tri_output], args.overwrite)
    if not args.skip_feedback:
        feedback_output = project / "qa" / "feedback_summary.json"
        _stage(project, "export_feedback", "export_feedback.py", ["--project", str(project), "--max-yolo", str(args.max_yolo), "--max-rtmpose", str(args.max_rtmpose)], [project / "tracks" / "tracks.jsonl", poses, events], [feedback_output], args.overwrite)
    print(json.dumps({"posec3d": str(pose_output), "tridet": str(tri_output), "human_verified_only": True}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    _run([sys.executable, str(_script("evaluate.py")), "--project", str(project)], project / "logs" / "evaluate.log")
    return 0


def command_features(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    manifest = read_json(project / "manifest.json")
    outputs = [project / "exports" / "tridet" / "features" / f"{manifest['video_id']}__fighter{fighter}.npy" for fighter in range(int(manifest.get("expected_fighters", 2)))]
    arguments = ["--project", str(project), "--model", str(args.model.expanduser().resolve()), "--device", args.device, "--stride", str(args.stride), "--clip-frames", str(args.clip_frames)]
    _stage(project, "videomae_features", "extract_videomae_features.py", arguments, [Path(manifest["paths"]["proxy_video"]), project / "tracks" / "tracks.jsonl", args.model.expanduser().resolve()], outputs, args.overwrite)
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    _run([sys.executable, str(_script("aggregate_dataset.py")), "--projects-root", str(args.projects_root.resolve()), "--output", str(args.output.resolve()), "--seed", str(args.seed)])
    return 0


def command_status(args: argparse.Namespace) -> int:
    project = args.project.expanduser().resolve()
    print(json.dumps({"project": str(project), "stages": StageState(project).rows()}, indent=2))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    checks = audit_environment(args.profile, args.require_gpu)
    checks["executables"] = {
        "ffmpeg": shutil.which("ffmpeg"),
        "docker": shutil.which("docker"),
        "wsl": shutil.which("wsl") or shutil.which("wsl.exe"),
    }
    if not checks["executables"]["ffmpeg"]:
        checks["errors"].append("ffmpeg executable was not found")
        checks["ok"] = False
    print(json.dumps(checks, indent=2, default=str))
    return 0 if checks["ok"] else 1


def command_train(args: argparse.Namespace) -> int:
    extras = args.extra or []
    if extras and extras[0] == "--":
        extras = extras[1:]
    if args.model == "yolo":
        command = [sys.executable, str(PROJECT_ROOT / "train_yolo.py"), *extras]
    elif args.model == "rtmpose":
        command = [sys.executable, str(PROJECT_ROOT / "train_rtmpose.py"), *extras]
    elif args.model == "posec3d":
        if not args.config:
            raise PipelineError("PoseC3D training requires --config")
        command = ["mim", "train", "mmaction2", str(args.config.resolve()), *extras]
    else:
        if not args.config or not args.repo:
            raise PipelineError("TriDet training requires --config and --repo pointing to OpenTAD")
        command = [sys.executable, str(args.repo.resolve() / "tools" / "train.py"), str(args.config.resolve()), *extras]
    print(json.dumps({"model": args.model, "command": command, "execute": args.execute}, indent=2))
    if args.execute:
        _run(command)
    return 0


def command_benchmark_trackers(args: argparse.Namespace) -> int:
    command = [sys.executable, str(_script("benchmark_trackers.py")), "--gold", str(args.gold.resolve()), "--output", str(args.output.resolve()), "--iou", str(args.iou)]
    for candidate in args.candidate:
        command.extend(["--candidate", candidate])
    _run(command)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtdata", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="Create one resumable project per source video")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "generated" / "action_annotations_v2")
    ingest.add_argument("--yolo-weights", type=Path, default=PROJECT_ROOT / "weights" / "yolov11" / "best.pt")
    ingest.add_argument("--rtmpose-config", type=Path, default=PROJECT_ROOT / "weights" / "rtmpose" / "rtmpose_s_combat.py")
    ingest.add_argument("--rtmpose-checkpoint", type=Path, default=PROJECT_ROOT / "weights" / "rtmpose" / "best_coco_AP_epoch_70.pth")
    ingest.add_argument("--proxy-fps", type=float, default=30.0)
    ingest.add_argument("--proxy-width", type=int, default=1280)
    ingest.add_argument("--detector-confidence", type=float, default=0.25)
    ingest.add_argument("--keypoint-confidence", type=float, default=0.20)
    ingest.add_argument("--skip-proxy", action="store_true")
    ingest.add_argument("--overwrite", action="store_true")
    ingest.set_defaults(func=command_ingest)

    infer = subparsers.add_parser("infer", help="Run resumable detection, pose and proposal stages")
    infer.add_argument("project", type=Path)
    infer.add_argument("--profile", choices=("core", "full"), default="core")
    infer.add_argument("--device", default="auto")
    infer.add_argument("--tracker", default="botsort.yaml")
    infer.add_argument("--vlm-model", type=Path)
    infer.add_argument("--pose-teacher", action="append", nargs=3, metavar=("NAME", "CONFIG", "CHECKPOINT"), help="Optional MMPose-compatible RTMW/teacher; repeatable")
    infer.add_argument("--sam2-config", type=Path)
    infer.add_argument("--sam2-checkpoint", type=Path)
    infer.add_argument("--vlm-limit", type=int, default=0)
    infer.add_argument("--max-frames", type=int, default=0, help="Smoke tests only")
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(func=command_infer)

    actions = subparsers.add_parser("actions", help="Run only action proposal, classification, fusion and diagnostics stages")
    actions.add_argument("project", type=Path)
    actions.add_argument("--mode", choices=("pose", "hybrid"), default="pose")
    actions.add_argument("--device", default="cuda:0")
    actions.add_argument("--vlm-model", type=Path)
    actions.add_argument("--vlm-limit", type=int, default=0)
    actions.add_argument("--scan-mode", choices=("candidates", "hybrid", "dense"), default="hybrid")
    actions.add_argument("--window-seconds", type=float, default=3.0)
    actions.add_argument("--hop-seconds", type=float, default=1.5)
    actions.add_argument("--model-branch", type=Path, action="append", help="Optional PoseC3D, Boxing6 or TriDet proposal JSONL; repeatable")
    actions.add_argument("--overwrite", action="store_true")
    actions.set_defaults(func=command_actions)

    diagnose_actions = subparsers.add_parser("diagnose-actions", help="Report action-pipeline temporal coverage and classifier health")
    diagnose_actions.add_argument("project", type=Path)
    diagnose_actions.set_defaults(func=command_diagnose_actions)

    review = subparsers.add_parser("review", help="Prepare or launch the FiftyOne review dataset")
    review.add_argument("project", type=Path)
    review.add_argument("--launch", action="store_true")
    review.add_argument("--port", type=int, default=5151)
    review.set_defaults(func=command_review)

    sync = subparsers.add_parser("sync-review", help="Validate and promote reviewed JSONL to ground truth")
    sync.add_argument("project", type=Path)
    sync.add_argument("input", type=Path)
    sync.add_argument("--require-full-review", action="store_true")
    sync.add_argument("--overwrite", action="store_true")
    sync.set_defaults(func=command_sync_review)

    export = subparsers.add_parser("export", help="Create PoseC3D, TriDet and feedback datasets")
    export.add_argument("project", type=Path)
    export.add_argument("--allow-partial", action="store_true")
    export.add_argument("--skip-feedback", action="store_true")
    export.add_argument("--max-yolo", type=int, default=200)
    export.add_argument("--max-rtmpose", type=int, default=200)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(func=command_export)

    evaluate_parser = subparsers.add_parser("evaluate", help="Score proposals and review readiness")
    evaluate_parser.add_argument("project", type=Path)
    evaluate_parser.set_defaults(func=command_evaluate)

    features = subparsers.add_parser("features", help="Extract actor-conditioned local VideoMAE features")
    features.add_argument("project", type=Path)
    features.add_argument("--model", type=Path, required=True)
    features.add_argument("--device", default="cuda")
    features.add_argument("--stride", type=int, default=4)
    features.add_argument("--clip-frames", type=int, default=16)
    features.add_argument("--overwrite", action="store_true")
    features.set_defaults(func=command_features)

    freeze = subparsers.add_parser("freeze", help="Aggregate per-video exports with leakage-free splits")
    freeze.add_argument("projects_root", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=42)
    freeze.set_defaults(func=command_freeze)

    status = subparsers.add_parser("status", help="Show resumable stage state")
    status.add_argument("project", type=Path)
    status.set_defaults(func=command_status)

    doctor = subparsers.add_parser("doctor", help="Check local/GPU/optional dependencies")
    doctor.add_argument(
        "--profile",
        choices=("auto", "base", "perception", "action", "full", "annotation", "rtmpose-training"),
        default="auto",
    )
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.set_defaults(func=command_doctor)

    benchmark = subparsers.add_parser("benchmark-trackers", help="Select a tracker from reviewed fighter-ID clips")
    benchmark.add_argument("--gold", type=Path, required=True)
    benchmark.add_argument("--candidate", action="append", required=True, help="NAME=tracks.jsonl")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--iou", type=float, default=0.30)
    benchmark.set_defaults(func=command_benchmark_trackers)

    train = subparsers.add_parser("train", help="Print or execute an existing trainer command")
    train.add_argument("model", choices=("posec3d", "tridet", "yolo", "rtmpose"))
    train.add_argument("--config", type=Path)
    train.add_argument("--repo", type=Path, help="OpenTAD repository for TriDet")
    train.add_argument("--execute", action="store_true")
    train.add_argument("extra", nargs=argparse.REMAINDER)
    train.set_defaults(func=command_train)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
