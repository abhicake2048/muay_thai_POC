"""Canonical v2 action schema helpers and v1 compatibility migration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_LIMB = {
    "punch": "hand", "kick": "leg", "knee": "knee", "elbow": "elbow",
    "clinch": "both_arms", "defense": "whole_body", "balance": "whole_body",
}
DEFAULT_SURFACE = {"punch": "fist", "kick": "shin", "knee": "knee", "elbow": "elbow"}


def technique_family(schema: dict[str, Any], technique: str) -> str | None:
    for family, techniques in schema.get("action_families", {}).items():
        if technique in techniques:
            return str(family)
    if technique in schema.get("balance_classes", []):
        return "balance"
    return None


def effector_id(side: str, limb: str) -> str:
    if side in {"left", "right"} and limb not in {"unknown", "not_applicable", "whole_body"}:
        return f"{side}_{limb}"
    if side == "bilateral" or limb in {"both_arms", "whole_body"}:
        return limb
    return "unknown"


def display_label(event: dict[str, Any], include_target: bool = True) -> str:
    technique = str(event.get("technique") or event.get("label") or "unknown_action")
    pieces = [technique]
    effector = str(event.get("effector_id") or effector_id(
        str(event.get("anatomical_side", event.get("side", "unknown"))),
        str(event.get("effector_limb", "unknown")),
    ))
    if effector not in {"unknown", "not_applicable", "whole_body"}:
        pieces.append(effector)
    if include_target:
        target = str(event.get("target_zone", "unknown"))
        if target not in {"unknown", "not_applicable"}:
            pieces.append(target)
    return "__".join(pieces)


def canonicalize_event(event: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Return a non-destructive v2 representation of a v1 or v2 event."""
    result = deepcopy(event)
    event_type = str(result.get("event_type", "action"))
    technique = str(result.get("technique") or result.get("label") or "")
    migration = schema.get("v1_label_migrations", {}).get(technique, {})
    if migration:
        original = technique
        technique = str(migration.get("technique", technique))
        result.setdefault("legacy_label", original)
        for key, value in migration.items():
            result.setdefault(key, value)
    if event_type == "action":
        family = str(result.get("action_family") or technique_family(schema, technique) or "")
        side = str(result.get("anatomical_side", result.get("side", "unknown")))
        limb = str(result.get("effector_limb") or DEFAULT_LIMB.get(family, "unknown"))
        result.update({
            "technique": technique, "label": technique, "action_family": family,
            "anatomical_side": side, "side": side, "effector_limb": limb,
            "effector_id": effector_id(side, limb),
        })
        result.setdefault("stance_at_start", "unknown")
        if technique == "jab":
            result.setdefault("stance_role", "lead")
        elif technique == "cross":
            result.setdefault("stance_role", "rear")
        else:
            result.setdefault("stance_role", "unknown")
        result.setdefault("target_zone", "unknown")
        result.setdefault("outcome", "uncertain")
        result.setdefault("contact_surface", DEFAULT_SURFACE.get(family, "unknown"))
        result.setdefault("visibility", "unknown")
        result.setdefault("occluded", result["visibility"] in {"partially_occluded", "heavily_occluded"})
        result.setdefault("uncertainty_flags", [])
        result["display_label"] = display_label(result)
    else:
        result.setdefault("technique", technique)
    result.setdefault("annotation_revision", 1)
    result.setdefault("provenance", {"source": result.get("source", "manual")})
    return result


def flatten_label(event: dict[str, Any], include_target: bool = False) -> str:
    return display_label(event, include_target=include_target)
