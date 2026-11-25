"""Framework-neutral masked multi-task head for PoseC3D clip features."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError("factorized_head requires PyTorch") from exc


class FactorizedActionHead(nn.Module):
    """Technique plus masked attribute heads.

    Input is a pooled PoseC3D feature tensor shaped ``[batch, channels]``.
    Attribute labels use -1 for unavailable values; masks prevent unknown and
    non-applicable values from contributing to their respective losses.
    """

    def __init__(
        self,
        in_channels: int,
        technique_classes: int,
        attribute_classes: dict[str, int],
        dropout: float = 0.5,
        attribute_loss_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.technique = nn.Linear(in_channels, technique_classes)
        self.attributes = nn.ModuleDict({name: nn.Linear(in_channels, count) for name, count in attribute_classes.items()})
        self.attribute_loss_weight = float(attribute_loss_weight)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(features)
        return {"technique": self.technique(features), **{name: head(features) for name, head in self.attributes.items()}}

    def loss(
        self,
        logits: dict[str, torch.Tensor],
        technique_labels: torch.Tensor,
        attribute_labels: dict[str, torch.Tensor],
        attribute_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {"loss_technique": F.cross_entropy(logits["technique"], technique_labels)}
        attribute_total = logits["technique"].sum() * 0.0
        for name in self.attributes:
            mask = attribute_masks[name].bool()
            if mask.any():
                value = F.cross_entropy(logits[name][mask], attribute_labels[name][mask])
            else:
                value = logits[name].sum() * 0.0
            losses[f"loss_{name}"] = value
            attribute_total = attribute_total + value
        losses["loss"] = losses["loss_technique"] + self.attribute_loss_weight * attribute_total
        return losses

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: values.softmax(dim=-1) for name, values in self(features).items()}
