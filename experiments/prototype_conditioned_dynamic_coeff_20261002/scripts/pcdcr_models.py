"""Stage I coefficient adapters; all outputs are residuals added to frozen c0.

Inputs contain no ground-truth quantities. ``h`` is the native 64-dimensional
coefficient feature. ``e`` has exactly this order:
    fg[32], bg[32], uncertainty[32], response[32], c0[32],
    normalized_width, normalized_height, normalized_area, level_one_hot[3].
Levels are integer indices 0/1/2 for P3/P4/P5. Each level owns independent
parameters. Original coefficients and the final P @ c decoder are outside
this module and are never replaced here.

PCDCR zero-initializes U, keeps alpha fixed at 1, and randomly initializes V
and the condition encoder. At the first nondegenerate backward pass U can
receive gradients while V/G gradients are necessarily zero because U is
zero. After the first U update, V/G can receive gradients. A full-chain smoke
must check this two-step behavior; requiring all gradients to be nonzero on
the first step would be incorrect. Static/concat MLP final layers are also
zero-initialized and have the analogous delayed hidden-layer gradients.
"""

from __future__ import annotations

from numbers import Integral
from typing import Optional, Union

import torch
from torch import Tensor, nn


H_DIM = 64
CONDITION_DIM = 166
COEFFICIENT_DIM = 32
LEVEL_COUNT = 3
RANK = 8
STATIC_HIDDEN = 123
CONCAT_HIDDEN = 45
LevelInput = Union[int, Tensor]


class DynamicLevel(nn.Module):
    """U(sigmoid(G(e)) * V(h)), with an implicit fixed alpha of one."""

    def __init__(self) -> None:
        super().__init__()
        self.V = nn.Linear(H_DIM, RANK, bias=False)
        self.G = nn.Sequential(
            nn.Linear(CONDITION_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, RANK),
            nn.Sigmoid(),
        )
        self.U = nn.Linear(RANK, COEFFICIENT_DIM, bias=False)
        nn.init.zeros_(self.U.weight)

    def forward(self, h: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        gate = self.G(e)
        return self.U(gate * self.V(h)), gate


class MLPLevel(nn.Module):
    """Capacity-matched static or concatenated-condition residual control."""

    def __init__(self, use_condition: bool) -> None:
        super().__init__()
        self.use_condition = use_condition
        input_dim = H_DIM + CONDITION_DIM if use_condition else H_DIM
        hidden_dim = CONCAT_HIDDEN if use_condition else STATIC_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, COEFFICIENT_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: Tensor, e: Optional[Tensor]) -> tuple[Tensor, None]:
        features = torch.cat((h, e), dim=-1) if self.use_condition else h
        return self.net(features), None


class CoefficientAdapter(nn.Module):
    """Common API for parameter-matched per-scale Stage I adapters.

    ``forward(h, e, level)`` returns ``delta`` with shape [N, 32].
    ``return_gate=True`` returns ``(delta, gate)``; gate is [N, 8] for
    PCDCR and None for controls. ``level`` is either one integer for the
    whole batch or an integer tensor of shape [N]. Static can receive
    ``e=None`` because it never reads condition features.

    This module does not freeze or detach its inputs. Stage I's caller must
    freeze the original model and construct h/e from that frozen model.
    This keeps the declared gradient path explicit in the training code.
    """

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in ("static", "concat", "pcdcr"):
            raise ValueError(f"Unknown adapter kind: {kind!r}")
        self.kind = kind
        self.levels = nn.ModuleList(
            DynamicLevel() if kind == "pcdcr" else MLPLevel(kind == "concat")
            for _ in range(LEVEL_COUNT)
        )

    def forward(
        self,
        h: Tensor,
        e: Optional[Tensor],
        level: LevelInput,
        return_gate: bool = False,
    ):
        if h.ndim != 2 or h.shape[1] != H_DIM:
            raise ValueError(f"h must have shape [N, {H_DIM}], got {tuple(h.shape)}")
        if self.kind != "static":
            if e is None or e.shape != (h.shape[0], CONDITION_DIM):
                actual = None if e is None else tuple(e.shape)
                raise ValueError(
                    f"e must have shape [N, {CONDITION_DIM}], got {actual}"
                )
            if e.device != h.device or e.dtype != h.dtype:
                raise ValueError("h and e must have the same device and dtype")

        if isinstance(level, Integral):
            if not 0 <= int(level) < LEVEL_COUNT:
                raise ValueError("level must be 0/1/2 for P3/P4/P5")
            delta, gate = self.levels[int(level)](h, e)
        else:
            if not isinstance(level, Tensor) or level.shape != (h.shape[0],):
                raise ValueError("level must be an integer or an integer tensor [N]")
            if level.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                raise ValueError("level tensor must have integer dtype")
            level = level.to(device=h.device)
            if bool(((level < 0) | (level >= LEVEL_COUNT)).any()):
                raise ValueError("level entries must be 0/1/2 for P3/P4/P5")
            delta = h.new_zeros((h.shape[0], COEFFICIENT_DIM))
            gate = h.new_zeros((h.shape[0], RANK)) if self.kind == "pcdcr" else None
            for index, block in enumerate(self.levels):
                selected = level == index
                if bool(selected.any()):
                    part_e = e[selected] if self.kind != "static" else None
                    part_delta, part_gate = block(h[selected], part_e)
                    delta[selected] = part_delta
                    if gate is not None:
                        gate[selected] = part_gate
        return (delta, gate) if return_gate else delta


def build_adapter(kind: str) -> CoefficientAdapter:
    """Build static / concat / pcdcr, in exact baseline-equivalent state."""
    return CoefficientAdapter(kind)


def parameter_count(module: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters, including biases; fixed alpha adds no parameter."""
    return sum(
        p.numel() for p in module.parameters()
        if not trainable_only or p.requires_grad
    )


def parameter_counts(adapter: CoefficientAdapter) -> dict:
    """Actual parameter totals and per-scale totals for the run manifest."""
    return {
        "kind": adapter.kind,
        "total": parameter_count(adapter),
        "trainable": parameter_count(adapter, trainable_only=True),
        "per_level": {
            f"P{index + 3}": parameter_count(block)
            for index, block in enumerate(adapter.levels)
        },
    }


EXPECTED_PARAMETER_COUNTS = {
    # Per level: 97 * 123 + 32; 263 * 45 + 32; 512 + 10688 + 520 + 256.
    "static": {"per_level": 11963, "total": 35889},
    "concat": {"per_level": 11867, "total": 35601},
    "pcdcr": {"per_level": 11976, "total": 35928},
}
