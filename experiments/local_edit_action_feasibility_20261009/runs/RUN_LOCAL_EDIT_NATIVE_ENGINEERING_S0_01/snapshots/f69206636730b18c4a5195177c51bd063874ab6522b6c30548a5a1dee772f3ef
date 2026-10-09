"""Generate single-component edits from an adapter's locked native mask decode.

The adapter supplies four bool NumPy arrays on the SAME 640-input pixel grid:
baseline (z > 0), positive_trial (z > +0.25), negative_trial (z > -0.25),
and predicted-box decode support. This module does not decode logits, resize,
inspect RGB, access a model/GT, rank predicted quality, or combine actions.
Other nonempty 2-D sizes are accepted for engineering checks only.

The zero action is always first, then at most eight removal actions, then at
most eight addition actions. Within each sign, components are ordered by
descending area, minimum row, minimum column, then lexicographic sorted flat
indices. Each nonzero action is one 8-connected component of the specified
threshold change within a four-pixel Chebyshev boundary band and support.
Image exterior is False for both erosion and dilation. Bounding boxes are
(row_start, column_start, row_end, column_end), with exclusive ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np


BAND_RADIUS = 4
MAX_ACTIONS_PER_SIGN = 8
TRIAL_THRESHOLD = 0.25


@dataclass(frozen=True, eq=False)
class LocalAction:
    """Sparse edit relative to its baseline; indices use C-order flattening.

    sign is -1 for removal, +1 for addition, and 0 for the baseline action.
    tau is the generating threshold, not the direction of the edit. Indices
    returned by generate_actions are sorted, unique, owned, and read-only.
    action_id is deterministic from the operation, shape and component pixels;
    it is local to these pixels, not a detection/image identity.
    """

    action_id: str
    sign: int
    tau: float
    shape: tuple[int, int]
    bbox: tuple[int, int, int, int] | None
    flat_indices: np.ndarray = field(repr=False)

    @property
    def area(self) -> int:
        return int(self.flat_indices.size)

    @property
    def operation(self) -> str:
        return {-1: "remove", 0: "zero", 1: "add"}[self.sign]

    def metadata(self) -> dict:
        """Return JSON-compatible metadata without expanding sparse pixels."""
        return {
            "action_id": self.action_id,
            "sign": self.sign,
            "operation": self.operation,
            "tau": self.tau,
            "area": self.area,
            "shape": list(self.shape),
            "bbox": None if self.bbox is None else list(self.bbox),
        }


def _validate_mask(mask: np.ndarray, name: str, shape=None) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, not {type(mask).__name__}")
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have dtype numpy.bool_, got {mask.dtype}")
    if mask.ndim != 2 or any(size == 0 for size in mask.shape):
        raise ValueError(f"{name} must be a nonempty 2-D array, got {mask.shape}")
    if shape is not None and mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match {shape}")


def _morphology(mask: np.ndarray, erode: bool) -> np.ndarray:
    """Four unit 8-neighborhood steps with False image-exterior padding."""
    result = mask.copy()
    height, width = mask.shape
    for _ in range(BAND_RADIUS):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = padded[0:height, 0:width].copy()
        for dr in range(3):
            for dc in range(3):
                if dr == 0 and dc == 0:
                    continue
                neighbor = padded[dr : dr + height, dc : dc + width]
                if erode:
                    result &= neighbor
                else:
                    result |= neighbor
    return result


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Top components via row runs and union-find, with no external image API.

    Component pixels are only materialized after applying the cap. Two row
    runs on adjacent rows connect if they overlap or touch diagonally. The
    first flat index suffices for the final lexicographic tie-break: different
    components are disjoint, so their sorted first indices cannot coincide.
    """
    height, width = mask.shape
    parents: list[int] = []
    sizes: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parents[right] = left
        sizes[left] += sizes[right]

    for row in range(height):
        # Transitions in a False-padded row identify [start, end) runs.
        padded = np.empty(width + 2, dtype=np.bool_)
        padded[0] = padded[-1] = False
        padded[1:-1] = mask[row]
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        current: list[tuple[int, int, int]] = []
        cursor = 0
        for start, end in zip(transitions[::2], transitions[1::2]):
            start, end = int(start), int(end)
            label = len(parents)
            parents.append(label)
            sizes.append(1)
            runs.append((row, start, end, label))
            current.append((start, end, label))
            while cursor < len(previous) and previous[cursor][1] < start:
                cursor += 1
            neighbor = cursor
            while neighbor < len(previous) and previous[neighbor][0] <= end:
                union(label, previous[neighbor][2])
                neighbor += 1
        previous = current

    groups: dict[int, list[tuple[int, int, int]]] = {}
    for row, start, end, label in runs:
        groups.setdefault(find(label), []).append((row, start, end))

    def order(component):
        return (
            -sum(end - start for _, start, end in component),
            component[0][0],
            min(start for _, start, _ in component),
            component[0][0] * width + component[0][1],
        )

    selected = sorted(groups.values(), key=order)[:MAX_ACTIONS_PER_SIGN]
    return [
        np.concatenate(
            [np.arange(row * width + start, row * width + end, dtype=np.int64)
             for row, start, end in component]
        )
        for component in selected
    ]


def _action(sign: int, shape: tuple[int, int], indices: np.ndarray) -> LocalAction:
    indices = np.array(indices, dtype=np.int64, copy=True)
    indices.flags.writeable = False
    if sign == 0:
        return LocalAction("baseline", 0, 0.0, shape, None, indices)
    rows, columns = np.divmod(indices, shape[1])
    bbox = (int(rows.min()), int(columns.min()), int(rows.max()) + 1,
            int(columns.max()) + 1)
    operation = "remove" if sign < 0 else "add"
    digest = hashlib.sha256(
        f"{operation}:{shape[0]}:{shape[1]}:".encode("ascii")
        + indices.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return LocalAction(
        f"{operation}-{digest}", sign, -sign * TRIAL_THRESHOLD, shape,
        bbox, indices,
    )


def generate_actions(
    baseline: np.ndarray,
    positive_trial: np.ndarray,
    negative_trial: np.ndarray,
    support: np.ndarray,
) -> tuple[LocalAction, ...]:
    """Return zero + up to 8 removals + up to 8 additions without mutating input.

    Baseline foreground outside support is an adapter error and is rejected.
    Trial foreground outside support is deliberately ignored: negative trials
    can contain crop-padding foreground, which must never become an action.
    No ordering/nesting assumptions about the two trial masks are required.
    """
    _validate_mask(baseline, "baseline")
    for name, mask in (("positive_trial", positive_trial),
                       ("negative_trial", negative_trial), ("support", support)):
        _validate_mask(mask, name, baseline.shape)
    if np.any(baseline & ~support):
        raise ValueError("baseline contains foreground outside support")
    inner_band = baseline & ~_morphology(baseline, erode=True)
    outer_band = ~baseline & _morphology(baseline, erode=False)
    removal = baseline & ~positive_trial & inner_band & support
    addition = ~baseline & negative_trial & outer_band & support
    actions = [_action(0, baseline.shape, np.empty(0, dtype=np.int64))]
    for sign, candidate in ((-1, removal), (1, addition)):
        actions.extend(_action(sign, baseline.shape, component)
                       for component in _components(candidate))
    return tuple(actions)


def apply_action(baseline: np.ndarray, action: LocalAction) -> np.ndarray:
    """Copy baseline and apply exactly one action, never an action combination.

    The caller must apply a generated action to its generating baseline. A
    shape mismatch, invalid indices or an incompatible edit sign is rejected.
    The returned C-contiguous bool mask always owns its data, including zero.
    """
    _validate_mask(baseline, "baseline")
    if not isinstance(action, LocalAction):
        raise TypeError("action must be a LocalAction")
    if action.shape != baseline.shape:
        raise ValueError("action and baseline shapes do not match")
    if action.sign not in (-1, 0, 1):
        raise ValueError("action sign must be -1, 0, or 1")
    indices = action.flat_indices
    if not isinstance(indices, np.ndarray) or indices.ndim != 1:
        raise TypeError("action indices must be a 1-D NumPy array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("action indices must have integer dtype")
    if indices.size:
        if indices[0] < 0 or indices[-1] >= baseline.size:
            raise ValueError("action indices are out of bounds")
        if np.any(indices[1:] <= indices[:-1]):
            raise ValueError("action indices must be sorted and unique")
    if (action.sign == 0) != (indices.size == 0):
        raise ValueError("only the zero action may have no indices")
    output = baseline.copy(order="C")
    flat = output.reshape(-1)
    if action.sign == -1:
        if not np.all(flat[indices]):
            raise ValueError("removal action contains baseline-background pixels")
        flat[indices] = False
    elif action.sign == 1:
        if np.any(flat[indices]):
            raise ValueError("addition action contains baseline-foreground pixels")
        flat[indices] = True
    return output
