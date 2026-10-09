"""Self-contained CPU engineering checks, not model/GT performance evaluation.

Run: python experiments/local_edit_action_feasibility_20261009/scripts/check_local_actions.py
JSON is emitted on stdout; any failed check yields a nonzero process exit code.
Only Python's standard library and NumPy are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from local_actions import apply_action, generate_actions


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(exception_type, function, *args):
    try:
        function(*args)
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def connected(indices, shape):
    """Independent flood traversal of one returned component for checking."""
    remaining = {divmod(int(index), shape[1]) for index in indices}
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        row, column = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = row + dr, column + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return not remaining


def mixed_case():
    baseline = np.zeros((24, 24), dtype=np.bool_)
    baseline[5:19, 5:19] = True
    positive = baseline.copy()
    positive[5, 6:8] = False
    positive[6, 8] = False  # diagonal contact must join these three pixels.
    positive[18, 16] = False
    positive[11, 11] = False  # too deep to form a removal action.
    negative = baseline.copy()
    negative[3, 6:9] = True
    negative[4, 9] = True  # diagonal contact joins this addition.
    negative[2, 17] = True
    negative[0, 5] = True  # distance five, outside the four-pixel band.
    support = np.ones_like(baseline)
    return baseline, positive, negative, support


def check_zero_and_locality():
    inputs = mixed_case()
    saved = [mask.copy() for mask in inputs]
    actions = generate_actions(*inputs)
    require(len(actions) == 5, "expected zero, two removals and two additions")
    baseline = inputs[0]
    zero = actions[0]
    require(zero.sign == 0 and zero.area == 0 and zero.tau == 0,
            "invalid zero metadata")
    for action in actions:
        output = apply_action(baseline, action)
        changed = np.flatnonzero(output != baseline)
        require(np.array_equal(changed, action.flat_indices),
                "changed pixels differ from this single action")
        require(not np.shares_memory(output, baseline) and output.flags.owndata,
                "output must own a separate copy")
        require(output.dtype == np.bool_, "output dtype changed")
        require(not action.flat_indices.flags.writeable, "indices are not read-only")
        if action.sign:
            require(connected(action.flat_indices, baseline.shape),
                    "action is not 8-connected")
            require(action.tau == -0.25 * action.sign, "wrong threshold metadata")
            require(action.area == changed.size, "area does not describe action")
            require(np.all(output.flat[changed] == (action.sign == 1)),
                    "wrong direction of edit")
            rows, columns = np.unravel_index(changed, baseline.shape)
            require(action.bbox == (int(rows.min()), int(columns.min()),
                                    int(rows.max()) + 1, int(columns.max()) + 1),
                    "incorrect half-open bbox")
        else:
            require(np.array_equal(output, baseline), "zero not exact")
            output[:] = ~output  # a consumer changing output cannot mutate input.
    require(all(np.array_equal(before, after) for before, after in zip(saved, inputs)),
            "generation or application mutated an input")
    require([action.area for action in actions] == [0, 3, 1, 4, 1],
            "component area ordering or diagonal adjacency incorrect")
    require(11 * 24 + 11 not in np.concatenate([a.flat_indices for a in actions]),
            "deep interior pixel became an action")
    require(5 not in np.concatenate([a.flat_indices for a in actions]),
            "distance-five pixel became an action")


def check_stable_order_and_cap():
    baseline = np.zeros((27, 27), dtype=np.bool_)
    # Twelve isolated pixels: each removal has area one; row/column tie-break
    # must retain the first eight in row-major order.
    points = [(row, column) for row in (3, 9, 15) for column in (3, 9, 15, 21)]
    for row, column in points:
        baseline[row, column] = True
    positive = np.zeros_like(baseline)
    negative = baseline.copy()
    for row, column in points:
        negative[row, column + 1] = True
    support = np.ones_like(baseline)
    actions = generate_actions(baseline, positive, negative, support)
    require(len(actions) == 17, "each sign must cap separately at eight")
    for sign, column_shift in ((-1, 0), (1, 1)):
        selected = [a for a in actions if a.sign == sign]
        expected = [(row * 27 + column + column_shift) for row, column in points[:8]]
        require([int(a.flat_indices[0]) for a in selected] == expected,
                "stable tied-area ordering failed")
    repeated = generate_actions(baseline, positive, negative, support)
    for left, right in zip(actions, repeated):
        require(left.metadata() == right.metadata()
                and np.array_equal(left.flat_indices, right.flat_indices),
                "repeated generation is not deterministic")
    require(len({a.action_id for a in actions}) == len(actions),
            "action IDs are not unique within candidate set")


def check_area_precedes_position():
    baseline = np.ones((16, 16), dtype=np.bool_)
    positive = baseline.copy()
    positive[0, 0] = False
    positive[15, 13:16] = False
    actions = generate_actions(baseline, positive, baseline, baseline)
    require([a.area for a in actions] == [0, 3, 1],
            "area must precede the earlier row/column")


def check_complete_bridged_component():
    baseline = np.ones((8, 24), dtype=np.bool_)
    positive = baseline.copy()
    # The first two rows contain two separate runs. The third row bridges
    # their existing labels; one complete component, not two connected
    # fragments, must survive the final union-find grouping.
    positive[0:2, 2:4] = False
    positive[0:2, 8:10] = False
    positive[2, 3:9] = False
    actions = generate_actions(baseline, positive, baseline, baseline)
    expected = np.array([2, 3, 8, 9, 26, 27, 32, 33,
                         51, 52, 53, 54, 55, 56], dtype=np.int64)
    require(len(actions) == 2 and actions[1].sign == -1,
            "bridged removal region must yield exactly one nonzero action")
    require(np.array_equal(actions[1].flat_indices, expected),
            "bridged component was split or lost/added target pixels")
    require(actions[1].area == 14, "bridged component has incorrect area")


def check_equal_area_minimum_column_order():
    baseline = np.zeros((14, 18), dtype=np.bool_)
    # A: 17-pixel L, starts at (2,12), but reaches minimum column 4.
    baseline[2:11, 12] = True
    baseline[10, 4:12] = True
    # B: separate 17-pixel compact component, starts at (2,6), minimum col 6.
    # Equal area and minimum row mean A must come first by minimum column,
    # although B's first sorted flat index is smaller.
    baseline[2:5, 6:11] = True
    baseline[5, 6:8] = True
    actions = generate_actions(baseline, np.zeros_like(baseline), baseline,
                               np.ones_like(baseline))
    require(len(actions) == 3 and [a.area for a in actions] == [0, 17, 17],
            "expected two separate equal-area multi-pixel components")
    require([a.bbox for a in actions[1:]] == [(2, 4, 11, 13), (2, 6, 6, 11)],
            "minimum column did not break the equal-area/minimum-row tie")
    require([int(a.flat_indices[0]) for a in actions[1:]] == [48, 42],
            "first flat index incorrectly took precedence over minimum column")


def check_support_and_invalid_baseline():
    baseline = np.zeros((15, 15), dtype=np.bool_)
    baseline[5:10, 5:10] = True
    support = np.zeros_like(baseline)
    support[4:11, 4:11] = True
    positive = baseline.copy()
    negative = np.ones_like(baseline)  # simulates negative-threshold crop padding.
    actions = generate_actions(baseline, positive, negative, support)
    require(len(actions) == 2 and actions[1].area == 24,
            "only the in-support one-pixel ring should be added")
    for action in actions:
        require(not np.any(apply_action(baseline, action) & ~support),
                "action leaked outside support")
    contaminated_positive = positive | ~support
    contaminated = generate_actions(baseline, contaminated_positive, negative, support)
    require([a.metadata() for a in actions] == [a.metadata() for a in contaminated],
            "positive trial outside support affected actions")
    illegal = baseline.copy()
    illegal[0, 0] = True
    rejects(ValueError, generate_actions, illegal, positive, negative, support)


def check_edges_empty_full_and_narrow():
    empty = np.zeros((11, 13), dtype=np.bool_)
    full = np.ones_like(empty)
    actions = generate_actions(empty, empty, full, full)
    require(len(actions) == 1, "empty baseline has no boundary to expand")
    actions = generate_actions(full, empty, full, full)
    require(len(actions) == 2 and actions[1].area == 11 * 13 - 3 * 5,
            "False exterior must leave exactly the four-pixel inner border")
    edited = apply_action(full, actions[1])
    expected = np.zeros_like(full)
    expected[4:-4, 4:-4] = True
    require(np.array_equal(edited, expected), "full-image boundary band incorrect")
    narrow = np.zeros_like(empty)
    narrow[:, 0] = True
    actions = generate_actions(narrow, empty, full, full)
    require([a.area for a in actions] == [0, 11, 44],
            "edge-touching narrow object bands incorrect")
    require(not np.any(apply_action(narrow, actions[1])), "narrow removal not complete")
    one = np.ones((1, 1), dtype=np.bool_)
    actions = generate_actions(one, ~one, one, one)
    require([a.area for a in actions] == [0, 1], "one-pixel image failed")


def check_band_boundary_distances():
    baseline = np.zeros((15, 15), dtype=np.bool_)
    baseline[7, 7] = True
    # A single pixel expands to a 9x9 Chebyshev square, not a Manhattan diamond.
    actions = generate_actions(baseline, baseline, np.ones_like(baseline),
                               np.ones_like(baseline))
    require(len(actions) == 2 and actions[1].area == 80,
            "four-step Chebyshev square must include all 80 neighboring pixels")
    output = apply_action(baseline, actions[1])
    expected = np.zeros_like(baseline)
    expected[3:12, 3:12] = True
    require(np.array_equal(output, expected), "distance-four corners missing")


def check_noncontiguous_readonly_inputs():
    inputs = mixed_case()
    expected = generate_actions(*inputs)
    converted = [np.asfortranarray(mask) for mask in inputs]
    for mask in converted:
        mask.flags.writeable = False
    actual = generate_actions(*converted)
    require([a.metadata() for a in expected] == [a.metadata() for a in actual],
            "input memory layout affected actions")
    for action in actual:
        require(apply_action(converted[0], action).flags.c_contiguous,
                "output must use C-order matching sparse indices")
    # Negative-stride input is accepted without writing through its view.
    reversed_inputs = [mask[::-1, ::-1] for mask in inputs]
    actions = generate_actions(*reversed_inputs)
    require(len(actions) == len(expected), "negative strides not supported")
    for action in actions:
        apply_action(reversed_inputs[0], action)


def check_invalid_inputs():
    inputs = list(mixed_case())
    for index in range(4):
        invalid = inputs.copy()
        invalid[index] = invalid[index].astype(np.uint8)
        rejects(TypeError, generate_actions, *invalid)
        invalid[index] = inputs[index].tolist()
        rejects(TypeError, generate_actions, *invalid)
        invalid[index] = np.zeros((3, 4), dtype=np.bool_)
        rejects(ValueError, generate_actions, *invalid)
    for malformed in (np.zeros(3, dtype=np.bool_), np.zeros((0, 3), dtype=np.bool_),
                      np.zeros((2, 2, 2), dtype=np.bool_)):
        rejects(ValueError, generate_actions, malformed, malformed, malformed, malformed)
    action = generate_actions(*inputs)[1]
    rejects(TypeError, apply_action, inputs[0].astype(np.uint8), action)
    rejects(ValueError, apply_action, np.zeros((3, 3), dtype=np.bool_), action)
    rejects(ValueError, apply_action, np.zeros_like(inputs[0]), action)
    rejects(TypeError, apply_action, inputs[0], object())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="write the complete engineering-check JSON to this path")
    arguments = parser.parse_args()
    checks = []
    for function in (
        check_zero_and_locality,
        check_stable_order_and_cap,
        check_area_precedes_position,
        check_complete_bridged_component,
        check_equal_area_minimum_column_order,
        check_support_and_invalid_baseline,
        check_edges_empty_full_and_narrow,
        check_band_boundary_distances,
        check_noncontiguous_readonly_inputs,
        check_invalid_inputs,
    ):
        try:
            function()
            checks.append({"name": function.__name__, "passed": True})
        except Exception as error:
            checks.append({"name": function.__name__, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})
    passed = all(check["passed"] for check in checks)
    directory = Path(__file__).resolve().parent
    result = {
        "scope": "synthetic CPU engineering checks; no model or GT performance",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "synthetic_only": True,
        "real_native_parity": False,
        "reads_gt": False,
        "reads_rgb": False,
        "performs_training": False,
        "checks": checks,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "executable": sys.executable, "platform": platform.platform()},
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("local_actions.py", "check_local_actions.py")
        },
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
