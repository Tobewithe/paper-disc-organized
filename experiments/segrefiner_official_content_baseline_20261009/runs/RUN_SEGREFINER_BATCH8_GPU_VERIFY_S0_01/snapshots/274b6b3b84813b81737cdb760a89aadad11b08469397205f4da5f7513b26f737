"""Single fixed hardware adaptation: author batch_max8, all other math intact."""
import ast
import copy
from pathlib import Path

import segrefiner_runtime as original

BASE_RUNTIME_SHA = "5a296b083991adb6918db09a9f69204a0a2e4f65b582d5c9a2f2335d4183c107"
RUNTIME_VERSION = "segrefiner_author_lr_batch8_hardware_v1"


def adapted_counter_method():
    path = Path(original.__file__)
    if original.sha256(path) != BASE_RUNTIME_SHA:
        raise ValueError("Batch8 requires unchanged locked original framework shim")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OfficialSegRefiner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_refine_seeded")
    matches = 0
    for node in method.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "expected_calls":
            if ast.unparse(node.value) != "6 * ((valid_count + 31) // 32)":
                raise ValueError("Original author forward-count guard differs")
            node.value = ast.parse("6 * ((valid_count + 7) // 8)").body[0].value
            matches += 1
    if matches != 1:
        raise ValueError("One explicit batch-dependent diagnostic guard required")
    scope = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
    return scope["_refine_seeded"]


class OfficialSegRefinerBatch8(original.OfficialSegRefiner):
    _refine_seeded = adapted_counter_method()

    def __init__(self, upstream, checkpoint, device="cuda", base_seed=20261009):
        super().__init__(upstream, checkpoint, device, base_seed)
        official = copy.deepcopy(self.model.test_cfg)
        if official["batch_max"] != 32:
            raise ValueError("Author default batch32 source changed")
        self.model.test_cfg = {**official, "batch_max": 8}
        self.provenance["runtime_version"] = RUNTIME_VERSION
        self.provenance["configuration"]["test_cfg"] = dict(self.model.test_cfg)
        self.provenance["official_default_test_cfg"] = official
        self.provenance["hardware_adaptation"] = {"batch_max": 8, "base_runtime_sha256": BASE_RUNTIME_SHA,
                                                  "original_author_methods_changed": False,
                                                  "diagnostic_forward_count_guard_changed": "ceil(valid_count/8), six calls per batch",
                                                  "rand_consumption_distribution_changed_from_batch32": True,
                                                  "batch32_output_bitwise_equivalence_claimed": False,
                                                  "hyperparameter_or_quality_search": False}
