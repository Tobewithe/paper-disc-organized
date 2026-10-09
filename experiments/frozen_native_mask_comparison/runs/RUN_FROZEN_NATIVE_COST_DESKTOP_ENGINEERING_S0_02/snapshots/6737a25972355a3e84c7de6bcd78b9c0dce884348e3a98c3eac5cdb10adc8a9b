"""Explicit desktop engineering profile for the locked independent cost code.

The AST adaptation changes only the declared Torch/profile and engineering
asset count. It does not patch torch, a model, any action, threshold or loss.
Original laptop source/Run/protocol bytes remain untouched.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

BASE_SHA = "8b6ccf6d770cdacc10ed43f625b59c15fb37f8df6ec5554692cb4a44be09bcd6"
DESKTOP_PROTOCOL_SHA = "85ba9fb8b0534d1f34723ece2d8f1809ba9bb8ee446afe4757a1c68e7fc6c13f"
DESKTOP_MAP_SHA = "ab5fcb3c11956ceccd55779a342fea5d9c4800d5c5b33d7841aefc1c28fdc56b"
DESKTOP_TORCH = "2.9.1+cu128"


class ProfileOnly(ast.NodeTransformer):
    def __init__(self):
        self.function = None
        self.version_guards = self.asset_slices = 0

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        node = self.generic_visit(node)
        if node.name == "archive_inputs":
            node.body.insert(0, ast.parse("sample_count = 4 if args.engineering else 32").body[0])
        self.function = previous
        return node

    def visit_Compare(self, node):
        if (self.function == "child" and ast.unparse(node.left) == "torch.__version__"
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "2.5.1"):
            node.comparators[0].value = DESKTOP_TORCH
            self.version_guards += 1
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if (self.function == "archive_inputs" and isinstance(node.value, ast.Name) and node.value.id == "all_paths"
                and isinstance(node.slice, ast.Slice) and isinstance(node.slice.upper, ast.Constant)
                and node.slice.upper.value == 32):
            node.slice.upper = ast.Name(id="sample_count", ctx=ast.Load())
            self.asset_slices += 1
        return self.generic_visit(node)


def load_program():
    source = Path(__file__).with_name("benchmark_individual_cost.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError("Desktop profile requires the unchanged locked laptop cost source")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    adapter = ProfileOnly()
    tree = ast.fix_missing_locations(adapter.visit(tree))
    if adapter.version_guards != 1 or adapter.asset_slices != 4:
        raise RuntimeError("Expected explicit environment/engineering asset-count adaptation differs")
    program = types.ModuleType("desktop_individual_cost_profile")
    program.__file__ = str(Path(__file__).resolve())
    sys.modules[program.__name__] = program
    exec(compile(tree, str(source), "exec"), program.__dict__)
    program.VERSION = "frozen_individual_cost_desktop_v1"
    program.PROTOCOL_SHA = DESKTOP_PROTOCOL_SHA
    program.LIST_SHA = DESKTOP_MAP_SHA
    program.SOURCES = (*program.SOURCES, Path(__file__).name)
    original_archive = program.archive_inputs

    def archive_inputs(args, run):
        mapping = Path(args.images_list)
        source_receipt = program.load_json(mapping.with_name("DESKTOP_PANEL_ASSETS_RECEIPT.json"))
        native = program.load_json(Path(args.reference_run) / "EVALUATION_INPUTS.json")["configuration"]
        ids = [int(Path(line.strip()).stem) for line in mapping.read_text().splitlines() if line.strip()]
        if (ids != native["image_ids"] or source_receipt.get("original_images_list_sha256") != native["images_list_sha256"]
                or source_receipt.get("all_jpeg_sha256_verified") is not True):
            raise ValueError("Actual desktop mapping/source JPEG receipt does not preserve original panel order")
        inputs = original_archive(args, run)
        inputs["desktop_execution"] = {"base_cost_source_sha256": BASE_SHA,
                                       "profile_source_sha256": program.sha256(__file__),
                                       "expected_torch_version": DESKTOP_TORCH,
                                       "protocol_sha256": DESKTOP_PROTOCOL_SHA,
                                       "source_asset_receipt_sha256": program.sha256(mapping.with_name("DESKTOP_PANEL_ASSETS_RECEIPT.json")),
                                       "actual_source_asset_receipt": source_receipt,
                                       "timing_claim": "GPU functionality/parity engineering under concurrent desktop CPU load; no formal cost claim" if args.engineering else "independent desktop profile",
                                       "controlled_changes": {"environment_version_guard": adapter.version_guards,
                                                               "archive_engineering_only_asset_slices": adapter.asset_slices}}
        program.dump_json(run / "COST_INPUTS.json", inputs)
        return inputs
    program.archive_inputs = archive_inputs
    original_records = program.records_and_verify

    def observed_records(result, iid, reference, categories, mask_utils):
        import os
        import torch
        location = Path(arguments.child_output)
        evidence = {"image_id": iid, "arm": arguments.arm, "pid": os.getpid(),
                    "actual_gpu": torch.cuda.get_device_name(), "actual_torch": torch.__version__,
                    "actual_gpu_forward_completed": True, "normal_cpu_mask_shape": list(result["masks"].shape),
                    "all_native_rows_exported_cpu": len(result["masks"]) == result["native_rows"],
                    "output_binary_sha256": hashlib.sha256(result["masks"].tobytes()).hexdigest(),
                    "input_tensor_sha256": hashlib.sha256(result["input_tensor_for_untimed_audit"].cpu().contiguous().numpy().tobytes()).hexdigest(),
                    "evidence_scope": "actual post-endpoint forward output; reference parity is separately tested, not presumed"}
        program.dump_json(location / "FIRST_REAL_GPU_OUTPUT.json", evidence)
        return original_records(result, iid, reference, categories, mask_utils)
    program.records_and_verify = observed_records
    return program


if __name__ == "__main__":
    active = load_program()
    arguments = active.parse()
    if not arguments.engineering:
        raise RuntimeError("This immutable desktop profile is engineering-only; formal cost needs separate resource-clear authorization/profile")
    raise SystemExit(active.child(arguments) if arguments.child else active.main(arguments))
