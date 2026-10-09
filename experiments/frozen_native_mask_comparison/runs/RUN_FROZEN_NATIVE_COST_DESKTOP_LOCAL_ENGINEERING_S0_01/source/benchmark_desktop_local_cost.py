"""Same-desktop-reference cost profile, preserving locked isolated algorithms.

Explicit adaptations: genuine Torch environment, truthful32 reference input
contract, reference lookup, and finite child-lifetime resource sampling.
Original laptop cost code and failed desktop profiles remain unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import types

BASE_SHA = "8b6ccf6d770cdacc10ed43f625b59c15fb37f8df6ec5554692cb4a44be09bcd6"
PROTOCOL_SHA = "f67d773459da9786f312d49acb77a8745ce83ded405e3ac0a6994e83f95b79a0"
MAP_SHA = "ab5fcb3c11956ceccd55779a342fea5d9c4800d5c5b33d7841aefc1c28fdc56b"


class EnvironmentAndLookup(ast.NodeTransformer):
    def __init__(self):
        self.current = None
        self.guards = self.lookups = 0

    def visit_FunctionDef(self, node):
        previous, self.current = self.current, node.name
        node = self.generic_visit(node)
        self.current = previous
        return node

    def visit_Compare(self, node):
        if (self.current == "child" and ast.unparse(node.left) == "torch.__version__"
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "2.5.1"):
            node.comparators[0].value = "2.9.1+cu128"
            self.guards += 1
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (self.current == "child" and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "reference_root"):
            node.value = ast.parse("Path(args.reference_run) / args.arm").body[0].value
            self.lookups += 1
        return self.generic_visit(node)


def resource_state():
    import psutil
    data = {"time": time.time(), "cpu_percent_since_last_sample": psutil.cpu_percent(),
            "system_available_memory_bytes": psutil.virtual_memory().available, "python_processes": []}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["name"].lower().startswith("python"):
                command = " ".join(process.info["cmdline"] or [])
                data["python_processes"].append({"pid": process.pid, "command": command,
                                                "cpu_boundary_heavy_candidate": "boundary" in command.lower() and "verify" not in command.lower() and "workbench" not in command.lower()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    data["gpu_query_return_code"], data["gpu_snapshot"] = query.returncode, query.stdout.strip()
    return data


def load_program():
    path = Path(__file__).with_name("benchmark_individual_cost.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Original isolated algorithms do not match locked source")
    adapter = EnvironmentAndLookup()
    tree = ast.fix_missing_locations(adapter.visit(ast.parse(path.read_text(encoding="utf-8"))))
    if (adapter.guards, adapter.lookups) != (1, 1):
        raise ValueError("Explicit environment/reference lookup adaptation differs")
    program = types.ModuleType("desktop_local_reference_cost")
    program.__file__ = str(Path(__file__).resolve())
    sys.modules[program.__name__] = program
    exec(compile(tree, str(path), "exec"), program.__dict__)
    program.VERSION, program.PROTOCOL_SHA, program.LIST_SHA = "frozen_individual_cost_desktop_local_v1", PROTOCOL_SHA, MAP_SHA
    program.SOURCES = (*program.SOURCES, Path(__file__).name)

    def archive_inputs(args, run):
        if program.sha256(args.protocol) != PROTOCOL_SHA or program.sha256(args.images_list) != MAP_SHA:
            raise ValueError("Desktop revision protocol/path-map saved bytes differ")
        paths = [Path(line.strip()).resolve() for line in Path(args.images_list).read_text().splitlines() if line.strip()]
        if len(paths) != 5000 or not all(path.is_file() for path in paths[:32]):
            raise ValueError("Declared identity map or original32 actual JPEGs incomplete")
        reference = Path(args.reference_run)
        recorded, summary, reference_inputs = [program.load_json(reference / name) for name in ("run.json", "SUMMARY.json", "INPUTS.json")]
        if (recorded["status"] != "completed" or recorded["return_code"] != 0 or recorded["artifact_completeness"] != "complete"
                or summary.get("image_count") != 32 or summary.get("complete_common_baseline_exact") is not True
                or summary.get("isolated_cost_executor_used") is not False or summary.get("actual_torch") != "2.9.1+cu128"
                or summary.get("gt_used") is not False or program.load_json(reference / "BASELINE_PARITY.json").get("passed") is not True):
            raise ValueError("Actual desktop original-path reference32 contract did not pass")
        ids = [int(path.stem) for path in paths[:32]]
        if ids != reference_inputs["image_ids"]:
            raise ValueError("Desktop32 source order differs")
        tf = Path(args.triflow_reference)
        sources = {name: program.sha256(Path(__file__).with_name(name)) for name in program.SOURCES}
        bindings = {"source_sha256": sources, "protocol_sha256": PROTOCOL_SHA,
                    "response_sha256": program.sha256(args.response_model), "multi_sha256": program.sha256(args.multi_model),
                    "weights_sha256": program.sha256(args.weights), "images_list_sha256": MAP_SHA,
                    "native_reference_summary_sha256": program.sha256(reference / "SUMMARY.json"),
                    "native_reference_parity_sha256": program.sha256(reference / "BASELINE_PARITY.json"),
                    "head_sha256": program.sha256(tf / "head_epoch_08.pt"),
                    "head_provenance_sha256": program.sha256(tf / "HEAD_PROVENANCE.json"),
                    "core_sha256": program.sha256(tf / "source/triflow_model.py"),
                    "runtime_sha256": program.sha256(tf / "diagnostics_provenance/diagnostics_runtime.py"),
                    "reference_scope": "desktop32 original-seven-arm plus original-final8 paths, no cross-device quality claim"}
        if (bindings["weights_sha256"], bindings["response_sha256"], bindings["multi_sha256"], bindings["head_sha256"], bindings["core_sha256"], bindings["runtime_sha256"]) != (
                program.OFFICIAL_SHA256, program.RESPONSE_SHA256, program.MULTI_SHA256, program.HEAD_SHA, program.CORE_SHA, program.RUNTIME_SHA):
            raise ValueError("Original fixed algorithm/head/asset hashes differ")
        bindings["vendor_source_sha256"] = {name: program.sha256(Path(args.vendor) / name) for name in (
            "ultralytics/utils/ops.py", "ultralytics/models/yolo/segment/val.py", "ultralytics/utils/nms.py", "ultralytics/data/augment.py")}
        source = run / "source"
        source.mkdir(exist_ok=True)
        for name in program.SOURCES:
            shutil.copy2(Path(__file__).with_name(name), source / name)
        shutil.copy2(args.protocol, source / "COST_PANEL_DESKTOP_LOCAL_REFERENCE_PROTOCOL.md")
        for name, origin in (("triflow_model.py", tf / "source/triflow_model.py"), ("diagnostics_runtime.py", tf / "diagnostics_provenance/diagnostics_runtime.py"), ("HEAD_PROVENANCE.json", tf / "HEAD_PROVENANCE.json")):
            (source / "triflow_reference").mkdir(exist_ok=True)
            shutil.copy2(origin, source / "triflow_reference" / name)
        for name, origin in (("response.json", args.response_model), ("multi_local.json", args.multi_model)):
            (run / "assets").mkdir(exist_ok=True)
            shutil.copy2(origin, run / "assets" / name)
        for iid in ids:
            for arm in program.ARMS:
                if not (reference / arm / "images" / f"{iid:012d}.json").is_file():
                    raise FileNotFoundError("Actual same-machine reference missing")
        initial_resource = resource_state()
        if not args.engineering and any(row["cpu_boundary_heavy_candidate"] for row in initial_resource["python_processes"]):
            raise RuntimeError("Desktop Boundary CPU heavy candidate is still active; formal cost must wait")
        program.dump_json(run / "RESOURCES_BEFORE.json", initial_resource)
        inputs = {"version": program.VERSION, "source_binding": bindings, "source_binding_sha256": program.canonical_sha(bindings),
                  "image_paths": [str(path) for path in paths[:32]], "image_ids": ids,
                  "image_files": [{"image_id": int(path.stem), "path": str(path), "bytes": path.stat().st_size, "sha256": program.sha256(path)} for path in paths[:32]],
                  "arm_names": list(program.ARMS), "engineering": args.engineering, "measured_images": 4 if args.engineering else 32,
                  "warmup_images": 4, "repeat_blocks": 1 if args.engineering else 3,
                  "endpoint": "preloaded original RGB ndarray through preprocessing/native forward/method to all original binary masks on CPU",
                  "reference_run": str(reference), "triflow_reference": str(tf), "interpreter": sys.executable,
                  "reference_environment": "same actual desktop Torch2.9.1+cu128/RTX5060Ti",
                  "cross_device_quality_equivalence_claimed": False, "desktop_full_ap_measured": False,
                  "deployment_excludes": ["disk/RGB prep", "model loading", "warmup", "RLE/audit", "writing", "COCO/GT"],
                  "sampling_rule": "original locked5000 ordered identity map first32; engineering first4, no selection by results"}
        program.dump_json(run / "COST_INPUTS.json", inputs)
        return inputs
    program.archive_inputs = archive_inputs

    real_popen = subprocess.Popen
    class ResourceObservedProcess:
        def __init__(self, command, **kwargs):
            self.process = real_popen(command, **kwargs)
            self.pid = self.process.pid
            self.output = Path(command[command.index("--child-output") + 1])
            self.stop = threading.Event()
            self.samples = []
            def observe():
                while not self.stop.is_set():
                    self.samples.append(resource_state())
                    if self.stop.wait(2):
                        break
            self.thread = threading.Thread(target=observe, daemon=True)
            self.thread.start()
        def wait(self):
            code = self.process.wait()
            self.stop.set()
            self.thread.join()
            program.dump_json(self.output / "RESOURCE_SAMPLES.json", {"samples": self.samples, "sampling_interval_seconds": 2,
                               "scope": "whole actual child lifetime including initialization, warmup, deployment and audit; not aligned to deployment-only intervals",
                               "sampler_in_cost_child": False, "sampler_overhead_exactly_measured": False})
            return code
    program.subprocess = types.SimpleNamespace(Popen=ResourceObservedProcess)
    return program


if __name__ == "__main__":
    active = load_program()
    arguments = active.parse()
    raise SystemExit(active.child(arguments) if arguments.child else active.main(arguments))
