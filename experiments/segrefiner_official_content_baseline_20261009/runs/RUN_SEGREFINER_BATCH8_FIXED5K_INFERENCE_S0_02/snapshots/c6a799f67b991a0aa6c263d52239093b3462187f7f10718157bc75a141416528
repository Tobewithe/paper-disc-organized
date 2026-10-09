"""Explicit batch8 hardware profile of the locked GT-free producer."""
import ast
import json
from pathlib import Path
import shutil
import sys
import types

from segrefiner_runtime import sha256
from segrefiner_batch8_runtime import OfficialSegRefinerBatch8
from wddm_memory_probe import WDDMMemoryProbe

BASE_INFERENCE_SHA = "7f550762d2285453098e2d99b9b3211cdc0e2f9c6cb7e233307d85445d855445"
ADAPTATION_SHA = "59ce9e80d87947f63741f872d6421fe80ffc3884fa53267d0a6ecd2744fc7a1d"


class ProducerSourceAdaptation(ast.NodeTransformer):
    def __init__(self):
        self.runtime_hash_nodes = self.source_nodes = self.observer_nodes = 0
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        target = node.targets[0] if len(node.targets) == 1 else None
        if isinstance(target, ast.Name) and target.id == "runtime_sha":
            if ast.unparse(node.value) != "sha256(Path(__file__).with_name('segrefiner_runtime.py'))":
                raise ValueError("Expected locked runtime-source guard expression differs")
            node.value = ast.parse("sha256(Path(__file__).with_name('segrefiner_batch8_runtime.py'))").body[0].value
            self.runtime_hash_nodes += 1
        if isinstance(target, ast.Name) and target.id == "source_hashes":
            node.value.generators[0].iter = ast.parse("('run_segrefiner_inference.py', 'segrefiner_runtime.py', 'run_segrefiner_batch8.py', 'segrefiner_batch8_runtime.py', 'wddm_memory_probe.py')").body[0].value
            self.source_nodes += 1
        if isinstance(target, ast.Tuple) and [ast.unparse(value) for value in target.elts] == ["peak_allocated", "peak_reserved"]:
            self.observer_nodes += 1
            return [node, ast.parse("runtime.finish_resource_observation()").body[0]]
        return node


class ResourceObservedBatch8(OfficialSegRefinerBatch8):
    def refine(self, *args, **kwargs):
        self.last_probe = WDDMMemoryProbe(interval_seconds=1.0)
        self.last_probe.start()
        try:
            result, diagnostics = super().refine(*args, **kwargs)
            self.last_diagnostics = diagnostics
            return result, diagnostics
        finally:
            self.last_probe.stop()
    def finish_resource_observation(self):
        self.last_diagnostics["wddm_memory"] = self.last_probe.finish()


def load_program():
    path = Path(__file__).with_name("run_segrefiner_inference.py")
    if sha256(path) != BASE_INFERENCE_SHA:
        raise ValueError("Original default producer source changed")
    adapter = ProducerSourceAdaptation()
    tree = ast.fix_missing_locations(adapter.visit(ast.parse(path.read_text(encoding="utf-8"))))
    if (adapter.runtime_hash_nodes, adapter.source_nodes, adapter.observer_nodes) != (1, 1, 1):
        raise ValueError("Explicit hardware source/runtime/observer adaptation differs")
    program = types.ModuleType("segrefiner_fixed_batch8_inference")
    program.__file__ = str(Path(__file__).resolve())
    sys.modules[program.__name__] = program
    exec(compile(tree, str(path), "exec"), program.__dict__)
    program.OfficialSegRefiner = ResourceObservedBatch8
    program.ARM = "SegRefiner_LR_first64_batch8"
    original_write = program.write
    def write(path, value):
        path = Path(path)
        if path.name == "INFERENCE_INPUTS.json":
            adaptation = Path(arguments.root) / "EXECUTION_BATCH8_ADAPTATION.md"
            if sha256(adaptation) != ADAPTATION_SHA:
                raise ValueError("Locked hardware adaptation protocol differs")
            value["configuration"].update(hardware_batch_max=8, official_default_batch_max=32,
                                           hardware_adaptation_sha256=ADAPTATION_SHA,
                                           rand_consumption_distribution_changed_from_batch32=True,
                                           batch32_output_bitwise_equivalence_claimed=False)
            shutil.copy2(adaptation, path.parent / "source" / adaptation.name)
        if path.name == "SUMMARY.json":
            rows = [json.loads(line) for line in (path.parent / "IMAGE_DIAGNOSTICS.jsonl").read_text(encoding="utf-8").splitlines()]
            probes = [row["actual"]["wddm_memory"] for row in rows]
            def peak(field):
                values = [probe[field] for probe in probes if probe[field] is not None]
                return max(values) if values else None
            value.update(hardware_batch_max=8, official_default_batch_max=32,
                         hardware_adaptation_sha256=ADAPTATION_SHA, rand_consumption_distribution_changed_from_batch32=True,
                         batch32_output_bitwise_equivalence_claimed=False,
                         wddm={"all_images_have_observation": all(probe["all_samples_present"] for probe in probes),
                               "dedicated_sampled_peak_bytes": peak("dedicated_sampled_peak_bytes"),
                               "shared_sampled_peak_bytes": peak("shared_sampled_peak_bytes"),
                               "committed_sampled_peak_bytes": peak("committed_sampled_peak_bytes"),
                               "sampling_interval_seconds": 1.0, "observer_overhead_exactly_measured": False,
                               "observer_join_outside_endpoint": True, "scope": "owned-process whole WDDM adapter instances; sampled peaks, not exact peaks"})
            value["limitations"].append("batch8 hardware adaptation changes stochastic draw partition; not untouched default32 output or a quality-selected batch")
        return original_write(path, value)
    program.write = write
    return program


if __name__ == "__main__":
    program = load_program()
    # Parse only the unchanged explicit producer CLI, without executing its
    # original __main__ branch during namespace construction.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "upstream", "checkpoint", "native-run", "images-list", "image-meta", "runtime-verification"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--engineering-receipt")
    arguments = parser.parse_args()
    raise SystemExit(program.main(arguments))
