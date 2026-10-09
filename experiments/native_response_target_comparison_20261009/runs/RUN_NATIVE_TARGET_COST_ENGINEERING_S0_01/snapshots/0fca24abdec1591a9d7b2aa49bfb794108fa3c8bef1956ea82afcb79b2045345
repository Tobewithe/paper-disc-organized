"""Isolated three-arm cost; use the original reviewed necessary response path.

Only the numeric gate identity changes: each fresh target child maps explicitly
to original RCMC_first64 and binds that fitted model SHA before construction.
The immutable original benchmark source is not edited. No GT or AP is read.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from target_common import ARMS, MODEL_PARAMS, VAL_LIST_SHA, append, canonical_sha, dump, load_config, now, paths_from_list, sha

BASE_COST_SHA = "8b6ccf6d770cdacc10ed43f625b59c15fb37f8df6ec5554692cb4a44be09bcd6"


def resource_snapshot(out):
    import psutil
    heavy = ("extract_targets.py", "fit_target.py", "infer_targets.py", "evaluate_target_comparison.py", "bootstrap_target_ap.py")
    observed = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = process.info.get("cmdline") or []
            if any(any(Path(value).name == name for value in command) for name in heavy):
                observed.append(dict(pid=process.pid, command=command))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    dump(out / "RESOURCE_BEFORE.json", dict(captured_at=now(), active_target_heavy_processes=observed,
         nvidia_smi_return_code=gpu.returncode, nvidia_smi_stdout=gpu.stdout, nvidia_smi_stderr=gpu.stderr,
         scope="single snapshot before all costs; no interval sampling or claim to observe every background process"))
    if observed:
        raise ValueError("Target extraction/fit/inference/scoring heavy process still active; cost must be serial")


def reviewed_executor(path, model_sha):
    if sha(path) != BASE_COST_SHA:
        raise ValueError("Reviewed necessary-path benchmark source changed")
    spec = importlib.util.spec_from_file_location("target_reviewed_cost", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Explicit new asset profile, not a claim to be the old1500 model.
    module.RESPONSE_SHA256 = model_sha
    return module


def child(a, c):
    began = time.perf_counter()
    import cv2
    import numpy as np
    import torch
    out = Path(a.child_output)
    if out.exists():
        raise FileExistsError("Fresh arm-repeat process requires a fresh directory")
    out.mkdir(parents=True)
    inputs = json.loads((Path(a.output) / "COST_INPUTS.json").read_text())
    arm, original_arm = a.arm, "baseline" if a.arm == "baseline" else "RCMC_first64"
    model_path = inputs["models"].get(arm)
    model_sha = sha(model_path) if model_path else None
    original = reviewed_executor(inputs["original_benchmark"], model_sha)
    dependency_seconds = time.perf_counter()-began
    if torch.__version__ != "2.5.1" or not torch.cuda.is_available():
        raise ValueError("Cost requires registered laptop CUDA2.5.1")
    paths = [Path(p) for p in inputs["image_paths"]]
    read_start = time.perf_counter()
    rgbs = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError("Cost image decode failed")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb_seconds = time.perf_counter()-read_start
    dump(out / "RAW_RGB_INPUTS.json", dict(images=[dict(image_id=int(p.stem), shape=list(rgb.shape),
             rgb_bytes_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()) for p, rgb in zip(paths, rgbs)]))
    initialize = time.perf_counter()
    executor = original.IndividualExecutor(SimpleNamespace(arm=original_arm, device="cuda", weights=c["weights"],
                                                         vendor=c["vendor"], response_model=model_path))
    initialization_seconds = time.perf_counter()-initialize
    from pycocotools import mask as mask_utils
    from ultralytics.data.converter import coco80_to_coco91_class
    categories = coco80_to_coco91_class()
    reference = Path(a.reference)
    wanted = {int(p.stem) for p in paths}
    expected_inputs = {}
    with (reference / "BASELINE_PARITY_IMAGES.jsonl").open() as f:
        for line in f:
            v = json.loads(line)
            if v["image_id"] in wanted:
                expected_inputs[v["image_id"]] = v["input_sha256"]
    if set(expected_inputs) != wanted:
        raise ValueError("Reference input receipts incomplete")
    warm, samples = [], []
    try:
        with torch.inference_mode():
            for path, rgb in zip(paths[:4], rgbs[:4]):
                with original.Timer("cuda") as timer:
                    result = executor.execute(rgb)
                original.check_isolation(original_arm, result, executor.loaded_assets)
                warm.append(dict(image_id=int(path.stem), seconds=timer.seconds))
                del result
            postwarm = original.memory(torch)
            for path, rgb in zip(paths, rgbs):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = original.memory(torch)
                with original.Timer("cuda") as timer:
                    result = executor.execute(rgb)
                after = original.memory(torch)
                original.check_isolation(original_arm, result, executor.loaded_assets)
                iid = int(path.stem)
                reference_path = reference / arm / "images" / f"{iid:012d}.json"
                read_start = time.perf_counter()
                expected = json.loads(reference_path.read_text())
                ref_read = time.perf_counter()-read_start
                records, rle_seconds, parity_seconds = original.records_and_verify(result, iid, expected, categories, mask_utils)
                audit = time.perf_counter()
                input_sha = hashlib.sha256(result["input_tensor_for_untimed_audit"].cpu().contiguous().numpy().tobytes()).hexdigest()
                if input_sha != expected_inputs[iid]:
                    raise ValueError("Isolated input differs from native production")
                sample = dict(arm=arm, repeat_index=a.repeat, image_id=iid, deployment_seconds=timer.seconds,
                              native_rows=result["native_rows"], method_work=result["method"], loaded_assets=executor.loaded_assets,
                              all_rows_identity_and_rle_exact=True, isolated_method_and_assets=True,
                              native_input_tensor_sha256=input_sha, native_input_tensor_exact=True,
                              reference_file=str(reference_path), reference_sha256=sha(reference_path),
                              output_mask_binary_sha256=hashlib.sha256(result["masks"].tobytes()).hexdigest(),
                              reference_read_seconds=ref_read, rle_records_encoding_seconds=rle_seconds,
                              identity_rle_parity_seconds=parity_seconds, input_tensor_audit_seconds=time.perf_counter()-audit,
                              memory_before_endpoint=before, memory_after_endpoint=after,
                              cuda_peak_scope="reset immediately before endpoint; includes resident own model and retained allocator cache",
                              cpu_peak_scope="fresh process lifetime including imports/init/warmup/audit; OS cumulative peak not reset")
                write = time.perf_counter()
                dump(out / "outputs" / f"{iid:012d}.json", dict(image_id=iid, detections=records))
                sample["output_write_seconds"] = time.perf_counter()-write
                append(out / "COST_SAMPLES.jsonl", sample)
                samples.append(sample)
                del result, expected, records
        frozen = executor.integrity()
        dump(out / "CHILD_COMPLETE.json", dict(passed=True, arm=arm, original_method_path=original_arm,
             gate_profile="fresh20k target-specific numeric HGB; only asset identity substituted", model_sha256=model_sha,
             repeat_index=a.repeat, image_count=len(paths), samples_sha256=sha(out / "COST_SAMPLES.jsonl"),
             source_binding_sha256=inputs["source_binding_sha256"], dependency_seconds=dependency_seconds,
             image_read_rgb_seconds=rgb_seconds, initialization_seconds=initialization_seconds,
             warmup_samples=warm, warmup_seconds=sum(v["seconds"] for v in warm), postwarm_memory=postwarm,
             process_final_memory=original.memory(torch), loaded_assets=executor.loaded_assets, frozen_integrity=frozen,
             pid=os.getpid(), runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                          gpu=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32,
                                          cudnn_tf32=torch.backends.cudnn.allow_tf32, threads=torch.get_num_threads()),
             total_child_seconds=time.perf_counter()-began, gt_used=False, ap_measured=False, completed_at=now()))
    except Exception as exc:
        dump(out / "CHILD_FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_at=now()))
        raise
    finally:
        executor.close()


def describe(v):
    import numpy as np
    x = np.asarray(v, dtype=np.float64)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), std_population=float(x.std()),
                p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)), min=float(x.min()), max=float(x.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "protocol-sha256", "cost-protocol", "cost-protocol-sha256", "reference", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--engineering", action="store_true")
    p.add_argument("--engineering-receipt")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--repeat", type=int)
    p.add_argument("--child-output")
    a = p.parse_args()
    c = load_config(a.config, a.protocol_sha256)
    if sha(a.cost_protocol) != a.cost_protocol_sha256:
        raise ValueError("Predeclared cost protocol changed")
    if a.child_output:
        child(a, c)
        return
    out, reference = Path(a.output).resolve(), Path(a.reference).resolve()
    if (out / "COST_INPUTS.json").exists():
        raise FileExistsError("Cost retry requires new Run")
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    source = json.loads((reference / "SUMMARY.json").read_text())
    receipt = json.loads((reference / "COMPLETE.json").read_text())
    metadata = json.loads((reference / "run.json").read_text())
    if not source["passed"] or source["engineering"] or source["image_count"] != 5000 or receipt["summary_sha256"] != sha(reference / "SUMMARY.json"):
        raise ValueError("Cost needs completed formal new-model native5k")
    if metadata["status"] != "completed" or metadata["return_code"] != 0 or metadata["artifact_completeness"] != "complete":
        raise ValueError("Formal5k runner completion not yet observed")
    production_inputs = json.loads((reference / "INPUTS.json").read_text())
    paths, ids = paths_from_list(c["val"]["images_list"], VAL_LIST_SHA, 5000)
    count, repeats = (4, 1) if a.engineering else (32, 3)
    models = {arm: str(Path(production_inputs["model_runs"][arm]) / "model.json") for arm in ARMS[1:]}
    fit_profiles = {}
    for arm in ARMS[1:]:
        model_root = Path(production_inputs["model_runs"][arm])
        fit = json.loads((model_root / "SUMMARY.json").read_text())
        fit_receipt = json.loads((model_root / "COMPLETE.json").read_text())
        if (not fit["passed"] or fit["engineering"] or fit["iterations"] != 100 or fit["portable_sign_mismatches"] != 0
                or fit["portable_max_abs_error"] >= 1e-12 or fit_receipt["summary_sha256"] != sha(model_root / "SUMMARY.json")
                or any(fit["full_hgb_parameters"][key] != value for key, value in MODEL_PARAMS.items())):
            raise ValueError("Completed formal fit/parameters/portable sign profile differs")
        fit_profiles[arm] = dict(summary_sha256=sha(model_root / "SUMMARY.json"), complete_sha256=sha(model_root / "COMPLETE.json"),
                                 full_hgb_parameters=fit["full_hgb_parameters"], portable_sign_mismatches=fit["portable_sign_mismatches"],
                                 portable_max_abs_error=fit["portable_max_abs_error"], training_feature_bytes_sha256=fit["training_feature_bytes_sha256"],
                                 supervised_binding_sha256=fit["supervised_binding_sha256"], weight_bytes_sha256=fit["weight_bytes_sha256"])
    equality = ("full_hgb_parameters", "training_feature_bytes_sha256", "supervised_binding_sha256", "weight_bytes_sha256")
    if any(fit_profiles["target_I"][key] != fit_profiles["target_H"][key] for key in equality):
        raise ValueError("Formal target capacity/features/row/weight profile differs")
    original = Path(c["frozen_scripts"]) / "benchmark_individual_cost.py"
    if sha(original) != BASE_COST_SHA:
        raise ValueError("Original cost executor source identity differs")
    binding = dict(config_sha256=sha(a.config), protocol_sha256=a.protocol_sha256,
                   cost_protocol_sha256=a.cost_protocol_sha256, script_sha256=sha(__file__), original_benchmark_sha256=BASE_COST_SHA,
                   reference_summary_sha256=sha(reference / "SUMMARY.json"), reference_complete_sha256=sha(reference / "COMPLETE.json"),
                   reference_run_metadata_sha256=sha(reference / "run.json"), fit_profiles=fit_profiles,
                   models_sha256={arm: sha(path) for arm, path in models.items()},
                   producer_source_sha256=c["producer_source_sha256"], weights_sha256=sha(c["weights"]))
    if binding["models_sha256"] != production_inputs["model_sha256"]:
        raise ValueError("Formal numeric models changed")
    binding_sha = canonical_sha(binding)
    if not a.engineering:
        e = json.loads(Path(a.engineering_receipt).read_text())
        if not e["passed"] or not e["engineering"] or e["source_binding_sha256"] != binding_sha:
            raise ValueError("Exact same cost profile requires passing engineering")
    resource_snapshot(out)
    dump(out / "COST_INPUTS.json", dict(source_binding=binding, source_binding_sha256=binding_sha, models=models,
         original_benchmark=str(original), image_paths=[str(path) for path in paths[:count]], image_ids=ids[:count],
         engineering=a.engineering, repeats=repeats, endpoint="preloaded RGB -> preprocess/native forward/own smooth+gate -> all original binary masks CPU",
         outside="RGB read/init/warm/RLE/identity+input audit/reference read/output JSON", gt_used=False))
    samples, children = [], []
    for repeat in range(repeats):
        order = ARMS[repeat:] + ARMS[:repeat]
        for arm in order:
            child_out = out / "measurements" / f"repeat{repeat}" / arm
            command = [sys.executable, str(Path(__file__).resolve()), "--config", a.config, "--protocol-sha256", a.protocol_sha256,
                       "--cost-protocol", a.cost_protocol, "--cost-protocol-sha256", a.cost_protocol_sha256,
                       "--reference", str(reference), "--output", str(out), "--arm", arm, "--repeat", str(repeat), "--child-output", str(child_out)]
            t = time.perf_counter()
            child_out.parent.mkdir(parents=True, exist_ok=True)
            logroot = child_out.parent / (arm + "_process")
            logroot.mkdir()
            with (logroot / "stdout.log").open("wb") as stdout, (logroot / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                dump(logroot / "PROCESS_STARTED.json", dict(pid=process.pid, command=command, started_at=now()))
                code = process.wait()
            wall = time.perf_counter()-t
            dump(logroot / "PROCESS_OBSERVED.json", dict(pid=process.pid, return_code=code, process_wall_seconds=wall, completed_at=now()))
            if code:
                raise RuntimeError("Cost child failed; preserve Run and stop")
            receipt = json.loads((child_out / "CHILD_COMPLETE.json").read_text())
            if not receipt["passed"] or receipt["source_binding_sha256"] != binding_sha or receipt["samples_sha256"] != sha(child_out / "COST_SAMPLES.jsonl"):
                raise ValueError("Cost child receipts differ")
            children.append(dict(path=str(child_out), receipt_sha256=sha(child_out / "CHILD_COMPLETE.json"),
                                 process_wall_seconds=wall, child_seconds=receipt["total_child_seconds"],
                                 startup_supervision_shutdown_residual_seconds=wall-receipt["total_child_seconds"],
                                 pure_startup_seconds=None))
            samples.extend(json.loads(line) for line in (child_out / "COST_SAMPLES.jsonl").read_text().splitlines())
    lookup = {(v["repeat_index"], v["image_id"], v["arm"]): v for v in samples}
    stats = {}
    for arm in ARMS:
        items = [v for v in samples if v["arm"] == arm]
        delta = [v["deployment_seconds"]-lookup[v["repeat_index"], v["image_id"], "baseline"]["deployment_seconds"] for v in items]
        stats[arm] = dict(deployment_seconds=describe([v["deployment_seconds"] for v in items]),
                          paired_signed_increment_seconds=describe(delta), negative_increment_count=sum(v < 0 for v in delta))
    if sha(__file__) != binding["script_sha256"] or sha(original) != BASE_COST_SHA or any(sha(models[k]) != v for k, v in binding["models_sha256"].items()):
        raise ValueError("Cost source/model changed during Run")
    dump(out / "COST_SAMPLES.json", dict(samples=samples))
    dump(out / "SUMMARY.json", dict(passed=True, status="cost_complete", engineering=a.engineering, source_binding_sha256=binding_sha,
         images_per_arm_repeat=count, repeats=repeats, fresh_children=len(children), samples=len(samples), statistics=stats, children=children,
         total_elapsed_seconds=time.perf_counter()-began, gt_used=False, ap_measured=False, confidence_intervals=None,
         source_unchanged=True, completed_at=now(), limitations=["Fixed panel/device/endpoint only", "CPU OSpeak includes initialization/warm/audit; GPU includes retained resident allocator", "Pure process startup not separately measured"]))
    dump(out / "COST_COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), samples_sha256=sha(out / "COST_SAMPLES.json")))


if __name__ == "__main__":
    main()
