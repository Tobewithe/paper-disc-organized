"""Shared contracts for one native action and two frozen supervision targets."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

TRAIN_IDS_SHA = "ef3453e65b80a2d07997c882b0624a88f6d8161fcd32e2d629cdff3285ba3c63"
TRAIN_ANN_SHA = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
TRAIN_LIST_SHA = "be017cfda05e13ffab85c79aa877c4c6b4920edf835f1913d6f62524f2c60dc3"
VAL_LIST_SHA = "b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db"
OFFICIAL_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
FROZEN_SOURCE = {
    "frozen_io.py": "cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293",
    "native_mask_adapter.py": "0a11d7cee6f73ec680ce7895f9501a2f87be5d97a8904d3fe6b3a62a958fa0c2",
    "risk_calibration.py": "b4d1fbae794e6e1de0a8d00bb99a721687224e5b530ac816638e113d3a09ae6f",
    "portable_risk.py": "b92dee9681af044ff5dd555a39f1b6c66d35314159be7ab26331ba6c321dccb9",
    "mask_calibration.py": "d238d586df78987b2288f1927599199e2a489ae855de48854d827246da40cac8",
    "local_features.py": "a0bdefe767f2fa38536e7c6261e32a1bc066238cc90bb7585de70490e8b0caf6",
}
MODEL_PARAMS = dict(loss="squared_error", learning_rate=.05, max_iter=100,
                    max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1.,
                    early_stopping=False, random_state=20260915)
ARMS = ("baseline", "target_I", "target_H")
PRODUCER_FILES = ("target_common.py", "native_response.py", "extract_targets.py", "fit_target.py",
                  "infer_targets.py", "run_target_pipeline.py", "preflight_targets.py", "collect_target_runs.py")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def load_config(path, expected_protocol):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c.get("schema") != "native_target_comparison_v1":
        raise ValueError("Unknown execution config schema")
    if c.get("protocol_sha256") != expected_protocol or sha(c["protocol"]) != expected_protocol:
        raise ValueError("Frozen protocol identity differs")
    for name, expected in FROZEN_SOURCE.items():
        if sha(Path(c["frozen_scripts"]) / name) != expected:
            raise ValueError("Reviewed frozen primitive changed: " + name)
    for name, expected in c["producer_source_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Execution producer source differs from preflight lock: " + name)
    sys.path.insert(0, str(Path(c["frozen_scripts"]).resolve()))
    return c


def runtime(require_gpu=False):
    import numpy as np
    import sklearn
    import torch
    if torch.__version__ != "2.5.1" or sklearn.__version__ != "1.6.1":
        raise ValueError("Registered laptop runtime requires Torch2.5.1 and sklearn1.6.1")
    if require_gpu and not torch.cuda.is_available():
        raise ValueError("Declared laptop CUDA unavailable")
    return dict(interpreter=sys.executable, torch=torch.__version__, sklearn=sklearn.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                pycocotools=importlib.metadata.version("pycocotools"), threads=torch.get_num_threads(),
                actual_ultralytics_version=getattr(sys.modules.get("ultralytics"), "__version__", None),
                actual_ultralytics_file=getattr(sys.modules.get("ultralytics"), "__file__", None))


def source_lock(run, config, config_path, extra_paths=()):
    """Archive every actual study/frozen/vendor Python source used by the loader.

    Vendor's full Python tree is hashed; core files are copied. This captures
    dependency identity without pretending all files executed in this Run.
    """
    run = Path(run)
    files = {}
    paths = [("config.json", Path(config_path)), ("PROTOCOL.md", Path(config["protocol"]))]
    paths += [("study/" + name, Path(__file__).with_name(name)) for name in PRODUCER_FILES]
    paths += [("frozen/" + name, Path(config["frozen_scripts"]) / name) for name in FROZEN_SOURCE]
    for key, original in paths:
        target = run / "source" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(original)
        if target.exists() and sha(target) != digest:
            raise ValueError("Archive source differs: " + key)
        if not target.exists():
            shutil.copy2(original, target)
        files[key] = dict(path=str(original.resolve()), sha256=digest, snapshot=str(target.relative_to(run)))
    # Immutable input hashes, without duplicating the large checkpoint/GT JSON.
    for i, p in enumerate(extra_paths):
        original = Path(p)
        key = "input/" + str(i) + "_" + original.name
        files[key] = dict(path=str(original.resolve()), sha256=sha(original), snapshot=None)
    vendor = Path(config["vendor"]).resolve()
    tree = {str(p.relative_to(vendor)).replace("\\", "/"): sha(p)
            for p in sorted(vendor.rglob("*.py"))}
    lock = dict(files=files, vendor_root=str(vendor), vendor_python_files=tree,
                vendor_tree_sha256=canonical_sha(tree), scope="identity of complete vendor Python tree; not an execution trace")
    dump(run / "SOURCE_LOCK.json", lock)
    return lock


def verify_lock(lock):
    for key, v in lock["files"].items():
        if sha(v["path"]) != v["sha256"]:
            raise ValueError("Execution input/source changed: " + key)
    vendor = Path(lock["vendor_root"])
    after = {str(p.relative_to(vendor)).replace("\\", "/"): sha(p) for p in sorted(vendor.rglob("*.py"))}
    if after != lock["vendor_python_files"]:
        raise ValueError("Vendor source changed")


def paths_from_list(path, expected_sha, expected_count):
    if sha(path) != expected_sha:
        raise ValueError("Original image-list bytes differ")
    paths = [Path(line.strip()).resolve() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [int(p.stem) for p in paths]
    if len(ids) != expected_count or len(set(ids)) != len(ids) or not all(p.is_file() for p in paths):
        raise ValueError("Original image-list scope incomplete")
    return paths, ids


def integer_targets(base_tp, base_union, trial_tp, trial_union, fallback=False):
    """Targets require valid GT pixels; caller leaves unassociated rows unknown."""
    if min(base_tp, trial_tp) < 0 or min(base_union, trial_union) <= 0:
        raise ValueError("Invalid GT/prediction pixel ledger")
    if fallback:
        return 0., 0., 0
    difference = sum(int(20 * trial_tp >= p * trial_union) - int(20 * base_tp >= p * base_union)
                     for p in range(10, 20))
    return trial_tp / trial_union - base_tp / base_union, difference / 10., difference


def apply_gain(gain, nonempty, supported=True):
    return bool(supported and nonempty and gain > 0)


def synthetic_contract():
    import numpy as np
    # Every exact COCO threshold, including upward and downward crossings.
    for p in range(10, 20):
        for tp in (p - 1, p, p + 1):
            _, h, ticks = integer_targets(p, 20, tp, 20)
            expected = sum(int(tp >= q) - int(p >= q) for q in range(10, 20))
            if ticks != expected or h != expected / 10:
                raise AssertionError("Signed rational threshold contract failed")
    if integer_targets(19, 20, 0, 20, fallback=True) != (0., 0., 0):
        raise AssertionError("Empty-trial fallback must have both targets zero")
    if apply_gain(1., False) or apply_gain(1., True, False) or apply_gain(0., True) or apply_gain(-1., True):
        raise AssertionError("Forced empty/scope/gain fallback must preserve baseline")
    if not apply_gain(1., True):
        raise AssertionError("Fixed positive-gain gate failed")
    unknown = np.array([np.nan, np.nan], dtype=np.float64)
    if not np.isnan(unknown).all():
        raise AssertionError("Unknown target marker changed")
    return dict(passed=True, exact_signed_threshold_cases=30, forced_empty_fallback=True,
                forced_empty_deployment_gate=True, zero_negative_and_unsupported_gate_fallback=True,
                unassociated_targets_remain_nan=True, gt_used=False)


def export_numeric(model, path):
    import numpy as np
    trees = [stage[0].nodes for stage in model._predictors]
    if any(len(stage) != 1 for stage in model._predictors) or any(t["is_categorical"].any() for t in trees):
        raise ValueError("Expected numeric single-output HGB")
    fields = dict(value="value", feature="feature_idx", threshold="num_threshold",
                  missing_left="missing_go_to_left", left="left", right="right", leaf="is_leaf")
    size = max(map(len, trees))
    nodes = {key: [np.pad(t[field], (0, size - len(t))).tolist() for t in trees] for key, field in fields.items()}
    dump(path, dict(format="numeric_hgb_v1", features=int(model.n_features_in_),
                    baseline=float(model._baseline_prediction[0, 0]),
                    max_depth=int(max(t["depth"].max() for t in trees)), nodes=nodes))


def peak_memory():
    import torch
    import psutil
    mem = psutil.Process().memory_info()
    return dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                cpu_peak_working_set_bytes=getattr(mem, "peak_wset", None), cpu_current_rss_bytes=mem.rss,
                scope="this process; not incremental model cost")
