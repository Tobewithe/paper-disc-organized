"""Server CPU-only, six-file check of one-pass gzip loading; no model runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time
import traceback

import numpy as np
import torch

import asset_runtime as current


def exact_compare(left, right, path="root"):
    """Check every nested leaf, including tensor bytes (not float tolerance)."""
    if type(left) is not type(right):
        raise AssertionError(f"{path}: types changed: {type(left)}, {type(right)}")
    if torch.is_tensor(left):
        if left.device.type != "cpu" or right.device.type != "cpu":
            raise AssertionError(f"{path}: non-CPU tensor")
        if (left.dtype != right.dtype or left.shape != right.shape
                or left.layout != right.layout or left.stride() != right.stride()):
            raise AssertionError(f"{path}: tensor metadata changed")
        a = left.detach().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            raise AssertionError(f"{path}: tensor bytes changed")
        return 1, a.numel(), 1
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
            raise AssertionError(f"{path}: array bytes changed")
        return 0, left.nbytes, 1
    if isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f"{path}: keys changed")
        stats = [exact_compare(left[k], right[k], f"{path}.{k}") for k in left]
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: sequence length changed")
        stats = [exact_compare(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))]
    else:
        equal = struct.pack("!d", left) == struct.pack("!d", right) if isinstance(left, float) else left == right
        if not equal:
            raise AssertionError(f"{path}: scalar changed")
        return 0, 0, 1
    return tuple(sum(row[j] for row in stats) for j in range(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-loader", required=True)
    parser.add_argument("--image-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if os.name == "nt" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use the Linux server with CUDA_VISIBLE_DEVICES empty")
    if len(args.image_ids) != 6 or len(set(args.image_ids)) != 6:
        raise ValueError("This bounded check requires exactly six fixed distinct image IDs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = current.resolve_runtime_config(current.load_json(args.config))
    spec = importlib.util.spec_from_file_location("asset_runtime_stream_before", args.old_loader)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    files = []
    for iid in args.image_ids:
        path = Path(cfg["assets"]) / "images" / f"{iid:012d}.pt.gz"
        meta = path.with_name(f"{iid:012d}.meta.json")
        metadata = current.load_json(meta)
        actual_sha = current.sha256(path)
        if actual_sha != metadata["compressed_sha256"]:
            raise AssertionError(f"{iid}: compressed file differs before test")
        files.append(dict(image_id=iid, split=metadata["split"], path=str(path),
                          compressed_bytes=path.stat().st_size, compressed_sha256=actual_sha,
                          metadata_sha256=current.sha256(meta)))
    # The fixed selection and exact inputs are recorded before observing timings.
    current.dump(out / "IO_FILES.json", files)
    rows = []
    started = time.monotonic()
    for index, item in enumerate(files):
        path = Path(item["path"])
        with gzip.open(path, "rb") as stream:
            payload = stream.read()
        once_sha, decompressed_bytes = hashlib.sha256(payload).hexdigest(), len(payload)
        del payload
        stream_sha = hashlib.sha256()
        with gzip.open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                stream_sha.update(block)
        if stream_sha.hexdigest() != once_sha:
            raise AssertionError("One-shot and streaming decompressed bytes differ")
        times = {"old_stream": [], "new_bytesio": []}
        checked = []
        for repeat in range(2):
            order = ["old_stream", "new_bytesio"] if (index + repeat) % 2 == 0 else ["new_bytesio", "old_stream"]
            values = {}
            for name in order:
                module = previous if name == "old_stream" else current
                t0 = time.perf_counter()
                values[name] = module.load_asset(cfg, item["image_id"], verify=True)
                times[name].append(time.perf_counter() - t0)
            checked.append(exact_compare(values["old_stream"], values["new_bytesio"]))
            del values
        if current.sha256(path) != item["compressed_sha256"]:
            raise AssertionError("Input changed during bounded I/O check")
        row = dict(**item, decompressed_bytes=decompressed_bytes, decompressed_sha256=once_sha,
                   decompressed_bytes_equal=True, all_nested_leaves_equal=True,
                   tensor_leaves_including_aliases=checked[0][0], compared_tensor_bytes_including_aliases=checked[0][1],
                   total_leaves=checked[0][2], seconds=times,
                   median_old_seconds=statistics.median(times["old_stream"]),
                   median_new_seconds=statistics.median(times["new_bytesio"]))
        row["median_speedup"] = row["median_old_seconds"] / row["median_new_seconds"]
        rows.append(row)
        current.dump(out / "IO_PROGRESS.json", dict(completed_images=len(rows), total_images=6, rows=rows))
        print(json.dumps({k: row[k] for k in ("image_id", "median_old_seconds", "median_new_seconds", "median_speedup")}), flush=True)
    old_total = sum(r["median_old_seconds"] for r in rows)
    new_total = sum(r["median_new_seconds"] for r in rows)
    report = dict(passed=True, kind="cpu_io_equivalence", images=len(rows), rows=rows,
                  old_loader_sha256=current.sha256(args.old_loader),
                  new_loader_sha256=current.sha256(current.__file__),
                  compressed_and_decompressed_bytes_unchanged=True, all_tensor_bytes_equal=True,
                  all_other_leaves_equal=True, compressed_integrity_checks_preserved=True,
                  old_total_median_seconds=old_total, new_total_median_seconds=new_total,
                  aggregate_speedup=old_total / new_total, elapsed_seconds=time.monotonic() - started,
                  repeats_per_file_per_loader=2, order="alternating per file/repeat", cache_policy="OS cache retained; no drop_caches",
                  cpu_threads=1, cuda_initialized=torch.cuda.is_initialized(), model_execution=False,
                  training_steps=0, method_or_budget_changed=False,
                  limitation="Six fixed available files; timing is not a training throughput or whole-cache estimate.",
                  finished_at=datetime.now(timezone.utc).isoformat())
    if report["cuda_initialized"]:
        raise AssertionError("Unexpected CUDA initialization in CPU-only check")
    current.dump(out / "IO_REPORT.json", report)
    current.dump(out / "COMPLETE.json", dict(passed=True, kind="cpu_io_equivalence", images=6,
                 all_tensor_bytes_equal=True, model_execution=False, training_steps=0))


if __name__ == "__main__":
    main()
