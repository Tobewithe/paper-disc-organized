#!/usr/bin/env python3
"""Safely inspect non-tensor metadata of the official SegRefiner LR checkpoint.

The loader is deliberately CPU-only and uses PyTorch's restricted
``weights_only=True`` unpickler.  This tool has no unsafe-deserialization
fallback: an unreadable file is recorded as unknown rather than loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_under(root: Path, candidate: Path) -> Path:
    root, candidate = root.resolve(), candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing path outside study: {candidate}") from exc
    return candidate


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 32 and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return list(value)
    return None


def safe_metadata(value: Any, depth: int = 0) -> Any:
    """Return only non-tensor scalar metadata, bounded in depth and key count."""
    primitive = scalar(value)
    if primitive is not None:
        return primitive
    if depth >= 4:
        return {"_omitted": f"depth_limit_type={type(value).__name__}"}
    if isinstance(value, Mapping):
        output = {}
        for index, key in enumerate(sorted(value, key=lambda item: str(item))):
            if index >= 128:
                output["_omitted"] = "key_limit"
                break
            child = value[key]
            if isinstance(child, torch.Tensor):
                continue
            output[str(key)] = safe_metadata(child, depth + 1)
        return output
    return {"_omitted": f"unsupported_type={type(value).__name__}"}


def summarize_top_level_mapping(mapping: Mapping) -> dict:
    """Summarize a state dict without emitting every tensor name/shape."""
    tensors = [value for value in mapping.values() if isinstance(value, torch.Tensor)]
    non_tensors = {str(key): safe_metadata(value) for key, value in mapping.items() if not isinstance(value, torch.Tensor)}
    prefixes = Counter(str(key).split(".", 1)[0] for key in mapping)
    return {
        "mapping_key_count": len(mapping),
        "direct_tensor_count": len(tensors),
        "direct_tensor_numel": sum(tensor.numel() for tensor in tensors),
        "key_prefix_counts": dict(sorted(prefixes.items())),
        "non_tensor_top_level_entries": non_tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    run_dir = ensure_under(study_root, args.run_dir)
    checkpoint = ensure_under(study_root, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "audit_version": "segrefiner_checkpoint_metadata_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_unknown_metadata",
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "loader": {
            "torch_version": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "map_location": "cpu",
            "mmap": True,
            "weights_only": True,
            "unsafe_fallback_attempted": False,
        },
        "expected_locked_configuration": {
            "source_revision": "53419a2d38ea3da0b6e2be77e5b45e139195a0b3",
            "train_ann_file": "lvis_v1_train.json",
            "train_max_iters": 120000,
        },
    }
    try:
        loaded = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        result["safe_load"] = {"ok": True, "loaded_type": type(loaded).__name__}
        if isinstance(loaded, Mapping):
            result["top_level"] = summarize_top_level_mapping(loaded)
            metadata_keys = [key for key in loaded if str(key).lower() in {"meta", "metadata", "config", "cfg", "runner", "data"}]
            result["non_tensor_metadata"] = {
                str(key): safe_metadata(loaded[key]) for key in metadata_keys
            }
            result["checkpoint_manifest_status"] = (
                "metadata fields captured below; explicit dataset manifest remains unknown"
                if metadata_keys
                else "no non-tensor metadata/config/runner/data container was stored at checkpoint top level; training manifest unknown"
            )
        else:
            result["checkpoint_manifest_status"] = "safe-loaded object is not a mapping; manifest unknown"
    except Exception as exc:  # Keep the restricted-loader failure as an audit fact.
        result["safe_load"] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        result["checkpoint_manifest_status"] = "unknown: restricted CPU loader failed; no unsafe fallback used"
        result["status"] = "completed_restricted_loader_blocked"

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "audit_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "run_id": run_dir.name,
        "study_id": "STUDY_SEGREFINER_OFFICIAL_CONTENT_BASELINE_20261009",
        "kind": "checkpoint_metadata_provenance_audit",
        "status": result["status"],
        "result": "audit_results.json",
        "no_gpu": True,
        "restricted_weights_only_cpu_loader": True,
        "unsafe_deserialization_fallback": False,
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "safe_load": result["safe_load"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
