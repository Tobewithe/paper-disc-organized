"""Laptop-only CPU continuation after the original 47 GiB storage cap fails.

The original packer and historical MIGRATION_IDENTITY are not edited. The
identity remains the content/source lineage. A separate, explicit storage
policy supersedes its historical resource budget, without changing any tensor,
sample, payload field, compression setting or previously committed archive.
No model forward is performed. Run through the existing research runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid


OLD_CAP = 47 * 1024**3
NEW_CAP = 49 * 1024**3
LAPTOP_PYTHON = Path("C:/Users/28358/anaconda3/envs/pytorch/python.exe")


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def update_run(out, **fields):
    path = Path(out) / "run.json"
    value = load(path) if path.exists() else {}
    value.update(fields)
    dump(path, value)


def validate_original_failure(previous, identity, original_script):
    failure_path = previous / "FAILURE.json"
    failure = load(failure_path)
    run = load(previous / "run.json")
    message = str(failure.get("error", ""))
    allowed = (
        "47 GiB conservative migration budget would be exceeded",
        "47 GiB serialized migration budget exceeded",
    )
    if not any(text in message for text in allowed):
        raise RuntimeError("Continuation is permitted only after the original 47 GiB cap failure")
    if run.get("status") not in {"failed", "error"}:
        raise RuntimeError("Original packing Run must have finished with failed status")
    if (previous / "COMPLETE.json").exists() and load(previous / "COMPLETE.json").get("complete"):
        raise RuntimeError("Original pack already completed; a storage continuation is unnecessary")
    if identity.get("max_asset_bytes") != OLD_CAP:
        raise AssertionError("Historical migration budget is not the expected 47 GiB")
    if sha(original_script) != identity["inputs"]["script"]:
        raise AssertionError("Original packer differs from the content-lineage snapshot")
    return failure, run


def make_archive_writer(module, out, amendment_sha256):
    """The only replaced packer function: byte-identical serializer, new guard."""
    def write_archive(path, payload, used_bytes, assets):
        path, assets = Path(path), Path(assets)
        images = (assets / "images").resolve()
        if path.resolve().parent != images or path.exists():
            raise RuntimeError("Continuation may write only a new archive in assets/images")
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            # Keep the exact failed file instead of overwriting failed evidence.
            retained = out / "retained_partial_archives"
            retained.mkdir(exist_ok=True)
            if temporary.resolve().parent != images:
                raise RuntimeError("Unsafe interrupted archive path")
            old_hash = sha(temporary)
            destination = retained / (temporary.name + "." + old_hash + "." + uuid.uuid4().hex)
            if destination.resolve().parent != retained.resolve():
                raise RuntimeError("Unsafe retained archive destination")
            os.replace(temporary, destination)
            dump(destination.with_name(destination.name + ".json"), {
                "source": str(temporary), "sha256": old_hash,
                "reason": "preserved interrupted archive before storage continuation",
            })
        estimate = module.tensor_bytes(payload) + 4 * 1024**2
        if used_bytes + estimate > NEW_CAP:
            raise RuntimeError("49 GiB amended conservative storage budget would be exceeded")
        if shutil.disk_usage(assets).free < estimate + module.RESERVE_BYTES:
            raise RuntimeError("Laptop free-space reserve reached under amended storage policy")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
                module.torch.save(payload, compressed)
            raw.flush()
            os.fsync(raw.fileno())
        size = temporary.stat().st_size
        if used_bytes + size > NEW_CAP:
            raise RuntimeError("49 GiB amended serialized storage budget exceeded; temporary retained")
        digest = sha(temporary)
        os.replace(temporary, path)
        return size, digest

    write_archive.storage_policy_sha256 = amendment_sha256
    return write_archive


def resume_storage(args):
    # A desktop can inspect/compile this script, but cannot accidentally execute
    # the CPU packing pipeline or import its torch/Ultralytics implementation.
    if os.name != "nt" or not LAPTOP_PYTHON.exists() or not Path(sys.executable).samefile(LAPTOP_PYTHON):
        raise RuntimeError("Run this CPU pack continuation only with the authorized laptop Python")
    cfg = load(args.config)
    assets = Path(args.assets or cfg.get("transfer_assets", Path(args.config).resolve().parent / "assets")).resolve()
    out, previous = Path(args.out).resolve(), Path(args.previous_run).resolve()
    if out == previous or out.name != args.run_id or not args.run_id.startswith("RUN_"):
        raise ValueError("Use a new, independent runs/RUN_* output directory")
    original_script = Path(args.original_packer).resolve()
    identity_path = assets / "MIGRATION_IDENTITY.json"
    identity = load(identity_path)
    identity_sha256 = sha(identity_path)
    failure, old_run = validate_original_failure(previous, identity, original_script)
    if (assets / "COMPLETE.json").exists() and load(assets / "COMPLETE.json").get("complete"):
        raise RuntimeError("Assets are already complete; do not restart packing")
    out.mkdir(parents=True, exist_ok=True)
    initial_manifest_path = assets / "manifest.jsonl"
    initial_manifest_bytes = initial_manifest_path.read_bytes()
    initial = [json.loads(line) for line in initial_manifest_bytes.decode("utf-8-sig").splitlines() if line]
    (out / "PREEXISTING_MANIFEST.jsonl").write_bytes(initial_manifest_bytes)
    initial_manifest_sha256 = hashlib.sha256(initial_manifest_bytes).hexdigest()
    extension_path = assets / "STORAGE_POLICY_AMENDMENT.json"
    scientific_scope = {
        "candidate_membership": "unchanged: fixed 10000/1000/2000 split",
        "tensor_values_and_dtypes": "unchanged",
        "serializer": "original gzip level1 lossless torch.save; unchanged",
        "existing_archives": "immutable bytes and SHA256; reused after original validation",
        "migration_identity": "historical content/source lineage; historical storage cap superseded below",
    }
    stable = {
        "schema": "prototype-evidence-storage-policy-amendment-v1",
        "base_migration_fingerprint": identity["fingerprint"],
        "base_migration_identity_sha256": identity_sha256,
        "historical_max_asset_bytes": OLD_CAP,
        "effective_max_asset_bytes": NEW_CAP,
        "original_packer_sha256": sha(original_script),
        "continuation_packer_sha256": sha(__file__),
        "scope": scientific_scope,
        "server_placement": "new val archives may use the Study-specific system-disk overflow; same bytes via symlink",
    }
    if extension_path.exists():
        amendment = load(extension_path)
        if any(amendment.get(key) != value for key, value in stable.items()):
            raise RuntimeError("Existing storage amendment differs; preserve prior assets and investigate")
    else:
        amendment = dict(stable, created_at=now(), originating_resume_run_id=args.run_id,
                         original_run_id=old_run.get("run_id", previous.name),
                         original_failure_sha256=sha(previous / "FAILURE.json"),
                         original_failure=failure,
                         preexisting_manifest_sha256=initial_manifest_sha256,
                         preexisting_images=len(initial))
        dump(extension_path, amendment)
    amendment_bytes = extension_path.read_bytes()
    (out / "STORAGE_POLICY_AMENDMENT.json").write_bytes(amendment_bytes)
    amendment_sha256 = hashlib.sha256(amendment_bytes).hexdigest()
    prior_run = load(out / "run.json") if (out / "run.json").exists() else {}
    if prior_run.get("run_id", args.run_id) != args.run_id:
        raise AssertionError("Output directory belongs to a different Run")
    update_run(out, run_id=args.run_id, status="running", storage_resume_started_at=now(),
               storage_resume_kind="asset_storage_continuation",
               previous_run=str(previous), original_packer_sha256=sha(original_script),
               continuation_packer_sha256=sha(__file__), storage_policy_sha256=amendment_sha256,
               historical_budget_bytes=OLD_CAP, effective_budget_bytes=NEW_CAP,
               storage_resume_command=sys.argv, model_forward_performed=False)
    try:
        spec = importlib.util.spec_from_file_location("original_immutable_asset_packer", original_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.CAP_BYTES != OLD_CAP:
            raise AssertionError("Original packer constant changed")
        # __file__, CAP_BYTES and main() are original. Their historical identity
        # remains reproducible; the amendment records the effective resource
        # policy and the exact wrapper responsible for this single substitution.
        module.write_archive = make_archive_writer(module, out, amendment_sha256)
        original_dump = module.dump

        def dump_with_policy(path, value):
            path = Path(path)
            if path.name in {"COMPLETE.json", "PARTIAL.json", "ENVIRONMENT.json"} and path.parent.resolve() in {assets, out}:
                value = dict(value, storage_policy_sha256=amendment_sha256,
                             effective_max_asset_bytes=NEW_CAP,
                             storage_budget_supersedes_historical_identity=True)
            if path.name == "COMPLETE.json" and path.parent.resolve() in {assets, out}:
                # The original packer finishes before the additional immutable-
                # archive audit. Do not expose completion to the live relay yet.
                original_dump(out / "PROVISIONAL_COMPLETE.json", dict(value, completion_not_yet_published=True))
                return
            original_dump(path, value)

        module.dump = dump_with_policy
        module.main(argparse.Namespace(config=Path(args.config), out=out, assets=assets, limit_per_split=None))
        if sha(identity_path) != identity_sha256:
            raise AssertionError("Historical MIGRATION_IDENTITY bytes unexpectedly changed")
        # This is transport verification only, not an additional model pass.
        for metadata in initial:
            path = assets / metadata["path"]
            current = load(path.with_name(f"{int(metadata['image_id']):012d}.meta.json"))
            if (path.stat().st_size != metadata["compressed_bytes"]
                    or sha(path) != metadata["compressed_sha256"]
                    or current["compressed_sha256"] != metadata["compressed_sha256"]):
                raise AssertionError("Previously committed archive changed during continuation")
        receipt = load(out / "PROVISIONAL_COMPLETE.json")
        if not receipt.get("complete") or receipt.get("fingerprint") != identity["fingerprint"]:
            raise AssertionError("Storage continuation did not complete all original images")
        dump(out / "PREEXISTING_ARCHIVES_AUDIT.json", {
            "status": "passed", "images": len(initial), "manifest_sha256": initial_manifest_sha256,
            "compressed_bytes_and_hashes_unchanged": True, "at": now(),
        })
        receipt.pop("completion_not_yet_published", None)
        receipt["preexisting_archives_audit_sha256"] = sha(out / "PREEXISTING_ARCHIVES_AUDIT.json")
        receipt["storage_resume_run_id"] = args.run_id
        dump(out / "COMPLETE.json", receipt)
        dump(assets / "COMPLETE.json", receipt)
        update_run(out, status="completed", exit_code=0, completed_at=now(),
                   completed_images=sum(receipt["completed"].values()),
                   archive_bytes=receipt["archive_bytes"])
    except Exception as exc:
        dump(out / "STORAGE_CONTINUATION_FAILURE.json", {
            "at": now(), "error": repr(exc), "traceback": traceback.format_exc(),
            "storage_policy_sha256": amendment_sha256,
        })
        # The relay observes FAILURE.json, including failures after main() has
        # returned. Never leave a full-completion claim after a failed audit.
        if not (out / "FAILURE.json").exists():
            dump(out / "FAILURE.json", {"error_type": type(exc).__name__, "error": str(exc),
                                       "traceback": traceback.format_exc()})
        for path in (out / "COMPLETE.json", assets / "COMPLETE.json"):
            if path.exists() and load(path).get("storage_policy_sha256") == amendment_sha256:
                retained = out / (path.parent.name + "_UNCONFIRMED_COMPLETE_" + uuid.uuid4().hex + ".json")
                os.replace(path, retained)
        update_run(out, status="failed", exit_code=1, completed_at=now(), error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--original-packer", type=Path, default=Path(__file__).with_name("prepare_transfer_assets.py"))
    parser.add_argument("--previous-run", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    resume_storage(parser.parse_args())


if __name__ == "__main__":
    main()
