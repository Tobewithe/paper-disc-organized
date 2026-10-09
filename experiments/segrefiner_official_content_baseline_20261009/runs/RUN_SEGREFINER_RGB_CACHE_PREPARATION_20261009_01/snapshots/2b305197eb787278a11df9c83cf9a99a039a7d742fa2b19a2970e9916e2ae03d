"""Bind original COCO JPEG bytes to a reusable local RGB cache, without GT.

Run source and verify as independent executions through research_runner.
All outputs are new files; existing caches and native Run records are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as f:
        f.write(data)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def list_entries(path: Path, expected_sha: str) -> tuple[bytes, list[Path]]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha:
        raise ValueError(f"Original image-list SHA mismatch: {digest} != {expected_sha}")
    entries = [Path(line.strip()) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    if len(entries) != 5000 or len({p.name for p in entries}) != 5000:
        raise ValueError("Original val5k must have exactly 5000 unique filenames")
    if any(p.suffix.lower() != ".jpg" or not p.stem.isdigit() or len(p.stem) != 12 for p in entries):
        raise ValueError("Unexpected COCO JPEG name")
    return data, entries


def source(args) -> None:
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.images_root).resolve(strict=True)
    original_list = Path(args.images_list).resolve(strict=True)
    data, files = list_entries(original_list, args.original_list_sha256)
    records = []
    for ordinal, original_path in enumerate(files):
        p = original_path.resolve(strict=True)
        if not p.is_file() or p.parent != root:
            raise ValueError(f"Image outside declared source directory: {p}")
        records.append({"ordinal": ordinal, "image_id": int(p.stem), "filename": p.name,
                        "source_image_path": str(p), "size_bytes": p.stat().st_size,
                        "sha256": sha256(p)})
        if (ordinal + 1) % 500 == 0:
            print(f"source hashed {ordinal + 1}/5000", flush=True)
    write_bytes(out / "SOURCE_IMAGE_LIST.txt", data)
    payload = {"schema_version": 1, "created_at": now(), "host": platform.node(),
               "python": sys.executable, "source_images_root": str(root),
               "source_images_list": str(original_list),
               "source_images_list_sha256": args.original_list_sha256,
               "image_count": len(records), "total_bytes": sum(r["size_bytes"] for r in records),
               "images": records, "gt_read": False, "jpeg_bytes_modified": False}
    write_json(out / "SOURCE_IMAGE_MANIFEST.json", payload)
    summary = {"status": "complete", "passed": True, "host": platform.node(),
               "python": sys.executable, "image_count": len(records),
               "total_bytes": payload["total_bytes"], "gt_read": False,
               "source_list_sha256": args.original_list_sha256,
               "source_manifest_sha256": sha256(out / "SOURCE_IMAGE_MANIFEST.json"),
               "source_script_sha256": sha256(Path(__file__).resolve())}
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps(summary), flush=True)


def verify(args) -> None:
    # Pillow decodes only RGB JPEGs. Native baseline image JSONs contain predicted
    # detections; this preparation reads only their image-level identity/shape.
    from PIL import Image
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_run).resolve(strict=True)
    root = Path(args.images_root).resolve(strict=True)
    native_root = Path(args.native_run).resolve(strict=True)
    original_list = source_dir / "SOURCE_IMAGE_LIST.txt"
    _, entries = list_entries(original_list, args.original_list_sha256)
    source_manifest_path = source_dir / "SOURCE_IMAGE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records = source_manifest["images"]
    source_ids = [int(p.stem) for p in entries]
    if source_ids != [r["image_id"] for r in records] or len(records) != 5000:
        raise ValueError("Source manifest order/count differs from original list")
    if source_manifest["source_images_list_sha256"] != args.original_list_sha256:
        raise ValueError("Source manifest list binding differs")
    source_run_path = source_dir / "run.json"
    source_run = json.loads(source_run_path.read_text(encoding="utf-8"))
    if source_run["status"] != "completed" or source_run["return_code"] != 0:
        raise ValueError("Source metadata Run did not complete")
    complete_path = native_root / "baseline" / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    fingerprint = complete["fingerprint"]
    if fingerprint["image_ids"] != source_ids or fingerprint["images_list_sha256"] != args.original_list_sha256:
        raise ValueError("Native baseline image order/original-list binding differs")
    parity_path = native_root / "BASELINE_PARITY_IMAGES.jsonl"
    parity_rows = [json.loads(line) for line in parity_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [r["image_id"] for r in parity_rows] != source_ids:
        raise ValueError("Native parity rows differ from original list order")
    parity = {r["image_id"]: r for r in parity_rows}
    image_meta = []
    cache_manifest_lines = []
    image_list_lines = []
    for ordinal, record in enumerate(records):
        image_id = record["image_id"]
        p = (root / record["filename"]).resolve(strict=True)
        if p.parent != root or not p.is_file():
            raise ValueError(f"Local image outside declared cache: {p}")
        digest = sha256(p)
        native_row = parity[image_id]
        if digest != record["sha256"] or digest != native_row["image_sha256"]:
            raise ValueError(f"Source/local/native JPEG SHA mismatch for {image_id}")
        if p.stat().st_size != record["size_bytes"]:
            raise ValueError(f"JPEG size differs for {image_id}")
        image_json_path = native_root / "baseline" / "images" / f"{image_id:012d}.json"
        image_json = json.loads(image_json_path.read_text(encoding="utf-8"))
        if image_json["image_id"] != image_id or image_json["source_baseline_identity_sha256"] != native_row["native_identity_sha256"]:
            raise ValueError(f"Native baseline image identity binding differs for {image_id}")
        if Path(image_json["image_path"]).name != record["filename"]:
            raise ValueError(f"Native baseline original JPEG filename differs for {image_id}")
        with Image.open(p) as image:
            if image.format != "JPEG":
                raise ValueError(f"Expected JPEG decoder for {p}")
            width, height = image.size
            original_mode = image.mode
            rgb = image.convert("RGB")
            rgb.load()
            if rgb.mode != "RGB" or rgb.size != (width, height):
                raise ValueError(f"RGB decode failed for {image_id}")
        if [height, width] != image_json["original_shape"]:
            raise ValueError(f"Decoded dimensions differ from native baseline for {image_id}")
        local_path = p.as_posix()
        image_list_lines.append(local_path)
        cache_manifest_lines.append(f"{digest}  {local_path}")
        image_meta.append({"ordinal": ordinal, "image_id": image_id, "image_path": local_path,
                           "filename": p.name, "width": width, "height": height,
                           "original_shape": [height, width], "decoded_mode": "RGB",
                           "jpeg_mode": original_mode, "size_bytes": record["size_bytes"],
                           "image_sha256": digest, "source_image_path": record["source_image_path"],
                           "native_input_sha256": native_row["input_sha256"],
                           "native_source_image_cache_sha256": native_row["source_image_cache_sha256"],
                           "native_identity_sha256": native_row["native_identity_sha256"],
                           "baseline_image_json_sha256": sha256(image_json_path)})
        if (ordinal + 1) % 500 == 0:
            print(f"local SHA + source/native binding + RGB decoded {ordinal + 1}/5000", flush=True)
    write_bytes(out / "IMAGE_LIST.txt", ("\n".join(image_list_lines) + "\n").encode("utf-8"))
    write_bytes(out / "CACHE_MANIFEST.sha256", ("\n".join(cache_manifest_lines) + "\n").encode("utf-8"))
    local_list_sha = sha256(out / "IMAGE_LIST.txt")
    bindings = {"source_run_id": source_run["run_id"], "source_host": source_run["host"],
                "source_run_local_copy": str(source_run_path), "source_run_json_sha256": sha256(source_run_path),
                "source_manifest_local_copy": str(source_manifest_path),
                "source_manifest_sha256": sha256(source_manifest_path),
                "source_images_list": source_manifest["source_images_list"],
                "original_images_list_sha256": args.original_list_sha256,
                "local_images_list_sha256": local_list_sha,
                "native_run_id": native_root.name, "native_run_path": str(native_root),
                "native_baseline_complete_sha256": sha256(complete_path),
                "native_parity_images_sha256": sha256(parity_path),
                "native_predictions_sha256": complete["predictions_sha256"],
                "native_frozen_state_sha256": complete["frozen_state_sha256"]}
    write_json(out / "IMAGE_META.json", {"schema_version": 1, "gt_read": False,
               "image_count": 5000, "images_root": root.as_posix(), "bindings": bindings,
               "images": image_meta})
    write_json(out / "SOURCE_BINDING.json", bindings)
    summary = {"status": "complete", "passed": True, "image_count": 5000,
               "total_bytes": sum(r["size_bytes"] for r in records),
               "local_images_root": root.as_posix(), "images_reused": 5000,
               "images_copied": 0, "source_local_native_sha_exact_count": 5000,
               "rgb_decode_dimensions_match_count": 5000, "gt_read": False,
               "cache_modified": False, "baseline_modified": False,
               "original_list_sha256": args.original_list_sha256,
               "local_list_sha256": local_list_sha,
               "image_meta_sha256": sha256(out / "IMAGE_META.json"),
               "cache_manifest_sha256": sha256(out / "CACHE_MANIFEST.sha256"),
               "source_script_sha256": sha256(Path(__file__).resolve()),
               "host": platform.node(), "python": sys.executable,
               "decode_library": f"Pillow {Image.__version__}"}
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps(summary), flush=True)


def seal(args) -> None:
    """After runner completion, freeze run files in a relative SHA manifest."""
    out = Path(args.output).resolve(strict=True)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    if run["status"] not in ("completed", "failed", "cancelled"):
        raise ValueError("Seal only a terminal recorder state")
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("manifest.sha256", "transfer.json", ".run.claim"):
            lines.append(f"{sha256(p)}  {p.relative_to(out).as_posix()}")
    write_bytes(out / "manifest.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    print(json.dumps({"sealed_run": run["run_id"], "file_count": len(lines),
                      "manifest_sha256": sha256(out / "manifest.sha256")}))


def receive(args) -> None:
    """Verify a verbatim SCP-returned metadata Run before creating its receipt."""
    out = Path(args.output).resolve(strict=True)
    manifest = out / "manifest.sha256"
    verified = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        p = (out / relative).resolve(strict=True)
        p.relative_to(out)  # Every returned path must stay within this Run.
        actual = sha256(p)
        if actual != expected:
            raise ValueError(f"Returned source file SHA mismatch: {relative}")
        verified.append({"path": relative, "size_bytes": p.stat().st_size, "sha256": actual})
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    receipt = {"schema_version": 1, "transferred_at": now(), "status": "complete",
               "method": "scp_original_metadata_run_no_images", "ssh_host_alias": args.ssh_host,
               "source_host": run["host"], "source_run_path": args.remote_run_path,
               "destination_host": platform.node(), "destination_run_path": str(out),
               "source_run_json_preserved_verbatim": True, "manifest_verified": True,
               "manifest_sha256": sha256(manifest), "returned_file_count": len(verified),
               "returned_bytes": sum(r["size_bytes"] for r in verified),
               "images_transferred": 0, "files": verified}
    write_json(out / "transfer.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["source", "verify", "seal", "receive"])
    p.add_argument("--output", required=True)
    p.add_argument("--images-root")
    p.add_argument("--images-list")
    p.add_argument("--source-run")
    p.add_argument("--native-run")
    p.add_argument("--ssh-host")
    p.add_argument("--remote-run-path")
    p.add_argument("--original-list-sha256", default="b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"source": source, "verify": verify, "seal": seal, "receive": receive}[args.mode](args)
