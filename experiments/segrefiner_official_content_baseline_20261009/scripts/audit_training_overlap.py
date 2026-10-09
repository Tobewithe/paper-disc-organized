#!/usr/bin/env python3
"""Audit SegRefiner's declared LVIS-v1 training images against COCO val2017.

This deliberately streams only the top-level ``images`` arrays.  It never
loads, decodes, or traverses LVIS instance annotations/polygons.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator


CHUNK_CHARS = 1 << 20
CHUNK_BYTES = 1 << 20
EXPECTED_LVIS_ENTRY = "lvis_v1_train.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids: list[int]) -> str:
    payload = "".join(f"{image_id}\n" for image_id in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def ensure_under(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing path outside study: {resolved_candidate}") from exc
    return resolved_candidate


def extract_lvis_json(zip_path: Path, target: Path) -> dict:
    """Extract the one expected JSON member without writing any other ZIP path."""
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if names != [EXPECTED_LVIS_ENTRY]:
            raise ValueError(f"Unexpected ZIP entries: {names}")
        entry = entries[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != entry.file_size:
                raise ValueError(
                    f"Existing extraction has {target.stat().st_size} bytes; "
                    f"archive declares {entry.file_size}. Refusing overwrite."
                )
            extraction = "reused_existing_exact_size"
        else:
            with archive.open(entry, "r") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=CHUNK_BYTES)
            if target.stat().st_size != entry.file_size:
                raise ValueError("Extraction byte count differs from ZIP central directory")
            extraction = "extracted"
    return {
        "status": extraction,
        "zip_entry": entry.filename,
        "zip_declared_uncompressed_bytes": entry.file_size,
        "json_bytes": target.stat().st_size,
    }


def iter_images(path: Path) -> Iterator[dict]:
    """Incrementally decode top-level ``images`` entries with only stdlib JSON."""
    decoder = json.JSONDecoder()
    marker = '"images"'
    buffer = ""
    found = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if not found:
                index = buffer.find(marker)
                if index >= 0:
                    cursor = index + len(marker)
                    while True:
                        colon = buffer.find(":", cursor)
                        if colon >= 0:
                            opening = buffer.find("[", colon + 1)
                            if opening >= 0:
                                buffer = buffer[opening + 1 :]
                                found = True
                                break
                        extra = handle.read(CHUNK_CHARS)
                        if not extra:
                            raise ValueError(f"Could not find start of top-level images array in {path}")
                        buffer += extra
                    continue
                extra = handle.read(CHUNK_CHARS)
                if not extra:
                    raise ValueError(f"Could not find top-level images key in {path}")
                # Retain a small marker-sized suffix to catch a chunk boundary.
                buffer = (buffer + extra)[-CHUNK_CHARS:]
                continue

            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                break
            if buffer.startswith("]"):
                return
            if not buffer:
                extra = handle.read(CHUNK_CHARS)
                if not extra:
                    raise ValueError(f"Unexpected EOF inside images array in {path}")
                buffer = extra
                continue
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                extra = handle.read(CHUNK_CHARS)
                if not extra:
                    raise ValueError(f"Malformed or truncated images record in {path}")
                buffer += extra
                continue
            if not isinstance(item, dict):
                raise ValueError(f"images item is not an object in {path}")
            yield item
            # Do not retain already-decoded image records in the input buffer.
            buffer = buffer[end:]


def collect_images(path: Path, label: str) -> tuple[Dict[int, dict], list[int], dict]:
    records: Dict[int, dict] = {}
    duplicates: list[int] = []
    metadata = Counter()
    for item in iter_images(path):
        image_id = item.get("id")
        if not isinstance(image_id, int):
            raise ValueError(f"{label}: image record without integer id")
        if image_id in records:
            duplicates.append(image_id)
            continue
        record = {
            "id": image_id,
            "file_name": item.get("file_name"),
            "coco_url": item.get("coco_url"),
        }
        records[image_id] = record
        metadata["records"] += 1
        file_name, coco_url = record["file_name"], record["coco_url"]
        if not isinstance(file_name, str):
            metadata["missing_file_name"] += 1
        if not isinstance(coco_url, str):
            metadata["missing_coco_url"] += 1
        if isinstance(file_name, str) and isinstance(coco_url, str) and coco_url.endswith("/" + file_name):
            metadata["coco_url_ends_with_file_name"] += 1
        if isinstance(coco_url, str) and "/train2017/" in coco_url:
            metadata["coco_url_has_train2017_segment"] += 1
        if isinstance(coco_url, str) and "/val2017/" in coco_url:
            metadata["coco_url_has_val2017_segment"] += 1
        if isinstance(file_name, str):
            try:
                if int(Path(file_name).stem) == image_id:
                    metadata["numeric_file_stem_equals_id"] += 1
            except ValueError:
                metadata["non_numeric_file_stem"] += 1
    if duplicates:
        raise ValueError(f"{label}: duplicate image IDs: {duplicates[:10]}")
    return records, sorted(records), dict(sorted(metadata.items()))


def write_id_list(path: Path, ids: list[int]) -> str:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for image_id in ids:
            handle.write(f"{image_id}\n")
    return sha256_file(path)


def write_overlap_records(path: Path, ids: list[int], lvis: Dict[int, dict], coco: Dict[int, dict]) -> tuple[str, dict]:
    counters = {
        "records": len(ids),
        "file_name_exact_equal": 0,
        "coco_url_exact_equal": 0,
        "lvis_url_has_val2017_segment": 0,
        "lvis_url_ends_with_coco_filename": 0,
        "missing_lvis_file_name": 0,
        "missing_coco_file_name": 0,
        "missing_lvis_coco_url": 0,
        "missing_coco_coco_url": 0,
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for image_id in ids:
            left, right = lvis[image_id], coco[image_id]
            left_name, right_name = left["file_name"], right["file_name"]
            left_url, right_url = left["coco_url"], right["coco_url"]
            if left_name is None:
                counters["missing_lvis_file_name"] += 1
            if right_name is None:
                counters["missing_coco_file_name"] += 1
            if left_url is None:
                counters["missing_lvis_coco_url"] += 1
            if right_url is None:
                counters["missing_coco_coco_url"] += 1
            if left_name == right_name and left_name is not None:
                counters["file_name_exact_equal"] += 1
            if left_url == right_url and left_url is not None:
                counters["coco_url_exact_equal"] += 1
            if isinstance(left_url, str) and "/val2017/" in left_url:
                counters["lvis_url_has_val2017_segment"] += 1
            if isinstance(left_url, str) and isinstance(right_name, str) and left_url.endswith("/" + right_name):
                counters["lvis_url_ends_with_coco_filename"] += 1
            row = {
                "image_id": image_id,
                "lvis": left,
                "coco_val2017": right,
                "file_name_exact_equal": left_name == right_name and left_name is not None,
                "coco_url_exact_equal": left_url == right_url and left_url is not None,
                "lvis_url_has_val2017_segment": isinstance(left_url, str) and "/val2017/" in left_url,
                "lvis_url_ends_with_coco_filename": isinstance(left_url, str) and isinstance(right_name, str) and left_url.endswith("/" + right_name),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(path), counters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lvis-zip", type=Path, required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    run_dir = ensure_under(study_root, args.run_dir)
    lvis_zip = ensure_under(study_root, args.lvis_zip)
    if not lvis_zip.is_file() or not args.coco_json.is_file():
        raise FileNotFoundError("LVIS ZIP or COCO JSON is unavailable")
    run_dir.mkdir(parents=True, exist_ok=False)
    extracted_json = ensure_under(study_root, study_root / "assets/training_provenance/extracted/lvis_v1_train.json")
    extraction = extract_lvis_json(lvis_zip, extracted_json)

    lvis_records, lvis_ids, lvis_metadata = collect_images(extracted_json, "LVIS v1 train")
    coco_records, coco_ids, coco_metadata = collect_images(args.coco_json, "COCO val2017")
    overlap_ids = sorted(set(lvis_ids).intersection(coco_ids))

    lvis_ids_path = run_dir / "lvis_train_image_ids.txt"
    coco_ids_path = run_dir / "coco_val2017_image_ids.txt"
    overlap_ids_path = run_dir / "overlap_image_ids.txt"
    overlap_records_path = run_dir / "overlap_url_audit.jsonl"
    lvis_ids_hash = write_id_list(lvis_ids_path, lvis_ids)
    coco_ids_hash = write_id_list(coco_ids_path, coco_ids)
    overlap_ids_hash = write_id_list(overlap_ids_path, overlap_ids)
    overlap_records_hash, url_checks = write_overlap_records(
        overlap_records_path, overlap_ids, lvis_records, coco_records
    )

    result = {
        "audit_version": "segrefiner_lvis_coco_image_overlap_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "scope": "streamed top-level images arrays only; no LVIS annotations/polygons decoded",
        "inputs": {
            "lvis_zip": str(lvis_zip),
            "lvis_zip_sha256": sha256_file(lvis_zip),
            "lvis_extracted_json": str(extracted_json),
            "lvis_extracted_json_sha256": sha256_file(extracted_json),
            "coco_val_json": str(args.coco_json.resolve()),
            "coco_val_json_sha256": sha256_file(args.coco_json),
        },
        "extraction": extraction,
        "counts": {
            "lvis_distinct_image_ids": len(lvis_ids),
            "coco_val2017_distinct_image_ids": len(coco_ids),
            "intersection_distinct_image_ids": len(overlap_ids),
        },
        "id_list_sha256": {
            "lvis_train_sorted": lvis_ids_hash,
            "coco_val2017_sorted": coco_ids_hash,
            "intersection_sorted": overlap_ids_hash,
        },
        "image_record_metadata_checks": {
            "lvis_v1_train": lvis_metadata,
            "coco_val2017": coco_metadata,
        },
        "overlap_url_checks": url_checks,
        "outputs": {
            "lvis_train_image_ids": lvis_ids_path.name,
            "coco_val2017_image_ids": coco_ids_path.name,
            "overlap_image_ids": overlap_ids_path.name,
            "overlap_url_audit": overlap_records_path.name,
            "overlap_url_audit_sha256": overlap_records_hash,
        },
        "interpretation": {
            "training_data_evidence": "The locked released config declares LVIS v1 train; this audit establishes image-ID overlap of the downloaded LVIS annotation, not actual checkpoint training history.",
            "checkpoint_manifest_status": "not available in this audit",
            "evaluation_scope_decision": "reserved for B/root",
        },
    }
    result_path = run_dir / "audit_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "run_id": run_dir.name,
        "study_id": "STUDY_SEGREFINER_OFFICIAL_CONTENT_BASELINE_20261009",
        "kind": "data_provenance_audit",
        "status": "completed",
        "started_and_completed_by": "scripts/audit_training_overlap.py",
        "result": "audit_results.json",
        "no_gpu": True,
        "no_lvis_annotations_or_polygons_decoded": True,
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
