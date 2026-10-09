"""Make the declared 20k image tar from the user's existing read-only COCO ZIP.

Selection uses original image IDs only. Original GT JSON is authenticated and
retained whole; no filtered annotation dataset or new download is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

VERSION = "triflow_existing_local_train20k_preparation_v1"
TRAIN_SHA = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
VAL_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
SELECTOR_PREFIX = "triflow_train20k_seed0:"
SELECTOR_NAME = "sha256_triflow_train20k_seed0_v1"
SELECTED_COUNT = 20000
ORIGINAL_COUNT = 118287
ORIGINAL_ARCHIVE_BYTES = 19336861798


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def select_declared_train_ids(image_ids):
    """The provider's exact source-ID-only selection, without importing Torch."""
    ids = [int(iid) for iid in image_ids]
    if len(ids) != ORIGINAL_COUNT or len(set(ids)) != ORIGINAL_COUNT:
        raise ValueError("Declared selection requires all 118287 original unique train image IDs")
    ranked = sorted(ids, key=lambda iid: (hashlib.sha256((SELECTOR_PREFIX + str(iid)).encode("utf-8")).digest(), iid))
    return sorted(ranked[:SELECTED_COUNT])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "archive", "annotations", "val-annotations"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = (root / "runs" / args.run_id).resolve()
    if run.parent != (root / "runs").resolve():
        raise ValueError("Run ID must name one directory immediately under this study's runs")
    run.mkdir(parents=True, exist_ok=True)
    owned_names = ("train20k_images.tar", "train20k_images.tar.partial", "SELECTED_IMAGES.sha256",
                   "SELECTED_IMAGE_IDS.json", "SELECTED_MEMBERS.json", "LOCAL20K_RECEIPT.json",
                   "LOCAL20K_FAILURE.json")
    if any((run / name).exists() for name in owned_names):
        raise FileExistsError("Existing preparation artifacts are retained; retries require a new Run ID")
    archive, annotations, val_annotations = args.archive.resolve(), args.annotations.resolve(), args.val_annotations.resolve()
    if any(path.is_relative_to(run) for path in (archive, annotations, val_annotations)):
        raise ValueError("Original existing inputs must remain outside this output Run")
    started, began = now(), time.monotonic()
    try:
        from PIL import Image
        print("LOCAL20K authenticating original JSONs and existing source ZIP", flush=True)
        train_sha, val_sha = sha(annotations), sha(val_annotations)
        if train_sha != TRAIN_SHA or val_sha != VAL_SHA:
            raise ValueError("Original train/val annotation bytes differ from locked COCO identities")
        with annotations.open("r", encoding="utf-8") as handle:
            train = json.load(handle)
        original_annotation_count = len(train["annotations"])
        images = train["images"]
        ids = [int(image["id"]) for image in images]
        by_id = {int(image["id"]): image for image in images}
        names = [str(image["file_name"]) for image in images]
        if len(ids) != ORIGINAL_COUNT or len(by_id) != ORIGINAL_COUNT or len(set(names)) != ORIGINAL_COUNT:
            raise ValueError("Original train image count or image/filename uniqueness differs")
        # Preserve original JSON identities; selection deliberately ignores GT.
        del train
        with val_annotations.open("r", encoding="utf-8") as handle:
            val = json.load(handle)
        val_ids = [int(image["id"]) for image in val["images"]]
        if len(val_ids) != 5000 or len(set(val_ids)) != 5000 or set(ids) & set(val_ids):
            raise ValueError("Original val split count/uniqueness differs or train-val image IDs overlap")
        del val
        selected = select_declared_train_ids(ids)
        if len(selected) != SELECTED_COUNT:
            raise ValueError("Declared selection is not exactly 20000 images")
        selector = {"selector_name": SELECTOR_NAME, "selection_seed": 0,
                    "rank_prefix": SELECTOR_PREFIX, "selection_size": SELECTED_COUNT,
                    "algorithm": "sort all original IDs by sha256('triflow_train20k_seed0:'+str(image_id)), take20000, then image_id ascending",
                    "universe_images": ORIGINAL_COUNT, "selected_images": SELECTED_COUNT,
                    "universe_image_identity_sha256": identity_sha(sorted(ids)),
                    "selected_image_identity_sha256": identity_sha(selected),
                    "uses_gt_categories_masks_or_image_contents": False}
        selector_sha = hashlib.sha256(json.dumps(selector, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        archive_stat = archive.stat()
        if archive_stat.st_size != ORIGINAL_ARCHIVE_BYTES:
            raise ValueError("Existing full train2017 ZIP size differs from the confirmed original archive")
        archive_sha = sha(archive)
        dump(run / "SELECTED_IMAGE_IDS.json", selected)
        partial_tar = run / "train20k_images.tar.partial"
        members = []
        selected_bytes = 0
        with zipfile.ZipFile(archive, "r") as zipped:
            entries = [entry for entry in zipped.infolist() if not entry.is_dir()]
            entry_names = [entry.filename for entry in entries]
            expected_names = {"train2017/" + name for name in names}
            if len(entries) != ORIGINAL_COUNT or len(set(entry_names)) != ORIGINAL_COUNT or set(entry_names) != expected_names:
                raise ValueError("ZIP member structure does not equal all original train2017 filenames")
            by_member = {entry.filename: entry for entry in entries}
            with tarfile.open(partial_tar, "w", format=tarfile.PAX_FORMAT) as tarred, (run / "SELECTED_IMAGES.sha256").open("x", encoding="utf-8") as manifest:
                for position, iid in enumerate(selected, 1):
                    info = by_id[iid]
                    filename = str(info["file_name"])
                    member_name = "train2017/" + filename
                    if Path(filename).name != filename or not filename.endswith(".jpg"):
                        raise ValueError("Unexpected original image filename")
                    member = by_member[member_name]
                    # zip.read consumes this member to EOF and validates its CRC.
                    content = zipped.read(member)
                    if len(content) != member.file_size:
                        raise ValueError("Selected JPEG decompressed byte count differs: " + member_name)
                    with Image.open(io.BytesIO(content)) as image:
                        observed_size = image.size
                        image_format = image.format
                    expected_size = (int(info["width"]), int(info["height"]))
                    if image_format != "JPEG" or observed_size != expected_size:
                        raise ValueError(f"Selected image JPEG header/GT dimensions differ: {iid}; {image_format}/{observed_size}/{expected_size}")
                    image_sha = hashlib.sha256(content).hexdigest()
                    target_info = tarfile.TarInfo(member_name)
                    target_info.size, target_info.mode, target_info.mtime = len(content), 0o644, 0
                    target_info.uid, target_info.gid, target_info.uname, target_info.gname = 0, 0, "", ""
                    tarred.addfile(target_info, io.BytesIO(content))
                    manifest.write(image_sha + "  " + filename + "\n")
                    members.append({"image_id": iid, "filename": filename, "member_name": member_name,
                                    "bytes": len(content), "sha256": image_sha, "width": observed_size[0],
                                    "height": observed_size[1], "zip_crc32": f"{member.CRC:08x}",
                                    "zip_crc_passed": True, "jpeg_header_dimensions_match_gt": True})
                    selected_bytes += len(content)
                    if position == 1 or position % 2000 == 0:
                        print(f"LOCAL20K bundled={position}/{SELECTED_COUNT} image_bytes={selected_bytes} elapsed_s={time.monotonic()-began:.1f}", flush=True)
        latest_archive_stat = archive.stat()
        if latest_archive_stat.st_size != archive_stat.st_size or latest_archive_stat.st_mtime_ns != archive_stat.st_mtime_ns:
            raise RuntimeError("Original archive changed while preparing selected members")
        if len(members) != SELECTED_COUNT or [record["image_id"] for record in members] != selected:
            raise RuntimeError("Selected member coverage does not equal the declared sorted IDs")
        final_tar = run / "train20k_images.tar"
        partial_tar.replace(final_tar)
        dump(run / "SELECTED_MEMBERS.json", members)
        receipt = {"version": VERSION, "status": "completed", "passed": True, "started_at": started,
                   "completed_at": now(), "date": datetime.now(timezone.utc).date().isoformat(),
                   "acquisition": "existing_local_dataset", "archive": str(archive), "source_archive": str(archive),
                   "archive_sha256": archive_sha, "archive_bytes": archive_stat.st_size, "published_sha256": None,
                   "original_train_images": ORIGINAL_COUNT, "original_annotations": original_annotation_count,
                   "annotations": str(annotations), "annotation_sha256": train_sha,
                   "train_annotations": str(annotations), "train_annotation_sha256": train_sha,
                   "val_annotations": str(val_annotations), "val_annotation_sha256": val_sha,
                   "val_images": 5000, "train_val_overlap": [], "train_val_overlap_count": 0,
                   "image_count": SELECTED_COUNT, "selected_image_count": SELECTED_COUNT,
                   "selected_image_fraction_of_original_train": SELECTED_COUNT / ORIGINAL_COUNT,
                   "selected_original_gt_fraction": None, "selected_class_coverage": None,
                   "selector_prefix": SELECTOR_PREFIX, "subset_selector": selector, "subset_selector_sha256": selector_sha,
                   "selected_image_identity_sha256": identity_sha(selected), "declared_subset_identity_sha256": identity_sha(selected),
                   "selected_image_ids": str(run / "SELECTED_IMAGE_IDS.json"), "selected_image_ids_file_sha256": sha(run / "SELECTED_IMAGE_IDS.json"),
                   "selected_members": str(run / "SELECTED_MEMBERS.json"), "selected_members_sha256": sha(run / "SELECTED_MEMBERS.json"),
                   "selected_image_manifest": str(run / "SELECTED_IMAGES.sha256"), "selected_image_manifest_sha256": sha(run / "SELECTED_IMAGES.sha256"),
                   "source_zip_all_member_names_match_original_split": True,
                   "selected_members_crc_all_passed": True, "all_selected_members_crc_passed": True,
                   "selected_jpeg_header_dimensions_all_match_gt": True, "archive_crc_all_entries_passed": None,
                   "crc_scope": "only the selected 20000 source ZIP members; unselected member CRC not checked",
                   "tar": str(final_tar), "tar_sha256": sha(final_tar), "tar_bytes": final_tar.stat().st_size,
                   "selected_image_bytes": selected_bytes, "output_directory": str(run),
                   "original_dataset_modified": False, "original_annotation_json_filtered": False,
                   "interpreter": sys.executable, "source_sha256": sha(Path(__file__)),
                   "elapsed_seconds": time.monotonic() - began}
        dump(run / "LOCAL20K_RECEIPT.json", receipt)
        print(json.dumps({"passed": True, "images": SELECTED_COUNT, "tar": str(final_tar),
                          "tar_sha256": receipt["tar_sha256"], "receipt": str(run / "LOCAL20K_RECEIPT.json")}), flush=True)
    except BaseException as exc:
        dump(run / "LOCAL20K_FAILURE.json", {"version": VERSION, "passed": False, "started_at": started,
                                             "failed_at": now(), "error": repr(exc), "traceback": traceback.format_exc(),
                                             "partial_outputs_retained": True})
        raise


if __name__ == "__main__":
    main()
