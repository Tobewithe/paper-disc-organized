"""Freeze the declared image panel; verify bytes and decode, not model outcomes."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

POOL_SHA = "ef3453e65b80a2d07997c882b0624a88f6d8161fcd32e2d629cdff3285ba3c63"
SPLIT_SHA = "1b75dfea2ca011dc43aaa8a7b863900c3bae913472b849f76934361366d9d8ce"
ANNOTATION_SHA = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
WEIGHT_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def canonical_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ("run", "pool", "calibration-split", "annotations", "image-root", "weights", "vendor"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    run = Path(args.run).resolve()
    if not (run / "run.json").is_file():
        raise RuntimeError("An actual runner record is required")
    if (run / "PANEL.json").exists():
        raise RuntimeError("Panel exists; do not overwrite a historical Run")
    paths = {k: Path(getattr(args, k)).resolve()
             for k in ("pool", "calibration_split", "annotations", "weights")}
    expected = {"pool": POOL_SHA, "calibration_split": SPLIT_SHA,
                "annotations": ANNOTATION_SHA, "weights": WEIGHT_SHA}
    before = {k: sha(path) for k, path in paths.items()}
    if before != expected:
        raise RuntimeError("Original input hash mismatch")
    source_ids = json.loads(paths["pool"].read_text(encoding="utf-8"))
    split = json.loads(paths["calibration_split"].read_text(encoding="utf-8"))
    fit, select = set(split["fit_ids"]), set(split["selection_ids"])
    if len(source_ids) != 20000 or len(set(source_ids)) != 20000:
        raise RuntimeError("Declared source pool is not 20,000 unique IDs")
    if len(fit) != 1500 or len(select) != 500 or fit & select:
        raise RuntimeError("Invalid original calibration split")
    excluded = sorted(fit | select)
    eligible = sorted(set(source_ids) - set(excluded))
    if len(eligible) != 19638:
        raise RuntimeError("The fixed eligible pool changed")
    selected = sorted(map(int, np.random.default_rng(20261009).choice(eligible, 512, replace=False)))
    annotation = json.loads(paths["annotations"].read_text(encoding="utf-8"))
    image_meta = {x["id"]: x for x in annotation["images"]}
    selected_set = set(selected)
    class_counts = {}
    gt_count = crowd_count = 0
    for item in annotation["annotations"]:
        if item["image_id"] in selected_set:
            gt_count += 1
            crowd_count += int(bool(item.get("iscrowd", 0)))
            key = str(item["category_id"])
            class_counts[key] = class_counts.get(key, 0) + 1
    del annotation
    images = []
    image_root = Path(args.image_root).resolve()
    for index, image_id in enumerate(selected):
        metadata = image_meta[image_id]
        path = image_root / metadata["file_name"]
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Selected image missing/empty; no resampling: {path}")
        digest = sha(path)
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is None or tuple(pixels.shape[:2]) != (metadata["height"], metadata["width"]):
            raise RuntimeError(f"Image decode/dimensions failed; no resampling: {path}")
        images.append({"image_id": image_id, "path": str(path), "sha256": digest,
                       "width": metadata["width"], "height": metadata["height"]})
        del pixels
        if (index + 1) % 128 == 0:
            print(json.dumps({"decoded_images": index + 1}), flush=True)
    eligible_path = run / "ELIGIBLE_IMAGE_IDS.json"
    exclusion_path = run / "CALIBRATION_EXCLUSION_IDS.json"
    dump(eligible_path, eligible)
    dump(exclusion_path, excluded)
    panel = {
        "schema_version": "local_edit_opportunity_panel_v1",
        "weights": {"path": str(paths["weights"]), "sha256": WEIGHT_SHA},
        "vendor": {"path": str(Path(args.vendor).resolve()), "version": "8.4.100"},
        "interpreter": sys.executable, "device": "cuda:0", "split": "train2017",
        "images": images, "engineering_image_ids": selected[:4],
        "annotations": {"path": str(paths["annotations"]), "sha256": ANNOTATION_SHA},
        "selection": {"kind": "uniform_without_replacement", "seed": 20261009,
                      "image_count": 512, "population_count": 19638,
                      "population_sha256": canonical_sha(eligible),
                      "algorithm": "numpy.default_rng(seed).choice(sorted_eligible_ids,512,replace=False); sort selected IDs"},
        "population": {"path": str(eligible_path), "sha256": sha(eligible_path),
                       "source": {"path": str(paths["pool"]), "sha256": POOL_SHA}},
        "calibration_exclusion": {"path": str(exclusion_path), "sha256": sha(exclusion_path),
                                  "image_ids": excluded,
                                  "source": {"path": str(paths["calibration_split"]), "sha256": SPLIT_SHA}},
        "use_boundary": "Route selection; not independent calibration, AP, training or novel-method confirmation",
    }
    after = {k: sha(path) for k, path in paths.items()}
    if after != before:
        raise RuntimeError("Original input bytes changed")
    dump(run / "PANEL.json", panel)
    dump(run / "PREPARATION_SUMMARY.json", {
        "status": "prepared", "passed": True, "source_pool_count": 20000,
        "excluded_known_calibration_count": 2000, "fit_intersection": len(set(source_ids) & fit),
        "selection_intersection": len(set(source_ids) & select), "eligible_pool_count": 19638,
        "selected_images": 512, "decoded_images": 512, "missing_or_corrupt_images": 0,
        "engineering_images": 4, "panel_sha256": sha(run / "PANEL.json"),
        "selected_id_identity_sha256": canonical_sha(selected),
        "eligible_id_identity_sha256": canonical_sha(eligible),
        "source_hashes_before": before, "source_hashes_after": after, "source_unchanged": True,
        "selected_annotation_count": gt_count, "selected_crowd_count": crowd_count,
        "selected_class_counts": class_counts, "model_executed": False,
        "opportunity": None, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {"python": sys.executable, "numpy": np.__version__, "opencv": cv2.__version__},
    })
    print(json.dumps({"status": "prepared", "panel_sha256": sha(run / "PANEL.json")}), flush=True)


if __name__ == "__main__":
    main()
