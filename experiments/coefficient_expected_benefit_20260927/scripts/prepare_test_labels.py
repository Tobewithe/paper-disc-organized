"""After gate selection, convert sealed 2k COCO val labels with official converter."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class


def main(a):
    manifest = json.loads(a.manifest.read_text())
    chosen = set(map(int, manifest["independent_test_images"]))
    assert len(chosen) == 2000
    assert (a.gate / "COMPLETE.json").is_file(), "Gate must be frozen before opening test GT"
    gate = json.loads((a.gate / "COMPLETE.json").read_text())
    assert gate["test_labels_read"] is False and gate["selected"]
    source = json.loads(a.annotations.read_text())
    source["images"] = [r for r in source["images"] if int(r["id"]) in chosen]
    source["annotations"] = [r for r in source["annotations"] if int(r["image_id"]) in chosen]
    assert len(source["images"]) == 2000
    images = {int(r["id"]): r for r in source["images"]}
    by_image = defaultdict(list)
    for ann in source["annotations"]:
        by_image[int(ann["image_id"])].append(ann)
    classmap = coco91_to_coco80_class()
    identities, exclusions = {}, Counter()
    for iid in chosen:
        kept, seen = [], []
        for ann in by_image[iid]:
            if ann.get("iscrowd", False):
                exclusions["crowd"] += 1
                continue
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2
            box[[0, 2]] /= images[iid]["width"]
            box[[1, 3]] /= images[iid]["height"]
            if box[2] <= 0 or box[3] <= 0:
                exclusions["invalid_box"] += 1
                continue
            key = [classmap[ann["category_id"] - 1], *box.tolist()]
            if key in seen:
                exclusions["duplicate_box"] += 1
                continue
            assert isinstance(ann.get("segmentation"), list) and ann["segmentation"], ann["id"]
            seen.append(key)
            kept.append(int(ann["id"]))
        identities[iid] = kept
    a.out.mkdir(parents=True, exist_ok=True)
    source_dir = a.out / "conversion_input"
    source_dir.mkdir()
    (source_dir / "instances_val2017.json").write_text(json.dumps(source), encoding="utf-8")
    converted = a.out / "official_data"
    convert_coco(str(source_dir), str(converted), use_segments=True)
    image_link = converted / "images" / "val2017"
    image_link.parent.mkdir(parents=True, exist_ok=True)
    image_link.symlink_to(a.images, target_is_directory=True)
    file_list = [image_link / images[i]["file_name"] for i in manifest["independent_test_images"]]
    assert all(p.is_file() for p in file_list)
    (converted / "test.txt").write_text("\n".join(map(str, file_list)) + "\n", encoding="utf-8")
    mapping = {}
    for iid in chosen:
        path = converted / "labels" / "val2017" / f"{iid:012d}.txt"
        lines = path.read_text().strip().splitlines() if path.exists() else []
        assert len(lines) == len(identities[iid]), (iid, len(lines), len(identities[iid]))
        mapping[str(iid)] = identities[iid]
    (a.out / "IDENTITIES.json").write_text(json.dumps(mapping), encoding="utf-8")
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(images=2000,
        annotations=len(source["annotations"]), exclusions=exclusions,
        gate_model=gate["selected"], conversion="official convert_coco(use_segments=True)"),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("manifest", "annotations", "images", "gate", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
