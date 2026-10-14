"""Prepare isolated native validation labels/lists from the official label archive.

No network, inference, or changes to the original COCO dataset. Existing output is refused.
"""
import json
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, require, sha256, write_new
from native_output_bridge import GT_SHA


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "assets/datasets/coco")
    original = parser.parse_args().original
    out = ROOT / "assets/datasets/coco_native_s0_20261013"
    archive = ROOT / "assets/dependencies/baseline_20261013/coco2017labels-segments.zip"
    require(not out.exists(), "prepared dataset already exists; verify it rather than overwrite")
    gt = original / "annotations/instances_val2017.json"
    require(sha256(gt) == GT_SHA, "wrong original annotation file")
    annotations = read_json(gt)
    imgs = sorted(annotations["images"], key=lambda x: x["id"])
    require(len(imgs) == 5000 and len({i["id"] for i in imgs}) == 5000, "invalid image inventory")
    counts = Counter(a["image_id"] for a in annotations["annotations"])
    crowd = sorted({a["image_id"] for a in annotations["annotations"] if a.get("iscrowd", 0)})
    empty = sorted(i["id"] for i in imgs if not counts[i["id"]])
    # Predeclared before model output: an empty-GT image, a crowd image, densest image,
    # then the earliest IDs. Empty predictions and ties are covered by synthetic tests.
    smoke = list(dict.fromkeys(empty[:1] + crowd[:1] + [max(counts, key=lambda k: (counts[k], -k))]
                              + [i["id"] for i in imgs]))[:16]
    (out / "images/val2017").mkdir(parents=True)
    (out / "labels/val2017").mkdir(parents=True)
    (out / "annotations").mkdir()
    shutil.copyfile(gt, out / "annotations/instances_val2017.json")
    records = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        require("coco/val2017.txt" in names, "unexpected official archive structure")
        official_list = z.read("coco/val2017.txt").decode("utf-8").splitlines()
        require({int(Path(p).stem) for p in official_list} == {i["id"] for i in imgs},
                "official validation list differs from annotation images")
        for im in imgs:
            name = im["file_name"]
            source = original / "images/val2017" / name
            require(source.is_file(), "missing original image: " + name)
            target = out / "images/val2017" / name
            shutil.copyfile(source, target)
            digest = sha256(source)
            require(sha256(target) == digest, "image copy changed")
            label_name = "coco/labels/val2017/" + Path(name).with_suffix(".txt").name
            label_sha = None
            if label_name in names:
                label = out / "labels/val2017" / Path(name).with_suffix(".txt").name
                with label.open("xb") as f:
                    f.write(z.read(label_name))
                label_sha = sha256(label)
            records.append({"image_id": im["id"], "file_name": name,
                            "image_sha256": digest, "label_sha256": label_sha})
    # Read the pinned YAML directly; preparing files does not need to import Torch.
    template_path = ROOT / "shared/vendor/ultralytics_8_4_100/ultralytics/cfg/datasets/coco.yaml"
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    for scope, ids in (("full", [i["id"] for i in imgs]), ("smoke", smoke)):
        # A nested list filename ending in val2017.txt keeps the native COCO detector active.
        folder = out / scope
        folder.mkdir()
        with (folder / "val2017.txt").open("x", encoding="utf-8") as f:
            f.write("".join(str((out / "images/val2017" / f"{i:012d}.jpg").resolve()) + "\n" for i in ids))
        write_new(folder / "image_ids.json", ids)
        data = {"path": str(out.resolve()), "train": str((folder / "val2017.txt").resolve()),
                "val": str((folder / "val2017.txt").resolve()), "names": template["names"]}
        with (folder / "coco.yaml").open("x", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    with (out / "IMAGE_MANIFEST.jsonl").open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    write_new(out / "PREPARATION.json", {
        "annotations_sha256": GT_SHA, "images": len(imgs), "original_images_unchanged": True,
        "official_archive_path": str(archive), "official_archive_sha256": sha256(archive),
        "official_archive_source": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip",
        "dataset_template_sha256": sha256(template_path),
        "manifest_sha256": sha256(out / "IMAGE_MANIFEST.jsonl"),
        "smoke_image_ids": smoke, "selection": "empty GT, crowd, largest GT count, ascending IDs; no model outputs",
        "source_script_sha256": sha256(__file__)})
    print(json.dumps({"prepared": str(out), "images": len(imgs), "smoke": smoke}))


if __name__ == "__main__":
    main()
