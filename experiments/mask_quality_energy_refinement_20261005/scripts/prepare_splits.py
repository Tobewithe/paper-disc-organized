"""Create deterministic image-level train2017 FIT/DEV lists.

No model inference is run here. The lists are frozen before remote training.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANN = Path(r"C:\Dpan\codexproject\paper-disc-organized\assets\datasets\coco\annotations\instances_train2017.json")
OUT = ROOT / "IMAGE_SPLIT.json"


def key(image_id: int) -> str:
    return hashlib.sha256(f"qcr20261005:{image_id}".encode()).hexdigest()


def main():
    data = json.loads(ANN.read_text(encoding="utf-8"))
    ordinary = {int(i["id"]) for i in data["images"]}
    counts = {i: 0 for i in ordinary}
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0) and int(ann["image_id"]) in counts:
            counts[int(ann["image_id"])] += 1
    ids = sorted((i for i, n in counts.items() if n > 0), key=key)
    if len(ids) < 22000:
        raise RuntimeError(f"train2017 ordinary images available: {len(ids)}")
    fit = ids[:20000]
    dev = ids[20000:22000]
    result = {
        "study": "STUDY_6f0c44af49e846f0a14f44a4b43d38e4",
        "seed_key": "qcr20261005",
        "dataset": "COCO train2017",
        "fit_images": fit,
        "dev_images": dev,
        "fit_image_count": len(fit),
        "dev_image_count": len(dev),
        "fit_instance_count": sum(counts[i] for i in fit),
        "dev_instance_count": sum(counts[i] for i in dev),
        "image_overlap": len(set(fit) & set(dev)),
        "ordinary_train_images_considered": len(ids),
    }
    if result["image_overlap"]:
        raise RuntimeError("FIT/DEV overlap")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k.endswith("count") or k.endswith("overlap")}, indent=2))


if __name__ == "__main__":
    main()
