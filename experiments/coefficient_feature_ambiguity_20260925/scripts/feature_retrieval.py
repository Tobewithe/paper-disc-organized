import argparse
import json
from pathlib import Path

import numpy as np
import torch


def annotations(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(a["id"]): (int(a["category_id"]), float(a["area"])) for a in obj["annotations"]}


def unpack(path, ann, keep_gram):
    data = torch.load(path, map_location="cpu", weights_only=True)
    h = np.stack([r["h"].numpy() for r in data]).astype(np.float32)
    delta = np.stack([r["delta"].numpy() for r in data]).astype(np.float32)
    out = {
        "h": h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12),
        "direction": delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12),
        "radius": np.linalg.norm(delta, axis=1),
        "image_id": np.array([int(r["image_id"]) for r in data]),
        "annotation_id": np.array([int(r["annotation_id"]) for r in data]),
        "category": np.array([ann[int(r["annotation_id"])][0] for r in data]),
        "area_bin": np.digitize(
            np.sqrt([ann[int(r["annotation_id"])][1] for r in data]), [32, 96]
        ),
    }
    if keep_gram:
        out["gram"] = np.stack([r["gram"].numpy() for r in data]).astype(np.float32)
    del data
    return out


def functional_cos(a, b, gram):
    ab = a @ gram @ b
    aa = max(float(a @ gram @ a), 1e-12)
    bb = max(float(b @ gram @ b), 1e-12)
    return float(ab / np.sqrt(aa * bb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--train-annotations", type=Path, required=True)
    ap.add_argument("--val-annotations", type=Path, required=True)
    ap.add_argument("--original-rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fit = unpack(args.fit, annotations(args.train_annotations), False)
    val = unpack(args.val, annotations(args.val_annotations), True)
    original_rows = json.loads(args.original_rows.read_text(encoding="utf-8"))
    original = {(int(r["image_id"]), int(r["annotation_id"])): r for r in original_rows}
    rng = np.random.default_rng(20260925)
    sim = val["h"] @ fit["h"].T
    rows = []
    for i in range(len(val["h"])):
        same_class = np.flatnonzero(fit["category"] == val["category"][i])
        matched = same_class[fit["area_bin"][same_class] == val["area_bin"][i]]
        fallback = len(matched) < 5
        eligible = same_class if fallback else matched
        if len(eligible) < 5:
            eligible = np.arange(len(fit["h"]))
            fallback = True
        nearest = eligible[np.argsort(sim[i, eligible])[-5:][::-1]]
        random = rng.choice(eligible, size=(20, 5), replace=True)
        target = val["direction"][i]
        gram = val["gram"][i]
        retrieved_1 = functional_cos(fit["direction"][nearest[0]], target, gram)
        random_1 = float(np.mean([functional_cos(fit["direction"][j[0]], target, gram) for j in random]))
        retrieved_5 = functional_cos(fit["direction"][nearest].mean(axis=0), target, gram)
        random_5 = float(np.mean([functional_cos(fit["direction"][j].mean(axis=0), target, gram) for j in random]))
        key = (int(val["image_id"][i]), int(val["annotation_id"][i]))
        rr = original[key]
        rows.append({
            "image_id": key[0], "annotation_id": key[1],
            "category": int(val["category"][i]), "area_bin": int(val["area_bin"][i]),
            "eligible": len(eligible), "fallback": fallback,
            "h_nearest_cos": float(sim[i, nearest[0]]),
            "h_random_cos": float(np.mean(sim[i, random[:, 0]])),
            "oracle_radius": float(val["radius"][i]),
            "box_iou": float(rr["box_iou"]),
            "original_mask_iou": float(rr["original_image_iou"]),
            "h_only_learned_effect_cos": float(rr["arms"]["h_only"]["effect_cos"]),
            "retrieved_1_effect_cos": retrieved_1,
            "random_1_effect_cos": random_1,
            "retrieved_5_effect_cos": retrieved_5,
            "random_5_effect_cos": random_5,
        })
    (args.out / "ROWS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"fit": len(fit["h"]), "val": len(rows), "fallback": sum(x["fallback"] for x in rows)}, indent=2), encoding="utf-8")
    print(json.dumps({"fit": len(fit["h"]), "val": len(rows), "fallback": sum(x["fallback"] for x in rows)}))


if __name__ == "__main__":
    main()
