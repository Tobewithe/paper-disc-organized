"""Paired image-cluster analysis of saved official Mask75 GT matches."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(args):
    ids = read(args.controls / "LOCAL_SCORE_IDS.json")
    control = {}
    for name in ("global_spatial", "native_coeff_mlp", "native_quad_coeff_mlp"):
        folder = args.controls / "runs" / ids[name]
        assert (folder / "COMPLETE.json").exists(), name
        control[name] = read(folder / "MATCHED_GT75.json")
    native = read(args.native / "MATCHED_GT75.json")
    baseline = set(native["baseline"])
    assert all(set(data["baseline"]) == baseline for data in control.values())

    annotations = read(args.annotations)
    images = sorted(image["id"] for image in annotations["images"])
    image_index = {image_id: i for i, image_id in enumerate(images)}
    rows = [a for a in annotations["annotations"] if not a.get("iscrowd", 0) and not a.get("ignore", 0)]
    assert len(images) == 5000 and len(rows) == 36335
    gt_ids = np.array([a["id"] for a in rows])
    image_of = np.array([image_index[a["image_id"]] for a in rows])
    area = np.array([a["area"] for a in rows])
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "small": area < 32**2,
        "medium": (area >= 32**2) & (area < 96**2),
        "large": area >= 96**2,
    }
    matched = {"baseline": baseline, "native_spatial": set(native["native_spatial"])}
    matched.update({name: set(data[name]) for name, data in control.items()})
    assert all(ids <= set(gt_ids) for ids in matched.values())
    success = {name: np.isin(gt_ids, list(ids)).astype(np.int8) for name, ids in matched.items()}
    rng = np.random.default_rng(20260923)
    draws = rng.integers(0, len(images), size=(2000, len(images)), dtype=np.int32)
    contrasts = [
        ("native_spatial", "native_coeff_mlp"),
        ("native_spatial", "global_spatial"),
        ("native_coeff_mlp", "native_quad_coeff_mlp"),
    ]
    result = {}
    for group, select in groups.items():
        denominator = np.bincount(image_of[select], minlength=len(images))
        boot_den = denominator[draws].sum(axis=1)
        entry = {
            "gt": int(select.sum()),
            "micro_R75_pct": {name: float(values[select].mean() * 100) for name, values in success.items()},
            "paired_differences_pp": {},
        }
        for a, b in contrasts:
            delta = success[a].astype(np.int8) - success[b].astype(np.int8)
            numerator = np.bincount(image_of[select], weights=delta[select], minlength=len(images))
            boot = numerator[draws].sum(axis=1) / boot_den * 100
            entry["paired_differences_pp"][f"{a}-minus-{b}"] = {
                "estimate": float(delta[select].mean() * 100),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "a_only": int(((success[a] == 1) & (success[b] == 0) & select).sum()),
                "b_only": int(((success[b] == 1) & (success[a] == 0) & select).sum()),
            }
        result[group] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "PAIRED_MATCHED75.json").write_text(json.dumps({
        "scope": "Saved official Mask75 GT matches; one trained model per new control, seed-0 native spatial reference; 2000 image-cluster bootstrap resamples conditional on these fixed predictions; not AP uncertainty; subgroup comparisons exploratory without multiplicity correction",
        "groups": result,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("controls", "native", "annotations", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
