"""Pair valid 7P 8x8 scores with 7D's exact official 640-grid support scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from oracle_resolution import grouped_summary


def main(args):
    rows = json.loads(args.grid_rows.read_text(encoding="utf-8"))
    source = json.loads(args.exact_rows.read_text(encoding="utf-8"))
    exact = {(int(r["image_id"]), int(r["annotation_id"])): r for r in source
             if abs(float(r["penalty"]) - float(args.penalty)) < 1e-12}
    assert len(rows) == len(exact) == 200
    original_difference, oracle_difference = [], []
    for r in rows:
        key = int(r["image_id"]), int(r["annotation_id"])
        q = exact[key]
        assert abs(r["box_iou"] - q["box_iou"]) < 1e-5
        assert abs(r["original_full_iou"] - q["original_mask_iou"]) < 1e-5
        original_difference.append(r["original640"]["iou"] - q["initial_iou"])
        oracle_difference.append(r["oracle640"]["iou"] - q["new_iou"])
        r["original640"] = {"iou": q["initial_iou"], "auc": q["initial_auc"]}
        r["oracle640"] = {"iou": q["new_iou"], "auc": q["new_auc"]}
        r["gain640"] = {m: r["oracle640"][m] - r["original640"][m]
                        for m in ("iou", "auc")}
    groups = {"all": lambda r: True,
              "original_failure": lambda r: r["original_full_iou"] < .75,
              "good_box_failure": lambda r: r["box_iou"] >= .75 and
              r["original_full_iou"] < .75,
              "original_success": lambda r: r["original_full_iou"] >= .75}
    summary = {name: grouped_summary(rows, name, select) for name, select in groups.items()}
    summary["audit"] = {"instances": len(rows),
                        "original_approx_minus_exact_mean": float(np.mean(original_difference)),
                        "oracle_approx_minus_exact_mean": float(np.mean(oracle_difference)),
                        "exact_source": str(args.exact_rows),
                        "grid_source": str(args.grid_rows)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ROWS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"instances": len(rows),
        "images": len(set(r["image_id"] for r in rows)),
        "high_resolution": "exact 7D official crop_mask 640 support",
        "low_resolution": "7O matched 8x8 pooled grid"}, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-rows", type=Path, required=True)
    parser.add_argument("--exact-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--penalty", default="0.003")
    main(parser.parse_args())
