"""Cross-tabulate established COCO Mask75 matched sets with fixed GT cohorts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main(args):
    groups = defaultdict(set)
    with args.grid_objects.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["group"]].add(row["annotation_id"])
    coeff = json.loads(args.coeff_matches.read_text(encoding="utf-8"))
    spatial = json.loads(args.spatial_matches.read_text(encoding="utf-8"))
    matched = {"baseline": set(coeff["baseline"]), "native_coeff_mlp": set(coeff["native_coeff_mlp"]),
               "native_spatial": set(spatial["native_spatial"])}
    assert matched["baseline"] == set(spatial["baseline"]), "Baseline GT identity mismatch across studies"
    result = {}
    for group, ids in groups.items():
        baseline = ids & matched["baseline"]
        result[group] = {"gt": len(ids), "methods": {}}
        for name, predictions in matched.items():
            current = ids & predictions
            result[group]["methods"][name] = {
                "matched75": len(current), "matched75_fraction": len(current) / len(ids),
                "rescued_from_baseline": len(current - baseline),
                "damaged_from_baseline": len(baseline - current),
                "net_from_baseline": len(current) - len(baseline),
            }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"groups": len(result)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("grid_objects", "coeff_matches", "spatial_matches", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    main(parser.parse_args())
