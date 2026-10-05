"""Posthoc paired comparison of two entirely GT-free gradient proxies."""
import argparse
import json
from pathlib import Path

import numpy as np

from crossfit_target_moment import MODES, cluster_ci, cosine


def main(args):
    data = np.load(args.moments, allow_pickle=False)
    indirect = np.load(args.indirect, allow_pickle=False)
    direct = np.load(args.direct, allow_pickle=False)
    for key in ("image_id", "annotation_id", "fold"):
        assert np.array_equal(data[key], indirect[key])
        assert np.array_equal(data[key], direct[key])
    g = data["g"]
    goodbox = data["box_iou"] >= .75
    failure = data["original_iou"] < .75
    masks = {"all": np.ones(len(g), bool), "failure": failure,
             "failure_good_box": failure & goodbox,
             "success_good_box": ~failure & goodbox,
             "severe_failure": data["original_iou"] < .5}
    result = {"posthoc_exploratory": True, "definition":
              "direct predicted g versus predicted-box s minus predicted t; GT gradient only as evaluation target",
              "groups": {}}
    for name, mask in masks.items():
        result["groups"][name] = {}
        for mode in MODES:
            a = cosine(direct[mode], g)
            b = cosine(data["s_predbox"] - indirect[mode], g)
            result["groups"][name][mode] = {"direct_mean": float(a[mask].mean()),
                "proxy_reconstruction_mean": float(b[mask].mean()),
                "direct_minus_proxy": cluster_ci(data["image_id"], a - b, mask)}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"failure_good_box": result["groups"]["failure_good_box"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("moments", "indirect", "direct", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
