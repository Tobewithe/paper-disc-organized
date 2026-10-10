"""Probe the retained cache NPZ schema for argmax_maskfail_position_oracle_20261005.

Laptop execution (authorized environment). Prints the NPZ keys/shapes and the index of a
target (annotation_id, raw_id) pair so the oracle-definition re-solve can be scripted
against the real schema.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(a):
    data = np.load(a.npz, allow_pickle=True)
    keys = list(data.keys())
    print("NPZ_KEYS " + json.dumps({k: (tuple(np.asarray(data[k]).shape), str(np.asarray(data[k]).dtype)) for k in keys}), flush=True)
    index = {}
    for field in ("annotation_id", "raw_id", "annotation_ids", "raw_ids"):
        if field in keys:
            values = np.asarray(data[field]).ravel()
            index[field] = {"min": int(values.min()), "max": int(values.max()), "count": int(values.size)}
            if field in ("annotation_id", "raw_id"):
                hits = np.nonzero(values == (a.annotation_id if field == "annotation_id" else a.raw_id))[0]
                index[field]["target_hits"] = hits.tolist()[:10]
    print("INDEX " + json.dumps(index), flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--annotation_id", type=int, required=True)
    p.add_argument("--raw_id", type=int, required=True)
    raise SystemExit(main(p.parse_args()))
