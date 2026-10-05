"""Measure frozen 7J-N correction norms without GT at predictor input."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main(a):
    sys.path.insert(0, str(a.neck_source / "scripts"))
    from train_neck import SpatialNet, prepare

    torch.set_num_threads(6)
    rows = torch.load(a.targets / "DATA.pt", weights_only=True, map_location="cpu")
    metadata = torch.load(a.metadata / "METADATA.pt", weights_only=True, map_location="cpu")
    neck = torch.load(a.neck / "NECK.pt", weights_only=True, map_location="cpu")
    normalization = torch.load(a.training / "NORMALIZATION.pt", weights_only=True, map_location="cpu")
    val, _ = prepare(rows, metadata, neck, normalization)
    output = []
    magnitude = {}
    for arm in ("h_only", "true_local"):
        checkpoint = torch.load(a.training / f"{arm}_best.pt", weights_only=True, map_location="cpu")
        model = SpatialNet().cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        values = []
        with torch.no_grad():
            for lo in range(0, len(rows), 128):
                hi = min(lo + 128, len(rows))
                unit, logradius = model(val["h"][lo:hi], val["inputs"][arm][lo:hi])
                values.append(logradius.expm1().cpu().numpy())
        magnitude[arm] = np.concatenate(values)
    assert len(magnitude["h_only"]) == len(rows)
    for i, row in enumerate(rows):
        output.append(dict(image_id=int(row["image_id"]), annotation_id=int(row["annotation_id"]),
                           h_only=float(magnitude["h_only"][i]),
                           true_local=float(magnitude["true_local"][i])))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "MAGNITUDES.json").write_text(json.dumps(output), encoding="utf-8")
    print(json.dumps(dict(n=len(output), h_only_mean=float(magnitude["h_only"].mean()),
                          true_local_mean=float(magnitude["true_local"].mean()))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("neck_source", "targets", "metadata", "neck", "training", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    main(p.parse_args())
