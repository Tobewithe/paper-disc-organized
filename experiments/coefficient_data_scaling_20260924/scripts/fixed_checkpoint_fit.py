"""Evaluate the last coefficient-head checkpoint over its entire fit subset."""
import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import load, roi_losses, setup, target_rois, write
    from feature_probes import Probe

    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    selected = json.loads((a.training_run / "SELECTION.json").read_text())["images"]
    history = json.loads((a.training_run / "HISTORY.json").read_text())
    epoch = history[-1]["epoch"]
    state = load(a.training_run / f"epoch{epoch}.pt")["state_dict"]
    net = Probe("mlp_large", state["h_mean"], state["h_std"],
        state["p_mean"], state["p_std"], state["c_scale"]).cuda()
    net.load_state_dict(state)
    net.eval()
    total_before, total_after, positives = 0.0, 0.0, 0
    images_with_positive = 0
    with torch.no_grad():
        for step, iid in enumerate(selected, 1):
            path = a.bank / "images" / f"{iid:012d}.pt"
            if not path.exists():
                continue
            image = load(path)
            count = len(image["rows"])
            if not count:
                continue
            mask = np.frombuffer(zlib.decompress(image.pop("mask_zlib")), dtype="<u2").copy()
            image["masks"] = torch.from_numpy(mask.reshape(image.pop("mask_shape")).astype(np.int64))
            image["proto"] = image["proto"].float()
            coeff = image["coeff"].cuda()
            updated = coeff + net(image["h"].cuda(),
                torch.zeros(count, 512, device="cuda"), image["levels"].cuda())
            rois = target_rois(image)
            total_before += float(roi_losses(coeff[None], rois)[0])
            total_after += float(roi_losses(updated[None], rois)[0])
            positives += count
            images_with_positive += 1
            if step % 1000 == 0 or step == len(selected):
                write(a.out / "PROGRESS.json", dict(images=step, total=len(selected),
                    positives=positives))
                print(json.dumps(dict(images=step, total=len(selected))), flush=True)
    before, after = total_before / positives, total_after / positives
    write(a.out / "SUMMARY.json", dict(images=len(selected), images_with_positive=images_with_positive,
        positives=positives, last_epoch=epoch, original_fit_bce=before,
        last_checkpoint_fit_bce=after, absolute_improvement=before-after,
        relative_improvement=(before-after)/before,
        scope="Fixed last checkpoint on all selected train positives; no per-instance oracle"))
    write(a.out / "COMPLETE.json", dict(images=len(selected), positives=positives))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("source", "bank", "training_run", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    main(parser.parse_args())
