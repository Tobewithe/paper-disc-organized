"""Method-specific neighbor counterfactual experiment.

Uses the frozen S047 image pairs and reruns the stock YOLO backbone on the
already materialized neighbor/background interventions. The candidate source
indices are fixed from the original image. Four readout arms are evaluated on
the same backbone response: S032, Ada, Center, and Ada+Center.

This is a mechanism probe, not a full COCO AP evaluation. It estimates the
paired effect of retaining the selected neighbor versus replacing it with the
same texture in a matched background location.
"""
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "4")
os.environ.setdefault("YOLO_CONFIG_DIR", str(__import__("pathlib").Path(__file__).parent / "settings"))

import argparse
import contextlib
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops

from ada_calib_readout import AdaCalibHead
from bifurcated_spatial_readout import BifurcatedSpatialHead
from contrastive_spatial_loss import apply_centerness_prior
from eval_readout_input_pilot import ici
from frozen_mechanism_probe import ownership, sha
from neighbor_background_probe import InputCapture, gt_geometry, spatial
from rich_pixel_readout import GlobalHead, instance_features


ROOT = Path(__file__).resolve().parent
S047 = ROOT / "diagnostics" / "neighbor_background_20260912"
WEIGHT = ROOT / "weights" / "yolo26m-seg.pt"
ANN = ROOT.parent.parent / "datasets" / "coco" / "annotations" / "instances_train2017.json"
NORMALIZER = ROOT / "diagnostics" / "shared_label_controls_20260912" / "normalizer.pt"
S032_DIR = ROOT / "diagnostics" / "shared_label_controls_20260912"
HEAD_DIR = ROOT / "runs" / "innovation_reaudit_20260913" / "corrected_training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def csvsave(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm(x, normalizer):
    return (x - normalizer["mean"].to(x.device)) / normalizer["std"].to(x.device)


def read_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_heads():
    heads = {}
    for seed in range(3):
        s032 = GlobalHead().to(DEVICE)
        ckpt = S032_DIR / f"raw_coco_s{seed}" / "checkpoints" / "epoch015.pt"
        s032.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["model"])
        s032.eval()
        heads[f"s032_s{seed}"] = s032

        ada = AdaCalibHead().to(DEVICE)
        ckpt = HEAD_DIR / f"ada_s{seed}" / "epoch015.pt"
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["model"]
        # The corrected-suite checkpoint wraps AdaCalibHead in CorrectedHead,
        # so its otherwise identical parameters carry a single ``base.`` prefix.
        if state and all(key.startswith("base.") for key in state):
            state = {key[len("base."):]: value for key, value in state.items()}
        ada.load_state_dict(state)
        ada.eval()
        heads[f"ada_s{seed}"] = ada
    return heads


def source_level(source, maps):
    sizes = [int(maps[f"h{k}"].shape[-2] * maps[f"h{k}"].shape[-1]) for k in range(3)]
    if source < sizes[0]:
        return 0
    if source < sizes[0] + sizes[1]:
        return 1
    return 2


def inferred_relaxed(model, image, hook_maps):
    """Run the frozen backbone without replaying post-NMS row identities.

    S047 stores the target's dense raw-grid source index. Edited pixels can
    change tie ordering in the predictor's postprocess output, so this probe
    reads the same raw tensor directly and keeps that original source index.
    No detector candidate is reselected from the edited image.
    """
    hook_maps.clear()
    with torch.inference_mode():
        model.predict(image, predictor=InputCapture, imgsz=640, rect=False,
                      conf=.001, iou=.7, max_det=300, half=False,
                      retina_masks=False, device=0, verbose=False)
        pred = model.predictor
        return dict(raw=pred.dense, cap=pred.capture, maps=dict(hook_maps))


def candidate_inputs(current, source, normalizer):
    """Build the exact 73D readout input used during training."""
    raw = current["raw"]
    maps = current["maps"]
    h_all = torch.cat([maps[f"h{k}"][0].flatten(1).T for k in range(3)], dim=0)
    h = h_all[source : source + 1]
    level = torch.tensor([source_level(source, maps)], device=h.device, dtype=torch.long)
    box = ops.xywh2xyxy(raw[0, :4, source : source + 1].T)
    x = instance_features(h, level, box, (640, 640))
    return norm(x, normalizer), raw[0, 84:, source : source + 1].T, current["cap"]["proto"], box


def logits_for_arm(arm, seed, x, coeff, proto, box):
    if arm == "s032":
        model = HEADS[f"s032_s{seed}"]
        z = torch.einsum("bc,chw->bhw", coeff + model(x), proto)
    elif arm == "ada":
        model = HEADS[f"ada_s{seed}"]
        z = model.forward_inference(x, coeff, proto)[0]
    elif arm == "center":
        model = HEADS[f"s032_s{seed}"]
        z = torch.einsum("bc,chw->bhw", coeff + model(x), proto)
        z = apply_centerness_prior(z, box * 160.0 / 640.0, (160, 160), margin_scale=0.5)
    elif arm == "ada_center":
        model = HEADS[f"ada_s{seed}"]
        z = model.forward_inference(x, coeff, proto)[0]
        z = apply_centerness_prior(z, box * 160.0 / 640.0, (160, 160), margin_scale=0.5)
    else:
        raise ValueError(arm)
    return z


def decode_logit(z, box, shape):
    up = F.interpolate(z[:, None], (640, 640), mode="bilinear", align_corners=False)[:, 0]
    cropped = ops.crop_mask(up, box)
    return (ops.scale_masks(cropped[:, None], shape)[:, 0] > 0.0).cpu().numpy()[0].astype(bool)


def region_mean(z, mask, shape):
    up = F.interpolate(z[:, None], (640, 640), mode="bilinear", align_corners=False)[:, 0]
    full = ops.scale_masks(up[:, None], shape)[:, 0][0]
    m = torch.as_tensor(mask, device=full.device, dtype=torch.bool)
    return float(full[m].mean().detach().cpu()) if bool(m.any()) else float("nan")


def arm_response(current, source, arm, seed, own, neighbor, union, crowd, normalizer, shape):
    # Backbone outputs are inference-mode tensors.  Readout evaluation is
    # deliberately gradient-free, so avoid autograd trying to save them.
    with torch.no_grad():
        x, coeff, proto, box = candidate_inputs(current, source, normalizer)
        z = logits_for_arm(arm, seed, x, coeff, proto, box)
        mask = decode_logit(z, box, shape)
        metrics = spatial(mask, own, neighbor, union, crowd)
        metrics.update(
            target_logit_own=region_mean(z, own, shape),
            target_logit_neighbor=region_mean(z, neighbor, shape),
            target_logit_background=region_mean(z, ~(union | crowd), shape),
            box_iou=float("nan"),
        )
    return metrics, z, box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "pairs").mkdir()

    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    normalizer = torch.load(NORMALIZER, map_location=DEVICE, weights_only=False)
    manifest = json.loads((S047 / "manifest.json").read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(__import__("io").StringIO()):
        gt = COCO(str(ANN))
    dump(out / "protocol.json", {
        "experiment": "METHOD_SPECIFIC_NEIGHBOR_COUNTERFACTUAL",
        "source_manifest": str(S047 / "manifest.json"),
        "source_manifest_sha256": sha(S047 / "manifest.json"),
        "pairs": len(manifest["pairs"]),
        "arms": ["s032", "ada", "center", "ada_center"],
        "seeds": [0, 1, 2],
        "interventions": "existing deterministic S047 texture replacements; neighbor and matched background control use identical pixel values",
        "candidate": "source index fixed from original image; no response-based reselection",
        "decoder": "bilinear logit upsample to 640, predicted-source box crop, scale to original shape, threshold logit at 0",
        "limits": "64 train2017 images from S047; mechanism probe, not full COCO AP; trained readout heads evaluated on backbone responses to edited images",
    })

    model = YOLO(str(WEIGHT))
    model.model.eval().requires_grad_(False)
    hook_maps = {}
    hooks = []
    for index in [4, 6, 10, 16, 19, 22]:
        def hook(module, inputs, output, index=index):
            hook_maps[f"layer{index}"] = output.detach().clone()
        hooks.append(model.model.model[index].register_forward_hook(hook))
    head = model.model.model[-1]
    # S047's frozen source indices were recorded from the one-to-many
    # 8,400-cell path.  Recent Ultralytics defaults enable end2end inference,
    # which returns only 300 one-to-one detections and cannot replay those
    # indices.  Keep the same model weights but select the recorded path.
    head.end2end = False
    for level, branch in enumerate(head.cv4):
        def hook(module, inputs, output, level=level):
            hook_maps[f"h{level}"] = inputs[0].detach().clone()
        hooks.append(branch[-1].register_forward_hook(hook))

    rows = []
    started = time.monotonic()
    for number, item in enumerate(manifest["pairs"], 1):
        iid, aid, bid = item["image_id"], item["target"], item["neighbor"]
        folder = S047 / "pairs" / str(iid)
        original = read_npz(folder / "original.npz")
        source = int(original["source_index"])
        if source < 0:
            continue
        image_path = S047 / "images" / gt.imgs[iid]["file_name"]
        image = cv2.imread(str(image_path))
        shape = image.shape[:2]
        own, neighbor, edit, union, crowd, background, _ = gt_geometry(gt, iid, gt.anns[aid], gt.anns[bid])

        variants = [("original", image)]
        for seed in range(3):
            path = folder / f"neighbor_texture_{seed}.png"
            if path.exists():
                variants.append((f"neighbor_{seed}", cv2.imread(str(path))))
            path = folder / f"background_control_texture_{seed}.png"
            if path.exists():
                variants.append((f"control_{seed}", cv2.imread(str(path))))

        for variant, changed in variants:
            current = inferred_relaxed(model, changed, hook_maps)
            # Keep the original source candidate fixed. A changed image may
            # alter NMS, but the counterfactual is evaluated on that same raw
            # source slot whenever it remains available.
            if source >= current["raw"].shape[-1]:
                continue
            current_source = source
            for arm in ["s032", "ada", "center", "ada_center"]:
                metrics, z, box = arm_response(current, current_source, arm, 0, own, neighbor, union, crowd, normalizer, shape)
                # Evaluate all three trained seeds without another backbone pass.
                for seed in [0, 1, 2]:
                    metrics, z, box = arm_response(current, current_source, arm, seed, own, neighbor, union, crowd, normalizer, shape)
                    row = dict(image_id=iid, target=aid, neighbor=bid, relation=item["relation"], ici=item["ici"],
                               variant=variant, arm=arm, seed=seed, source_index=source, **metrics)
                    rows.append(row)
        if number % 8 == 0 or number == len(manifest["pairs"]):
            print(json.dumps({"completed": number, "total": len(manifest["pairs"]), "rows": len(rows),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)

    for hook in hooks:
        hook.remove()
    csvsave(out / "metrics.csv", rows)

    # Paired neighbor-minus-background effects at the same image/fill seed.
    import pandas as pd
    df = pd.DataFrame(rows)
    effects = []
    value_cols = ["mask_iou", "coverage", "old_neighbor_error", "background_error",
                  "target_logit_own", "target_logit_neighbor", "target_logit_background"]
    for keys, group in df.groupby(["image_id", "relation", "ici", "arm", "seed"]):
        lookup = group.set_index("variant")
        for variant in ["neighbor_0", "neighbor_1", "neighbor_2"]:
            control = "control_" + variant.rsplit("_", 1)[1]
            if variant not in lookup.index or control not in lookup.index:
                continue
            row = dict(image_id=keys[0], relation=keys[1], ici=keys[2], arm=keys[3], seed=keys[4],
                       contrast="neighbor_minus_control", variant=variant)
            for key in value_cols:
                row[key] = float(lookup.loc[variant, key] - lookup.loc[control, key])
            effects.append(row)
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(out / "paired_effects.csv", index=False)

    summary = []
    if len(effects_df):
        for (arm, relation), group in effects_df.groupby(["arm", "relation"]):
            high = group[group.ici > 0.5]
            for state, part in [("all", group), ("high_ici", high)]:
                if not len(part):
                    continue
                row = dict(arm=arm, relation=relation, state=state, n_images=int(part.image_id.nunique()), n_rows=len(part))
                for key in value_cols:
                    row[key + "_mean"] = float(part[key].mean())
                    row[key + "_median"] = float(part[key].median())
                summary.append(row)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    dump(out / "COMPLETE.json", {"status": "COMPLETE", "rows": len(rows), "seconds": time.monotonic() - started,
                                  "source_manifest_sha256": sha(S047 / "manifest.json")})


if __name__ == "__main__":
    HEADS = load_heads()
    main()
