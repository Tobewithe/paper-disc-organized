"""Single-seed final-candidate evaluation on the frozen COCO transfer split.

This is a screening run for the proposed Ada+Center readout.  Backbone
features, boxes, scores, classes, prototypes and coefficients are reused from
the predeclared 1200-fit/300-transfer cache.  Only the four readout arms are
changed; all arms use the same candidate set and official COCOeval wrapper.
"""
from pathlib import Path
import json
import hashlib
import torch

from eval_innovations_suite import evaluate_transfer
from rich_pixel_readout import GlobalHead
from ada_calib_readout import AdaCalibHead

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "diagnostics/readout_input_scale1200_20260912/cache"
NORMALIZER = ROOT / "diagnostics/shared_label_controls_20260912/normalizer.pt"
S032 = ROOT / "diagnostics/shared_label_controls_20260912/raw_coco_s0/checkpoints/epoch015.pt"
ADA = ROOT / "runs/innovation_reaudit_20260913/corrected_training/ada_s0/epoch015.pt"
OUT = ROOT / "diagnostics/final_candidate_s0_20260914"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_s032(device):
    m = GlobalHead().to(device)
    ckpt = torch.load(S032, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model"])
    return m.eval()


def load_ada(device):
    m = AdaCalibHead().to(device)
    ckpt = torch.load(ADA, map_location=device, weights_only=False)["model"]
    # CorrectedHead(mode='ada') wraps AdaCalibHead under a `base.` prefix.
    mapped = {k[5:] if k.startswith("base.") else k: v for k, v in ckpt.items()}
    m.load_state_dict(mapped)
    return m.eval()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel = json.loads((CACHE / "selection.json").read_text(encoding="utf-8"))
    transfer = list(sel["transfer"])
    normalizer = torch.load(NORMALIZER, map_location=device, weights_only=True)
    s032 = load_s032(device)
    ada = load_ada(device)
    models = {
        "s032_s0": {"type": "s032", "model": s032},
        "ada_s0": {"type": "ada_calib", "model": ada, "center_prior": False},
        "center_s0": {"type": "s032", "model": s032, "center_prior": True},
        "ada_center_s0": {"type": "ada_calib", "model": ada, "center_prior": True},
    }
    # evaluate_transfer uses the model type to choose the readout path.  Its
    # `center_prior` flag is honored for both s032 and adaptive arms.
    out = evaluate_transfer(models, normalizer, transfer, OUT)
    protocol = {
        "status": "COMPLETE",
        "seed": 0,
        "split": "predeclared 300-image transfer split from readout_input_scale1200",
        "fit_images": 1200,
        "transfer_images": len(transfer),
        "arms": list(models),
        "GT_used_in_prediction": False,
        "backbone_recomputed": False,
        "candidate_set": "frozen cached candidates; same boxes/scores/classes for all arms",
        "decoder": "640 bilinear logits, original candidate crop, threshold 0",
        "evaluator": "pycocotools COCOeval on conversion_input/instances_probe.json",
        "limitations": "This is a single-seed screening evaluation on a previously explored transfer split, not the final blind COCO test.",
        "source_sha256": {
            "s032_checkpoint": sha256(S032),
            "ada_checkpoint": sha256(ADA),
            "normalizer": sha256(NORMALIZER),
            "selection": sha256(CACHE / "selection.json"),
        },
        "summary": out.to_dict(orient="records"),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
