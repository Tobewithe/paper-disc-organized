"""Read-only asset probe for the coefficient oracle-alignment study on the laptop.

Checks, without executing any model:
  1. interpreter / torch / ultralytics versions;
  2. the official TAL affine cache transferred to the laptop (per-image banks);
  3. the ACD baseline/method coefficient-head overlays;
  4. the official COCO-pretrained checkpoint and list/annotation locations;
  5. the keys and shapes of one sample bank file and of the two overlays.
"""
import json
import sys
from pathlib import Path

import torch

CHECKS = []


def note(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(json.dumps({"check": name, "ok": bool(ok), "detail": str(detail)}), flush=True)


def probe_cache(root: Path):
    if not root.is_dir():
        note("official_cache_root", False, str(root))
        return
    note("official_cache_root", True, str(root))
    for name in ("official_data", "images"):
        p = root / name
        note(f"official_cache/{name}", p.is_dir(),
            f"{len(list(p.iterdir()))} entries" if p.is_dir() else "missing")
    images = root / "images"
    banks = sorted(images.glob("*.pt")) if images.is_dir() else []
    note("bank_count", len(banks) > 0, f"{len(banks)} per-image banks")
    if banks:
        sample = torch.load(banks[0], weights_only=False, map_location="cpu")
        note("bank_keys", True, sorted(sample.keys()))
        note("bank_meta", True, {k: (tuple(v.shape) if torch.is_tensor(v) else v)
                                 for k, v in sample.items() if k != "rows"})
        note("bank_rows", True, f"{len(sample['rows'])} rows; first row keys="
                                f"{sorted(sample['rows'][0].keys()) if sample['rows'] else 'none'}")


def probe_overlay(path: Path, label: str):
    if not path.is_file():
        note(f"overlay_{label}", False, str(path))
        return
    note(f"overlay_{label}_exists", True, f"{path.stat().st_size} bytes")
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:  # noqa: BLE001
        note(f"overlay_{label}_load", False, repr(error))
        return
    if isinstance(state, dict) and "state_dict" in state:
        meta = {k: v for k, v in state.items() if k != "state_dict"}
        note(f"overlay_{label}_meta", True, meta)
        state = state["state_dict"]
    note(f"overlay_{label}_keys", True, sorted(state.keys()))
    shapes = {k: tuple(v.shape) for k, v in state.items() if torch.is_tensor(v)}
    note(f"overlay_{label}_shapes", True, shapes)


def main():
    import ultralytics
    note("torch", True, torch.__version__)
    note("ultralytics", True, ultralytics.__version__)

    cache = Path(r"D:/coco_wire/data/official_tal_affine_20260930/runs/official_cache")
    probe_cache(cache)

    for name in ("fit.txt", "dev.txt", "val.txt"):
        p = cache / "official_data" / name
        note(f"list/{name}", p.is_file(), f"{len(p.read_text().split())} ids" if p.is_file() else "missing")

    acd = Path(r"D:/coco_wire/experiments/acd_native_coefficient_20261006/runs")
    probe_overlay(acd / "RUN_ACD_FEASIBILITY_S0/coeff_final_ema.pt", "acd_ema")
    probe_overlay(acd / "RUN_BASELINE_FEASIBILITY_S0_RETRY1/coeff_final_ema.pt", "baseline_ema")

    for candidate in (Path(r"D:/coco_wire/data/yolo26m-seg.pt"),
                      Path(r"D:/coco_wire/data/official_tal_affine_20260930/YOLO26m_seg_OFFICIAL_SHARED.pt")):
        note(f"weights/{candidate.name}", candidate.is_file(), str(candidate))

    ann = Path(r"D:/coco_wire/data/annotations")
    for name in ("instances_train2017.json", "instances_val2017.json"):
        p = ann / name
        note(f"annotations/{name}", p.is_file(), f"{p.stat().st_size} bytes" if p.is_file() else "missing")
    for split in ("train2017", "val2017"):
        p = Path(r"D:/coco_wire/data/images") / split
        note(f"images/{split}", p.is_dir(), f"{len(list(p.iterdir()))} files" if p.is_dir() else "missing")

    summary = {"checks": CHECKS, "all_ok": all(c["ok"] for c in CHECKS)}
    print("PROBE_SUMMARY " + json.dumps(summary), flush=True)
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
