"""Construct the old 160-pixel and official 640-pixel objectives on one cache.

This module is deliberately a data/target adapter.  It never fits parameters,
reruns assignment, or calls a model.  ``load_image_rows`` returns one explicit
record per cached candidate, making the target pixels and area auditable before
any solver is run.

``old160`` follows the old ``native_bank.py`` construction: a COCO polygon is
rasterized, letterboxed with nearest-neighbour interpolation to 640, reduced
to 160 with nearest-neighbour interpolation, and restricted by the letterboxed
COCO bounding box on the 160 grid.  ``official640`` follows the official TAL
cache: the saved overlap raster/owner, 640 bilinear prototype, and saved TAL
target box are used.  The two labels are intentionally never substituted for
one another.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


OFFICIAL_SEG_GAIN = 9.83241
LAMBDA = 0.003


def _value(image: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a cache value while accepting the historic dict-like format."""
    if key not in image and default is not None:
        return default
    return image[key]


def _xyxy_from_xywh(box: torch.Tensor) -> torch.Tensor:
    return torch.cat((box[:2], box[:2] + box[2:]))


def _letterbox_geometry(image: Mapping[str, Any], ann: Mapping[str, Any]) -> tuple[float, int, int, int, int, torch.Tensor]:
    """Return old letterbox gain/shape/padding and the 640-space GT box."""
    gain_value = image.get("gain", image.get("letterbox_gain"))
    pad_value = None
    if gain_value is None and image.get("ratio_pad") is not None:
        # Ultralytics stores ratio_pad as ((gain_x, gain_y), (pad_x, pad_y));
        # old caches store scalar gain/left/top.  Both forms are accepted.
        ratio_pad = image["ratio_pad"]
        try:
            gain_raw = ratio_pad[0]
            if hasattr(gain_raw, "__len__") and not isinstance(gain_raw, (str, bytes)):
                gain_raw = gain_raw[0]
            gain_value = float(gain_raw)
            pad_value = ratio_pad[1]
        except (IndexError, TypeError, ValueError):
            gain_value = None
    if gain_value is None:
        raise KeyError("old160 requires saved gain/letterbox_gain or ratio_pad")
    gain = float(gain_value)
    original_shape = image.get("original_shape")
    if original_shape is None:
        raise KeyError("old160 requires image['original_shape']")
    oh, ow = (int(original_shape[0]), int(original_shape[1]))
    # native_bank stores the rounded shape and integer left/top padding.  Use
    # the saved geometry when available so the adapter cannot silently change
    # a historical cache's rounding convention.
    nh = int(image.get("resized_height", round(oh * gain)))
    nw = int(image.get("resized_width", round(ow * gain)))
    left_default = round((640 - nw) / 2 - 0.1)
    top_default = round((640 - nh) / 2 - 0.1)
    left = int(image.get("left", pad_value[0] if pad_value is not None else left_default))
    top = int(image.get("top", pad_value[1] if pad_value is not None else top_default))
    xywh = torch.as_tensor(ann["bbox"], dtype=torch.float32)
    box640 = _xyxy_from_xywh(xywh) * torch.tensor([gain, gain, gain, gain])
    box640 = box640 + torch.tensor([left, top, left, top], dtype=torch.float32)
    return gain, nh, nw, left, top, box640


def _old_gt_and_support(image: Mapping[str, Any], ann: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    """Build old native-grid GT and support exactly from one COCO annotation."""
    if "original_shape" not in image:
        raise KeyError("old160 requires image['original_shape']")
    gain, nh, nw, left, top, box640 = _letterbox_geometry(image, ann)
    mask = torch.as_tensor(image.get("_coco_mask"), dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError("internal COCO mask must be HxW")
    # native_bank.py: padded = F.pad(F.interpolate(gt,(nh,nw),nearest), ...)
    resized = F.interpolate(mask[None, None], (nh, nw), mode="nearest")[0, 0]
    right = 640 - nw - left
    bottom = 640 - nh - top
    if min(left, right, top, bottom) < 0:
        raise ValueError(f"invalid letterbox padding {(left, right, top, bottom)}")
    padded = F.pad(resized, (left, right, top, bottom))
    native_gt = F.interpolate(padded[None, None], (160, 160), mode="nearest")[0, 0]
    native_box = box640 / 4.0
    yy, xx = torch.meshgrid(torch.arange(160), torch.arange(160), indexing="ij")
    support = ((xx >= native_box[0]) & (xx < native_box[2]) &
               (yy >= native_box[1]) & (yy < native_box[3]))
    area = float((native_box[2:] - native_box[:2]).clamp_min(0).prod())
    meta = {
        "label_mode": "coco_annToMask_letterbox_nearest_640_then_nearest_160",
        "support_mode": "letterboxed_coco_bbox_half_open_160",
        "gain_letterbox": gain,
        "resized_shape": [nh, nw],
        "padding_left_top": [left, top],
        "box_640": box640.tolist(),
        "box_160": native_box.tolist(),
        "area_definition": "continuous GT box area in 160 coordinates",
    }
    return native_gt, support, area, meta


def _official_gt_and_support(image: Mapping[str, Any], k: int) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    """Build the official 640-grid overlap target for candidate index k."""
    masks = torch.as_tensor(image["masks"], dtype=torch.float32)
    boxes = torch.as_tensor(image["target_boxes"], dtype=torch.float32)
    owners = torch.as_tensor(image["owners"], dtype=torch.long)
    box = boxes[k]
    yy, xx = torch.meshgrid(torch.arange(masks.shape[-2]), torch.arange(masks.shape[-1]), indexing="ij")
    support = ((xx >= box[0]) & (xx < box[2]) &
               (yy >= box[1]) & (yy < box[3]))
    if not bool(support.any()):
        raise ValueError(f"empty official support for candidate {k}")
    y = masks[support] == (owners[k] + 1)
    area = float((box[2:] - box[:2]).clamp_min(0).prod())
    return y.float(), support, area, {
        "label_mode": "official_overlap_raster_owner_plus_one",
        "support_mode": "saved_TAL_target_box_half_open_640",
        "box_640": box.tolist(),
        "owner": int(owners[k]),
        "area_definition": "continuous GT box area in 640 coordinates",
    }


def _identity(record: Mapping[str, Any], image_id: int) -> dict[str, Any]:
    """Copy all stable identity fields without relying on list order."""
    row_image_id = int(record.get("image_id", image_id))
    return {
        "image_id": row_image_id,
        "annotation_id": int(record["annotation_id"]),
        "branch": str(record.get("branch", "one2one")),
        "raw_id": int(record["raw_id"]),
        "pyramid_level": int(record.get("pyramid_level", record.get("level", record.get("pyramid_level", 0)))),
        "target_gt_idx": int(record.get("target_gt_idx", record.get("gt_index", -1))),
    }


def _proto_for_mode(image: Mapping[str, Any], mode: str) -> torch.Tensor:
    proto = torch.as_tensor(image["proto"], dtype=torch.float32)
    if tuple(proto.shape) != (32, 160, 160):
        raise ValueError(f"expected native proto (32,160,160), got {tuple(proto.shape)}")
    if mode == "old160":
        return proto
    if mode == "official640":
        return F.interpolate(proto[None], (640, 640), mode="bilinear", align_corners=False)[0]
    raise ValueError("mode must be old160 or official640")


def load_image_rows(image: Mapping[str, Any], coco: Any, mode: str) -> list[dict[str, Any]]:
    """Export fixed objective rows for one cached image.

    Parameters
    ----------
    image:
        One ``images/<image_id>.pt`` dictionary from the official cache.  It
        must contain ``proto``, ``h``, ``coeff`` and ``rows``.  Official mode
        additionally requires ``masks``, ``owners`` and ``target_boxes``;
        old mode requires ``original_shape`` and COCO annotation lookup.
        The image id should be present as ``image['image_id']`` or in every
        row's ``image_id`` field; otherwise the exported identity uses ``-1``
        so a caller cannot mistake an invented id for a verified one.
    coco:
        A ``pycocotools.coco.COCO`` object, or an object exposing ``anns`` and
        ``annToMask``.  It is used only by old160.  Passing None to old160 is
        an error; official640 does not inspect it.
    mode:
        ``"old160"`` or ``"official640"``.

    Returns
    -------
    list of dict
        Each dict has identity fields plus CPU tensors ``p`` (pixels×32),
        ``y`` (pixels), ``h`` (64), ``c`` (32), continuous ``area``, scalar
        ``gain`` and ``target_meta``.  For old160, ``gain=1``; official640
        uses ``gain=9.83241``.  Rows preserve cache order but identity fields
        are explicit so callers can audit matching independently.
    """
    if mode not in {"old160", "official640"}:
        raise ValueError("mode must be old160 or official640")
    rows = image["rows"]
    image_id = int(image.get("image_id", rows[0].get("image_id", -1))) if rows else int(image.get("image_id", -1))
    proto = _proto_for_mode(image, mode)
    h_all = torch.as_tensor(image["h"], dtype=torch.float32)
    c_all = torch.as_tensor(image["coeff"], dtype=torch.float32)
    exported: list[dict[str, Any]] = []
    for k, record in enumerate(rows):
        ident = _identity(record, image_id)
        rid = ident["raw_id"]
        if rid < 0 or rid >= len(h_all) or rid >= len(c_all):
            raise IndexError(f"raw_id {rid} outside cached h/coeff")
        if mode == "old160":
            if coco is None:
                raise ValueError("old160 requires a COCO object")
            ann_id = ident["annotation_id"]
            try:
                ann = coco.anns[ann_id]
            except (KeyError, TypeError, AttributeError) as exc:
                raise KeyError(f"annotation {ann_id} missing from COCO object") from exc
            # Keep the mask local to this call; do not add it to the cache row.
            old_mask = torch.as_tensor(coco.annToMask(ann), dtype=torch.float32)
            local = dict(image)
            local["_coco_mask"] = old_mask
            y_full, support, area, target_meta = _old_gt_and_support(local, ann)
            y = y_full[support].contiguous()
            p = proto[:, support].T.contiguous()
            gain = 1.0
        else:
            y, support, area, target_meta = _official_gt_and_support(image, k)
            p = proto[:, support].T.contiguous()
            gain = OFFICIAL_SEG_GAIN
        if p.shape[0] != y.shape[0]:
            raise AssertionError("p/y pixel count mismatch")
        exported.append({
            **ident,
            "p": p.cpu(),
            "y": y.cpu(),
            "h": h_all[rid].cpu(),
            "c": c_all[rid].cpu(),
            "area": float(area),
            "gain": float(gain),
            "lambda": LAMBDA,
            "support_pixels": int(y.numel()),
            "positive_pixels": int((y > 0.5).sum()),
            "proto_resolution": [int(proto.shape[-2]), int(proto.shape[-1])],
            "target_meta": target_meta,
        })
    return exported


def load_coco_for_split(annotations: Path | str, split: str):
    """Load a COCO annotation file without coupling the solver to pycocotools."""
    from pycocotools.coco import COCO

    path = Path(annotations)
    if path.is_dir():
        path = path / f"instances_{split}2017.json"
    return COCO(str(path))


def objective_row(row: Mapping[str, Any], delta: torch.Tensor | None = None) -> torch.Tensor:
    """Evaluate one exported row for a coefficient increment (no optimizer)."""
    p = torch.as_tensor(row["p"], dtype=torch.float64)
    y = torch.as_tensor(row["y"], dtype=torch.float64)
    c = torch.as_tensor(row["c"], dtype=torch.float64)
    if delta is None:
        delta = torch.zeros_like(c)
    z = p @ (c + delta)
    bce = float(row["gain"]) * F.binary_cross_entropy_with_logits(z, y, reduction="sum") / float(row["area"])
    return bce + LAMBDA * delta.square().sum() / 2


def compare_identities(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Report identity overlap and target-shape differences between modes."""
    key = lambda r: (r["image_id"], r["annotation_id"], r["raw_id"], r["pyramid_level"], r["target_gt_idx"])
    a, b = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    common = sorted(set(a) & set(b))
    return {
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "common_identity_rows": len(common),
        "only_a": len(set(a) - set(b)),
        "only_b": len(set(b) - set(a)),
        "common_pixel_count_different": sum(int(a[k]["p"].shape[0] != b[k]["p"].shape[0]) for k in common),
        "common_area_different": sum(int(abs(float(a[k]["area"]) - float(b[k]["area"])) > 1e-6) for k in common),
        "common_positive_pixel_count_different": sum(int(a[k]["positive_pixels"] != b[k]["positive_pixels"]) for k in common),
        "common_gain_different": sum(int(abs(float(a[k]["gain"]) - float(b[k]["gain"])) > 1e-12) for k in common),
        "identity_fields": ["image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export and audit old160/official640 target rows")
    parser.add_argument("--image", type=Path, required=True, help="one cached image .pt")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO annotation JSON or directory")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    image = torch.load(args.image, map_location="cpu", weights_only=False)
    coco = load_coco_for_split(args.annotations, args.split)
    old = load_image_rows(image, coco, "old160")
    official = load_image_rows(image, None, "official640")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"mode": "old160", "rows": old}, args.out / "ROWS_old160.pt")
    torch.save({"mode": "official640", "rows": official}, args.out / "ROWS_official640.pt")
    summary = compare_identities(old, official)
    summary.update({"old_target": "COCO polygon nearest 640 -> nearest 160; GT bbox support", "official_target": "cached overlap owner raster; 640 bilinear proto; TAL bbox support", "seg_gain_old160": 1.0, "seg_gain_official640": OFFICIAL_SEG_GAIN, "lambda": LAMBDA})
    (args.out / "OBJECTIVE_DIFFERENCES.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _main()
