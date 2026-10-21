"""S1-only input geometry and bounded prediction reading; no model forward."""
from __future__ import annotations

import json
from pathlib import Path


def iter_prediction_array(path, chunk_size=1024 * 1024):
    """Stream a strict top-level JSON array of objects, retaining original order."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, eof = "", False

        def more():
            nonlocal buffer, eof
            block = stream.read(chunk_size)
            buffer += block
            eof = not block

        def nonempty():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                more()
                buffer = buffer.lstrip()
            if not buffer:
                raise ValueError("truncated prediction JSON")

        nonempty()
        if buffer[0] != "[":
            raise ValueError("predictions must be an array")
        buffer = buffer[1:]
        first = True
        while True:
            nonempty()
            if buffer[0] == "]":
                if not first:
                    raise ValueError("trailing comma")
                buffer = buffer[1:]
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    more()
            if not isinstance(value, dict):
                raise ValueError("each prediction must be an object")
            buffer = buffer[end:]
            yield value
            nonempty()
            if buffer[0] == "]":
                buffer = buffer[1:]
                break
            if buffer[0] != ",":
                raise ValueError("missing prediction delimiter")
            buffer = buffer[1:]
            first = False
        if (buffer + stream.read()).strip():
            raise ValueError("trailing data after prediction array")


def geometry(cache):
    """Recover actual integer resize/pad dimensions, not an imgsz approximation."""
    oh, ow = cache["original_hw"]
    ih, iw = cache["input_hw"]
    gains, pads = cache["ratio_pad"]
    rh, rw = round(oh * gains[0]), round(ow * gains[1])
    # LetterBox.apply_instances stores INTEGER left/top, not half the total pad.
    left, top = round(pads[0]), round(pads[1])
    if [float(left), float(top)] != list(pads):
        raise ValueError("pinned native cache requires integer left/top padding")
    right, bottom = iw - left - rw, ih - top - rh
    if min(oh, ow, ih, iw, rh, rw) <= 0 or min(left, top, right, bottom) < 0:
        raise ValueError("invalid cached resize/padding")
    if right - left not in (0, 1) or bottom - top not in (0, 1):
        raise ValueError("cache geometry differs from pinned centered LetterBox")
    # Native decoding uses the recorded pad symmetrically. With odd total pad,
    # its unpadded extent can exceed the actual resized image by one pixel.
    # Preserve that decoder unchanged; GT projection uses the real image transform.
    return {"original_hw": [oh, ow], "input_hw": [ih, iw],
            "resized_hw": [rh, rw], "ltrb_padding": [left, top, right, bottom],
            "native_unpadded_hw": [ih - 2 * top, iw - 2 * left]}


def project_gt(mask, cache):
    """Original COCO binary mask -> actual resized/padded image -> proto grid."""
    import torch
    import torch.nn.functional as functional
    info = geometry(cache)
    target = torch.as_tensor(mask.copy(), dtype=torch.float32)
    if list(target.shape) != info["original_hw"]:
        raise ValueError("GT/cache image shape mismatch")
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("COCO GT must be binary")
    resized = functional.interpolate(target[None, None], info["resized_hw"], mode="nearest")
    left, top, right, bottom = info["ltrb_padding"]
    padded = functional.pad(resized, (left, right, top, bottom), value=0)
    if list(padded.shape[-2:]) != info["input_hw"]:
        raise ValueError("GT padded shape changed")
    return functional.interpolate(padded, list(cache["proto"].shape[-2:]), mode="nearest")[0, 0]


def predicted_roi(cache, rank):
    """Pinned crop_mask's integer-pixel support, mapped to the proto grid."""
    import torch
    ph, pw = cache["proto"].shape[-2:]
    ih, iw = cache["input_hw"]
    box = cache["boxes_input"][rank].detach().cpu().float()
    box = box * torch.tensor([pw / iw, ph / ih, pw / iw, ph / ih])
    x, y = torch.arange(pw), torch.arange(ph)
    return ((y[:, None] >= box[1]) & (y[:, None] < box[3])
            & (x[None, :] >= box[0]) & (x[None, :] < box[2]))


def full_rle_iou(prediction, ground_truth):
    """The full original-image GT contributes to the union, including outside ROI."""
    from pycocotools import mask as masks
    return float(masks.iou([prediction], [ground_truth], [0])[0, 0])
