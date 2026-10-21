"""Shared frozen V2 replay; no optimizer or trainable parameters."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sources"))
import numpy as np
import torch
from pycocotools import mask as mask_api
from native_stream import NativeBatches, batch_identity
from native_output_bridge import decode_native
from topology import Context
from matching import associate, boxes_iou


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def lines(p):
    return [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def append(p, x):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, allow_nan=False) + "\n")


def translate(a, dx, dy, fill=0):
    h, w = a.shape[:2]
    out = np.full_like(a, fill)
    sx, sy = max(0, -dx), max(0, -dy)
    tx, ty = max(0, dx), max(0, dy)
    nx, ny = w - abs(dx), h - abs(dy)
    out[ty:ty+ny, tx:tx+nx] = a[sy:sy+ny, sx:sx+nx]
    return out


def rle(a):
    x = mask_api.encode(np.asfortranarray(a.astype(np.uint8)))
    return dict(size=x["size"], counts=x["counts"].decode())


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def cpu_cache(c):
    fields = ("image_id", "proto", "c0", "boxes_input", "scores", "classes",
              "raw_ids", "raw_locations", "input_hw", "native_decode_hw",
              "original_hw", "ratio_pad", "batch_index", "batch_position",
              "image_sha256", "input_tensor_sha256")
    return {k: (c[k].detach().cpu().clone() if torch.is_tensor(c[k]) else c[k]) for k in fields} | {"device": "cuda:0"}


def continuous(c, coefficients=None, proto=None):
    """Unthresholded, uncropped native logit field in original coordinates."""
    from ultralytics.utils import ops
    p = c["proto"].to("cuda:0") if proto is None else proto.to("cuda:0")
    coeff = c["c0"].to("cuda:0") if coefficients is None else coefficients.to("cuda:0")
    logits = (coeff @ p.float().view(p.shape[0], -1)).view(-1, *p.shape[1:])
    value = ops.scale_masks(logits[None], c["input_hw"])[0]
    return ops.scale_masks(value[None], c["original_hw"], ratio_pad=c["ratio_pad"])[0]


class Inputs:
    def __init__(self):
        self.cfg = read(ROOT / "CONFIG.json")
        self.base = Path(self.cfg.pop("v2_run"))
        self.contract = read(self.base / "prepared/DATA_CONTRACT.json")
        self.frames = {x["image_id"]: x for x in lines(self.base / "dataset/FRAMES.jsonl")}
        self.identity = lines(self.base / "NATIVE_IDENTITY.jsonl")
        self.pred = {}
        self.id_to_rank = {}
        for p in read(self.base / "PREDICTIONS.json"):
            group = self.pred.setdefault(p["image_id"], [])
            self.id_to_rank[p["output_detection_id"]] = len(group)
            group.append(p)
        self.timeline = {(t["image_id"], t["track_id"]): t for t in lines(self.base / "TIMELINE.jsonl")}

    def labels(self, iid):
        with np.load(self.base / f"dataset/labels/{iid:012d}.npz") as z:
            return z["instance_id"].copy()

    def prepared(self, run, offset=(0, 0)):
        import cv2
        dx, dy = offset
        prep = run / f"prepared_{dx}_{dy}"; prep.mkdir()
        contract = copy.deepcopy(self.contract)
        paths = []
        for b in contract["val_batches"]:
            for iid in b["image_ids"]:
                src = self.base / f"dataset/images/{iid:012d}.png"
                assert sha(src) == contract["val_image_hashes"][str(iid)]
                if dx or dy:
                    dst = prep / "images" / src.name
                    dst.parent.mkdir(exist_ok=True)
                    original = cv2.imread(str(src))
                    shifted = translate(original, dx, dy, 114)
                    back = translate(shifted, -dx, -dy, 114)
                    assert np.array_equal(back[4:-4, 4:-4], original[4:-4, 4:-4])
                    assert cv2.imwrite(str(dst), shifted)
                    contract["val_image_hashes"][str(iid)] = sha(dst)
                    paths.append(dst)
                else:
                    paths.append(src)
        (prep / "val.txt").write_text("".join(str(p) + "\n" for p in paths), encoding="utf-8")
        save(prep / "DATA_CONTRACT.json", contract)
        return prep

    def check_base(self, bi, caches, rows, features):
        got = batch_identity(caches, rows, features)
        assert got == self.identity[bi], ("V2_tensor_identity", bi)
        for c, group in zip(caches, rows):
            expected = [{k: v for k, v in p.items() if k != "output_detection_id"} for p in self.pred[c["image_id"]]]
            assert group == expected, ("V2_native_predictions", c["image_id"])
        return got

    def matches(self, iid, rows, offset=(0, 0)):
        dx, dy = offset
        gt = copy.deepcopy(self.frames[iid]["records"])
        for g in gt:
            g["bbox"] = [v + (dx if j % 2 == 0 else dy) for j, v in enumerate(g["bbox"])]
        operating = [(j, p) for j, p in enumerate(rows) if p["score"] > .25]
        mapping = associate(gt, [p for _, p in operating])
        return {g["track_id"]: (operating[mapping[j]][0] if j in mapping else None) for j, g in enumerate(gt)}


def resource_check():
    import resource
    gpu = torch.cuda.max_memory_reserved() / 2**30
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    assert gpu <= 12 and ram < 15, ("resource_limit", gpu, ram)
    return {"peak_cuda_reserved_GiB": gpu, "peak_host_RSS_GiB": ram}
