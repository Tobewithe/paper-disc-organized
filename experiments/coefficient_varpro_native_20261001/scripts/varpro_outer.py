from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops


def load_input(path: Path, cache: dict) -> torch.Tensor:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    rh, rw = (float(x) for x in cache["ratio_pad"][0])
    left = int(round(float(cache["ratio_pad"][1][0])))
    top = int(round(float(cache["ratio_pad"][1][1])))
    nh, nw = round(im.shape[0] * rh), round(im.shape[1] * rw)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = im
    x = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).float()
    return x.div_(255).unsqueeze(0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def capture_features(model, tensor):
    head = model.model[-1]
    captured = {}

    def hook(_module, inputs):
        captured["x"] = [v.detach() for v in inputs[0]]

    handle = head.register_forward_pre_hook(hook)
    with torch.inference_mode():
        model(tensor)
    handle.remove()
    # Tensors returned by inference_mode carry the inference-tensor tag and
    # cannot be saved for backward when the trainable coefficient branch uses
    # them. Clone into ordinary tensors before the outer gradient pass.
    return [v.clone() for v in captured["x"]]


def box_area(box):
    # Official reference loss uses normalized target-box area, not row["area"].
    return float((((box[2:] - box[:2]) / 640.0).prod() * 640.0 * 640.0).item())


def candidate_loss(c, cache, k, proto, masks, gain, lam):
    row = cache["rows"][k]
    box = cache["target_boxes"][k].float()
    support = ops.crop_mask(torch.ones((1, 640, 640), device=proto.device), box[None].to(proto.device))[0].bool()
    p = proto[:, support].T
    y = (masks[support] == int(cache["owners"][k]) + 1).float()
    z = p @ c
    bce = float(gain) * (F.softplus(z) - y * z).sum() / box_area(box)
    return bce, support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--c-final", type=Path, required=True)
    ap.add_argument("--readout", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--outer-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum-images", type=int, default=8)
    ap.add_argument("--lambda-eval", type=float, default=0.003)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("refusing CPU execution")

    index = json.loads((args.cache / "INDEX.json").read_text(encoding="utf-8"))
    items = list(index["fit"])
    if args.max_images:
        items = items[: args.max_images]
    n_total = sum(len(torch.load(args.cache / "images" / f"{int(i['image_id']):012d}.pt", map_location="cpu", weights_only=False)["rows"]) for i in items)

    yolo = YOLO(str(args.weights))
    model = yolo.model.float().to(device).eval()
    head = model.model[-1]
    cstate = torch.load(args.c_final, map_location="cpu", weights_only=False)
    head.one2one_cv4.load_state_dict(cstate["cv4"], strict=True)
    readout = torch.load(args.readout, map_location="cpu", weights_only=False)
    readouts = [{"W": r["W"].float().to(device), "b": r["b"].float().to(device)} for r in readout["readouts"]]
    for r in readouts:
        r["W"].requires_grad_(False)
        r["b"].requires_grad_(False)

    # Only the native pre-readout convolutions are trainable; all other model
    # parameters and buffers remain frozen. BatchNorms stay in eval mode.
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for branch in head.one2one_cv4:
        for layer in branch[:-1]:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad_(True)
                trainable.append(p)
    if not trainable:
        raise RuntimeError("no trainable pre-readout parameters")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    gain_default = 9.83241
    logs = []
    started = time.monotonic()

    for epoch in range(args.outer_epochs):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        images_since_step = 0
        for j, item in enumerate(items):
            iid = int(item["image_id"])
            cache = torch.load(args.cache / "images" / f"{iid:012d}.pt", map_location="cpu", weights_only=False)
            feats = capture_features(model, load_input(args.data / "images" / "train2017" / f"{iid:012d}.jpg", cache).to(device))
            proto = F.interpolate(cache["proto"].float()[None].to(device), (640, 640), mode="bilinear", align_corners=False)[0]
            masks = cache["masks"].to(device)
            image_loss = torch.zeros((), device=device)
            for k, row in enumerate(cache["rows"]):
                level = int(row["level"])
                # Keep gradients through the native pre-readout branch only.
                hidden = head.one2one_cv4[level][:-1](feats[level]).reshape(1, 64, -1).squeeze(0).T
                start = sum(int(feats[l].shape[-2] * feats[l].shape[-1]) for l in range(level))
                h = hidden[int(row["raw_id"]) - start]
                c = F.linear(h, readouts[level]["W"], readouts[level]["b"])
                c0 = cache["coeff"][int(row["raw_id"])].float().to(device)
                bce, _ = candidate_loss(c, cache, k, proto, masks, cache.get("segmentation_gain", gain_default), args.lambda_eval)
                image_loss = image_loss + bce + args.lambda_eval * 0.5 * (c - c0).square().sum()
                count += 1
            image_loss = image_loss / max(n_total, 1)
            image_loss.backward()
            total += float(image_loss.detach().cpu()) * max(n_total, 1)
            images_since_step += 1
            if images_since_step >= args.accum_images or j + 1 == len(items):
                grad = float(torch.nn.utils.clip_grad_norm_(trainable, float("inf")))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                images_since_step = 0
            if (j + 1) % 25 == 0 or j + 1 == len(items):
                print(json.dumps({"stage": "outer_feature_update", "epoch": epoch + 1, "images": j + 1, "total": len(items)}), flush=True)
        logs.append({"epoch": epoch + 1, "objective_estimate": total / max(count, 1), "candidates": count, "last_grad_norm": grad, "elapsed_s": time.monotonic() - started})
        print(json.dumps({"stage": "outer_epoch", **logs[-1]}), flush=True)

    final = {
        "mode": "varpro_outer_pre_readout",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "images": len(items),
        "candidates": n_total,
        "outer_epochs": args.outer_epochs,
        "lr": args.lr,
        "accum_images": args.accum_images,
        "lambda_eval": args.lambda_eval,
        "weights_sha256": sha256(args.weights),
        "logs": logs,
        "cv4": {k: v.detach().cpu() for k, v in head.one2one_cv4.state_dict().items()},
    }
    torch.save(final, args.out / "VARPRO_OUTER.pt")
    (args.out / "SUMMARY.json").write_text(json.dumps({"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "result": {k: v for k, v in final.items() if k not in ("cv4",)}}, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"stage": "complete", "summary": str(args.out / "SUMMARY.json")}), flush=True)


if __name__ == "__main__":
    main()
