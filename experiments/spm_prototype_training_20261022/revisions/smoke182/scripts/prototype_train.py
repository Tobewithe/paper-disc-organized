"""Train a real+Blender local projector, then build a frozen prototype memory."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pycocotools import mask as cm


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for z in iter(lambda: f.read(1 << 20), b""): h.update(z)
    return h.hexdigest()


def dump(path, x):
    Path(path).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def letterbox(image, ids, bad=None):
    h, w = image.shape[:2]; scale = 640 / max(h, w)
    hh, ww = round(h * scale), round(w * scale); top, left = (640-hh)//2, (640-ww)//2
    rgb = np.full((640, 640, 3), 114, np.uint8)
    target = np.zeros((640, 640), np.int32); valid = np.zeros((640, 640), np.uint8)
    rgb[top:top+hh, left:left+ww] = cv2.resize(image, (ww, hh), interpolation=cv2.INTER_LINEAR)
    target[top:top+hh, left:left+ww] = cv2.resize(ids, (ww, hh), interpolation=cv2.INTER_NEAREST)
    valid[top:top+hh, left:left+ww] = 1
    if bad is not None:
        region = cv2.resize(bad.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)
        valid[top:top+hh, left:left+ww] &= 1-cv2.dilate(region, np.ones((3, 3), np.uint8))
    return rgb, target, valid


def real_image(row, cfg):
    src = Path(cfg["real_root"]) / row["file_name"]
    assert sha(src) == row["source_sha256"]
    image = cv2.imread(str(src)); assert image is not None
    ids = np.zeros(image.shape[:2], np.int32); bad = np.zeros_like(ids, bool)
    for k, ann in enumerate(row["annotations"], 1):
        seg = ann["segmentation"]
        rle = cm.merge(cm.frPyObjects(seg, image.shape[0], image.shape[1])) if isinstance(seg, list) else seg
        mask = cm.decode(rle).astype(bool); bad |= mask & (ids > 0); ids[mask] = k
    return letterbox(image, ids, bad)


def synthetic_image(row, cfg):
    src, label = Path(row["rgb_path"]), Path(row["label_path"])
    assert sha(src) == row["rgb_sha256"] and sha(label) == row["label_sha256"]
    image = cv2.imread(str(src)); assert image is not None
    raw = np.load(label)["instance_id"]
    ids = np.where((raw >= 1) & (raw <= 4), raw, 0).astype(np.int32)
    ids = cv2.resize(ids, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return letterbox(image, ids)


def sample_tokens(ids, valid, rng, maximum):
    good = cv2.resize(valid.astype(np.float32), (80, 80), interpolation=cv2.INTER_AREA) >= .999
    fg = cv2.resize((ids > 0).astype(np.float32), (80, 80), interpolation=cv2.INTER_AREA)
    dominant = np.zeros((80, 80), np.float32)
    for identity in np.unique(ids):
        if identity > 0:
            dominant = np.maximum(dominant, cv2.resize((ids == identity).astype(np.float32), (80, 80), interpolation=cv2.INTER_AREA))
    selected, labels = [], []
    for cls, mask in [(1, (fg >= .8) & (dominant >= .8) & good), (0, (fg <= .02) & good)]:
        ix = np.flatnonzero(mask.ravel())
        if len(ix):
            ix = rng.choice(ix, min(maximum, len(ix)), replace=False)
            selected.extend(ix.tolist()); labels.extend([cls] * len(ix))
    return np.asarray(selected, np.int64), np.asarray(labels, np.int64)


def state_hash(model):
    h = hashlib.sha256()
    for k, t in sorted(model.state_dict().items()): h.update(k.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def encoder(cfg, device):
    from ultralytics import YOLO
    import ultralytics
    assert ultralytics.__version__ == "8.4.100"
    assert sha(cfg["weights"]) == cfg["weights_sha256"]
    assert sha(Path(ultralytics.__file__).parent / "nn/modules/head.py") == cfg["head_source_sha256"]
    model = YOLO(cfg["weights"]).model.to(device).float().eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def cache_features(run, cfg, manifest, smoke):
    device = "cuda"; model = encoder(cfg, device); original_hash = state_hash(model)
    tap = {}; hook = model.model[-1].register_forward_pre_hook(lambda mod, args: tap.update(f=args[0][0].detach().clone()))
    rows = manifest["rows"]
    if smoke:
        rows = [z for domain in ["real", "synthetic"] for split in ["train", "dev"]
                for z in [r for r in rows if r["domain"] == domain and r["split"] == split][:2]]
    features, pairs, labels, record = [], [], [], []
    counts = Counter()
    for i, row in enumerate(rows):
        rng = np.random.default_rng(cfg["seed"] + row["ordinal"])
        image, ids, valid = real_image(row, cfg) if row["domain"] == "real" else synthetic_image(row, cfg)
        idx, cls = sample_tokens(ids, valid, rng, cfg["tokens_per_class_image"])
        x = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float().to(device) / 255
        mask = torch.from_numpy(valid).to(device).bool()[None]
        aug = ((x-.5)*float(rng.uniform(.8, 1.2))+.5+float(rng.uniform(-.08, .08))).clamp(0, 1)
        aug = torch.where(mask, aug, x)
        with torch.no_grad(): model(torch.stack([x, aug]))
        f = tap["f"]; assert tuple(f.shape[-2:]) == (80, 80)
        pos = torch.as_tensor(idx, device=device)
        features.append(f[0].flatten(1)[:, pos].T.cpu()); pairs.append(f[1].flatten(1)[:, pos].T.cpu())
        labels.append(torch.from_numpy(cls))
        for j, label in zip(idx, cls):
            record.append({"ordinal": row["ordinal"], "source_id": row["source_id"], "domain": row["domain"],
                           "split": row["split"], "group": row["group"], "label": int(label),
                           "grid_x": int(j % 80), "grid_y": int(j // 80)})
            counts[f"{row['split']}/{row['domain']}/{label}"] += 1
        if i % 20 == 0: print("features", i+1, len(rows), dict(counts), flush=True)
    hook.remove(); assert state_hash(model) == original_hash
    data = {"a": torch.cat(features).float(), "b": torch.cat(pairs).float(), "labels": torch.cat(labels), "records": record,
            "encoder_state_sha256": original_hash, "weights_sha256": cfg["weights_sha256"], "counts": dict(counts)}
    torch.save(data, run / "FEATURES.pt")
    dump(run / "FEATURE_SUMMARY.json", {"images": len(rows), "tokens": len(record), "channels": data["a"].shape[1],
          "counts": dict(counts), "encoder_unchanged": True, "encoder_state_sha256": original_hash,
          "gpu": torch.cuda.get_device_name(0)})
    del model
    return data


class Projector(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.linear = nn.Linear(c, d); self.fg_head = nn.Linear(d, 1)

    def forward(self, x): return self.linear(x)


def losses(model, a, b, y):
    ha, hb = model(a), model(b)
    inv = F.mse_loss(ha, hb)
    var = (F.relu(1-torch.sqrt(ha.var(0, unbiased=False)+1e-4)).mean()+F.relu(1-torch.sqrt(hb.var(0, unbiased=False)+1e-4)).mean())/2
    covloss = ha.new_zeros(())
    for h in [ha, hb]:
        hc = h-h.mean(0); cov = hc.T@hc/max(len(h)-1, 1)
        covloss = covloss+(cov.square().sum()-cov.diagonal().square().sum())/h.shape[1]/2
    fg = (F.binary_cross_entropy_with_logits(model.fg_head(ha).flatten(), y)+F.binary_cross_entropy_with_logits(model.fg_head(hb).flatten(), y))/2
    return 25*inv+25*var+covloss+fg, {"invariance": inv, "variance": var, "covariance": covloss, "fg_bce": fg}


def train_projector(run, cfg, data, smoke, record):
    device = "cuda"; torch.manual_seed(cfg["seed"])
    rows = data["records"]; train = torch.tensor([r["split"] == "train" for r in rows])
    assert train.any()
    mu = data["a"][train].mean(0); std = data["a"][train].std(0).clamp_min(.01)
    a = ((data["a"]-mu)/std).to(device); b = ((data["b"]-mu)/std).to(device)
    y = data["labels"].float().to(device)
    pools = []
    for domain in ["real", "synthetic"]:
        for label in [0, 1]:
            ix = [i for i, r in enumerate(rows) if r["split"] == "train" and r["domain"] == domain and r["label"] == label]
            assert len(ix) >= 32, (domain, label, len(ix))
            pools.append(torch.tensor(ix, device=device))
    model = Projector(a.shape[1], cfg["dimension"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    steps = cfg["smoke_steps"] if smoke else cfg["optimizer_steps"]
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    initial_sha = state_hash(model); history = []; grad_seen = False
    with open(run / "TRAIN_LOG.jsonl", "w") as log:
        for step in range(1, steps+1):
            idx = torch.cat([p[torch.randint(len(p), (128,), generator=generator, device=device)] for p in pools])
            total, parts = losses(model, a[idx], b[idx], y[idx]); assert torch.isfinite(total)
            opt.zero_grad(); total.backward()
            assert torch.isfinite(model.linear.weight.grad).all()
            grad_seen |= bool(model.linear.weight.grad.abs().sum() > 0)
            nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); record["optimizer_steps"] = step
            if step == 1 or step % 100 == 0 or step == steps:
                row = {"step": step, "total": float(total.detach()), **{k: float(v.detach()) for k, v in parts.items()}}
                log.write(json.dumps(row)+"\n"); log.flush(); history.append(row); print("train", row, flush=True)
    assert grad_seen and state_hash(model) != initial_sha
    payload = {"projector_state": {k: v.detach().cpu() for k, v in model.linear.state_dict().items()},
               "auxiliary_head_state": {k: v.detach().cpu() for k, v in model.fg_head.state_dict().items()},
               "input_mean": mu, "input_std": std, "input_channels": a.shape[1], "dimension": cfg["dimension"],
               "optimizer_steps": steps, "encoder_weights_sha256": cfg["weights_sha256"],
               "input_manifest_sha256": cfg["input_manifest_sha256"], "feature_tap": "P3_before_Segment26"}
    torch.save(payload, run / "projector.pt")
    with torch.no_grad():
        ha, hb = model(a), model(b); q = F.normalize(ha, dim=1); qb = F.normalize(hb, dim=1)
        fg_score = model.fg_head(ha).flatten().sigmoid()
    metrics = {}
    for split in ["train", "dev"]:
        for domain in ["real", "synthetic"]:
            mask = torch.tensor([r["split"] == split and r["domain"] == domain for r in rows], device=device)
            if mask.any():
                metrics[f"{split}/{domain}"] = {"tokens": int(mask.sum()), "fg_accuracy_at_0_5": float(((fg_score[mask] >= .5) == y[mask].bool()).float().mean()),
                     "paired_cosine": float((q[mask]*qb[mask]).sum(1).mean()), "mean_coordinate_std": float(ha[mask].std(0).mean())}
    return model, payload, q.cpu(), metrics, history


def discover(q, rows, cfg, label):
    ids = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == label]
    rng = np.random.default_rng(cfg["seed"]+label); rng.shuffle(ids)
    vectors = q.numpy(); centers = []; sums = []; cap_hits = 0
    for ix in ids:
        z = vectors[ix]
        if not centers: centers.append(z.copy()); sums.append(z.copy()); continue
        scores = np.asarray(centers)@z; nearest = int(scores.argmax())
        if 2-2*float(scores[nearest]) > cfg["cluster_squared_distance_threshold"]:
            if len(centers) < cfg["maximum_per_partition"]:
                centers.append(z.copy()); sums.append(z.copy()); continue
            cap_hits += 1
        sums[nearest] += z; centers[nearest] = sums[nearest]/max(float(np.linalg.norm(sums[nearest])), 1e-12)
    centers = np.asarray(centers, np.float32)
    for _ in range(cfg["reassignment_passes"]):
        assignment = (vectors[ids]@centers.T).argmax(1)
        keep = []; updated = []
        for k in range(len(centers)):
            members = [ids[j] for j in np.flatnonzero(assignment == k)]
            if len(members) >= cfg["minimum_tokens_per_centroid"] and len({rows[j]["source_id"] for j in members}) >= 2:
                keep.append(k); z = vectors[members].sum(0); updated.append(z/max(np.linalg.norm(z), 1e-12))
        assert updated, "all prototypes pruned"
        centers = np.asarray(updated, np.float32)
    assignment = (vectors[ids]@centers.T).argmax(1); cards = []
    for k in range(len(centers)):
        members = [ids[j] for j in np.flatnonzero(assignment == k)]
        ranking = sorted(members, key=lambda j: -float(vectors[j]@centers[k]))
        cards.append({"centroid": k, "tokens": len(members), "source_images": len({rows[j]["source_id"] for j in members}),
                      "domain_counts": dict(Counter(rows[j]["domain"] for j in members)),
                      "examples": [rows[j] for j in ranking[:8]]})
    return torch.from_numpy(centers), cards, {"partition": label, "K": len(centers), "training_tokens": len(ids), "capacity_hits": cap_hits,
          "quantization_cosine": float(np.max(vectors[ids]@centers.T, axis=1).mean())}


def build_memory(run, cfg, data, projector, q, metrics):
    fg, fg_cards, fg_summary = discover(q, data["records"], cfg, 1)
    bg, bg_cards, bg_summary = discover(q, data["records"], cfg, 0)
    memory = {"format_version": 1, "keys_fg": fg, "values_fg": fg.clone(), "keys_bg": bg, "values_bg": bg.clone(),
              "projector_state": projector["projector_state"], "input_mean": projector["input_mean"], "input_std": projector["input_std"],
              "input_channels": projector["input_channels"], "dimension": cfg["dimension"], "projector_sha256": sha(run / "projector.pt"),
              "encoder_weights_sha256": cfg["weights_sha256"], "feature_tap": "P3_before_Segment26", "topk": 4, "temperature": .1,
              "input_manifest_sha256": cfg["input_manifest_sha256"], "construction_split": "train_only",
              "source_domains": ["real", "synthetic"], "protocol_sha256": sha(run.parent.parent / "PROTOCOL.md"),
              "query_contract": "raw P3 -> train mean/std -> frozen linear projector -> L2 normalization -> per-partition topk"}
    torch.save(memory, run / "memory.pt")
    loaded = torch.load(run / "memory.pt", map_location="cpu", weights_only=False)
    assert torch.equal(loaded["keys_fg"], loaded["values_fg"]) and torch.equal(loaded["keys_bg"], loaded["values_bg"])
    assert torch.allclose(fg.norm(dim=1), torch.ones(len(fg)), atol=1e-5)
    assert torch.allclose(bg.norm(dim=1), torch.ones(len(bg)), atol=1e-5)
    assert all(torch.isfinite(loaded[k]).all() for k in ["keys_fg", "keys_bg", "input_mean", "input_std"])
    from memory_query import FrozenPrototypeMemory
    query = FrozenPrototypeMemory(loaded)
    dev = [i for i, r in enumerate(data["records"]) if r["split"] == "dev"][:512]
    assert dev
    with torch.no_grad(): response = query(data["a"][dev])
    assert torch.isfinite(response["foreground"]).all() and response["foreground"].shape == (len(dev), cfg["dimension"])
    expected = F.normalize(F.linear((data["a"][dev]-projector["input_mean"])/projector["input_std"], **projector["projector_state"]), dim=1)
    assert torch.allclose(expected, q[dev], atol=1e-5, rtol=1e-4)
    dump(run / "PROTOTYPE_CARDS.json", {"foreground": fg_cards, "background": bg_cards})
    summary = {"status": "trained_and_memory_built", "projector_steps": projector["optimizer_steps"],
               "foreground": fg_summary, "background": bg_summary, "representation_metrics": metrics,
               "memory_sha256": sha(run / "memory.pt"), "projector_sha256": sha(run / "projector.pt"),
               "reload_query_passed": True, "query_samples": len(dev), "unit_norm_passed": True,
               "not_measured": ["YOLO_mask_AP_gain", "mechanism_causality", "physical_pen_generalization"]}
    dump(run / "VALIDATION.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["smoke", "train"]); p.add_argument("--root", required=True)
    args = p.parse_args(); root = Path(args.root); cfg = json.loads((root / "CONFIG.json").read_text())
    manifest = json.loads((root / "INPUT_MANIFEST.json").read_text()); assert sha(root / "INPUT_MANIFEST.json") == cfg["input_manifest_sha256"]
    freeze = json.loads((root / "FREEZE_MANIFEST.json").read_text())
    for name, digest in freeze["files"].items(): assert sha(root / name) == digest, name
    lock = root / ("LOCK_"+args.stage); fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    run = root / "runs" / ("RUN_SPM_PROTO_"+args.stage.upper()+"_"+uuid.uuid4().hex[:12]); run.mkdir(parents=True)
    meta = {"study_id": "STUDY_SPM_PROTOTYPE_TRAINING_20261022", "run_id": run.name, "stage": args.stage,
            "started_at": utc(), "execution_status": "running", "scientific_status": "artifact_training_not_AP_evaluation",
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "executor": os.uname().nodename, "pid": os.getpid(),
            "command": sys.argv, "code_commit": freeze["git_head"], "source_sha256": sha(__file__),
            "freeze_manifest_sha256": sha(root/"FREEZE_MANIFEST.json"), "protocol_sha256": sha(root/"PROTOCOL.md"),
            "config_sha256": sha(root/"CONFIG.json"), "input_manifest_sha256": cfg["input_manifest_sha256"],
            "optimizer_steps": 0, "detector_optimizer_steps": 0, "prototype_means_optimization": "offline_spherical_clustering_after_projection_training"}
    dump(run/"run.json", meta); dump(run/"EVAL_CONTRACT.json", cfg); print("RUN", run.name, flush=True)
    t = time.monotonic()
    try:
        torch.set_num_threads(4); cv2.setNumThreads(4); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        data = cache_features(run, cfg, manifest, args.stage == "smoke")
        model, projector, q, metrics, history = train_projector(run, cfg, data, args.stage == "smoke", meta)
        summary = build_memory(run, cfg, data, projector, q, metrics)
        summary["elapsed_seconds"] = time.monotonic()-t
        meta.update(execution_status="completed", ended_at=utc(), exit_code=0)
        dump(run/"SUMMARY.json", summary); dump(run/"run.json", meta)
        dump(run/"DECISION.json", {"status": "artifact_validated" if args.stage == "train" else "smoke_passed",
              "claim_scope": "trained projector and source-traceable frozen dual-domain memory, no detector-utility claim",
              "next_direction": "wire validated frozen memory into YOLO P3 with zero-residual equivalence"})
        dump(root/(args.stage.upper()+"_POINTER.json"), {"run_id": run.name, "memory_sha256": sha(run/"memory.pt")})
        print("COMPLETE", json.dumps(summary), flush=True)
    except BaseException:
        (run/"ERROR.log").write_text(traceback.format_exc()); meta.update(execution_status="failed", ended_at=utc(), exit_code=1)
        dump(run/"run.json", meta); raise
    finally:
        artifact = {str(p.relative_to(run)): {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in run.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
        dump(run/"ARTIFACT_MANIFEST.json", artifact); lock.unlink()


if __name__ == "__main__": main()
