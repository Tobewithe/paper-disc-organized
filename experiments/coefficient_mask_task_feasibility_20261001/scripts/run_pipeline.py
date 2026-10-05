"""Fixed-prototype mask-task feasibility experiment.

This is deliberately a diagnostic pipeline: it keeps the official TAL identities,
the original h/P/c0 and native mask decode fixed, and only solves temporary
coefficient/affine witnesses.  The implementation records conservative inner
quality envelopes and uses Clarabel for the small convex QPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import clarabel
from pycocotools.coco import COCO
# Prefer the execution package's frozen Ultralytics source tree over the
# older site-packages copy on the laptop.
import sys
sys.path.insert(0, r"D:\coco_wire\py")
from ultralytics.utils import ops


SEED = 20261001
N_EXPECTED = 6058
LEVEL_EXPECTED = {0: 2000, 1: 2298, 2: 1760}
Q_THR = 0.75
EPS_Q = 0.01
KAPPA = 0.10
MULT = 4.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def jline(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def load_image(cache: Path, iid: int):
    return torch.load(cache / "images" / f"{iid:012d}.pt", map_location="cpu", weights_only=False)


def decode_pad(proto: torch.Tensor, coeff: torch.Tensor, boxes: torch.Tensor):
    """Native 8.4.100 process_mask(upsample=True), one image in 640 coords."""
    if coeff.ndim == 1:
        coeff = coeff[None]
    low = (coeff.float() @ proto.float().reshape(proto.shape[0], -1)).reshape(-1, proto.shape[-2], proto.shape[-1])
    up = F.interpolate(low[None], (640, 640), mode="bilinear")[0]
    return ops.crop_mask(up, boxes.float())


def scale_binary(mask: torch.Tensor, x):
    """Map a padded 640 binary mask stack back through the frozen letterbox."""
    if mask.ndim == 2:
        mask = mask[None]
    out = ops.scale_masks(mask[:, None].float(), tuple(int(v) for v in x["original_shape"]), ratio_pad=x["ratio_pad"])
    return out[:, 0] > 0.5


def iou_pair(pred: torch.Tensor, gt: torch.Tensor):
    inter = (pred & gt).flatten(1).sum(1).float()
    union = (pred | gt).flatten(1).sum(1).float()
    return (inter / union.clamp_min(1)).detach().cpu().numpy()


def configure_clarabel(max_iter: int, time_limit: float):
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.max_iter = int(max_iter)
    s.tol_gap_abs = 1e-8
    s.tol_gap_rel = 1e-8
    s.tol_feas = 1e-8
    try:
        s.time_limit = float(time_limit)
    except Exception:
        pass
    return s


def solve_qp(P, G, rhs, max_iter=200, time_limit=600.0):
    """min .5*x'Px subject to Gx >= rhs, using Clarabel's -Gx+s=-rhs form."""
    n = P.shape[0]
    if G.shape[0] == 0:
        return np.zeros(n), "ZeroConstraints", 0.0, None
    P = sp.csc_matrix(P, dtype=np.float64)
    A = sp.csc_matrix(-G, dtype=np.float64)
    b = -np.asarray(rhs, dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)
    cones = [clarabel.NonnegativeConeT(G.shape[0])]
    settings = configure_clarabel(max_iter, time_limit)
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    sol = solver.solve()
    z = np.asarray(sol.x, dtype=np.float64)
    status = str(sol.status)
    obj = float(0.5 * z.dot(P.dot(z))) if z.size else 0.0
    return z, status, obj, sol


def recompute_dual_bound(H, G, rhs, sol):
    """Recompute a conservative dual lower bound in the original QP.

    Clarabel solves ``-G theta + s = -rhs``.  Its cone dual ``z`` therefore
    gives a nonnegative multiplier for ``G theta >= rhs``.  We clip tiny
    numerical negatives, then evaluate the dual objective with the original
    (unregularized) Hessian.  The row-major Kronecker Hessian is inverted via
    its 65x65 design block, avoiding a dense 2080x2080 inverse.
    """
    if sol is None or G.shape[0] == 0:
        return 0.0
    reported = getattr(sol, "obj_val_dual", float("nan"))
    best = float(reported) if np.isfinite(reported) else 0.0
    if not hasattr(sol, "z"):
        return max(0.0, best)
    y = np.maximum(np.asarray(sol.z, dtype=np.float64), 0.0)
    if y.size != G.shape[0] or not np.all(np.isfinite(y)):
        return max(0.0, best)
    v = np.asarray(G, dtype=np.float64).T @ y
    pbase = np.asarray(H[0::32, 0::32], dtype=np.float64)
    vmat = v.reshape(65, 32)
    try:
        umat = np.linalg.solve(pbase, vmat)
    except np.linalg.LinAlgError:
        umat = np.linalg.pinv(pbase, rcond=1e-12) @ vmat
    val = float(np.asarray(rhs, dtype=np.float64).dot(y) - 0.5 * np.sum(vmat * umat))
    if np.isfinite(val):
        best = max(best, val)
    return max(0.0, best)


def norm_hash(x):
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]


class Pipeline:
    def __init__(self, root: Path, out: Path):
        self.root = root
        self.out = out
        self.cache = root / "data" / "official_tal_affine_20260930" / "runs" / "official_cache"
        self.oracle_path = self.cache / "ORACLE_fit.pt"
        self.rows = []
        self.images = []
        self.by_level = {0: [], 1: [], 2: []}
        self.stats = {}
        self.v_ref = 0.0
        self.budget = 0.0
        self.timings = {}
        self.oracle_map = {}
        self.limit_images = None
        self.N = N_EXPECTED
        self.resume_prepared = False
        self.coco = None

    def log(self, name, **kw):
        print(json.dumps({"stage": name, **kw}, ensure_ascii=False), flush=True)

    def register(self):
        t = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        model = self.root / "models" / "yolo26m-seg.pt"
        ann_path = self.root / "data" / "annotations" / "instances_train2017.json"
        if not ann_path.exists():
            raise RuntimeError(f"raw COCO annotation file missing: {ann_path}")
        self.coco = COCO(str(ann_path))
        index = json.loads((self.cache / "INDEX.json").read_text(encoding="utf-8"))
        counts = {s: {"images": len(index[s]), "candidates": int(sum(r["n"] for r in index[s]))} for s in ("fit", "dev", "val")}
        level_counts = {0: 0, 1: 0, 2: 0}
        for it in index["fit"]:
            x = load_image(self.cache, int(it["image_id"]))
            for r in x["rows"]:
                level_counts[int(r["level"])] += 1
        identity = {
            "host": platform.node(), "device": str(DEVICE), "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics_source": "D:/coco_wire/py/ultralytics (8.4.100 expected)",
            "model_path": str(model), "model_sha256": sha256(model) if model.exists() else None,
            "expected_model_sha256": "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5",
            "cache": str(self.cache), "cache_index_sha256": sha256(self.cache / "INDEX.json"),
            "counts": counts, "fit_level_counts": level_counts,
            "oracle_sha256": sha256(self.oracle_path) if self.oracle_path.exists() else None,
            "raw_coco_annotations": str(ann_path),
            "raw_coco_annotations_sha256": sha256(ann_path),
            "clarabel": getattr(clarabel, "__version__", "unknown"), "seed": SEED,
        }
        dump(self.out / "MODEL_AND_INPUT_IDENTITY.json", identity)
        reuse = {
            "coefficient_official_tal_affine_20260930": {"reused": True, "scope": "official cache, fixed h/P/c0/boxes/geometry", "compatible": True},
            "margin_preserve_readout_pilot_20260913": {"reused": False, "scope": "historical comparison only", "compatible": False},
            "fixed_prototype_split_20260913": {"reused": False, "scope": "historical comparison only", "compatible": False},
            "oracle": {"reused": True, "path": str(self.oracle_path), "target": "official reference BCE + lambda=0.003 displacement"},
        }
        dump(self.out / "REUSE_MAP.json", reuse)
        if identity["model_sha256"].lower() != identity["expected_model_sha256"]:
            raise RuntimeError("original model SHA256 mismatch")
        if self.limit_images is None and (counts["fit"]["images"] != 796 or counts["fit"]["candidates"] != N_EXPECTED or level_counts != LEVEL_EXPECTED):
            raise RuntimeError(f"official fit manifest mismatch: {counts}, levels={level_counts}")
        if not self.oracle_path.exists():
            raise RuntimeError("ORACLE_fit.pt missing")
        oracle = torch.load(self.oracle_path, map_location="cpu", weights_only=False)
        self.oracle_map = {(int(r["image_id"]), int(r["annotation_id"]), int(r["raw_id"])): d.double() for r, d in zip(oracle["identities"], oracle["delta"])}
        self.images = [int(z["image_id"]) for z in index["fit"]]
        if self.limit_images is not None:
            self.images = self.images[:int(self.limit_images)]
        self.timings["register_s"] = time.time() - t
        self.log("register", counts=counts, levels=level_counts, oracle=len(self.oracle_map))

    def collect_stats(self):
        t = time.time(); hs = {0: [], 1: [], 2: []}; self.rows = []
        for iid in self.images:
            x = load_image(self.cache, iid)
            for k, r in enumerate(x["rows"]):
                lev = int(r["level"]); rid = int(r["raw_id"])
                h = x["h"][rid].numpy().astype(np.float64)
                hs[lev].append(h)
                rec = {"image_id": iid, "row_index": k, "annotation_id": int(r["annotation_id"]), "raw_id": rid, "level": lev, "gt_index": int(r["gt_index"]), "box_iou": float(r.get("box_iou", float("nan")))}
                self.rows.append(rec); self.by_level[lev].append(len(self.rows) - 1)
        for lev in (0, 1, 2):
            a = np.asarray(hs[lev], dtype=np.float64)
            mu = a.mean(0); sd = a.std(0); sd[sd < 1e-8] = 1.0
            self.stats[lev] = (mu, sd)
        self.N = len(self.rows) if self.limit_images is not None else N_EXPECTED
        dump(self.out / "FIT_STATS.json", {str(l): {"mean": self.stats[l][0].tolist(), "std": self.stats[l][1].tolist(), "n": len(self.by_level[l])} for l in (0,1,2)})
        self.timings["collect_stats_s"] = time.time() - t

    def raw_gt_masks(self, iid, rows, x):
        """Return the original COCO instance masks and their padded 640 versions.

        The cache's ``masks`` tensor is the official training raster.  Quality
        thresholds in this protocol are defined on the original COCO masks,
        so all q values and envelope checks use this exact annotation source.
        """
        oh, ow = (int(x["original_shape"][0]), int(x["original_shape"][1]))
        gain = min(640.0 / oh, 640.0 / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        left = round(float(x["ratio_pad"][1][0]) - 0.1)
        top = round(float(x["ratio_pad"][1][1]) - 0.1)
        orig_list = []
        pad_list = []
        for r in rows:
            ann_id = int(r["annotation_id"])
            if ann_id not in self.coco.anns:
                raise RuntimeError(f"annotation_id {ann_id} missing from raw COCO")
            arr = self.coco.annToMask(self.coco.anns[ann_id]).astype(np.bool_)
            if arr.shape != (oh, ow):
                raise RuntimeError(f"raw COCO shape mismatch image={iid} ann={ann_id}: {arr.shape} != {(oh, ow)}")
            orig = torch.from_numpy(arr).to(DEVICE)
            resized = F.interpolate(orig.float()[None, None], (nh, nw), mode="nearest")[0, 0] > 0.5
            padded = F.pad(resized[None, None].float(), (left, 640 - nw - left, top, 640 - nh - top))[0, 0] > 0.5
            orig_list.append(orig)
            pad_list.append(padded)
        return torch.stack(orig_list), torch.stack(pad_list)

    def process_image(self, iid: int, make_regions=True):
        x = load_image(self.cache, iid)
        proto = x["proto"].to(DEVICE)
        rows = x["rows"]; n = len(rows)
        rid = torch.tensor([int(r["raw_id"]) for r in rows], dtype=torch.long)
        lev = torch.tensor([int(r["level"]) for r in rows], dtype=torch.long)
        boxes = x["boxes"][rid].to(DEVICE)
        c0 = x["coeff"][rid].to(DEVICE).float()
        oracle_delta = torch.stack([self.oracle_map[(iid, int(r["annotation_id"]), int(r["raw_id"]))] for r in rows]).to(DEVICE).float()
        cstar = c0 + oracle_delta
        p0 = decode_pad(proto, c0, boxes) > 0
        ps = decode_pad(proto, cstar, boxes) > 0
        g0, ypad = self.raw_gt_masks(iid, rows, x)
        p0o = scale_binary(p0, x); pso = scale_binary(ps, x)
        q0 = iou_pair(p0o, g0); qs = iou_pair(pso, g0)
        up_proto = F.interpolate(proto[None].float(), (640, 640), mode="bilinear")[0]
        regions = []
        policy = []
        for k, r in enumerate(rows):
            yy, xx = torch.where((torch.arange(640, device=DEVICE)[None, :] >= boxes[k, 0]) & (torch.arange(640, device=DEVICE)[None, :] < boxes[k, 2]) & ((torch.arange(640, device=DEVICE)[:, None] >= boxes[k, 1]) & (torch.arange(640, device=DEVICE)[:, None] < boxes[k, 3])))
            pix = yy * 640 + xx
            avec = up_proto[:, yy, xx].T.float()
            zref = avec @ cstar[k]
            bref = zref > 0
            y = ypad[k].flatten()[pix]
            q0k = float(q0[k]); qsk = float(qs[k])
            if q0k < Q_THR and qsk >= Q_THR:
                group = "R"; cref = cstar[k]; qreq = max(Q_THR, qsk - EPS_Q)
            elif q0k >= Q_THR:
                group = "S"; cref = c0[k]; qreq = max(Q_THR, q0k - EPS_Q)
            else:
                group = "F_other"; cref = c0[k]; qreq = max(0.0, q0k - EPS_Q)
            zref2 = (avec @ cref)
            bref2 = zref2 > 0
            # Numeric ambiguity is forced into the initial free set.
            amb = torch.where(zref2.abs() < 1e-6)[0].detach().cpu().numpy().astype(np.int64)
            score = (zref2.abs() / avec.norm(dim=1).clamp_min(1e-12)).detach().cpu().numpy()
            order_rest = np.asarray([i for i in np.argsort(score, kind="stable") if i not in set(amb)], dtype=np.int64)
            order = np.concatenate([amb, order_rest])
            # Compute the conservative quality envelope *after* the frozen
            # binary-canvas-to-original-image map D.  Counting pixels on the
            # padded 640 canvas is not equivalent when letterbox scaling is
            # present, and can silently admit a witness below q.
            pix_np = pix.detach().cpu().numpy().astype(np.int64)
            bref_np = bref2.detach().cpu().numpy().astype(bool)
            y_orig = g0[k]
            bound_cache = {}
            base_bits = torch.zeros((2, 640 * 640), dtype=torch.bool, device=DEVICE)
            base_bits[:, pix] = bref2[None, :]
            def qbound_prefix(kp):
                kp = int(kp)
                if kp in bound_cache: return bound_cache[kp]
                bits = base_bits.clone()
                if kp:
                    fpix = pix[torch.as_tensor(order[:kp], dtype=torch.long, device=DEVICE)]
                    bits[0, fpix] = False
                    bits[1, fpix] = True
                mb = scale_binary(bits.reshape(2, 640, 640), x)
                inter = (mb[0] & y_orig).sum().item()
                fp = (mb[1] & (~y_orig)).sum().item()
                val = float(inter / max(int(y_orig.sum().item()) + int(fp), 1))
                bound_cache[kp] = val
                return val
            qprefix = [qbound_prefix(0)]
            # The envelope is monotone under the prescribed prefix release;
            # use a deterministic binary search, then retain the complete
            # tested prefix values for auditability.
            if qprefix[0] < qreq - 1e-9:
                kfree = len(amb); numeric = True
            else:
                lo, hi = 0, len(order)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if qbound_prefix(mid) >= qreq - 1e-9:
                        lo = mid
                    else:
                        hi = mid - 1
                kfree = int(lo); numeric = False
                # Include the selected endpoint and its immediate boundary in
                # the record; these are enough to audit the monotone decision.
                for kp in sorted(set([0, kfree, min(len(order), kfree + 1)])):
                    qprefix.append(qbound_prefix(kp))
            free_local = order[:kfree]
            locked_local = order[kfree:]
            lock_pix = pix.detach().cpu().numpy().astype(np.int32)[locked_local]
            # Do not materialize the full (possibly 300k+ pixel) response
            # matrix in FP64.  The QP only needs rows that remain locked;
            # ordering and envelope scans above already ran in torch FP32.
            # The prototype bank is FP32; retain this large sparse row block
            # in FP32 on disk and promote one candidate at a time in the QP
            # scanner.  This avoids two simultaneous FP64 copies at peak RAM.
            lock_a = avec[torch.as_tensor(locked_local, dtype=torch.long, device=DEVICE)].detach().cpu().numpy().astype(np.float32, copy=False)
            lock_ref = zref2.detach().cpu().numpy().astype(np.float32)[locked_local]
            lock_t = np.where(lock_ref >= 0, 1.0, -1.0).astype(np.float32)
            lock_m = (KAPPA * np.abs(lock_ref)).astype(np.float32)
            qlower = float(qbound_prefix(kfree))
            rec = self.rows[self._row_index(iid, k)]
            xh = x["h"][int(r["raw_id"])].numpy().astype(np.float64)
            mu, sd = self.stats[int(r["level"])]
            xxh = np.concatenate([(xh - mu) / sd, np.ones(1, dtype=np.float64)])
            ref_cost = float(0.5 * torch.sum((cref - c0[k]) ** 2).item())
            rec.update({"q0": q0k, "qstar": qsk, "group": group, "q": qreq, "q_lower": qlower, "free_count": int(kfree), "support_count": int(len(pix)), "lock_count": int(len(lock_pix)), "numeric_unresolved": bool(numeric), "ref_cost_half": ref_cost})
            policy.append(rec.copy())
            if make_regions:
                regions.append({"image_id": iid, "annotation_id": int(r["annotation_id"]), "raw_id": int(r["raw_id"]), "level": int(r["level"]), "x": xxh, "c0": c0[k].detach().cpu().numpy().astype(np.float64), "cref": cref.detach().cpu().numpy().astype(np.float64), "q": qreq, "group": group, "q0": q0k, "qstar": qsk, "lock_a": lock_a, "lock_t": lock_t, "lock_m": lock_m, "lock_pix": lock_pix, "free_count": int(kfree), "numeric_unresolved": bool(numeric), "support_count": int(len(pix)), "ref_cost_half": ref_cost})
        return policy, regions

    def _row_index(self, iid, k):
        # fit order in INDEX is image-major; row_index is stable by image/raw key.
        # A direct lookup keeps the implementation robust to dictionary ordering.
        for idx in self.by_image_rows[iid]:
            if self.rows[idx]["row_index"] == k:
                return idx
        raise KeyError((iid, k))

    def build_regions(self):
        t = time.time(); self.by_image_rows = {}
        for idx, rec in enumerate(self.rows): self.by_image_rows.setdefault(rec["image_id"], []).append(idx)
        # Stream records one by one.  A single torch.save(list) duplicates a
        # very large nested object during pickling and can exceed RAM even
        # though the underlying region data itself fits.
        streams = {lev: (self.out / f"regions_level{lev}.pklstream").open("wb") for lev in (0,1,2)}
        policy_path = self.out / "REFERENCE_POLICY.jsonl"; policy_path.unlink(missing_ok=True)
        region_counts = {0: 0, 1: 0, 2: 0}; groups_count = {g: 0 for g in ("R", "S", "F_other")}; numeric = 0
        image_seen = 0
        try:
            for iid in self.images:
                pol, regs = self.process_image(iid, make_regions=True)
                for p in pol: jline(policy_path, p)
                for r in regs:
                    lev = int(r["level"]); pickle.dump(r, streams[lev], protocol=4)
                    region_counts[lev] += 1; groups_count[r["group"]] += 1; numeric += int(r["numeric_unresolved"]); self.v_ref += r["ref_cost_half"]
                image_seen += 1
                if image_seen % 25 == 0 or image_seen == len(self.images): self.log("build_regions", images=image_seen, total=len(self.images), candidates=sum(region_counts.values()))
        finally:
            for f in streams.values(): f.close()
        self.v_ref /= self.N; self.budget = MULT * self.v_ref
        dump(self.out / "REFERENCE_SUMMARY.json", {"N": self.N, "groups": groups_count, "v_ref": self.v_ref, "budget": self.budget, "numeric_unresolved": numeric, "level_counts": {str(l): region_counts[l] for l in (0,1,2)}, "region_storage": "pickle_stream", "q_min": min(float(r["q"]) for r in self.rows)})
        self.timings["build_regions_s"] = time.time() - t
        self.log("regions_complete", groups=groups_count, v_ref=self.v_ref, budget=self.budget, numeric_unresolved=numeric)

    def iter_regions(self, lev):
        """Stream one level from disk; P5 records exceed laptop RAM if materialized."""
        path = self.out / f"regions_level{lev}.pklstream"
        with path.open("rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def preflight(self):
        # Deterministic eight-image implementation check after full asset construction.
        ids = sorted(self.images, key=lambda z: hashlib.sha256(str(z).encode()).hexdigest())[:8]
        rows = [r for r in self.rows if r["image_id"] in ids]
        ok = all((r.get("q_lower", -1) >= r.get("q", 0.0) - 1e-9) or r.get("numeric_unresolved", False) for r in rows)
        valid = not any(r.get("numeric_unresolved", False) for r in rows)
        payload = {"status": "PASS" if ok and valid else "PARTIAL", "images": ids, "candidates": len(rows), "checks": {"identity": True, "reference_region": ok, "no_numeric_unresolved": valid, "native_decode": True, "scope": "preflight only"}}
        dump(self.out / "PREFLIGHT.json", payload)
        self.log("preflight", **payload)
        if not ok:
            raise RuntimeError("8-image preflight failed quality-envelope checks")

    def independent(self):
        t = time.time(); out_path = self.out / "INDEPENDENT_WITNESSES.jsonl"; out_path.unlink(missing_ok=True)
        all_rows = []
        for lev in (0,1,2):
            for j, r in enumerate(self.iter_regions(lev)):
                a = np.asarray(r["lock_a"], dtype=np.float64); tvec = np.asarray(r["lock_t"], dtype=np.float64); m = np.asarray(r["lock_m"], dtype=np.float64); c0 = np.asarray(r["c0"], dtype=np.float64)
                if len(a) == 0:
                    d = np.zeros(32); status = "ZERO_FEASIBLE"; opt = 0.0; upper = 0.0
                else:
                    G = tvec[:,None] * a; rhs = m - G.dot(c0); viol = float(np.min(G.dot(np.zeros(32)) - rhs))
                    if viol >= -1e-8:
                        d = np.zeros(32); status = "ZERO_FEASIBLE"; opt = 0.0; upper = 0.0
                    else:
                        d, status, opt, sol = solve_qp(np.eye(32), G, rhs, max_iter=200, time_limit=60.0)
                        feas = bool(np.min(G.dot(d) - rhs) >= -1e-6)
                        if not feas:
                            d = np.asarray(r["cref"]) - c0; upper = float(0.5*d.dot(d)); status = status + "+REFERENCE_FEASIBLE_UPPER"; opt = float("nan")
                        else: upper = float(opt)
                row = {"image_id": r["image_id"], "annotation_id": r["annotation_id"], "raw_id": r["raw_id"], "level": lev, "group": r["group"], "status": status, "objective_half": opt, "upper_half": upper, "reference_half": float(0.5*np.sum((np.asarray(r["cref"])-c0)**2)), "constraint_count": int(len(a))}
                jline(out_path, row); all_rows.append(row)
        finite = [float(r["objective_half"]) for r in all_rows if np.isfinite(r["objective_half"])]
        dump(self.out / "INDEPENDENT_SUMMARY.json", {"count": len(all_rows), "zero": sum(r["status"]=="ZERO_FEASIBLE" for r in all_rows), "mean_half": float(np.mean(finite)) if finite else 0.0, "reference_feasible_upper_total": float(sum(r["reference_half"] for r in all_rows)), "statuses": {s: sum(r["status"].startswith(s) for r in all_rows) for s in sorted(set(r["status"].split("+")[0] for r in all_rows))}})
        self.timings["independent_s"] = time.time() - t
        self.log("independent_complete", count=len(all_rows), elapsed_s=self.timings["independent_s"])

    def _initial_active_stream(self, lev):
        active = []; seen = set()
        for ci, r in enumerate(self.iter_regions(lev)):
            a = np.asarray(r["lock_a"], dtype=np.float64); tvec=np.asarray(r["lock_t"],dtype=np.float64); m=np.asarray(r["lock_m"],dtype=np.float64); c0=np.asarray(r["c0"],dtype=np.float64); x=np.asarray(r["x"],dtype=np.float64)
            if len(a)==0: continue
            score=m/(np.linalg.norm(a,axis=1)+1e-12)
            chosen=[]
            for sign in (1.0,-1.0):
                ids=np.where(tvec==sign)[0]
                chosen.extend(ids[np.argsort(score[ids],kind='stable')[:4]].tolist())
            for pi in chosen:
                key=(ci,pi)
                if key in seen: continue
                seen.add(key); active.append(self._row_tuple(r,pi,ci))
        return active, seen

    @staticmethod
    def _row_tuple(r, pi, ci):
        x=np.asarray(r["x"],dtype=np.float64); a=np.asarray(r["lock_a"],dtype=np.float64)[pi]; t=float(np.asarray(r["lock_t"])[pi]); m=float(np.asarray(r["lock_m"])[pi]); c0=np.asarray(r["c0"],dtype=np.float64); return (ci,pi,np.kron(x,t*a),m-t*float(a.dot(c0)))

    def solve_shared_level(self, lev):
        nvar=65*32
        # Only the 65-dimensional design rows are retained; the large pixel
        # response arrays remain on disk and are streamed during each scan.
        X=np.asarray([r["x"] for r in self.iter_regions(lev)],dtype=np.float64)
        # theta is reshaped row-major to A[65,32] and d=x@A, so the
        # corresponding Hessian is kron(X'X, I_32), not the column-major
        # ordering kron(I_32, X'X).
        H=np.kron((X.T@X)/self.N, np.eye(32))
        active, seen = self._initial_active_stream(lev); rounds=[]; status='UNRESOLVED'; theta=np.zeros(nvar); lower=0.0; upper=None
        for it in range(40):
            if active:
                G=np.vstack([q[2] for q in active]); rhs=np.asarray([q[3] for q in active],dtype=np.float64)
            else:
                G=np.zeros((0,nvar)); rhs=np.zeros(0)
            theta, st, obj, sol = solve_qp(H,G,rhs,max_iter=200,time_limit=600.0)
            dual = recompute_dual_bound(H, G, rhs, sol)
            lower=max(float(lower), float(dual)); maxv=np.inf; added=[]; total_viol=0
            # A valid dual certificate from any active subset is already a
            # certificate for the full constraint set.  Stop before an
            # expensive full scan when it exceeds the fixed global budget.
            if dual > self.budget * (1.0 + 1e-10):
                rounds.append({"iteration":it+1,"solver_status":st,"active_rows":len(active),"objective_active":float(obj),"dual_bound":float(dual),"lower_bound":float(lower),"violations":None,"added":0,"max_violation":None,"early_exit":"dual_over_budget"})
                self.log("shared_level_round", level=lev, **rounds[-1])
                status='CERTIFIED_OUTSIDE_BUDGET'; break
            for ci,r in enumerate(self.iter_regions(lev)):
                a=np.asarray(r["lock_a"],dtype=np.float64); tv=np.asarray(r["lock_t"],dtype=np.float64); m=np.asarray(r["lock_m"],dtype=np.float64); c0=np.asarray(r["c0"],dtype=np.float64); x=np.asarray(r["x"],dtype=np.float64); A=theta.reshape(65,32); d=x@A
                if len(a)==0: continue
                res=tv*(a@(c0+d))-m; bad=np.where(res < -1e-7)[0]; total_viol += len(bad)
                if len(bad):
                    order=bad[np.argsort(res[bad],kind='stable')[:8]]
                    for pi in order:
                        key=(ci,int(pi)); maxv=min(maxv,float(res[pi]))
                        if key not in seen: seen.add(key); active.append(self._row_tuple(r,int(pi),ci)); added.append(key)
            rounds.append({"iteration":it+1,"solver_status":st,"active_rows":len(active),"objective_active":float(obj),"dual_bound":float(dual),"lower_bound":float(lower),"violations":int(total_viol),"added":len(added),"max_violation":float(maxv) if np.isfinite(maxv) else 0.0})
            self.log("shared_level_round", level=lev, **rounds[-1])
            if total_viol==0:
                status='ACTIVE_ALL_LOCKED_FEASIBLE'; break
            if not added:
                status='UNRESOLVED_NO_NEW_CONSTRAINTS'; break
        A=theta.reshape(65,32); delta_cost=float(0.5*np.sum((X@A)**2)/self.N)
        return {"level":lev,"A":A,"H":H,"theta":theta,"lower_bound":lower,"primal_cost":delta_cost,"status":status,"rounds":rounds,"active_rows":len(active),"n_candidates":len(X)}

    def verify_shared_fit(self, sol_by_level):
        # Re-run native 640 decode for all fit candidates and verify q_i.  This also
        # checks the envelope rather than treating a QP status as a witness.
        failures=[]; metrics=[]; A_by={l:s["A"] for l,s in sol_by_level.items()}
        for iid in self.images:
            x=load_image(self.cache,iid); rows=x["rows"]; proto=x["proto"].to(DEVICE); rid=torch.tensor([int(r["raw_id"]) for r in rows]); levs=[int(r["level"]) for r in rows]; boxes=x["boxes"][rid].to(DEVICE); c0=x["coeff"][rid].to(DEVICE).float(); gt, _ = self.raw_gt_masks(iid, rows, x)
            cs=[]
            for k,r in enumerate(rows):
                lev=int(r["level"]); mu,sd=self.stats[lev]; h=x["h"][int(r["raw_id"])].numpy().astype(np.float64); xx=np.concatenate([(h-mu)/sd,np.ones(1)]); d=xx@A_by[lev]; cs.append(torch.from_numpy(d).to(DEVICE).float()+c0[k])
            cnew=torch.stack(cs).to(DEVICE); pred=decode_pad(proto,cnew,boxes)>0; pred_o=scale_binary(pred,x); qs=iou_pair(pred_o,gt)
            for k,r in enumerate(rows):
                g=self.rows[self._row_index(iid,k)]; q=float(g["q"]); ok=bool(qs[k]>=q-1e-6); metrics.append({"image_id":iid,"annotation_id":int(r["annotation_id"]),"raw_id":int(r["raw_id"]),"level":int(r["level"]),"q":q,"q_new":float(qs[k]),"ok":ok})
                if not ok: failures.append(metrics[-1])
        dump(self.out/"NATIVE_WITNESS_REPLAY.json",{"all_checked":len(metrics),"failures":len(failures),"passed":not failures,"min_q_margin":float(min(m["q_new"]-m["q"] for m in metrics)) if metrics else None,"failures_head":failures[:100]})
        return not failures, metrics

    def shared(self):
        t=time.time(); sols={}
        for lev in (0,1,2):
            sols[lev]=self.solve_shared_level(lev)
            if sols[lev]["status"] == 'CERTIFIED_OUTSIDE_BUDGET':
                break
        total_lower=float(sum(s["lower_bound"] for s in sols.values())); total_primal=float(sum(s["primal_cost"] for s in sols.values()))
        feasible, metrics=self.verify_shared_fit(sols) if all(s["status"]=='ACTIVE_ALL_LOCKED_FEASIBLE' for s in sols.values()) else (False,[])
        upper=total_primal if feasible else None
        finding='FEASIBLE_WITHIN_BUDGET' if upper is not None and upper<=self.budget else 'CERTIFIED_OUTSIDE_BUDGET' if total_lower>self.budget else 'UNRESOLVED'
        summary={"levels":{str(l):{k:v for k,v in s.items() if k not in ('A','H','theta')} for l,s in sols.items()},"global_lower_bound":total_lower,"global_primal_cost":total_primal,"global_upper_bound":upper,"budget":self.budget,"lower_over_budget":total_lower>self.budget,"finding":finding,"native_replay_passed":feasible,"candidate_metrics":len(metrics)}
        dump(self.out/"SHARED_PRIMAL_DUAL.json",summary)
        for lev,s in sols.items(): np.savez(self.out/f"SHARED_A_level{lev}.npz",A=s["A"],H=s["H"],theta=s["theta"])
        self.timings["shared_s"]=time.time()-t; self.log("shared_complete", **{k:summary[k] for k in ('global_lower_bound','global_primal_cost','global_upper_bound','budget','finding','native_replay_passed')})
        return summary, sols

    def closeout(self, shared_summary):
        status='VALID' if shared_summary["finding"]!='UNRESOLVED' else 'PARTIAL'
        report = ["# 固定原型下掩码任务可行性对照", "", "## 结论", ""]
        finding=shared_summary["finding"]
        if finding=='FEASIBLE_WITHIN_BUDGET':
            report.append(f"在固定官方TAL候选、原始 h/P/c0 和正常解码下，E1 共享仿射读出构造了全体候选可复核的预算内见证；共享成本 {shared_summary['global_upper_bound']:.8g}，预算 {self.budget:.8g}。这只证明声明质量包络的 fit 可行性，不证明新图泛化或COCO AP。")
        elif finding=='CERTIFIED_OUTSIDE_BUDGET':
            report.append(f"在声明的 E1 质量包络与有限预算下，共享任务得到条件数值超预算证据：有效下界 {shared_summary['global_lower_bound']:.8g} > 预算 {self.budget:.8g}。这不等于所有 Mask75 解都不可达，也不等于多数实例有同一根因。")
        else:
            report.append(f"共享任务在当前预算和数值资源内未判定；下界 {shared_summary['global_lower_bound']:.8g}，当前活动解成本 {shared_summary['global_primal_cost']:.8g}，预算 {self.budget:.8g}。证据不足，不能解释成模型做不到。")
        report += ["", "## 范围", "", f"官方 fit：{len(self.images)} 张图、{self.N} 个 one-to-one TAL 候选；完整运行预期 P3/P4/P5=2000/2298/1760。E1 使用 q=0.75、IoU 容差0.01、margin=0.10、预算=4*v_ref。", "", "## 运行产物", "", "见 PREFLIGHT.json、REFERENCE_SUMMARY.json、INDEPENDENT_SUMMARY.json、SHARED_PRIMAL_DUAL.json、NATIVE_WITNESS_REPLAY.json。"]
        (self.out/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
        dump(self.out/"RESOURCE_SUMMARY.json",{"host":platform.node(),"device":str(DEVICE),"timings_s":self.timings,"cuda_peak_allocated":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,"status":status,"finding":finding})
        dump(self.out/"SUMMARY.json",{"status":status,"finding":finding,"v_ref":self.v_ref,"budget":self.budget,"shared":shared_summary})
        return status

    def run(self):
        self.register(); self.collect_stats()
        if self.resume_prepared:
            ref = json.loads((self.out / "REFERENCE_SUMMARY.json").read_text(encoding="utf-8"))
            self.v_ref = float(ref["v_ref"]); self.budget = float(ref["budget"])
            if (self.out / "PREFLIGHT.json").exists():
                self.log("preflight_reused", source="prepared_run")
            else:
                self.preflight()
            if (self.out / "INDEPENDENT_SUMMARY.json").exists():
                self.log("independent_reused", count=int(ref["N"]))
            else:
                self.independent()
        else:
            self.build_regions(); self.preflight(); self.independent()
        shared_summary, sols=self.shared(); status=self.closeout(shared_summary); self.log("closeout", status=status, finding=shared_summary["finding"])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--through',default='closeout'); ap.add_argument('--root',type=Path,default=Path(r'D:\coco_wire')); ap.add_argument('--out',type=Path,default=None); ap.add_argument('--limit-images',type=int,default=None); ap.add_argument('--resume-prepared',action='store_true'); a=ap.parse_args()
    run_id=os.environ.get('RESEARCH_RUN_ID','RUN_mask_task_feasibility_20261001'); out=a.out or (a.root/'mask_task_feasibility_20261001'/'runs'/run_id); torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(min(8,os.cpu_count() or 1));
    p=Pipeline(a.root,out); p.limit_images=a.limit_images; p.resume_prepared=a.resume_prepared; p.run()


if __name__=='__main__': main()
