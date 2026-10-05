"""S081: image-disjoint ownership probes on the frozen COCO readout cache.

The probe does not change model parameters or run a new forward pass.  COCO
annotations are used only to define held-out pixel ownership labels.  Models
are fitted on fit images and evaluated on disjoint transfer images.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "diagnostics/readout_input_scale1200_20260912/cache"
ANNOTATION = ROOT / "local_readout_runtime_20260912/data/annotations/instances_train2017.json"
OUT = ROOT / "diagnostics/representation_ownership_probe_20260913_area_stratified"
CROP_STRATA = ROOT / "diagnostics/crop_support_failure_cross_20260913/per_target.csv"
SEED = 20260913
MAX_PER_CLASS = 32
FEATURES = {
    "coord": lambda p, c, h, xy: xy,
    "head_input": lambda p, c, h, xy: np.broadcast_to(h, (len(p), len(h))),
    "prototype": lambda p, c, h, xy: p,
    "coefficient": lambda p, c, h, xy: np.broadcast_to(c, (len(p), len(c))),
    "logit": lambda p, c, h, xy: (p * c[None, :]).sum(1, keepdims=True),
    "proto_coeff": lambda p, c, h, xy: p * c[None, :],
    "head_plus_proto_coeff": lambda p, c, h, xy: np.concatenate([np.broadcast_to(h, (len(p), len(h))), p * c[None, :]], 1),
    "all": lambda p, c, h, xy: np.concatenate([xy, np.broadcast_to(h, (len(p), len(h))), p * c[None, :]], 1),
}

TARGET_AUC_ROWS: list[dict] = []


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox_map(shape: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map native 640 pixel centers to original COCO pixel indices."""
    height, width = map(int, shape)
    gain = min(640.0 / height, 640.0 / width)
    nh, nw = round(height * gain), round(width * gain)
    top, left = round((640 - nh) / 2 - 0.1), round((640 - nw) / 2 - 0.1)
    yy, xx = np.divmod(positions.astype(np.int64), 640)
    inside = (yy >= top) & (yy < top + nh) & (xx >= left) & (xx < left + nw)
    oy = np.floor((yy - top) / gain).astype(np.int64).clip(0, height - 1)
    ox = np.floor((xx - left) / gain).astype(np.int64).clip(0, width - 1)
    return np.stack([oy, ox], 1), inside


def sample_balanced(rng: np.random.Generator, indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return rng.choice(indices, size=limit, replace=False)


def add_rows(rows: list[dict], target_meta: list[dict], raw: dict[str, list[dict]], task: str, split: str,
             image_id: int, ann_id: int, p: np.ndarray, c: np.ndarray, h: np.ndarray,
             pos: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
    # y=-1 marks pixels outside the requested binary comparison.  Those
    # pixels must not silently become negatives (e.g. other instances in the
    # own-vs-same task).
    valid = np.flatnonzero(y >= 0)
    own = valid[y[valid] == 1]
    other = valid[y[valid] == 0]
    if len(own) < 8 or len(other) < 8:
        return
    take = np.concatenate([sample_balanced(rng, own, MAX_PER_CLASS),
                           sample_balanced(rng, other, MAX_PER_CLASS)])
    # Keep a deterministic order while retaining the balanced class counts.
    take.sort()
    label = y[take].astype(np.int8)
    xy = np.stack([pos[take] % 640, pos[take] // 640], 1).astype(np.float32) / 639.0
    pp = p[take].astype(np.float32)
    raw.setdefault(f"{task}/{split}", []).append(dict(p=pp, c=c.astype(np.float32),
                                                       h=h.astype(np.float32), xy=xy,
                                                       y=label))
    record = dict(image_id=image_id, annotation_id=ann_id, split=split,
                  task=task, sampled_pixels=len(take), own_sampled=int(label.sum()),
                  other_sampled=int((label == 0).sum()))
    rows.append(record)
    target_meta.append(dict(task=task, split=split, image_id=image_id,
                            annotation_id=ann_id, sampled_pixels=len(take)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concat(arrays: dict[str, list[np.ndarray]], key: str, width: int | None = None) -> np.ndarray:
    values = arrays.get(key, [])
    if values:
        return np.concatenate(values, 0).astype(np.float32, copy=False)
    return np.empty((0, width or 0), dtype=np.float32)


def image_bootstrap(per_image: np.ndarray, seed: int = SEED, draws: int = 2000) -> tuple[float, float, float]:
    if len(per_image) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = per_image[rng.integers(0, len(per_image), size=(draws, len(per_image)))].mean(1)
    return float(per_image.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def build_matrix(records: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    make = FEATURES[name]
    for record in records:
        xs.append(make(record["p"], record["c"], record["h"], record["xy"]))
        ys.append(record["y"])
    return np.concatenate(xs, 0).astype(np.float32, copy=False), np.concatenate(ys, 0)


def fit_one(raw: dict[str, list[dict]], task: str, name: str) -> dict:
    global TARGET_AUC_ROWS
    xfit, yfit = build_matrix(raw[f"{task}/fit"], name)
    xtest, ytest = build_matrix(raw[f"{task}/transfer"], name)
    # SGD avoids the Windows SciPy L-BFGS backend, which is unstable for the
    # million-scale dense pixel matrix used by this probe.
    model = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", penalty="l2",
                                                          alpha=1e-4, max_iter=40,
                                                          tol=1e-3, average=True,
                                                          random_state=SEED))
    model.fit(xfit, yfit)
    score = model.decision_function(xtest)
    micro_auc = float(roc_auc_score(ytest, score))
    micro_ap = float(average_precision_score(ytest, score))
    # Macro target AUC is computed after fitting, using the same held-out
    # image split. It avoids letting images with many sampled pixels dominate.
    # Arrays are appended target-by-target in add_rows; recover slices from
    # one metadata record per target/task, rather than one record per pixel.
    per_target = []
    cursor = 0
    transfer_rows = [r for r in TARGET_META if r["task"] == task and r["split"] == "transfer"]
    for record in transfer_rows:
        n = int(record["sampled_pixels"])
        labels = ytest[cursor:cursor + n]
        values = score[cursor:cursor + n]
        cursor += n
        if len(np.unique(labels)) == 2:
            auc = float(roc_auc_score(labels, values))
            per_target.append(auc)
            TARGET_AUC_ROWS.append(dict(feature=name, task=task,
                                        image_id=int(record["image_id"]),
                                        annotation_id=int(record["annotation_id"]),
                                        auc=auc))
    macro, ci_low, ci_high = image_bootstrap(np.asarray(per_target, dtype=np.float64))
    result = dict(feature=name, task=task, fit_rows=int(len(yfit)), transfer_rows=int(len(ytest)),
                transfer_target_count=int(len(per_target)), micro_auc=micro_auc, micro_ap=micro_ap,
                macro_target_auc=macro, macro_target_auc_ci_low=ci_low,
                macro_target_auc_ci_high=ci_high)
    del xfit, xtest, yfit, ytest, model, score
    return result


def write_stratified(out: Path) -> list[dict]:
    """Join target AUCs with the audited S078 target strata and summarize them.

    The S078 labels are used only after inference to define diagnostic strata;
    they are never features of the ownership classifier.
    """
    with CROP_STRATA.open("r", newline="", encoding="utf-8") as handle:
        crop = {(int(r["image_id"]), int(r["annotation_id"])): r
                for r in csv.DictReader(handle)}
    with ANNOTATION.open("r", encoding="utf-8") as handle:
        annotation_area = {int(r["id"]): float(r.get("area", 0.0))
                           for r in json.load(handle)["annotations"]}

    def area_bin(r: dict) -> str:
        area = annotation_area.get(int(r["annotation_id"]), 0.0)
        if area < 1024.0:
            return "area_small_<1024"
        if area < 9216.0:
            return "area_medium_1024_9216"
        return "area_large_>=9216"

    strata = [
        ("all", lambda r: True),
        ("high_ici", lambda r: r["group"] == "high"),
        ("low_ici", lambda r: r["group"] == "low"),
        ("support_low_mask_bad", lambda r: r["state"] == "box_good_support_low_mask_bad"),
        ("support_sufficient_mask_bad", lambda r: r["state"] == "box_good_support_sufficient_mask_bad"),
        ("mask_good", lambda r: r["state"] == "box_good_mask_good"),
        ("area_small_<1024", lambda r: area_bin(r) == "area_small_<1024"),
        ("area_medium_1024_9216", lambda r: area_bin(r) == "area_medium_1024_9216"),
        ("area_large_>=9216", lambda r: area_bin(r) == "area_large_>=9216"),
    ]
    detail = []
    for row in TARGET_AUC_ROWS:
        meta = crop.get((row["image_id"], row["annotation_id"]))
        if meta is None:
            continue
        for stratum, predicate in strata:
            if predicate(meta):
                detail.append(dict(feature=row["feature"], task=row["task"],
                                   stratum=stratum, image_id=row["image_id"],
                                   annotation_id=row["annotation_id"], auc=row["auc"],
                                   ici=float(meta["ici"]), state=meta["state"],
                                   area=annotation_area.get(int(row["annotation_id"]), 0.0)))
    write_csv(out / "target_auc_stratified.csv", detail)

    summary = []
    for feature in FEATURES:
        for task in ["own_vs_same", "own_vs_background"]:
            for stratum, _ in strata:
                vals = np.asarray([r["auc"] for r in detail
                                   if r["feature"] == feature and r["task"] == task
                                   and r["stratum"] == stratum], dtype=np.float64)
                if len(vals) == 0:
                    continue
                rng = np.random.default_rng(SEED + len(summary))
                boot = vals[rng.integers(0, len(vals), size=(2000, len(vals)))].mean(1)
                summary.append(dict(feature=feature, task=task, stratum=stratum,
                                    n_targets=int(len(vals)), mean_auc=float(vals.mean()),
                                    ci_low=float(np.quantile(boot, .025)),
                                    ci_high=float(np.quantile(boot, .975))))
    write_csv(out / "STRATIFIED.csv", summary)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    with ANNOTATION.open("r", encoding="utf-8") as handle:
        gt = COCO()  # avoid COCO's stdout while keeping official RLE decoding
        gt.dataset = json.load(handle)
        gt.createIndex()
    raw: dict[str, list[dict]] = {}
    rows: list[dict] = []
    global TARGET_META
    TARGET_META = []
    image_count = 0
    target_count = 0
    for path in sorted((CACHE / "images").glob("*.npz")):
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        split = sidecar["split"]
        if split not in {"fit", "transfer"}:
            continue
        image_id = int(path.stem)
        image_count += 1
        with np.load(path, allow_pickle=False) as data:
            shape = data["shape"]
            ann_ids = data["annotation_ids"].astype(np.int64)
            pred_ids = data["prediction_indices"].astype(np.int64)
            p_all = data["sample_p"]
            y_own_all = data["sample_y"]
            pos_all = data["sample_positions"]
            coeff_all = data["coeff"]
            h_all = data["h"]
        anns = [a for a in gt.imgToAnns[image_id] if not a.get("iscrowd", 0)]
        masks = {int(a["id"]): gt.annToMask(a).astype(bool) for a in anns}
        categories = {int(a["id"]): int(a["category_id"]) for a in anns}
        for row_index, ann_id in enumerate(ann_ids):
            ann_id = int(ann_id)
            if ann_id not in masks:
                continue
            target_count += 1
            positions = pos_all[row_index]
            original_xy, inside = letterbox_map(shape, positions)
            oy, ox = original_xy[:, 0], original_xy[:, 1]
            target_cat = categories[ann_id]
            same = np.zeros(len(positions), dtype=bool)
            union = np.zeros(len(positions), dtype=bool)
            for other_id, mask in masks.items():
                values = mask[oy, ox] & inside
                union |= values
                if other_id != ann_id and categories[other_id] == target_cat:
                    same |= values
            own = y_own_all[row_index].astype(bool)
            own &= inside
            # Use the audited native ownership for the target itself. Remove
            # own pixels from every negative class to keep labels disjoint.
            same &= ~own
            background = (~union) & (~own)
            stable_seed = int(hashlib.sha256(
                f"{SEED}:{image_id}:{ann_id}".encode("ascii")).hexdigest()[:8], 16)
            rng = np.random.default_rng(stable_seed)
            common = dict(image_id=image_id, annotation_id=ann_id, split=split,
                          category_id=target_cat, same_pixels=int(same.sum()),
                          background_pixels=int(background.sum()), own_pixels=int(own.sum()))
            before = len(rows)
            add_rows(rows, TARGET_META, raw, "own_vs_same", split, image_id, ann_id,
                     p_all[row_index], coeff_all[pred_ids[row_index]], h_all[pred_ids[row_index]],
                     positions, np.where(own, 1, np.where(same, 0, -1)).astype(np.int8), rng)
            if len(rows) > before:
                rows[-1].update(common)
            before = len(rows)
            add_rows(rows, TARGET_META, raw, "own_vs_background", split, image_id, ann_id,
                     p_all[row_index], coeff_all[pred_ids[row_index]], h_all[pred_ids[row_index]],
                     positions, np.where(own, 1, np.where(background, 0, -1)).astype(np.int8), rng)
            if len(rows) > before:
                rows[-1].update(common)
    results = []
    errors = []
    global TARGET_AUC_ROWS
    TARGET_AUC_ROWS = []
    for task in ["own_vs_same", "own_vs_background"]:
        for name in FEATURES:
            key = f"{task}/fit"
            if not raw.get(key) or not raw.get(f"{task}/transfer"):
                continue
            try:
                results.append(fit_one(raw, task, name))
            except Exception as exc:
                errors.append(dict(task=task, feature=name, error=repr(exc)))
    summary = dict(status="COMPLETE", experiment="S081", image_count=image_count,
                   target_count=target_count, usable_target_task_rows=len(rows), max_pixels_per_class=MAX_PER_CLASS,
                   feature_names=list(FEATURES), results=results, errors=errors,
                   cache_sha256=sha(CACHE / "COMPLETE.json"), annotation_sha256=sha(ANNOTATION),
                   split="fit images fit the classifier; transfer images are image-disjoint evaluation only",
                   labels="COCO GT ownership; not used as model input")
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "per_target.csv", rows)
    stratified = write_stratified(OUT)
    with (OUT / "SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        if results:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    lines = ["# S081：COCO 图片级迁移的目标/邻居表示判别 probe", "",
             "状态：COMPLETE。复用冻结 readout cache，无新增前向、训练或 AP。fit 图片拟合线性判别器，transfer 图片只用于评估；COCO GT 只定义像素归属标签。", "",
             f"图片 {image_count}，匹配目标 {target_count}，可用目标/任务记录 {len(rows)}。每个目标最多平衡采样自身与负类各 {MAX_PER_CLASS} 个像素。", "",
             "特征含义：coord=位置；head_input=候选头输入 h（同一候选内为空间常量）；prototype=原型向量；coefficient=实例系数；logit=原型与系数的标量乘积；proto_coeff=逐通道贡献；all=位置+h+逐通道贡献。", "",
             "| task | feature | transfer micro AUC | transfer macro target AUC (95% image bootstrap) |", "|---|---|---:|---:|"]
    for r in results:
        lines.append(f"| {r['task']} | {r['feature']} | {r['micro_auc']:.4f} | {r['macro_target_auc']:.4f} [{r['macro_target_auc_ci_low']:.4f}, {r['macro_target_auc_ci_high']:.4f}] |")
    lines += ["", "## 分层结果", "", "AUC 按 transfer 目标分层；S078 的 ICI/失败状态仅用于事后分组。区间是目标级 bootstrap，不是图像级不确定性。", "", "| task | feature | stratum | n | mean AUC | 95% target bootstrap |", "|---|---|---|---:|---:|---:|"]
    for r in stratified:
        lines.append(f"| {r['task']} | {r['feature']} | {r['stratum']} | {r['n_targets']} | {r['mean_auc']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines += ["", "解释边界：高 AUC 说明冻结表示中存在可利用的像素区分信息，不证明某个可部署损失或模块已经有效；head_input 单独应接近随机，因为它对同一候选的所有像素相同。若 proto_coeff/head_plus_proto_coeff 显著优于 prototype，说明实例系数提供了必要的条件化；若加入 h 无增益，则瓶颈更可能在系数映射而非候选头缺少空间输入。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
