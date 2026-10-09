"""Derive damage diagnostics from immutable existing DEV/FINAL candidate rows.

No model inference, training, selection, or parameter updates. Subgroup and
margin-bin results are descriptive associations, not causal effects.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


TZ = timezone(timedelta(hours=8), "Asia/Shanghai")
ARMS = ("B", "D1", "D")
METRICS = ("iou", "coverage", "auc", "fpr")
IDENTITY = ("image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summary_vector(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    result = {"n": int(len(values)), "mean": float(values.mean())}
    result.update(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))
    return result


def rate(mask, population):
    n = int(population.sum())
    count = int((mask & population).sum())
    return {"count": count, "denominator": n, "fraction": count / n if n else None}


def correlation(x, y, mask):
    keep = mask & np.isfinite(x) & np.isfinite(y)
    xx, yy = x[keep], y[keep]
    n = len(xx)
    value = float(np.corrcoef(xx, yy)[0, 1]) if n >= 2 and np.std(xx) > 0 and np.std(yy) > 0 else None
    return {"n": n, "pearson": value}


class Data:
    def __init__(self, path, rho):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        identities = [tuple(row[k] for k in IDENTITY) for row in rows]
        if len(set(identities)) != len(rows):
            raise ValueError(f"Duplicate candidate identity: {path}")
        self.rows = rows
        self.path = path
        self.rho = rho
        self.n = len(rows)
        self.image = np.array([row["image_id"] for row in rows])
        self.split = rows[0]["split"]
        if any(row["split"] != self.split for row in rows):
            raise ValueError("Mixed splits")
        self.values = {}
        for metric in METRICS:
            for arm in ("A",) + ARMS:
                self.values[f"{metric}_{arm}"] = np.array([np.nan if row.get(f"{metric}_{arm}") is None else row[f"{metric}_{arm}"] for row in rows], dtype=float)
        for key in ("delta_norm_B", "delta_norm_D1", "delta_norm_D", "q_c0", "q_c1", "q_c2"):
            self.values[key] = np.array([np.nan if row.get(key) is None else row[key] for row in rows], dtype=float)
        self.success = self.values["iou_A"] >= .75
        self.guarded = np.array([bool(row["class_correct"]) and row["box_iou"] >= .75 for row in rows])
        self.size = np.array(["small" if row["area"] < 32 ** 2 else "medium" if row["area"] < 96 ** 2 else "large" for row in rows])
        self.layer = np.array(["P" + str(int(row["pyramid_level"]) + 3) for row in rows])
        if set(self.layer) - {"P3", "P4", "P5"}:
            raise ValueError("Unrecognized pyramid layer")
        self.support_negative = np.array([row["original_pixel_support_count"] - row["original_pixel_support_gt_positive"] for row in rows], dtype=float)
        for arm in ("A",) + ARMS:
            self.values[f"support_fp_pixels_{arm}"] = self.values[f"fpr_{arm}"] * self.support_negative

    def paired(self, metric, arm, mask, baseline="A"):
        x, y = self.values[f"{metric}_{baseline}"], self.values[f"{metric}_{arm}"]
        keep = mask & np.isfinite(x) & np.isfinite(y)
        buckets = defaultdict(list)
        for image, difference in zip(self.image[keep], (y - x)[keep]):
            buckets[int(image)].append(float(difference))
        return {
            "candidate_pairs": int(keep.sum()),
            "images": len(buckets),
            "candidate_mean_delta": float((y - x)[keep].mean()) if keep.any() else None,
            "image_macro_mean_delta": float(np.mean([np.mean(items) for items in buckets.values()])) if buckets else None,
        }

    def group(self, mask):
        n = int(mask.sum())
        result = {"n": n, "images": int(len(set(self.image[mask]))), "baseline_success": int((mask & self.success).sum())}
        result["arms"] = {}
        for arm in ARMS:
            outcome = self.values[f"iou_{arm}"] >= .75
            result["arms"][arm] = {
                "damage": rate(~outcome, mask & self.success),
                "repair": rate(outcome, mask & ~self.success),
                "iou_decreased": rate(self.values[f"iou_{arm}"] < self.values["iou_A"], mask),
                "delta_norm": summary_vector(self.values[f"delta_norm_{arm}"][mask]),
                "norm_above_success_training_max": rate(self.values[f"delta_norm_{arm}"] > self.rho / 4 + 1e-5, mask),
                "paired": {metric: self.paired(metric, arm, mask) for metric in METRICS + ("support_fp_pixels",)},
                "fpr_increased": rate(self.values[f"fpr_{arm}"] > self.values["fpr_A"], mask & np.isfinite(self.values[f"fpr_{arm}"]) & np.isfinite(self.values["fpr_A"])),
                "norm_vs_iou_change": correlation(self.values[f"delta_norm_{arm}"], self.values[f"iou_{arm}"] - self.values["iou_A"], mask),
                "norm_vs_fpr_change": correlation(self.values[f"delta_norm_{arm}"], self.values[f"fpr_{arm}"] - self.values["fpr_A"], mask),
            }
        return result

    def trajectory(self, mask):
        a, d1, d2 = (self.values[f"iou_{arm}"] >= .75 for arm in ("A", "D1", "D"))
        dd1 = a & ~d1 & mask
        dd2 = a & ~d2 & mask
        repair1 = ~a & d1 & mask
        stages = {}
        for label, qbase, qnext, ibase, inext in (("step1", "q_c0", "q_c1", "A", "D1"), ("step2", "q_c1", "q_c2", "D1", "D"), ("total", "q_c0", "q_c2", "A", "D")):
            dq = self.values[qnext] - self.values[qbase]
            diou = self.values[f"iou_{inext}"] - self.values[f"iou_{ibase}"]
            qgroups = {"up": mask & (dq > 0), "down": mask & (dq < 0), "tie": mask & (dq == 0)}
            stages[label] = {
                "mean_q_change": summary_vector(dq[mask]),
                "q_iou_pearson": correlation(dq, diou, mask),
                "iou_up": rate(diou > 0, mask),
                "iou_down": rate(diou < 0, mask),
                "iou_tie": rate(diou == 0, mask),
                "by_q_sign": {},
            }
            for labelq, qm in qgroups.items():
                stages[label]["by_q_sign"][labelq] = {
                    "n": int(qm.sum()),
                    "iou_up": rate(diou > 0, qm), "iou_down": rate(diou < 0, qm),
                    "final_damage": rate(~d2, qm & a),
                    "paired": {metric: self.paired(metric, inext, qm, ibase) for metric in METRICS + ("support_fp_pixels",)},
                }
        return {
            "n": int(mask.sum()),
            "baseline_success": int((mask & a).sum()),
            "step1_damage": int(dd1.sum()),
            "final_damage": int(dd2.sum()),
            "step1_damage_restored_by_step2": rate(d2, dd1),
            "step1_damage_persists": rate(~d2, dd1),
            "new_step2_damage": rate(~d2, mask & a & d1),
            "step1_repair": int(repair1.sum()),
            "step1_repair_lost_by_step2": rate(~d2, repair1),
            "new_step2_repair": rate(d2, mask & ~a & ~d1),
            "stages": stages,
        }

    def analyze(self):
        all_rows = np.ones(self.n, dtype=bool)
        groups = {
            "all": all_rows,
            "baseline_success_all": self.success,
            "baseline_success_classcorrect_box75": self.success & self.guarded,
            "baseline_failure_all": ~self.success,
            "baseline_failure_classcorrect_box75": ~self.success & self.guarded,
            "final_damaged_success": self.success & (self.values["iou_D"] < .75),
            "final_preserved_success": self.success & (self.values["iou_D"] >= .75),
            "guarded_final_damaged_success": self.success & self.guarded & (self.values["iou_D"] < .75),
            "guarded_final_preserved_success": self.success & self.guarded & (self.values["iou_D"] >= .75),
        }
        for scope, base in (("success_all", self.success), ("success_classcorrect_box75", self.success & self.guarded)):
            for label, lo, hi in (("margin75_80", .75, .8), ("margin80_90", .8, .9), ("margin90_100", .9, 1.00000001)):
                groups[f"{scope}_{label}"] = base & (self.values["iou_A"] >= lo) & (self.values["iou_A"] < hi)
            for size in ("small", "medium", "large"):
                groups[f"{scope}_{size}"] = base & (self.size == size)
            for layer in ("P3", "P4", "P5"):
                groups[f"{scope}_{layer}"] = base & (self.layer == layer)
        result = {
            "split": self.split,
            "rows": self.n,
            "images_with_rows": int(len(set(self.image))),
            "identity_unique": True,
            "groups": {name: self.group(mask) for name, mask in groups.items()},
            "trajectories": {name: self.trajectory(groups[name]) for name in ("all", "baseline_success_all", "baseline_success_classcorrect_box75", "final_damaged_success", "final_preserved_success")},
        }
        for arm in ARMS:
            transitions = result["groups"]["all"]["arms"][arm]
            transitions["net_mask75_count"] = transitions["repair"]["count"] - transitions["damage"]["count"]
        return result


def fmt(value, factor=1., digits=3):
    return "NA" if value is None else f"{value * factor:.{digits}f}"


def tables(summary):
    lines = ["# 现有 QCR 逐候选记录的损伤诊断", "", "生成时间：" + summary["finished_at"] + "（Asia/Shanghai）。只读取现有 FINAL/DEV 记录，未执行模型或调整参数。", "", "Mask75 成功为正常原图 IoU≥0.75。所有损伤率的分母为指定子组的 baseline 成功行；classcorrectBox75 还限制 argmax 类别正确及 BoxIoU≥0.75。Margin 分组左闭右开，最后一组包括 1.0。", "", "像素指标增量采用同一行配对；图片 macro 先同图平均有效配对，再平均图片。NA 不补零。support FP pixels 是固定原图预测框支持内的非目标像素阳性数，不能代表整图所有 FP 面积。所有分层与相关系数都是事后描述，不能证明因果，未用于调参。", "", f"Success 训练扰动最大半径 rho/4={summary['radius']['success_training_max_radius']:.6f}；第一步固定 rho/2={summary['radius']['step1_eta']:.6f}，第二步 rho/4={summary['radius']['step2_eta']:.6f}。半径源自现有 Quality MODEL.json 和实际训练 states 实现。"]
    for split, data in summary["splits"].items():
        lines += ["", f"## {split.upper()}（{data['rows']} 行 / {data['images_with_rows']} 图）", "", "### 损伤与分层", "", "| baseline 成功子组 | 分母 | B 损伤 | D1 损伤 | D 损伤 | D 损伤率 | D IoU Δ pp（行/图） | D FPR Δ pp（行/图） | D coverage Δ pp（行/图） |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, group in data["groups"].items():
            if not (name.startswith("baseline_success") or name.startswith("success_")):
                continue
            arms = group["arms"]
            p = arms["D"]["paired"]
            lines.append(f"| {name} | {group['n']} | {arms['B']['damage']['count']} | {arms['D1']['damage']['count']} | {arms['D']['damage']['count']} | {fmt(arms['D']['damage']['fraction'], 100)}% | {fmt(p['iou']['candidate_mean_delta'], 100)} / {fmt(p['iou']['image_macro_mean_delta'], 100)} | {fmt(p['fpr']['candidate_mean_delta'], 100)} / {fmt(p['fpr']['image_macro_mean_delta'], 100)} | {fmt(p['coverage']['candidate_mean_delta'], 100)} / {fmt(p['coverage']['image_macro_mean_delta'], 100)} |")
        lines += ["", "### 两步轨迹", "", "| 范围 | D1 损伤 | 第二步恢复 | D1 损伤保留 | 第二步新损伤 | 最终损伤 |", "|---|---:|---:|---:|---:|---:|"]
        for name in ("all", "baseline_success_classcorrect_box75"):
            t = data["trajectories"][name]
            lines.append(f"| {name} | {t['step1_damage']} | {t['step1_damage_restored_by_step2']['count']} | {t['step1_damage_persists']['count']} | {t['new_step2_damage']['count']} | {t['final_damage']} |")
        lines += ["", "### Q 增量与真实指标（全体）", "", "| 轨迹 | ΔQ 符号 | 行数 | 真 IoU 上升率 | 真 IoU 下降率 | IoU Δ pp（行/图） | FPR Δ pp（行/图） | coverage Δ pp（行/图） |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for step, s in data["trajectories"]["all"]["stages"].items():
            for sign, q in s["by_q_sign"].items():
                p = q["paired"]
                lines.append(f"| {step} | {sign} | {q['n']} | {fmt(q['iou_up']['fraction'], 100)}% | {fmt(q['iou_down']['fraction'], 100)}% | {fmt(p['iou']['candidate_mean_delta'], 100)} / {fmt(p['iou']['image_macro_mean_delta'], 100)} | {fmt(p['fpr']['candidate_mean_delta'], 100)} / {fmt(p['fpr']['image_macro_mean_delta'], 100)} | {fmt(p['coverage']['candidate_mean_delta'], 100)} / {fmt(p['coverage']['image_macro_mean_delta'], 100)} |")
        lines += ["", "### 最终损伤与保持成功", "", "| 子组 | 行数 | D 位移均值 / 中位 / p95 | D FPR 上升率 | FPR Δ pp（行/图） | coverage Δ pp（行/图） | AUC Δ pp（行/图） | 支持内 FP 像素 Δ（行/图） |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ("final_damaged_success", "final_preserved_success", "guarded_final_damaged_success", "guarded_final_preserved_success"):
            g = data["groups"][name]
            d = g["arms"]["D"]
            n, p = d["delta_norm"], d["paired"]
            lines.append(f"| {name} | {g['n']} | {fmt(n['mean'])} / {fmt(n['median'])} / {fmt(n['p95'])} | {fmt(d['fpr_increased']['fraction'], 100)}% | {fmt(p['fpr']['candidate_mean_delta'], 100)} / {fmt(p['fpr']['image_macro_mean_delta'], 100)} | {fmt(p['coverage']['candidate_mean_delta'], 100)} / {fmt(p['coverage']['image_macro_mean_delta'], 100)} | {fmt(p['auc']['candidate_mean_delta'], 100)} / {fmt(p['auc']['image_macro_mean_delta'], 100)} | {fmt(p['support_fp_pixels']['candidate_mean_delta'])} / {fmt(p['support_fp_pixels']['image_macro_mean_delta'])} |")
        lines += ["", "### 位移分位数", "", "| 范围/方法 | p05 | p25 | 中位 | p75 | p95 | 超过 Success 训练最大半径 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name in ("all", "baseline_success_all", "baseline_success_classcorrect_box75", "final_damaged_success", "final_preserved_success"):
            for arm in ARMS:
                d = data["groups"][name]["arms"][arm]
                n = d["delta_norm"]
                lines.append(f"| {name}/{arm} | {fmt(n['p05'])} | {fmt(n['p25'])} | {fmt(n['median'])} | {fmt(n['p75'])} | {fmt(n['p95'])} | {fmt(d['norm_above_success_training_max']['fraction'], 100)}% |")
    lines += ["", "数据 SHA256、定义、相关系数及完整配对分母见 SUMMARY.json；执行与代码身份见 run.json。这里比较成功与损伤子组属于按结局分层，其差异不能直接解释为 FPR 增长或位移造成损伤的独立效应。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    out = (args.out or study / "runs" / "RUN_damage_existing_rows_20261006").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "final": study / "runs" / "RUN_stage1_final_complete_seed0_retry1" / "PER_CANDIDATE.jsonl",
        "dev": study / "runs" / "RUN_stage1_dev_original_metrics_seed0" / "PER_CANDIDATE.jsonl",
        "quality_model_metadata": study / "runs" / "RUN_stage1_quality_seed0" / "MODEL.json",
        "trained_states_implementation": study / "scripts" / "qcr_train_fixed.py",
    }
    inputs = [{"name": name, "path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size} for name, path in paths.items()]
    rho = float(json.loads(paths["quality_model_metadata"].read_text(encoding="utf-8"))["rho"])
    run = {
        "schema_version": 2,
        "run_id": out.name,
        "study_id": json.loads((study / "study.json").read_text(encoding="utf-8"))["study_id"],
        "source_kind": "derived_diagnostic",
        "status": "running",
        "started_at": now(),
        "timezone": "Asia/Shanghai",
        "execution_environment": "local",
        "model_execution": False,
        "parameter_updates": False,
        "inputs": inputs,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "command": [sys.executable, str(Path(__file__).resolve()), "--study", str(study), "--out", str(out)],
        "working_directory": str(Path.cwd()),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "scope": "Descriptive post-hoc paired diagnostics of existing one-to-one TAL candidate rows; FINAL and DEV analyzed separately; no AP/all-GT recall or new inference.",
    }
    save(out / "run.json", run)
    try:
        datasets = {name: Data(paths[name], rho) for name in ("final", "dev")}
        summary = {
            "source_kind": "derived_diagnostic",
            "parameter_updates": False,
            "started_at": run["started_at"],
            "finished_at": now(),
            "input_sha256": {item["name"]: item["sha256"] for item in inputs},
            "radius": {"rho": rho, "success_training_max_radius": rho / 4, "step1_eta": rho / 2, "step2_eta": rho / 4, "description": "Actual Success states use rr=.5*rho and largest perturbation .5*rr; all constants are fixed existing values."},
            "definitions": {
                "baseline_success": "normal original-image iou_A >= .75",
                "classcorrect_box75": "class_correct == true and box_iou >= .75",
                "margin_bins": "[.75,.8), [.8,.9), [.9,1] on baseline original-image IoU",
                "size": "COCO annotation original area: small <32^2, medium [32^2,96^2), large >=96^2",
                "layers": "native pyramid_level 0/1/2 maps to P3/P4/P5",
                "support_fp_pixels": "fpr multiplied by original_pixel_support_count minus original_pixel_support_gt_positive; fixed predicted-box support only",
                "paired_delta": "same-row arm minus baseline; image macro averages defined within-image pairs then images; undefined metrics excluded only from that metric",
                "extrapolation": "Norm outside Success training ball is distributional geometric mismatch; Failure training states may cover larger norms, so this is not globally unseen coefficient space.",
                "statistical_limits": "post-hoc descriptive associations; no causal attribution; no step/radius/checkpoint selection; no bootstrap CI computed for these added subgroups",
            },
            "splits": {name: dataset.analyze() for name, dataset in datasets.items()},
        }
        summary["finished_at"] = now()
        save(out / "SUMMARY.json", summary)
        (out / "TABLES.md").write_text(tables(summary), encoding="utf-8")
        run.update(status="completed", finished_at=now(), artifact_completeness="complete", return_code=0,
                   artifacts=[{"path": str(out / name), "sha256": sha(out / name)} for name in ("SUMMARY.json", "TABLES.md")])
        save(out / "run.json", run)
        print(json.dumps({"out": str(out), "final_rows": datasets["final"].n, "dev_rows": datasets["dev"].n, "status": "completed"}, ensure_ascii=False))
    except Exception as error:
        run.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}", return_code=1, artifact_completeness="partial")
        save(out / "run.json", run)
        raise


if __name__ == "__main__":
    main()
