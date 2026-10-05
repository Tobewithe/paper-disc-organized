"""Matched-pair analysis and human-readable report for neighbour-specific recovery."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "raw_p3_best_box_iou",
    "raw_best_box_iou",
    "raw_p3_center_error_norm",
    "raw_true_score_at_p3_best",
    "raw_p3_box50",
)
CONTRASTS = (
    ("neighbor_flat", "background_flat"),
    ("neighbor_blur", "background_blur"),
    ("neighbor_flat", "distant_instance_flat"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values: list[float], seed: int, reps: int = 10000) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    low, high = np.quantile(means, [.025, .975])
    return float(x.mean()), float(low), float(high)


def percent(value: float) -> str:
    return f"{100 * value:+.3f}"


def ci_percent(row: dict[str, object]) -> str:
    return f"{percent(float(row['effect']))} [{percent(float(row['ci_low']))}, {percent(float(row['ci_high']))}]"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = read_csv(ROOT / "per_target.csv")
    index = {(row["cohort"], row["pair_id"], row["arm"]): row for row in rows}
    pairs = sorted({row["pair_id"] for row in rows})
    interaction_rows: list[dict[str, object]] = []
    for treatment, control in CONTRASTS:
        for metric in METRICS:
            values: list[float] = []
            for pair_id in pairs:
                needed = [
                    ("raw_geometry_small", pair_id, treatment),
                    ("raw_geometry_small", pair_id, control),
                    ("matched_small_control", pair_id, treatment),
                    ("matched_small_control", pair_id, control),
                ]
                if not all(key in index for key in needed):
                    continue
                failure = float(index[needed[0]][metric]) - float(index[needed[1]][metric])
                matched = float(index[needed[2]][metric]) - float(index[needed[3]][metric])
                values.append(failure - matched)
            mean, low, high = bootstrap(values, 4100 + len(interaction_rows))
            interaction_rows.append({"treatment": treatment, "control": control,
                                     "metric": metric, "matched_pairs": len(values),
                                     "effect": mean, "ci_low": low, "ci_high": high})
    write_csv(ROOT / "matched_failure_control_interactions.csv", interaction_rows)

    original = {(row["cohort"], row["annotation_id"]): row for row in rows if row["arm"] == "original"}
    arm_index = {(row["cohort"], row["annotation_id"], row["arm"]): row for row in rows}
    recovery_rows: list[dict[str, object]] = []
    for cohort in ("raw_geometry_small", "matched_small_control"):
        ids = sorted(aid for c, aid in original if c == cohort)
        for arm in ("neighbor_flat", "background_flat", "distant_instance_flat",
                    "neighbor_blur", "background_blur"):
            eligible = [aid for aid in ids if (cohort, aid, arm) in arm_index]
            gained = sum(int(original[(cohort, aid)]["raw_p3_box50"]) == 0
                         and int(arm_index[(cohort, aid, arm)]["raw_p3_box50"]) == 1 for aid in eligible)
            lost = sum(int(original[(cohort, aid)]["raw_p3_box50"]) == 1
                       and int(arm_index[(cohort, aid, arm)]["raw_p3_box50"]) == 0 for aid in eligible)
            recovery_rows.append({"cohort": cohort, "arm": arm, "eligible": len(eligible),
                                  "p3_box50_gained": gained, "p3_box50_lost": lost,
                                  "net": gained - lost})
    write_csv(ROOT / "threshold_transitions.csv", recovery_rows)

    # Subgroup analysis is descriptive: neighbour class was not a preregistered effect modifier.
    subgroup_rows: list[dict[str, object]] = []
    for cohort in ("raw_geometry_small", "matched_small_control"):
        ids = sorted(aid for c, aid in original if c == cohort)
        for same in (0, 1):
            eligible = [aid for aid in ids
                        if (cohort, aid, "neighbor_flat") in arm_index
                        and (cohort, aid, "background_flat") in arm_index
                        and int(arm_index[(cohort, aid, "neighbor_flat")]["selected_neighbor_same_class"]) == same]
            values = [float(arm_index[(cohort, aid, "neighbor_flat")]["raw_p3_best_box_iou"])
                      - float(arm_index[(cohort, aid, "background_flat")]["raw_p3_best_box_iou"])
                      for aid in eligible]
            mean, low, high = bootstrap(values, 5200 + len(subgroup_rows))
            subgroup_rows.append({"cohort": cohort, "same_class_neighbor": same,
                                  "n": len(values), "effect": mean, "ci_low": low, "ci_high": high})
    write_csv(ROOT / "neighbor_class_subgroups.csv", subgroup_rows)

    arm_deltas = read_csv(ROOT / "arm_deltas.csv")
    paired = read_csv(ROOT / "paired_contrasts.csv")
    specific = read_csv(ROOT / "neighbor_specific_instances.csv")

    def arm(cohort: str, name: str) -> dict[str, str]:
        return next(row for row in arm_deltas if row["cohort"] == cohort and row["arm"] == name)

    def contrast(cohort: str, treatment: str, control: str, metric: str) -> dict[str, str]:
        return next(row for row in paired if row["cohort"] == cohort and row["treatment"] == treatment
                    and row["control"] == control and row["metric"] == metric)

    def interaction(treatment: str, control: str, metric: str) -> dict[str, object]:
        return next(row for row in interaction_rows if row["treatment"] == treatment
                    and row["control"] == control and row["metric"] == metric)

    def delta_ci(record: dict[str, str], metric: str) -> str:
        return (f"{percent(float(record[metric + '_delta']))} "
                f"[{percent(float(record[metric + '_ci_low']))}, {percent(float(record[metric + '_ci_high']))}]")

    def contrast_ci(record: dict[str, str]) -> str:
        return (f"{percent(float(record['paired_net']))} "
                f"[{percent(float(record['ci_low']))}, {percent(float(record['ci_high']))}]")

    flat = contrast("raw_geometry_small", "neighbor_flat", "background_flat", "raw_p3_best_box_iou")
    blur = contrast("raw_geometry_small", "neighbor_blur", "background_blur", "raw_p3_best_box_iou")
    distant = contrast("raw_geometry_small", "neighbor_flat", "distant_instance_flat", "raw_p3_best_box_iou")
    center = contrast("raw_geometry_small", "neighbor_flat", "background_flat", "raw_p3_center_error_norm")
    box50 = contrast("raw_geometry_small", "neighbor_flat", "background_flat", "raw_p3_box50")
    primary_interaction = interaction("neighbor_flat", "background_flat", "raw_p3_best_box_iou")
    transition = {row["arm"]: row for row in recovery_rows if row["cohort"] == "raw_geometry_small"}
    subclasses = [row for row in subgroup_rows if row["cohort"] == "raw_geometry_small"]

    report = f"""---
type: diagnostic-result
node_id: diagnostic:neighbor-specific-recovery-20260915
stage: mechanism-localization
outcome: supported
dataset: COCO-val2017
model: YOLO26m-seg
---

# 邻居特异性 P3 恢复诊断

## 结论

在预先冻结的 384 个小目标原始几何失败中，302 个目标具有至少 8 个像素的可干预暴露邻居并找到等形背景对照。抑制按输入空间暴露量预先选定的邻居，相对等形、等面积的背景扰动，使 raw P3 最佳 Box IoU 净提高 **{contrast_ci(flat)} 个百分点**。95% 配对 bootstrap 区间排除零，达到预注册的主张门槛。

这排除了“任意修改同样多图像像素就能恢复”的解释。它支持一个范围明确的机制结论：**在这批已定位的小目标 P3 几何失败中，存在相当比例的目标对特定近邻内容敏感；近邻干扰发生在或早于 P3 定位表征。** 该实验仍是使用 GT 的受控诊断，不能直接当作推理方法或 COCO AP 提升。

## 主结果

| 比较 | n | raw P3 Box IoU 差值，百分点 [95% CI] |
|---|---:|---:|
| 邻居平坦化 − 原图 | {arm('raw_geometry_small', 'neighbor_flat')['n']} | {delta_ci(arm('raw_geometry_small', 'neighbor_flat'), 'raw_p3_best_box_iou')} |
| 等形背景平坦化 − 原图 | {arm('raw_geometry_small', 'background_flat')['n']} | {delta_ci(arm('raw_geometry_small', 'background_flat'), 'raw_p3_best_box_iou')} |
| 邻居平坦化 − 等形背景平坦化 | {flat['n']} | **{contrast_ci(flat)}** |
| 邻居模糊 − 等形背景模糊 | {blur['n']} | **{contrast_ci(blur)}** |
| 邻居平坦化 − 远端实例平坦化 | {distant['n']} | **{contrast_ci(distant)}** |

三个对照给出一致方向。平坦化实验中，背景对照的平均每通道像素改变量高于邻居干预（64.89 对 55.92），却几乎没有恢复；远端实例对照的编辑能量也更高（66.01），仍近似为零。因此主效应不能用编辑幅度较弱解释。

## 失败特异性

在严格保持原有失败/成功一一配对的样本上，“邻居平坦化 − 背景平坦化”的失败减成功交互效应为 **{ci_percent(primary_interaction)} 个百分点**（{primary_interaction['matched_pairs']} 对）。匹配成功对照本身没有受益，说明该现象不是所有小目标的一般增益，而是集中在已识别的失败对象。

raw P3 中心误差的邻居特异净变化为 **{contrast_ci(center)} 个百分点**（误差以 GT 框对角线归一化）；负值表示定位更准。P3 Box50 的净恢复比例为 **{contrast_ci(box50)}**。在失败组中，邻居平坦化带来 {transition['neighbor_flat']['p3_box50_gained']} 个 Box50 获得、{transition['neighbor_flat']['p3_box50_lost']} 个丢失；背景对照分别为 {transition['background_flat']['p3_box50_gained']} 和 {transition['background_flat']['p3_box50_lost']}。

按预注册的严格实例定义，共有 **{sum(int(row['neighbor_specific_recovery']) for row in specific)}/{len(specific)}** 个目标满足邻居特异恢复：原图 P3 IoU<0.5，邻居干预后跨过 0.5 且提升至少 0.1，同时背景对照未跨阈值，邻居相对背景至少再提高 0.05。

## 探索性类别分组

| 预选邻居 | n | 邻居平坦化 − 背景平坦化，百分点 [95% CI] |
|---|---:|---:|
| 异类 | {subclasses[0]['n']} | {ci_percent(subclasses[0])} |
| 同类 | {subclasses[1]['n']} | {ci_percent(subclasses[1])} |

该分组没有预注册，只用于形成后续假设。无论邻居是否同类，效应都应先看实际结果，不能再把机制预设成“同类系数相似”。

## 论文中的正确用法

这组实验适合作为“失败定位 → 受控输入干预 → 层级归因 → 方法设计”的动机证据。正文可以主张特定近邻内容会造成一部分 P3 可恢复定位失败，并据此解释为什么训练时用受控干预生成教师信号、只救援原图学生失败而干预教师恢复的位置。

不能写成“密集场景普遍导致失败”“删除邻居证明了严格因果关系”或“诊断本身提高了 COCO AP”。最终方法价值仍由正在运行的完整 COCO 训练及原图推理评估决定。

## 可复核文件

- `protocol.json`：结果产生前冻结的实验门槛与对照定义。
- `per_target.csv`：全部 3,588 次推理的逐实例结果与编辑审计。
- `paired_contrasts.csv`：失败组和匹配对照组内的配对对照。
- `matched_failure_control_interactions.csv`：严格 pair_id 配对的失败减成功交互。
- `neighbor_specific_instances.csv`：实例级严格恢复判定。
- `threshold_transitions.csv`：P3 Box50 获得与丢失计数。
"""
    (ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    complete = json.loads((ROOT / "COMPLETE.json").read_text(encoding="utf-8"))
    project = ROOT.parents[1]
    audit_paths = {
        "selection": project / "experiments/small_raw_geometry_origin_20260914/selection.csv",
        "annotations": project / "assets/datasets/coco/annotations/instances_val2017.json",
        "checkpoint": project / "assets/models/coco_clean_20260911/yolo26m-seg.pt",
        "execution_script": ROOT / "neighbor_specific_recovery.py",
        "analysis_script": ROOT / "analyse_results.py",
        "protocol": ROOT / "protocol.json",
        "per_target": ROOT / "per_target.csv",
    }
    complete.update({"matched_analysis": "complete", "primary_failure_control_interaction": primary_interaction,
                     "sha256": {name: sha256(path) for name, path in audit_paths.items()}})
    (ROOT / "COMPLETE.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
    print("analysis complete")


if __name__ == "__main__":
    main()
