"""Fixed-candidate normal-decode evaluation for candidate-relation readout.

This is a seed-0 development screen, not COCO AP or an independent test.
Prediction-only neighbor construction happens in relation_io, not here.
Official GT supplies training supervision, evaluation and descriptive strata;
it must never select the neighbors or modify final prediction support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from runtime_utils import dump, load, gpu_image, setup
from train_evidence import feature_channels
from training_engine import make_model, resolve_config
from relation_io import read_training_image, feed
from evaluate_feedback import bce_values, baseline_for, coco_for, identity, decode_smoke
import evaluate as metrics


ARMS = ("A", "N", "S", "T", "M")
PAIRS = (("T", "S"), ("T", "M"), ("T", "A"), ("T", "N"), ("N", "A"), ("S", "A"))
# evaluate_feedback sets these globals on import. This study deliberately
# resets them after that import; otherwise its unrelated arms contaminate us.
metrics.ARMS = ARMS
metrics.PAIRS = PAIRS
GROUPS = ("all", "box_good_mask_bad", "original_success", "original_failure", "with_neighbors", "without_neighbors")


def _metadata_value(relation, name, j, n):
    """Never broadcast an image summary as if it were instance metadata."""
    if name not in relation:
        raise KeyError(f"Missing candidate relation metadata: {name}")
    value = relation[name]
    if torch.is_tensor(value):
        value = value.detach().cpu()
    if len(value) != n:
        raise ValueError(f"{name}: relation metadata length differs from candidates")
    selected = value[j]
    if torch.is_tensor(selected):
        selected = selected.tolist()
    return selected


def attach_relation_metadata(decoded, x):
    relation = x["_relation"]
    n = len(decoded)
    for j, row in enumerate(decoded):
        count = int(_metadata_value(relation, "neighbor_count", j, n))
        same = int(_metadata_value(relation, "same_class_count", j, n))
        changed = float(_metadata_value(relation, "changed_fraction", j, n))
        raw = _metadata_value(relation, "chosen_raw", j, n)
        raw = [int(r) for r in (raw if isinstance(raw, (list, tuple)) else [raw])]
        chosen = [r for r in raw if r >= 0]
        if not 0 <= same <= count or count != len(chosen):
            raise AssertionError("Neighbor count/identity metadata disagree")
        if len(set(chosen)) != len(chosen) or int(row["raw_id"]) in chosen:
            raise AssertionError("Prediction pool must exclude the query and duplicate raw IDs")
        if not metrics.finite(changed) or not 0 <= changed <= 1:
            raise ValueError("Relation changed_fraction must be in [0,1]")
        row.update(
            neighbor_count=count, same_class_neighbor_count=same,
            relation_changed_fraction=changed, neighbor_raw_ids=chosen,
            with_neighbors=bool(count), neighbor_class_source="frozen predicted classes, not GT",
        )
        row["operator_valid"] = bool(x["_operator"]["valid"][j])


def _relation_summary(rows):
    with_neighbors = [r for r in rows if r["with_neighbors"]]
    return dict(
        candidates=len(rows), with_neighbors=len(with_neighbors),
        without_neighbors=len(rows)-len(with_neighbors),
        neighbor_count_mean=metrics.avg([r["neighbor_count"] for r in rows]),
        same_class_count_mean=metrics.avg([r["same_class_neighbor_count"] for r in rows]),
        mismatch_changed_fraction_all=metrics.avg([r["relation_changed_fraction"] for r in rows]),
        mismatch_changed_fraction_with_neighbors=metrics.avg([r["relation_changed_fraction"] for r in with_neighbors]),
        same_class_is_predicted_class=True,
        true_neighbors_meaning="prediction-defined geometric neighbors; not verified distinct GT instances",
    )


def summarize(rows, out, cfg, expected, audit, timing):
    tables = {}
    for split in ("fit", "dev"):
        ss = [r for r in rows if r["split"] == split]
        groups = {
            "all": ss,
            "box_good_mask_bad": [r for r in ss if r["box_good_mask_bad"]],
            "original_success": [r for r in ss if r["mask75_A"]],
            "original_failure": [r for r in ss if not r["mask75_A"]],
            "with_neighbors": [r for r in ss if r["with_neighbors"]],
            "without_neighbors": [r for r in ss if not r["with_neighbors"]],
        }
        for name, group in groups.items():
            tables[split+":"+name] = metrics.make_table(
                group, 20261004, cfg["bootstrap"], "exploratory paired whole-image development screen"
            )

    def delta(group, comparison):
        return tables[group]["comparisons"][comparison]["iou"]["image_macro"]["delta"]

    required = {name: delta("dev:all", name) for name in ("T_minus_S", "T_minus_M", "T_minus_A", "T_minus_N")}
    target = delta("dev:box_good_mask_bad", "T_minus_A")
    ta = required["T_minus_A"]
    all_threshold = float(cfg.get("screen_effect_all", .002))
    target_threshold = float(cfg.get("screen_effect_target", .005))
    guard_margin = float(cfg.get("screen_all_guard_margin", -.001))
    practical = bool((metrics.finite(ta) and ta >= all_threshold) or
                     (metrics.finite(target) and target >= target_threshold))
    exceeds_controls = all(metrics.finite(v) and v > 0 for v in required.values())
    guard = bool(metrics.finite(ta) and ta >= guard_margin)
    transition = tables["dev:all"]["comparisons"]["T_minus_A"]["mask75_transition"]
    net_nonnegative = bool(transition["valid_candidates"] and transition["net"] >= 0)
    complete_pairs = all(tables["dev:all"]["comparisons"][name]["iou"]["complete_pairs"] for name in required)
    passed = bool(exceeds_controls and practical and guard and net_nonnegative and complete_pairs)
    decision = "positive_screen_needs_confirmation" if passed else "stop_current_implementation_no_predeclared_positive_screen"
    criteria = dict(
        required_dev_all_iou_deltas=required, target_T_minus_A=target,
        all_effect_threshold=all_threshold, target_effect_threshold=target_threshold,
        all_guard_margin=guard_margin, exceeds_all_controls=exceeds_controls,
        practical_effect=practical, all_point_guard=guard,
        net_mask75_nonnegative=net_nonnegative, net_mask75=transition["net"], complete_pairs=complete_pairs,
        interpretation="Point-estimate screening gate only; CI reported, no confirmed benefit/noninferiority or causal claim.",
    )
    result = dict(
        tables=tables, expected=expected, audit=audit, seconds=timing, decision=decision, criteria=criteria,
        relation={split: _relation_summary([r for r in rows if r["split"] == split]) for split in ("fit", "dev")},
        primary="dev:all T_minus_S image_macro original_mask_iou; T_minus_M relation control",
        single_seed_exploratory=True, historical_development=True, automatic_followup=False,
        not_coco_ap=True,
    )
    metrics.write_json(out/"SUMMARY.json", result)
    with (out/"PER_IMAGE.jsonl").open("w", encoding="utf-8") as stream:
        for group in metrics.image_groups(rows):
            rec = dict(split=group[0]["split"], image_id=group[0]["image_id"], n=len(group))
            for arm in ARMS:
                for metric in metrics.METRICS:
                    rec[f"{metric}_{arm}"] = metrics.avg([metrics.metric_value(r, metric, arm) for r in group])
            rec["relation"] = _relation_summary(group)
            stream.write(json.dumps(metrics.clean(rec), ensure_ascii=False, allow_nan=False)+"\n")

    def fmt(value):
        return "NA" if value is None else f"{100*value:+.4f}"

    report = [
        "# 候选关系读取：固定预算短程结果", "",
        f"判定：{decision}。固定seed0、最后第3轮；属于历史开发数据筛选，不能称完整COCO AP或独立确认。", "",
        "A是原始冻结模型；N是原生系数分支普通微调。S/T/M采用同容量关系读取头与相同训练预算：S在邻居位置读取自身系数响应，T读取预测池中邻居的真实空间响应，M在邻居支持内滚动响应，破坏空间对应。这里的“邻居”来自预测，不保证是不同真实实例。", "",
        "自身预测框、原型、类别、分数、评价候选和解码保持冻结；GT参与监督和结果分层，不参与邻居选择。普通BCE已包含目标与非目标监督，本实验检验新增预测关系信息是否有额外收益。", "",
        "|分组|图片/候选|比较|图片macro IoU差(pp)|95%图片配对区间(pp)|候选IoU差(pp)|Mask75修复/损伤/净增|",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for group in ("fit:all", *("dev:"+name for name in GROUPS)):
        table = tables[group]
        for a, b in PAIRS:
            pair = table["comparisons"][f"{a}_minus_{b}"]
            m, tr = pair["iou"]["image_macro"], pair["mask75_transition"]
            report.append(
                f"|{group}|{table['n_images']}/{table['n_candidates']}|{a}-{b}|{fmt(m['delta'])}|"
                f"[{fmt(m['ci95'][0])},{fmt(m['ci95'][1])}]|{fmt(pair['iou']['candidate']['delta'])}|"
                f"{tr['repair']}/{tr['damage']}/{tr['net']:+d}|"
            )
    report += ["", "|分组|有邻居/全部候选|平均邻居数|M在有邻居候选中改变比例|", "|---|---:|---:|---:|"]
    for split, relation in result["relation"].items():
        nmean = relation["neighbor_count_mean"]
        report.append(f"|{split}|{relation['with_neighbors']}/{relation['candidates']}|{nmean if nmean is not None else 'NA'}|{fmt(relation['mismatch_changed_fraction_with_neighbors'])}%|")
    report += [
        "", "Coverage、固定预测框内像素AUC/FPR、官方训练标签BCE、原成功损伤、逐候选与逐图统计见SUMMARY.json。FPR包含邻居和背景，不直接等于邻居泄漏。无邻居、无效算子和空预测均保留；无正样本图片单列，不伪造实例指标。", "",
        "筛选条件：dev全体macro IoU的T-S、T-M、T-A、T-N均为正；T-A全体至少+0.2pp或原框好掩码差组至少+0.5pp；全体T-A不低于−0.1pp，Mask75净修复非负。仅点估计筛选，不用CI跨零宣称非劣，也不把GT辅助的冻结候选诊断称完整推理收益。", "",
        "解释边界：T-M可能包含空间分布变化影响，不能单独证明邻居导致失败；T还必须超过自身读取、原生微调和原模型。S不含邻居身份但保留邻居几何布局，检验的是邻居响应信息的增量。共享原型、固定候选和短程预算限定结论范围。", "",
        "结束后不自动加轮数、学习率搜索、更多随机种子或全量训练。正信号另做独立确认；负信号只停止当前实现，不宣布候选关系机制普遍无效。",
    ]
    (out/"REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoints", required=True)
    args = parser.parse_args()
    root = Path(args.config).parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    setup(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU host required")
    if (out/"PER_CANDIDATE.jsonl").exists() or (out/"COMPLETE.json").exists():
        raise RuntimeError("Existing evaluation output; a fresh independent Run is required")
    if set(cfg["arms"]) != set(ARMS)-{"A"}:
        raise ValueError("Exactly N/S/T/M training checkpoints are required")
    index = json.loads((root/"INDEX.json").read_text(encoding="utf-8-sig"))
    selected = {"fit": index["fit"][:cfg["fit_eval_images"]], "dev": index["dev"]}
    if {int(r["image_id"]) for r in selected["fit"]} & {int(r["image_id"]) for r in selected["dev"]}:
        raise AssertionError("Fit/dev image overlap")
    first = next(r for items in selected.values() for r in items if r["n"])
    channels = feature_channels(read_training_image(cfg, int(first["image_id"])))
    paths = json.loads(Path(args.checkpoints).read_text(encoding="utf-8-sig"))
    models = {}
    for mode in cfg["arms"]:
        checkpoint = load(paths[mode])
        if checkpoint["epoch"] != cfg["epochs"] or checkpoint["config"] != cfg or checkpoint["mode"] != mode:
            raise AssertionError(f"Checkpoint/config/fixed-epoch mismatch: {mode}")
        model = make_model(channels, mode, cfg).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[mode] = model

    baseline, coco = baseline_for(cfg), coco_for(cfg)
    rows, visited, expected = [], set(), {}
    started, base_error = time.monotonic(), 0.
    timing = {a: 0. for a in cfg["arms"]}
    for split, items in selected.items():
        expected[split] = dict(
            planned_images=len(items), effective_images=sum(r["n"] > 0 for r in items),
            candidates=sum(r["n"] for r in items), no_positive=[r["image_id"] for r in items if not r["n"]],
        )
        for position, item in enumerate(items):
            if not item["n"]:
                continue
            x = read_training_image(cfg, int(item["image_id"]))
            if x["split"] != split or len(x["rows"]) != item["n"]:
                raise AssertionError("Frozen INDEX/cache population mismatch")
            fs, selection = feed([x])
            coefficients = {}
            with torch.no_grad():
                for mode, model in models.items():
                    torch.cuda.synchronize()
                    t = time.monotonic()
                    coefficients[mode] = model(fs, selection)[0]
                    torch.cuda.synchronize()
                    timing[mode] += time.monotonic()-t
                gx = gpu_image(x)
                coefficients["A"] = gx["c0"]
                losses = {a: bce_values(gx, c) for a, c in coefficients.items()}
                decoded = metrics.evaluate_image(gx, coefficients, coco)
            attach_relation_metadata(decoded, x)
            for j, row in enumerate(decoded):
                key = identity(row)
                if key in visited or key not in baseline:
                    raise ValueError("Frozen identity/baseline join failed")
                visited.add(key)
                old = baseline[key]
                error = abs(row["iou_A"]-old[0])
                base_error = max(base_error, error)
                if error > 1e-6 or row["mask75_A"] != old[1]:
                    raise AssertionError("Original A normal decode changed")
                row["cached_baseline_iou_absolute_error"] = error
                for arm in ARMS:
                    row[f"bce_{arm}"] = float(losses[arm][j])
            metrics.append_rows(out/"PER_CANDIDATE.jsonl", decoded)
            rows.extend(decoded)
            if position % 10 == 0 or position+1 == len(items):
                state = dict(stage="evaluation", split=split, images=position+1, planned=len(items),
                             candidates=len(rows), elapsed_s=time.monotonic()-started)
                dump(out/"PROGRESS.json", state)
                print(json.dumps(state), flush=True)
            del x, gx, fs, selection, coefficients, decoded, losses
    if len(rows) != sum(r["candidates"] for r in expected.values()):
        raise AssertionError("Final evaluation candidate count differs from frozen manifest")
    audit = dict(
        baseline_max_error=base_error, all_identities_unique=True, all_candidates_retained=True,
        checkpoints=paths, identity_fields=["split", "image_id", "annotation_id", "raw_id"],
        frozen_instance_identity_fields=["image_id", "annotation_id", "raw_id"],
        normal_decode="unchanged inherited evaluator; original COCO annToMask and original predicted boxes",
        no_val_tuning=True, fixed_epoch=cfg["epochs"],
        neighbor_protocol="prediction-only pool, query excluded by raw ID; no GT/official-positive-list neighbor selection",
        caveat="Evaluation queries are frozen official TAL positives; this conditional candidate diagnostic is not full inference AP.",
    )
    dump(out/"EVALUATION_AUDIT.json", audit)
    result = summarize(rows, out, cfg, expected, audit, timing)
    dump(out/"COMPLETE.json", dict(completed=True, decision=result["decision"],
                                 elapsed_s=time.monotonic()-started, automatic_followup=False))


if __name__ == "__main__":
    main()

