"""Test whether the failure gate selects correctable failures and whether local ROI shrinks correction magnitude."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def key(row):
    return (int(row["image_id"]), int(row["annotation_id"]))


def ranking(label, score):
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=float)
    pos, neg = int(label.sum()), int((~label).sum())
    assert pos and neg
    auc = (rankdata(score)[label].sum() - pos * (pos + 1) / 2) / (pos * neg)
    order = np.argsort(-score, kind="stable")
    sorted_y = label[order].astype(float)
    ap = ((np.cumsum(sorted_y) / np.arange(1, len(label) + 1)) * sorted_y).sum() / pos
    return dict(n=len(label), positives=pos, prevalence=pos / len(label),
                auroc=float(auc), auprc=float(ap))


def ranking_cluster_ci(label, score, image_ids, repetitions=2000):
    images = np.unique(image_ids)
    members = [np.flatnonzero(image_ids == iid) for iid in images]
    rng = np.random.default_rng(20260927)
    values = []
    for _ in range(repetitions):
        sample = np.concatenate([members[i] for i in rng.integers(0, len(images), len(images))])
        if len(np.unique(label[sample])) < 2:
            continue
        rank = ranking(label[sample], score[sample])
        values.append([rank["auroc"], rank["auprc"]])
    return dict(auroc=np.quantile(values, [.025, .975], axis=0)[:, 0].tolist(),
                auprc=np.quantile(values, [.025, .975], axis=0)[:, 1].tolist())


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


def paired_ci(values, image_index, repetitions=3000):
    images = int(image_index.max()) + 1
    totals = np.bincount(image_index, weights=values, minlength=images)
    counts = np.bincount(image_index, minlength=images)
    samples = np.random.default_rng(20260927).integers(0, images, size=(repetitions, images))
    trial = totals[samples].sum(1) / np.maximum(counts[samples].sum(1), 1)
    return [float(x) for x in np.quantile(trial, [.025, .975])]


def main(a):
    scores = json.loads((a.scores / "SCORES.json").read_text())
    rows = json.loads((a.evaluation / "ROWS.json").read_text())
    magnitudes = json.loads((a.magnitudes / "MAGNITUDES.json").read_text())
    gate_result = json.loads((a.gate_result / "RESULTS.json").read_text())
    assert len(scores) == len(rows) == len(magnitudes) == 1346
    rowmap = {key(x): x for x in rows}
    magmap = {key(x): x for x in magnitudes}
    assert len(rowmap) == len(magmap) == len(rows)
    ordered = [rowmap[key(x)] for x in scores]
    mag = [magmap[key(x)] for x in scores]
    failure = np.array([bool(x["failure"]) for x in scores])
    assert np.array_equal(failure, [x["original_image_iou"] < .75 for x in ordered])
    image_id = np.array([x["image_id"] for x in scores])
    _, image_index = np.unique(image_id, return_inverse=True)
    repair = {arm: np.array([r["arms"][arm]["image_iou"] >= .75 for r in ordered]) & failure
              for arm in ("h_only", "true_local")}
    damage = {arm: np.array([r["arms"][arm]["image_iou"] < .75 for r in ordered]) & ~failure
              for arm in ("h_only", "true_local")}
    output = dict(repairable={}, gate_op={}, magnitude={}, repair_overlap={}, damage_overlap={}, geometry={})
    for arm, positive in repair.items():
        output["repairable"][arm] = {}
        output["geometry"][arm] = {}
        for field, values in (("original_mask_iou", np.array([r["original_image_iou"] for r in ordered])),
                              ("box_iou", np.array([r["box_iou"] for r in ordered]))):
            output["geometry"][arm][field] = dict(
                repairable_failure=distribution(values[positive]),
                nonrepairable_failure=distribution(values[failure & ~positive]))
        for feature, task in (("h_raw_detection", "failure"), ("box_score", "failure"),
                              ("h_raw_detection", "high_gap")):
            prediction = np.array([x["scores"][feature][task] for x in scores])
            item = ranking(positive[failure], prediction[failure])
            item["image_cluster_ci95"] = ranking_cluster_ci(positive[failure], prediction[failure], image_id[failure])
            item["score_repairable"] = distribution(prediction[positive])
            item["score_nonrepairable_failure"] = distribution(prediction[failure & ~positive])
            output["repairable"][arm][f"{feature}_{task}"] = item
    for gate, ops in gate_result["gates"].items():
        score_name = ("h_raw_detection", "high_gap") if gate.endswith("high_gap") else (
            ("box_score", "failure") if gate.startswith("box_score") else ("h_raw_detection", "failure"))
        prediction = np.array([x["scores"][score_name[0]][score_name[1]] for x in scores])
        output["gate_op"][gate] = {}
        for rate, op in ops.items():
            selected = prediction > op["threshold"]
            output["gate_op"][gate][rate] = {}
            for arm in repair:
                selected_failed = int((selected & failure).sum())
                selected_repair = int((selected & repair[arm]).sum())
                total_repair = int(repair[arm].sum())
                baseline_rate = total_repair / int(failure.sum())
                output["gate_op"][gate][rate][arm] = dict(
                    selected_failed=selected_failed, selected_repairable=selected_repair,
                    fraction_of_all_repairable=selected_repair / total_repair,
                    repairable_fraction_in_selected_failure=selected_repair / selected_failed,
                    enrichment=(selected_repair / selected_failed) / baseline_rate)
    for arm in ("h_only", "true_local"):
        norm = np.array([x[arm] for x in mag])
        output["magnitude"][arm] = dict(
            success=distribution(norm[~failure]), failure=distribution(norm[failure]),
            repairable_failure=distribution(norm[repair[arm]]),
            nonrepairable_failure=distribution(norm[failure & ~repair[arm]]))
    h = np.array([x["h_only"] for x in mag])
    local = np.array([x["true_local"] for x in mag])
    output["magnitude"]["true_minus_h"] = {}
    for name, selection in (("all", np.ones(len(failure), dtype=bool)),
                            ("original_success", ~failure), ("original_failure", failure)):
        output["magnitude"]["true_minus_h"][name] = dict(
            mean=float((local[selection] - h[selection]).mean()),
            ci95=paired_ci((local - h)[selection], np.unique(image_index[selection], return_inverse=True)[1]),
            true_over_h_mean=float(local[selection].mean() / h[selection].mean()))
    output["repair_overlap"] = dict(h_only=int(repair["h_only"].sum()),
        true_local=int(repair["true_local"].sum()), both=int((repair["h_only"] & repair["true_local"]).sum()))
    output["damage_overlap"] = dict(h_only=int(damage["h_only"].sum()),
        true_local=int(damage["true_local"].sum()), both=int((damage["h_only"] & damage["true_local"]).sum()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ANALYSIS.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(dict(repairable=output["repairable"], magnitude=output["magnitude"],
                          repair_overlap=output["repair_overlap"], damage_overlap=output["damage_overlap"]),
                     indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("scores", "evaluation", "magnitudes", "gate_result", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    main(p.parse_args())
