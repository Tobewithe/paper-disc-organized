"""CPU-only scoring contracts and actual COCOeval per-image match cache."""
from __future__ import annotations
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ARMS = ("baseline", "target_I", "target_H")
PROTOCOL_SHA = "cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057"
ANN_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
VAL_LIST_SHA = "b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db"
BASE_PRED_SHA = "6c757a3a3ab302b807de7b77646cd7acc7d84afa3b2360e50b070e32d80d8f43"
FINAL8_HEAD_SHA = "9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8"
COCO_KEYS = ("image_id", "category_id", "bbox", "score", "segmentation")
IDENTITY_KEYS = ("image_id", "category_id", "bbox", "score", "detection_index", "raw_input_box_xyxy", "raw_confidence", "box_xyxy", "model_class")
METRICS = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge")
PAIR_FIELDS = ("matched_count", "baseline_success", "baseline_failure", "damage", "repair", "baseline_iou_sum", "method_iou_sum", "delta_iou_sum", "success_delta_sum", "coverage_sum", "coverage_n", "purity_sum", "purity_n", "fp_g_sum", "fp_g_n", "fn_g_sum", "fn_g_n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def seal_outputs(root, relative_paths):
    root = Path(root).resolve(strict=True)
    result = {}
    for relative in relative_paths:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        check(path.is_file(), "Expected output is not a file: " + relative)
        result[relative] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    return result


def verify_output_seal(root, seal, required, lock=None):
    check(isinstance(seal, dict) and set(required).issubset(seal), "Producer output seal is incomplete")
    root = Path(root).resolve(strict=True)
    for relative, record in seal.items():
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        check(path.is_file() and path.stat().st_size == record["size_bytes"], "Sealed output size differs: " + relative)
        digest = lock.add(path) if lock else sha(path)
        check(digest == record["sha256"], "Sealed output SHA differs: " + relative)


def validate_completed_record(record, expected_status, summary, complete, summary_sha256, source_lock_sha256):
    """A SUMMARY left behind by a failed/cancelled process is insufficient."""
    check(record.get("status") == "completed" and record.get("return_code") == 0
          and record.get("artifact_completeness") == "complete", "Recorded process did not actually complete")
    check(summary.get("status") == expected_status and summary.get("passed") is True
          and complete.get("status") == expected_status and complete.get("summary_sha256") == summary_sha256,
          "Terminal completion/summary binding differs")
    check(summary.get("source_lock_sha256") == source_lock_sha256
          and complete.get("source_lock_sha256") == source_lock_sha256, "Terminal source lock binding differs")


def completed_receipt(root, expected_status, lock):
    root = Path(root)
    for name in ("run.json", "SUMMARY.json", "COMPLETE.json", "SOURCE_LOCK.json"):
        lock.add(root / name)
    observed, summary, complete = [read_json(root / name) for name in ("run.json", "SUMMARY.json", "COMPLETE.json")]
    validate_completed_record(observed, expected_status, summary, complete, sha(root / "SUMMARY.json"), sha(root / "SOURCE_LOCK.json"))
    # runner's own final hashes bind these receipts to the actual exited process,
    # including a verbatim run.json transferred from another Windows host.
    original_root = Path(observed["locations"][0]["path"])
    recorded_artifacts = {}
    for artifact in observed.get("artifacts", []):
        try:
            relative = Path(artifact["path"]).relative_to(original_root).as_posix()
        except ValueError:
            continue
        recorded_artifacts[relative] = artifact
    for relative in ("SUMMARY.json", "COMPLETE.json", "SOURCE_LOCK.json"):
        check(relative in recorded_artifacts and recorded_artifacts[relative].get("exists") is True
              and recorded_artifacts[relative].get("sha256") == sha(root / relative),
              "Actual runner terminal artifact hash missing/different: " + relative)
    source = read_json(root / "SOURCE_LOCK.json")
    for original, entry in source["files"].items():
        check(entry["sha256_before"] == entry["sha256_after"], "Original input/source changed during producer Run")
        if entry.get("snapshot"):
            check(lock.add(root / entry["snapshot"]) == entry["sha256_after"], "Executed source snapshot differs")
    check(summary.get("artifacts") == complete.get("artifacts"), "Terminal output seal binding differs")
    verify_output_seal(root, summary["artifacts"], [], lock)
    return summary, complete, source


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def json_array(path):
    decoder, buffer, position, first, value_next = json.JSONDecoder(), "", 0, True, True
    with Path(path).open(encoding="utf-8-sig") as f:
        buffer = f.read(1024 * 1024)
        position = len(buffer) - len(buffer.lstrip())
        check(position < len(buffer) and buffer[position] == "[", "COCO array opener missing")
        position += 1
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                buffer, position = f.read(1024 * 1024), 0
                check(bool(buffer), "Truncated COCO array")
            if buffer[position] == "]":
                check(first or not value_next, "COCO array trailing comma")
                check(not buffer[position + 1:].strip() and not f.read().strip(), "COCO array trailing data")
                return
            if not value_next:
                check(buffer[position] == ",", "COCO separator missing")
                position += 1
                value_next = True
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                extra = f.read(1024 * 1024)
                check(bool(extra), "Malformed COCO record")
                buffer, position = buffer[position:] + extra, 0
                continue
            check(isinstance(value, dict) and set(COCO_KEYS).issubset(value), "COCO fields missing")
            yield value
            position, first, value_next = end, False, False
            if position > 1024 * 1024:
                buffer, position = buffer[position:], 0


class InputLock:
    def __init__(self):
        self.files = {}

    def add(self, path):
        p = Path(path).resolve(strict=True)
        if str(p) not in self.files:
            self.files[str(p)] = {"sha256_before": sha(p), "size_bytes": p.stat().st_size}
        return self.files[str(p)]["sha256_before"]

    def finish(self):
        for path, record in self.files.items():
            record["sha256_after"] = sha(path)
            check(record["sha256_after"] == record["sha256_before"], "Input changed: " + path)
        return self.files


def encoded_rle(value):
    check(isinstance(value.get("counts"), str), "Require lossless compressed ASCII RLE")
    return {"size": value["size"], "counts": value["counts"].encode("ascii")}


def segm_records(path):
    # Fresh copies: official COCO.loadRes mutates dictionaries and adds bbox.
    return [{k: copy.deepcopy(row[k]) for k in ("image_id", "category_id", "score", "segmentation")}
            for row in json_array(path)]


def official_eval(coco, records, ids, COCO, COCOeval, kind="segm", **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        if records:
            detections = coco.loadRes(copy.deepcopy(records))
        else:
            detections = COCO()
            detections.dataset = {"images": list(coco.imgs.values()), "categories": list(coco.cats.values()), "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(coco, detections, iouType=kind, **kwargs)
        evaluator.params.imgIds = sorted(ids)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    values = {k: float(v) if v >= 0 else None for k, v in zip(METRICS, evaluator.stats)}
    return evaluator, values, stream.getvalue()


def cache_from_evaluator(evaluator):
    """Keep only actual all-area/max100 matching, including GT-only frames."""
    p = evaluator._paramsEval
    check(p.useCats == 1 and list(p.maxDets) == [1, 10, 100], "Unexpected evaluator configuration")
    check(len(p.iouThrs) == 10 and len(p.recThrs) == 101 and p.areaRngLbl[0] == "all", "Unexpected AP grid")
    n, k = len(p.imgIds), len(p.catIds)
    offsets = np.empty((k, n + 1), dtype=np.int64)
    gt_n = np.zeros((k, n), dtype=np.int64)
    scores, matches, ignored = [], [], []
    cursor = 0
    for cat in range(k):
        offsets[cat, 0] = cursor
        for img in range(n):
            e = evaluator.evalImgs[cat * len(p.areaRng) * n + img]
            if e is not None:
                check(e["image_id"] == p.imgIds[img] and e["category_id"] == p.catIds[cat], "evalImgs layout differs")
                gt_n[cat, img] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
                scores.append(np.asarray(e["dtScores"][:100], dtype=np.float64))
                matches.append(np.asarray(e["dtMatches"][:, :100], dtype=np.float64) != 0)
                ignored.append(np.asarray(e["dtIgnore"][:, :100], dtype=bool))
                cursor += len(scores[-1])
            offsets[cat, img + 1] = cursor
    return {"image_ids": np.asarray(p.imgIds, dtype=np.int64), "category_ids": np.asarray(p.catIds, dtype=np.int64),
            "offsets": offsets, "gt_nonignore": gt_n,
            "dt_scores": np.concatenate(scores) if scores else np.empty(0),
            "dt_matches": np.concatenate(matches, axis=1) if matches else np.empty((10, 0), bool),
            "dt_ignore": np.concatenate(ignored, axis=1) if ignored else np.empty((10, 0), bool),
            "iou_thresholds": np.asarray(p.iouThrs), "recall_thresholds": np.asarray(p.recThrs),
            "official_precision": evaluator.eval["precision"][:, :, :, 0, 2],
            "official_recall": evaluator.eval["recall"][:, :, 0, 2]}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def paired_summary(array):
    totals = np.asarray(array, dtype=np.float64).sum(axis=0)
    d = dict(zip(PAIR_FIELDS, map(float, totals)))
    def ratio(a, b):
        return d[a] / d[b] if d[b] else None
    output = {"counts": {k: int(d[k]) for k in PAIR_FIELDS[:5]},
              "baseline_mean_mask_iou": ratio("baseline_iou_sum", "matched_count"),
              "method_mean_mask_iou": ratio("method_iou_sum", "matched_count"),
              "mean_mask_iou_delta": ratio("delta_iou_sum", "matched_count"),
              "baseline_success_mean_iou_delta": ratio("success_delta_sum", "baseline_success"),
              "damage_rate_of_baseline_success": ratio("damage", "baseline_success"),
              "repair_rate_of_baseline_failure": ratio("repair", "baseline_failure"),
              "candidate_macro": {}, "image_macro": {}}
    for name, numerator, denominator in (("coverage", "coverage_sum", "coverage_n"), ("purity", "purity_sum", "purity_n"),
                                        ("FP_G", "fp_g_sum", "fp_g_n"), ("FN_G", "fn_g_sum", "fn_g_n"),
                                        ("mask_iou", "method_iou_sum", "matched_count"), ("delta_mask_iou", "delta_iou_sum", "matched_count")):
        ni, di = PAIR_FIELDS.index(numerator), PAIR_FIELDS.index(denominator)
        valid = array[:, di] > 0
        means = array[valid, ni] / array[valid, di]
        output["candidate_macro"][name] = {"value": ratio(numerator, denominator), "valid_count": int(d[denominator]),
                                         "missing_count": int(d["matched_count"] - d[denominator])}
        output["image_macro"][name] = {"value": float(means.mean()) if len(means) else None,
                                     "valid_images": int(valid.sum()), "missing_images": int((~valid).sum())}
    return output
