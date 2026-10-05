"""Generate the locked Stage-I PCDCR report and decision, without model execution.

Usage:
  python summarize_pcdcr.py --run <run-directory> [--study <study-directory>]
         [--config <frozen-RUN_CONFIG.json>] [--out <report-directory>]

Uses only JSON/JSONL artifacts and stdlib. It never imports torch, trains,
selects another checkpoint, re-estimates a threshold, or launches Stage II.
Missing evidence is reported as unknown, rather than filled with a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("A", "B", "C", "D", "M", "S")
PAIRS = ("D_minus_B", "D_minus_C", "D_minus_M", "D_minus_A", "D_minus_S")
EXPECTED_COUNTS = {"fit": [796, 6058], "dev": [197, 1402], "val": [196, 1346]}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get(obj, *keys, default=None):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


class Evidence:
    def __init__(self):
        self.sources = []
        self.missing = []
        self.errors = []

    def read(self, path, fallback=None, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                self.missing.append(str(path))
            return {} if fallback is None else fallback
        self.sources.append({"path": str(path.resolve()), "sha256": sha(path)})
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            self.errors.append({"path": str(path), "error": str(exc)})
            return {} if fallback is None else fallback

    def reference(self, path):
        path = Path(path)
        if path.is_file():
            self.sources.append({"path": str(path.resolve()), "sha256": sha(path)})
        else:
            self.missing.append(str(path))


def estimate(table, pair, metric="iou", estimand="image_macro"):
    item = get(table, "comparisons", pair, metric, estimand, default={})
    point = item.get("delta")
    ci = item.get("ci95", [None, None])
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        ci = [None, None]
    if not finite(point) or not all(finite(v) for v in ci):
        status = "unknown"
    elif ci[0] > 0:
        status = "positive_interval"
    elif ci[1] < 0:
        status = "negative_interval"
    elif point == 0 and ci[0] == ci[1] == 0:
        status = "observed_identical_zero"
    else:
        status = "interval_includes_zero"
    return {"delta": point, "ci95": list(ci), "status": status,
            "valid_images": item.get("valid_images"), "valid_candidates": item.get("valid_candidates")}


def fmt(value, scale=1.0, digits=4, signed=False):
    if not finite(value):
        return "未知"
    return f"{value*scale:+.{digits}f}" if signed else f"{value*scale:.{digits}f}"


def fmt_stat(stat, pp=True):
    scale = 100 if pp else 1
    ci = stat.get("ci95", [None, None])
    return f"{fmt(stat.get('delta'),scale,signed=True)} [{fmt(ci[0],scale,signed=True)},{fmt(ci[1],scale,signed=True)}]"


def answer_effect(stat, name):
    status = stat["status"]
    if status == "positive_interval":
        return f"{name}为正，图片配对95%区间排除0；效应大小仍须结合实用量级判断。"
    if status == "negative_interval":
        return f"{name}为负，图片配对95%区间支持本轮D较差。"
    if status == "observed_identical_zero":
        return f"{name}在本轮评价上为0，未提供优势证据。"
    if status == "unknown":
        return f"{name}缺少完整结果，未知。"
    return f"{name}的95%区间跨越或接触0，当前比较不确定；不能把点估计方向当作已建立优势。"


def distribution(values):
    values = sorted(float(x) for x in values if finite(x))
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": values[0], "max": values[-1],
            "population_std": statistics.pstdev(values)}


def distribution_metadata(obj, location="run_record"):
    """Find recorded distribution versions, distinct from imported source."""
    found=[]
    if isinstance(obj,dict):
        if str(obj.get("name", "")).lower()=="ultralytics" and "version" in obj:
            found.append({"location":location,"version":obj["version"]})
        for key,value in obj.items():
            here=location+"."+str(key)
            if str(key).lower()=="ultralytics" and isinstance(value,(str,int,float)):
                found.append({"location":here,"version":str(value)})
            elif isinstance(value,(dict,list)):
                found.extend(distribution_metadata(value,here))
    elif isinstance(obj,list):
        for i,value in enumerate(obj):
            found.extend(distribution_metadata(value,f"{location}[{i}]"))
    return found


def intermediates(path, evidence):
    """Scalar/gate summaries only; no prototype/checkpoint tensor reads."""
    if not path.is_file():
        evidence.missing.append(str(path))
        return {}
    evidence.reference(path)
    bank = {}
    counts = {}
    bad = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            split = row.get("split", "unknown")
            counts[split] = counts.get(split, 0) + 1
            for arm in ("B", "C", "D", "M"):
                bank.setdefault(f"{split}:delta_c_norm_{arm}", []).append(row.get(f"delta_c_norm_{arm}"))
            bank.setdefault(f"{split}:true_mismatch_gate_l2", []).append(row.get("gate_difference_norm"))
            for arm in ("D", "M"):
                gate = row.get(f"gate_{arm}", [])
                if isinstance(gate, list):
                    for rank, value in enumerate(gate):
                        bank.setdefault(f"{split}:gate_{arm}_rank{rank}", []).append(value)
    if bad:
        evidence.errors.append({"path": str(path), "invalid_jsonl_rows": bad})
    return {"candidate_counts": counts, "distributions": {k: distribution(v) for k, v in bank.items()}}


def generate(run, study, config_path, out):
    run, study, out = Path(run), Path(study), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    evidence = Evidence()
    cfg = evidence.read(config_path)
    for name in ("PROTOCOL.md", "USER_EXECUTION_INSTRUCTIONS.md"):
        evidence.reference(study/name)
    results = evidence.read(run/"RESULTS.json")
    complete = evidence.read(run/"COMPLETE.json")
    assets = evidence.read(run/"ASSET_MANIFEST.json")
    smoke = evidence.read(run/"SMOKE.json")
    train_complete = evidence.read(run/"TRAIN_COMPLETE.json")
    bias = evidence.read(run/"SCALAR_BIAS_SELECTION.json")
    mismatch = evidence.read(run/"MISMATCH_MANIFEST.json", fallback=[])
    condition_latency = evidence.read(run/"CONDITION_LATENCY.json")
    adapter_latency = evidence.read(run/"ADAPTER_LATENCY.json")
    run_record = evidence.read(run/"run.json",required=False)
    baseline_replay = evidence.read(run/"BASELINE_REPLAY_AUDIT.json",required=False)
    selections, histories = {}, {}
    for arm in ("B", "C", "D"):
        selections[arm] = evidence.read(run/arm/"SELECTION.json")
        histories[arm] = evidence.read(run/arm/"TRAIN_HISTORY.json", fallback=[])
        evidence.reference(run/arm/"best.pt")
    intermediate = intermediates(run/"PER_CANDIDATE.jsonl", evidence)
    tables = results.get("tables", {})
    val = tables.get("val:all", {})
    comparisons = {pair: {metric: estimate(val, pair, metric) for metric in ("iou", "mask75", "coverage", "auc", "fpr")} for pair in PAIRS}
    threshold = cfg.get("practical_iou_D_minus_B")
    db, dc, dm, da, ds = [comparisons[p]["iou"] for p in PAIRS]
    mask_db = comparisons["D_minus_B"]["mask75"]
    damage = {}
    for pair in PAIRS:
        raw = get(val, "comparisons", pair, "damage_rate_on_A_success", default={})
        damage[pair] = {"image_macro": raw.get("image_macro", {}), "candidate": raw.get("candidate", {}),
                        "population": raw.get("population", "unknown: expected A original success")}
    arithmetic = {}
    if finite(db["delta"]) and finite(threshold) and finite(mask_db["delta"]):
        arithmetic["capacity_D_B"] = bool(db["delta"] >= threshold or (db["delta"] > 0 and mask_db["delta"] > 0))
    else:
        arithmetic["capacity_D_B"] = None
    arithmetic.update({name: (stat["delta"] > 0 if finite(stat["delta"]) else None)
                       for name, stat in (("concat_D_C",dc),("correspondence_D_M",dm),("baseline_D_A",da))})
    damage_detected, damage_unknown = [], []
    for pair in PAIRS:
        # Both stated estimands are shown. A significant increase in either is
        # recorded; absence of detected increase is not a noninferiority proof.
        for estimand in ("candidate", "image_macro"):
            ci = damage[pair][estimand].get("ci95", [None, None])
            if not isinstance(ci, list) or len(ci)!=2 or not all(finite(v) for v in ci):
                damage_unknown.append(f"{pair}:{estimand}")
            elif ci[0] > 0:
                damage_detected.append(f"{pair}:{estimand}")
    # The archived rule says "damage not significantly increased" but does
    # not name a reference arm/estimand. Report all pairs and do not silently
    # invent five simultaneous noninferiority gates after seeing results.
    locked_damage_refs = cfg.get("stage2_damage_reference_arms")
    locked_damage_estimand = cfg.get("stage2_damage_estimand")
    if isinstance(locked_damage_refs,list) and locked_damage_refs and locked_damage_estimand in ("candidate","image_macro"):
        chosen_keys=[f"D_minus_{arm}:{locked_damage_estimand}" for arm in locked_damage_refs]
        arithmetic["damage_no_detected_increase"] = None if any(key in damage_unknown for key in chosen_keys) else not any(key in damage_detected for key in chosen_keys)
        damage_rule_scope={"reference_arms":locked_damage_refs,"estimand":locked_damage_estimand,"source":"frozen config"}
    else:
        arithmetic["damage_no_detected_increase"] = None
        damage_rule_scope={"status":"unknown","reason":"preregistered text does not specify damage comparator/estimand; all pairwise intervals are reported without choosing a favorable gate post hoc"}
    arithmetic_pass = all(value is True for value in arithmetic.values())
    negative_rules = [key for key,value in arithmetic.items() if value is False]
    unknown_rules = [key for key,value in arithmetic.items() if value is None]
    unresolved_effects = [pair for pair in PAIRS[:4] if comparisons[pair]["iou"]["status"] in ("unknown","interval_includes_zero")]
    ranking = comparisons["D_minus_A"]["auc"]
    if ds["status"] == "positive_interval" and ranking["status"] == "positive_interval":
        calibration_status = "beyond_this_scalar_bias_and_positive_ranking_evidence"
    elif finite(ds["delta"]) and ds["delta"] <= 0:
        calibration_status = "no_advantage_over_this_scalar_bias"
    else:
        calibration_status = "uncertain"

    integrity = []
    if complete.get("status") != "completed":
        integrity.append("COMPLETE.json does not confirm completed Stage I")
    if get(smoke,"checks","all_passed") is not True:
        integrity.append("smoke not confirmed passed")
    if assets.get("historical_official_identity_match") is not True:
        integrity.append("historical official candidate identity equivalence not confirmed")
    if assets.get("counts") != EXPECTED_COUNTS:
        integrity.append("asset split counts differ from frozen protocol or are missing")
    if [val.get("n_images"),val.get("n_candidates")] != EXPECTED_COUNTS["val"]:
        integrity.append("val evaluation counts differ from frozen protocol or are missing")
    if results.get("arms") != list(ARMS):
        integrity.append("evaluation does not contain all six frozen arms A/B/C/D/M/S")
    zero = results.get("zero_bias_replay_audit",{})
    if zero.get("padded_pixel_differences") != 0 or zero.get("original_pixel_differences") != 0:
        integrity.append("zero bias native replay missing or nonidentical")
    if bias.get("selected_before_val") is not True:
        integrity.append("scalar bias selection before val not confirmed")
    for arm in ("B","C","D"):
        history = histories[arm]
        if not isinstance(history,list) or len(history) != cfg.get("epochs"):
            integrity.append(f"{arm}: complete fixed-budget training history missing")
        if not isinstance(selections[arm].get("best_epoch"), int):
            integrity.append(f"{arm}: dev checkpoint selection missing")
    parameters = assets.get("parameter_counts",{})
    pd = parameters.get("pcdcr")
    capacity_ratios = {name:(abs(parameters[name]-pd)/pd if finite(pd) and pd>0 and finite(parameters.get(name)) else None) for name in ("static","concat")}
    if any(value is None or value>0.1 for value in capacity_ratios.values()):
        integrity.append("capacity controls not verified within 10% parameter difference")
    training_hours = train_complete.get("elapsed_s")
    training_hours = training_hours / 3600 if finite(training_hours) else None
    if finite(training_hours) and finite(cfg.get("max_training_hours")) and training_hours > cfg["max_training_hours"]:
        integrity.append("formal training elapsed time exceeds locked resource budget")

    if integrity or evidence.errors or evidence.missing:
        decision = "INCOMPLETE_EVIDENCE"
        reason = "必要产物或一致性记录未齐，先补齐既有运行记录，不能据此放行。"
    elif negative_rules:
        decision = "STOP_STAGE_I"
        reason = "至少一个已锁定数值条件未满足；PCDCR第一版在本轮结案。"
    elif unknown_rules or unresolved_effects:
        decision = "STOP_INCONCLUSIVE"
        reason = "数值条件未知或主要比较区间跨0；依协议保留不确定性，停止第一版，不自动进入Stage II。"
    else:
        decision = "STAGE_II_NUMERIC_EVIDENCE_ELIGIBLE"
        reason = "已锁定的容量、concat、错配、原模型比较及damage检查通过；本脚本仅给出证据状态，不执行Stage II。"
    # The protocol never fixed an excessive-latency threshold. Do not invent
    # one using the observed results, or treat adapter-only timings as end-to-end.
    stage2_authorized = decision == "STAGE_II_NUMERIC_EVIDENCE_ELIGIBLE"
    answers = {
        "Q1": answer_effect(db,"D−B") + (f" 锁定实用量级为{threshold*100:.3f} pp，Mask75替代条件单独列出。" if finite(threshold) else " 实用门槛缺失。"),
        "Q2": answer_effect(dc,"D−C"),
        "Q3": answer_effect(dm,"D−M") + " 即使为正，也只支持整套实例条件的对应关系有作用；e同时含P统计、c0和几何，不能据此证明prototype是唯一缺失变量。",
        "Q4": ("D超过本轮单bias对照，且D−A连续logit AUC区间为正，支持超出该标量校准的像素排序改善。" if calibration_status=="beyond_this_scalar_bias_and_positive_ranking_evidence" else "目前未形成同时超过单bias与提高连续像素排序的明确证据；不宣称已识别空间组合机制。") + " 这不能排除所有其他校准方式，也不是完整oracle恢复证明。",
        "Q5": ("检测到损伤率增加："+", ".join(damage_detected) if damage_detected else "未检测到损伤率显著增加，但区间跨0不等于已经证明非劣性。") if not damage_unknown else "损伤率配对区间缺失，未知。",
        "Q6": reason,
    }
    if selections["B"].get("best_epoch") == 0:
        answers["Q1"] += " B在dev选择了零初始化epoch0，因此本轮所选B输出等于A；该对照反映当前训练预算和选模规则的结果，不能代表静态函数类的最优水平。"
    answers["Q4"] += f" 实测D−A AUC为{fmt_stat(ranking)} pp；D−S原图macro IoU为{fmt_stat(ds)} pp。"
    if ranking["status"] == "negative_interval":
        answers["Q4"] += " 本轮AUC区间为负，没有支持像素排序改善。"
    if finite(ds["delta"]) and ds["delta"] <= 0:
        answers["Q4"] += " D未建立超过S的优势，但这不证明D与阈值校准完全等价，也不能说其行为被阈值完整解释。"
    transitions = get(val,"comparisons","D_minus_A","mask75_transition",default={})
    if all(isinstance(transitions.get(k), (int,float)) for k in ("repair","damage","net")):
        answers["Q5"] = (f"D相对A修复{int(transitions['repair'])}个、损伤{int(transitions['damage'])}个，"
                         f"净变化{int(transitions['net']):+d}个Mask75候选。" + answers["Q5"])
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "stage":"Stage I", "study_id":cfg.get("study_id"),
        "decision":decision,"stage_II_numeric_evidence_eligible":stage2_authorized,
        "automatic_stage_II_executed":False,"reason":reason,"questions":answers,
        "preregistered_rule_text":cfg.get("stage2_rule"),
        "numeric_rules":arithmetic,"numeric_rules_all_passed":arithmetic_pass,
        "failed_numeric_rules":negative_rules,"unknown_numeric_rules":unknown_rules,
        "uncertain_primary_comparisons":unresolved_effects,"comparisons":comparisons,
        "damage_on_A_original_success":damage,"detected_damage_increases":damage_detected,
        "damage_gate_definition":damage_rule_scope,
        "calibration_evidence":calibration_status,"integrity_issues":integrity,
        "capacity_relative_parameter_difference":capacity_ratios,"training_hours":training_hours,
        "checkpoint_selections":selections,"scalar_bias_selection":bias,
        "historical_baseline_replay_audit":baseline_replay,
        "latency":{"condition":condition_latency,"adapter":adapter_latency,
                   "excessive_latency_gate":"unknown: preregistration has no numeric cutoff; reported timings are not end-to-end latency"},
        "environment":{"actually_imported_ultralytics":assets.get("ultralytics"),
                       "actually_imported_path":assets.get("ultralytics_import_path"),
                       "recorded_distribution_metadata":distribution_metadata(run_record),
                       "interpretation":"An installed distribution version can differ from the source imported via sys.path. The experiment executes the verified imported source."},
        "unresolved_protocol_definitions":[
            "No numerical equivalence margin was preregistered for 'basically identical' or 'AUC basically unchanged'; no post-hoc margin is introduced.",
            "No numerical excessive-latency cutoff or full inference latency measurement is available.",
            "All intervals are pointwise 95% image-cluster bootstrap; not familywise multiplicity-adjusted.",
            "No-detected-damage-increase is not an equivalence/noninferiority guarantee.",
            "The damage comparator/estimand was not named in the current frozen configuration; this script does not add one after training.",
        ],
        "interpretation_boundaries":[
            "本轮没有oracle教师，只比较官方同候选mask监督和系数增量正则下的外挂读出结构。",
            "结论限于本次166维复合条件、rank=8、seed=0、12轮训练和已冻结的样本/选模方案；不能据此判定所有prototype条件机制不存在。",
            "GT仅用于训练监督与离线评价；class_id和原成功/失败标签属于评价元数据，未进入condition。",
            "固定官方TAL候选由GT条件的分配流程确定；本轮没有评价完整推理输出，也没有COCO AP结论。",
            "val未参加本轮训练或checkpoint选择，但已在历史研究中使用，因此不是从未查看的新盲测。",
            "M替换整个166维e，包含P统计、原系数c0和预测框几何；即使D优于M，也只能支持这套复合条件的实例对应关系，不能单独证明P是缺失信息。",
            "FPR统计预测框内的全部非目标像素，其中包含邻居实例；不能将其单独归因为纯背景泄漏。",
            "旧固定表示共享仿射证书只提供研究背景：其正则为lambda/2乘系数位移平方，本轮为lambda_delta乘系数增量平方，且函数类不同，不能拼接目标值或计算跨实验回收比例。",
            "旧证书不预先证明PCDCR有效，也不识别YOLO历史训练失败的唯一原因。",
        ],
        "intermediates":intermediate,"sources":evidence.sources,"missing_sources":evidence.missing,"source_errors":evidence.errors,
    }

    lines=["# PCDCR Stage I 实验结果", "", f"**结论状态：{decision}。{reason}**", "",
           "组别：A=原模型；B=参数量匹配的静态MLP；C=普通条件拼接；D=PCDCR动态读出；M=同一D模型使用错配条件；S=开发集选定并冻结的单标量logit校准。D−B表示动态读出相对静态容量对照的差值。", ""]
    for q,text in answers.items():
        lines += [f"**{q}：** {text}", ""]
    lines += ["## 主要比较", "", "所有效应及区间均为百分点（pp）；主统计量是val图片macro，候选均值是辅助。95%区间为整图配对bootstrap，未作多比较校正。", "",
              "| 比较 | 原图macro IoU差 [95% CI] | Mask75差 [95% CI] | AUC差 [95% CI] |", "|---|---:|---:|---:|"]
    for pair in PAIRS:
        item=comparisons[pair]
        lines.append(f"| {pair.replace('_minus_','−')} | {fmt_stat(item['iou'])} | {fmt_stat(item['mask75'])} | {fmt_stat(item['auc'])} |")
    lines += ["", "## 六组原图质量", "", "| 组 | 图片macro IoU | 候选IoU | Mask75候选数 | 修复/损伤（相对A） | Coverage | AUC | 框内非目标FPR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        tr=get(val,"comparisons",f"{arm}_minus_A","mask75_transition",default={})
        transition="0/0" if arm=="A" else f"{tr.get('repair','未知')}/{tr.get('damage','未知')}"
        lines.append(f"| {arm} | {fmt(get(val,'image_macro','iou',arm),100)} | {fmt(get(val,'candidate','iou',arm),100)} | {get(val,'mask75_counts',arm,default='未知')} | {transition} | {fmt(get(val,'image_macro','coverage',arm),100)} | {fmt(get(val,'image_macro','auc',arm),100)} | {fmt(get(val,'image_macro','fpr',arm),100)} |")
    lines += ["", "FPR人群为预测框内所有非目标像素，包括邻居实例，不能单独解释为纯背景泄漏。", "", "## 成功实例损伤", "", "共同人群固定为A原图MaskIoU≥0.75；damage_X=1[MaskIoU_X<0.75]。正差代表D损伤更多。", "",
              "| 比较 | 图片macro损伤率差 [95% CI] | 候选损伤率差 [95% CI] |", "|---|---:|---:|"]
    for pair in PAIRS:
        lines.append(f"| {pair.replace('_minus_','−')} | {fmt_stat(damage[pair]['image_macro'])} | {fmt_stat(damage[pair]['candidate'])} |")
    lines += ["", "未检出增加不能代替非劣性证明；本轮没有事后新增损伤容忍界。锁定文本未明确损伤率比较基准，因此保留全表；若主要比较已经不通过，直接按该已确定失败条件结案，不以这项定义歧义代替失败原因。", "", "## 训练、选择和运行核对", "",
              f"实际val：{val.get('n_images','未知')}张图 / {val.get('n_candidates','未知')}候选；协议范围196 / 1346。",
              f"训练：seed={cfg.get('seed','未知')}，epochs={cfg.get('epochs','未知')}，lambda_delta={cfg.get('lambda_delta','未知')}。训练损失为官方同候选mask项 + lambda_delta·mean(||delta_c||²)，正则没有1/2。无oracle教师项。",
              f"正式训练耗时：{fmt(training_hours,digits=3)}小时；预算{cfg.get('max_training_hours','未知')}小时。",
              f"实际导入Ultralytics：{assets.get('ultralytics','未知')}；路径：{assets.get('ultralytics_import_path','未知')}。运行记录中的发行包元数据：{json.dumps(distribution_metadata(run_record),ensure_ascii=False) or '未知'}。发行包版本（如8.4.27）与sys.path实际导入源码（如8.4.100）可不同，以核验的实际导入源码为执行依据。",
              (f"历史baseline逐候选复核：[BASELINE_REPLAY_AUDIT.json](BASELINE_REPLAY_AUDIT.json)。审计记录：{json.dumps(clean(baseline_replay),ensure_ascii=False)}。" if baseline_replay else "历史baseline专项审计文件未提供；该可选文件不影响已保存的zero-bias回放结果，不能在缺失时补写匹配数。"),
              "", "| 组 | 参数数 | dev选择epoch | dev macro IoU | 训练记录轮数 |", "|---|---:|---:|---:|---:|"]
    for arm,kind in (("B","static"),("C","concat"),("D","pcdcr")):
        sel=selections[arm]
        lines.append(f"| {arm} | {parameters.get(kind,'未知')} | {sel.get('best_epoch','未知')} | {fmt(sel.get('dev_macro_iou'),100)} | {len(histories[arm]) if isinstance(histories[arm],list) else '未知'} |")
    lines += ["", f"S固定bias：{bias.get('bias','未知')}；选择规则：{bias.get('rule','未知')}。",
              f"S−A AUC浮点审计：{json.dumps(clean(results.get('scalar_bias_auc_audit',{})),ensure_ascii=False)}。",
              f"条件计算均值：{fmt(condition_latency.get('per_candidate_mean_ms'),digits=4)} ms/candidate（包含完整原型上采样及ROI池化；不含主网络、IO及H2D）。",
              f"D adapter均值：{fmt(get(adapter_latency,'D','mean_ms_per_candidate'),digits=4)} ms/candidate。以上均不是端到端延迟。",
              "协议没有给出‘明显过高’的数值截止值，因此不基于结果临时造一个延迟门槛。", "", "## 分层与条件行为", "",
              "P3/P4/P5、A原成功/失败及COCO面积分层保留在RESULTS.json，仅作解释，不能以有利分层替代val全体。完整逐候选记录在PER_CANDIDATE.jsonl，逐图值在PER_IMAGE.json。",
              "门控各rank通道、true/mismatch gate差异与系数增量分布写入DECISION.json的intermediates；这里的sigmoid门控是动态读出内部结构，不是另加失败选择gate。", "", "## 结论范围与结束条件", ""]
    lines += ["- "+text for text in output["interpretation_boundaries"]]
    lines += ["", "完整机器判定和来源SHA256见DECISION.json。脚本只整理现有冻结结果，不训练、不改checkpoint、不改bias，不执行Stage II。"]
    if integrity or evidence.missing or evidence.errors:
        lines += ["", "## 未齐证据", ""]
        lines += ["- "+issue for issue in integrity]
        lines += ["- 缺失："+path for path in evidence.missing]
        lines += ["- 读取失败："+json.dumps(err,ensure_ascii=False) for err in evidence.errors]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"DECISION.json").write_text(json.dumps(clean(output),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--study",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--config",type=Path)
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    result=generate(args.run,args.study,args.config or args.study/"RUN_CONFIG.json",args.out or args.run)
    print(json.dumps({"decision":result["decision"],"reason":result["reason"],"stage_II_executed":False,
                      "missing_sources":len(result["missing_sources"]),"integrity_issues":len(result["integrity_issues"])},ensure_ascii=False))


if __name__=="__main__":
    main()
