"""Fixed epoch3 dev screening; normal full-prototype masks, no COCO AP.

All reported arms are reevaluated on this screen's own immutable candidates.
No historical performance numbers are mixed into A or the N training control.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Pin actual imports before runtime_utils/evaluation functions import Ultralytics.
if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config")+1]).read_text(encoding="utf-8-sig"))
    if _cfg.get("source_python"):
        sys.path.insert(0, _cfg["source_python"])

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from screen_models import build_model, build_operators
from runtime_utils import setup
import evaluation_metrics as em

ARMS = ("N", "U", "Q", "P", "L", "B")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
FINAL_EPOCH = 3
# Reuse the established native decoder, allowing this screen's arm names only.
em.ARMS = ("A", *ARMS)


@torch.no_grad()
def official_bce_values(image, coefficients, chunk_size=8):
    if tuple(image["input_shape"]) != (640, 640):
        raise ValueError("Fixed official label geometry must remain 640x640")
    proto = F.interpolate(image["proto"][None].float(), (640, 640), mode="bilinear", align_corners=False)[0]
    values = []
    for lo in range(0, len(coefficients), chunk_size):
        hi = min(lo + chunk_size, len(coefficients))
        target = (image["masks"][None] == (image["owners"][lo:hi] + 1)[:, None, None]).float()
        boxes = image["target_boxes"][lo:hi]
        area = ((boxes[:, 2:] - boxes[:, :2]) / 640).prod(1)
        if not bool((area > 0).all()):
            raise ValueError("Official target-box normalized area must be positive")
        logits = torch.einsum("in,nhw->ihw", coefficients[lo:hi].float(), proto)
        pixels = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        result = ops.crop_mask(pixels, boxes).mean((1, 2)) / area * float(image["segmentation_gain"])
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite official BCE")
        values.append(result.cpu())
    return torch.cat(values) if values else torch.empty(0)



def official_gpu_payload(image, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in image.items()
            if k not in ("F", "input_uint8", "operator", "_operator")}


def identity(row):
    return tuple(row[k] for k in ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx"))


def summary_table(rows, arms, bootstrap):
    groups = em.image_groups(rows)
    table = dict(images=len(groups), candidates=len(rows), candidate={}, image_macro={}, undefined={}, comparisons={})
    for metric in METRICS:
        table["candidate"][metric] = {a:em.avg([r.get(f"{metric}_{a}") for r in rows]) for a in arms}
        table["image_macro"][metric] = {a:em.avg([em.avg([r.get(f"{metric}_{a}") for r in g]) for g in groups]) for a in arms}
        table["undefined"][metric] = {a:sum(not em.finite(r.get(f"{metric}_{a}")) for r in rows) for a in arms}
    pairs = [(a,ref) for a in arms if a != "A" for ref in ("A","N") if ref in arms and a != ref]
    # Same-capacity selector comparisons are secondary exploratory screen readouts.
    pairs += [(a,b) for a,b in (("P","Q"),("P","U"),("L","Q"),("B","Q")) if a in arms and b in arms]
    for arm,ref in pairs:
        compared = {m:em.paired(groups,arm,ref,m,20261003,bootstrap) for m in METRICS}
        repair = sum(r[f"mask75_{ref}"] == 0 and r[f"mask75_{arm}"] == 1 for r in rows)
        damage = sum(r[f"mask75_{ref}"] == 1 and r[f"mask75_{arm}"] == 0 for r in rows)
        compared["crossings"] = dict(repair=repair,damage=damage,net=repair-damage,
                                     reference=ref,denominator_candidates=len(rows))
        table["comparisons"][f"{arm}_minus_{ref}"] = compared
    return table


def classify_screen(tables, modes, missing, cfg):
    """IoU screening signal only; retain other valuable improvements for review."""
    recommendations = {}
    for mode in modes:
        if mode == "N":
            continue
        checks = {}
        for ref in ("A","N"):
            if ref not in ("A",*modes):
                continue
            full = tables["all"]["comparisons"][f"{mode}_minus_{ref}"]
            target = tables["box_good_mask_bad"]["comparisons"][f"{mode}_minus_{ref}"]
            overall = full["iou"]["image_macro"]["delta"]
            targeted = target["iou"]["image_macro"]["delta"]
            checks[ref] = dict(overall_macro_iou_delta=overall,target_macro_iou_delta=targeted,
                practical_signal=(overall is not None and targeted is not None and
                    (overall >= float(cfg.get("screening_effect_all",.002)) or
                     (targeted >= float(cfg.get("screening_effect_target",.005)) and
                      overall >= float(cfg.get("all_iou_guard",-.001))))))
        other_metrics = {}
        for group in ("all", "box_good_mask_bad"):
            other_metrics[group] = {}
            for ref in checks:
                compared = tables[group]["comparisons"][f"{mode}_minus_{ref}"]
                other_metrics[group][ref] = {
                    metric: compared[metric]["image_macro"]
                    for metric in ("mask75", "coverage", "auc", "fpr")}
                other_metrics[group][ref]["crossings"] = compared["crossings"]
        recommendations[mode] = dict(comparisons=checks,
            status="promising_IoU_screen_signal_not_confirmed" if len(checks)==2 and all(x["practical_signal"] for x in checks.values())
                   else "IoU_screen_threshold_not_met_review_other_metrics" if len(checks)==2 else "incomplete_native_control",
            other_metric_observations=other_metrics, needs_metric_value_review=True,
            interpretation="This classification concerns the prespecified IoU screen only. A meaningful Mask75, coverage, AUC or FPR improvement can still be valuable; assess magnitude, uncertainty and trade-offs separately. Three epochs and reused dev do not establish mechanism, AP, or absence of potential.")
    return dict(kind="exploratory_screen", recommendations=recommendations,
        incomplete_arms=missing, incomplete_is_negative=False, automatic_followup=False,
        multiplicity="1000 image-cluster bootstrap intervals are descriptive, not familywise confirmatory tests")


def summarize(rows,out,modes,expected,info,cfg):
    arms=("A",*modes)
    definitions={"all":lambda r:True,
        "box_good_mask_bad":lambda r:bool(r["box_good_mask_bad"]),
        "original_success":lambda r:r["mask75_A"]==1,
        "original_failure":lambda r:r["mask75_A"]==0}
    tables={name:summary_table([r for r in rows if predicate(r)],arms,int(cfg["bootstrap"]))
            for name,predicate in definitions.items()}
    image_rows=[]
    for name,predicate in definitions.items():
        for group in em.image_groups([r for r in rows if predicate(r)]):
            item=dict(split="dev",image_id=int(group[0]["image_id"]),group=name,candidates=len(group))
            for metric in METRICS:
                for arm in arms:
                    item[f"{metric}_{arm}"]=em.avg([r.get(f"{metric}_{arm}") for r in group])
                    item[f"{metric}_{arm}_defined"]=sum(em.finite(r.get(f"{metric}_{arm}")) for r in group)
                for arm in modes:
                    for ref in ("A","N"):
                        if ref not in arms or arm==ref:continue
                        values=[r[f"{metric}_{arm}"]-r[f"{metric}_{ref}"] for r in group
                                if em.finite(r.get(f"{metric}_{arm}")) and em.finite(r.get(f"{metric}_{ref}"))]
                        item[f"delta_{metric}_{arm}_minus_{ref}"]=em.avg(values)
            image_rows.append(item)
    em.append_rows(out/"PER_IMAGE.jsonl",image_rows)
    assessment=classify_screen(tables,modes,info["missing_or_incomplete_arms"],cfg)
    result=dict(schema="prototype-readout-fast-screen-v1",population=expected,arms=list(arms),tables=tables,
        assessment=assessment,evaluation=info,bootstrap=dict(draws=1000,unit="paired image",seed=20261003),
        no_COCO_AP=True,no_new_blind_test=True,all_images_retained=True)
    dump(out/"SUMMARY.json",em.clean(result))
    lines=["# Prototype-to-coefficient rapid screen", "",
        "Fixed epoch 3 on 1,024 planned fit images; evaluation is the fixed 256-image dev slice. This is a short-budget developmental screen, not independent confirmation or COCO AP.", "",
        f"Evaluated {len(rows)} fixed official candidates in {tables['all']['images']} effective dev images. Missing or unfinished arms: {', '.join(info['missing_or_incomplete_arms']) or 'none'}. An unfinished arm is not a negative result.", "",
        "All effect sizes below are percentage points. Each CI resamples whole images 1,000 times; the table's point estimate and CI both use image macro.", "",
        "| Group | Comparison | Macro IoU delta [95% CI] | Candidate IoU delta | Net Mask75 | Repair / damage | Coverage delta | AUC delta | FPR delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def pp(v):return "undefined" if v is None else f"{100*v:+.4f}"
    for group,table in tables.items():
        for pair,c in table["comparisons"].items():
            d=c['iou']['image_macro'];lo,hi=d['ci95'];cross=c['crossings']
            lines.append(f"| {group} | {pair} | {pp(d['delta'])} [{pp(lo)}, {pp(hi)}] | {pp(c['iou']['candidate']['delta'])} | {cross['net']} | {cross['repair']} / {cross['damage']} | {pp(c['coverage']['image_macro']['delta'])} | {pp(c['auc']['image_macro']['delta'])} | {pp(c['fpr']['image_macro']['delta'])} |")
    lines += ["", "GT only defines fixed official supervision and evaluation groups; the selectors use original frozen P, c0 and predicted boxes. All arms use the same normal full-prototype decoder and original COCO instance masks. AUC/FPR use continuous input-space logits on the original predicted-box support; undefined AUC cases remain in IoU evaluation.", "",
        "N is the same-screen native coefficient-branch fine-tuning control. No historical method metrics are joined. U/Q/P/L/B have matched trainable architecture; their frozen point choices differ. Positive differences against another weak arm do not by themselves establish improvement over original A.", "",
        "The automated screen status concerns the prespecified IoU threshold only. It does not veto a meaningful Mask75, coverage, AUC or FPR improvement, and it does not require all metrics to improve together. Each improvement needs its own magnitude, uncertainty and trade-off assessment.", "",
        "The screen status applies only to this seed, fit slice and three-epoch budget. No mechanism or final negative conclusion is inferred, and no next experiment starts automatically.", ""]
    for mode,entry in assessment['recommendations'].items():lines.append(f"- {mode}: {entry['status']}")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def run(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Evaluation is permitted only on the authorized Linux GPU server")
    cfg=resolve_runtime_config(load_json(args.config))
    if cfg.get("arms") != list(ARMS) or cfg.get("epochs") != 3 or cfg.get("bootstrap") != 1000:
        raise ValueError("Fixed six-arm / epoch3 / 1000-bootstrap screen configuration required")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/"PER_CANDIDATE.jsonl").exists() or (out/"COMPLETE.json").exists():
        raise RuntimeError("Preserve prior evaluation; use a new independent Run")
    receipt=load_json(Path(cfg["assets"])/"COMPLETE.json")
    if receipt.get("complete") is not True and receipt.get("completed") is not True:
        raise RuntimeError("New screen assets are not complete")
    index=load_index(cfg)
    if len(index.get("fit",[]))!=1024 or len(index.get("dev",[]))!=256 or index.get("val"):
        raise AssertionError("Screen split changed")
    items=index['dev']
    if len({int(x['image_id']) for x in items})!=256:
        raise AssertionError("Duplicate dev images")
    expected=dict(planned_images=256,effective_images=sum(int(x['n'])>0 for x in items),
        candidates=sum(int(x['n']) for x in items),no_positive_image_ids=[int(x['image_id']) for x in items if not int(x['n'])])
    dump(out/"EVALUATION_INDEX.json",dict(dev=items))
    checkpoints=load_json(args.checkpoints)
    if set(checkpoints)-set(ARMS):raise ValueError("Unknown checkpoint arm")
    replay=FrozenReplay(cfg)
    modes=tuple(m for m in ARMS if m in checkpoints)
    models={};hashes={};counts={}
    for mode in modes:
        path=Path(checkpoints[mode])
        completed=load_json(path.parent/"COMPLETE.json")
        if completed.get('completed') is not True or completed.get('epochs')!=3:
            raise ValueError(f"{mode}: partial or unfinished arm cannot be evaluated as epoch3")
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('mode')!=mode or ck.get('epoch')!=3 or path.name!='final.pt':
            raise ValueError("Only fixed final epoch3 checkpoints are eligible")
        if resolve_runtime_config(ck['config'])!=cfg:raise ValueError("Checkpoint config differs")
        if list(ck['feature_channels'])!=list(replay.feature_channels):raise ValueError("Feature channels differ")
        setup(cfg['seed']);model=build_model(mode,replay,cfg).to(replay.device).float().eval()
        model.load_state_dict(ck['state_dict'],strict=True);model.requires_grad_(False)
        models[mode]=model;hashes[mode]=sha256(path);counts[mode]=model.parameter_counts()
    capacities=[counts[m] for m in modes if m!='N']
    if capacities and any(c!=capacities[0] for c in capacities):raise AssertionError("Selector-arm capacities differ")
    coco=COCO(str(cfg['annotations_train']))
    rows=[];seen=set();start=time.monotonic()
    for position,entry in enumerate(items):
        if not int(entry['n']):continue
        image=load_asset(cfg,int(entry['image_id']),verify=True)
        if image['split']!='dev' or len(image['rows'])!=int(entry['n']):raise AssertionError("Evaluation population changed")
        expected_keys=[identity(r) for r in image['rows']]
        if len(set(expected_keys))!=len(expected_keys) or seen.intersection(expected_keys):raise AssertionError("Duplicate candidate")
        features=replay.replay([image]);coefficients={'A':image['c0'].to(replay.device)};audit=[]
        with torch.no_grad():
            for mode,model in models.items():
                selected=build_operators(image,mode,cfg,replay.device)
                diagnostic=selected.pop('_operator_diagnostics',{})
                coefficients[mode]=model(features,[selected])[0]
                audit.append(dict(image_id=int(entry['image_id']),mode=mode,
                    tensor_sha256={k:tensor_sha(v) for k,v in selected.items() if torch.is_tensor(v)},diagnostics=diagnostic))
        gpu=official_gpu_payload(image,replay.device)
        bce={arm:official_bce_values(gpu,value) for arm,value in coefficients.items()}
        decoded=em.evaluate_image(gpu,coefficients,coco)
        if [identity(r) for r in decoded]!=expected_keys:raise AssertionError("Decoder changed candidate identity or order")
        for i,row in enumerate(decoded):
            for arm in coefficients:row[f'bce_{arm}']=float(bce[arm][i])
        em.append_rows(out/'PER_CANDIDATE.jsonl',decoded);em.append_rows(out/'OPERATOR_AUDIT.jsonl',audit)
        rows.extend(decoded);seen.update(expected_keys)
        if position%20==0 or position+1==len(items):
            progress=dict(stage='dev_evaluation',images_processed=position+1,planned_images=256,candidates=len(rows),elapsed_s=time.monotonic()-start)
            dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
        del image,features,coefficients,gpu,bce,decoded
    if len(rows)!=expected['candidates']:raise AssertionError("Incomplete fixed dev population")
    replay.assert_unchanged()
    info=dict(checkpoints=checkpoints,checkpoint_sha256=hashes,counts=counts,
        missing_or_incomplete_arms=[m for m in ARMS if m not in modes],
        checkpoint_rule='fixed epoch3; no dev checkpoint selection',source=replay.import_info,
        new_assets=True,old_historical_metrics_joined=False,source_buffers_unchanged=True,
        pixel_metrics='continuous logit on fixed input-space predicted-box support; original COCO GT',
        decoding='process_mask(upsample=True); scale_masks true ratio_pad; threshold >0 then scaled >.5')
    dump(out/'EVALUATION_AUDIT.json',info)
    result=summarize(rows,out,modes,expected,info,cfg)
    dump(out/'COMPLETE.json',dict(completed=True,kind='screen_evaluation',candidates=len(rows),
        evaluated_arms=['A',*modes],missing_or_incomplete_arms=info['missing_or_incomplete_arms'],
        scientific_scope='developmental screen; not confirmatory and no AP',automatic_followup=False,
        elapsed_s=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--checkpoints',required=True,help='JSON mapping completed arm to final.pt; omit unfinished arms')
    args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        dump(Path(args.out)/'FAILURE.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
