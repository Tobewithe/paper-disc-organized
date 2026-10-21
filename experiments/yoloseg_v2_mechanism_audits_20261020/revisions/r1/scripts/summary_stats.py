"""Frozen table-level adjudication, stdlib only, repeatable off-server."""
import collections
import json
from pathlib import Path


def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else None


def median(xs):
    xs=sorted(xs); n=len(xs)
    return None if not n else (xs[n//2] if n%2 else (xs[n//2-1]+xs[n//2])/2)


def readlines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def chain_summary(run):
    data=readlines(run/"CHAIN_ROWS.jsonl")
    pairs=readlines(run/"FACTORIAL_ROWS.jsonl")
    counts={}
    for variant in sorted({r["variant"] for r in data}):
        for population in ["all","focus"]:
            rows=[r for r in data if r["variant"]==variant and (population=="all" or r["track_id"]==1)]
            counts[f"{variant}_{population}"]=dict(n=len(rows),correct=sum(r["correct_class"] for r in rows),
                first_failure=dict(collections.Counter(str(r["first_failure"]) for r in rows if not r["correct_class"])))
    fail=[r for r in data if not r["correct_class"]]
    early=sum(r["first_failure"] in [1,2] for r in fail)
    late=sum(r["first_failure"] in [3,4,5,6] for r in fail)
    direction=("selection_confidence" if len(fail)>=10 and late/len(fail)>=.5 else
               "raw_localization_or_class_evidence" if len(fail)>=10 and early/len(fail)>=.5 else "insufficient")
    factorial={}
    for contrast in ["near_clear","front_rear"]:
        rows=[r for r in pairs if r["contrast"]==contrast]
        valid=[r for r in rows if r["eligible"]]
        result=dict(all_pairs=len(rows),eligible=len(valid),excluded=len(rows)-len(valid),anchors={})
        factor_pass={}
        for anchor in ["pressure","control"]:
            result["anchors"][anchor]={}
            for key in ["P0c0","P1c0","P0c1","P1c1"]:
                result["anchors"][anchor][key]={m:mean(r["results"][anchor][key][m] for r in valid)
                    for m in ["iou","coverage","bridge_mass","material_bridge","split_mass","material_split"]}
            for factor,key in [("restore_P","P0c1"),("restore_c","P1c0")]:
                bridge=lambda r:r["results"][anchor]["P1c1"]["bridge_mass"]-r["results"][anchor][key]["bridge_mass"]
                diou=lambda r:r["results"][anchor][key]["iou"]-r["results"][anchor]["P1c1"]["iou"]
                seed_values={str(seed):dict(n=sum(r["seed"]==seed for r in valid),
                    bridge_reduction=mean(bridge(r) for r in valid if r["seed"]==seed),
                    delta_iou=mean(diou(r) for r in valid if r["seed"]==seed)) for seed in [20261020,20261021,20261022]}
                br=mean(bridge(r) for r in valid); di=mean(diou(r) for r in valid)
                passed=len(valid)>=60 and br>=.01 and di>=-.01 and sum(v["bridge_reduction"] is not None and v["bridge_reduction"]>0 for v in seed_values.values())>=2
                result["anchors"][anchor][factor]=dict(bridge_reduction=br,delta_iou=di,seed=seed_values,passed=passed)
                factor_pass.setdefault(factor,[]).append(passed)
        result["supported_factor"]=[k for k,v in factor_pass.items() if all(v)]
        result["logit_rms"]={k:mean(r["logit_term_rms"][k] for r in valid) for k in ["P","c","interaction"]}
        result["decomposition_max_abs_error"]=max((r["logit_decomposition_max_abs_error"] for r in valid),default=None)
        factorial[contrast]=result
    return dict(experiment="chain",gt_observations=len(data),counts=counts,
        detection_failures=len(fail),early_failures=early,late_failures=late,
        first_failure=dict(collections.Counter(str(r["first_failure"]) for r in fail)),
        detection_priority=direction,factorial=factorial,training_steps=0,existing_method_comparison="not_tested")


def shift_summary(run):
    allrows=readlines(run/"SHIFT_ROWS.jsonl")
    rows=[r for r in allrows if r["offset"]!=[0,0] and r["eligible"]]
    groups={}
    for off in sorted({tuple(r["offset"]) for r in rows}):
        rr=[r for r in rows if tuple(r["offset"])==off]
        paired=[r for r in rr if "mask_consistency_iou" in r]
        groups[str(off)]=dict(n=len(rr),correct=sum(r["correct"] for r in rr),
            either_flip=sum(r["either_flip"] for r in rr),detection_flip=sum(r["detection_flip"] for r in rr),
            success_flip=sum(r["success_flip"] for r in rr),delta_iou=mean(r["delta_iou"] for r in rr),
            common_correct=len(paired),consistency_iou=mean(r["mask_consistency_iou"] for r in paired),
            fixed_box_consistency_iou=mean(r["fixed_box_consistency_iou"] for r in paired),
            logit_relative_rms_median=median(r["logit_relative_rms"] for r in paired))
    one=[r for r in rows if max(abs(x) for x in r["offset"])==1]
    instances={(r["image_id"],r["track_id"]) for r in one}
    assert len(one)==4*len(instances)
    flips=mean(float(r["either_flip"]) for r in one)
    seed={str(s):dict(n=sum(r["seed"]==s for r in one),
        flip_fraction=mean(float(r["either_flip"]) for r in one if r["seed"]==s),
        delta_iou=mean(r["delta_iou"] for r in one if r["seed"]==s)) for s in [20261020,20261021,20261022]}
    paired=[r for r in one if "mask_consistency_iou" in r]
    poor=mean(float(r["mask_consistency_iou"]<.95) for r in paired)
    rm=median(r["logit_relative_rms"] for r in paired)
    gate=len(instances)>=100 and flips>=.05 and sum(v["flip_fraction"] is not None and v["flip_fraction"]>=.05 for v in seed.values())>=2
    variants={}
    for v in sorted({r["variant"] for r in one}):
        rr=[r for r in one if r["variant"]==v]
        variants[v]=dict(n=len(rr),flip_fraction=mean(float(r["either_flip"]) for r in rr),
            delta_iou=mean(r["delta_iou"] for r in rr),
            detection_flip=sum(r["detection_flip"] for r in rr),success_flip=sum(r["success_flip"] for r in rr))
    return dict(experiment="shift",frames=len(allrows)//3,eligible_instances=len(instances),
        one_pixel_n=len(one),one_pixel_flip_fraction=flips,one_pixel_delta_iou=mean(r["delta_iou"] for r in one),
        phase_gate_passed=gate,seed=seed,variant=variants,offsets=groups,
        common_correct_1px=len(paired),poor_mask_consistency_fraction=poor,logit_relative_rms_median=rm,
        decode_amplification_screen=poor is not None and poor>=.1 and rm<.05,
        training_steps=0,existing_method_comparison="not_tested")


def summarize(run, kind):
    return chain_summary(Path(run)) if kind=="chain" else shift_summary(Path(run))
