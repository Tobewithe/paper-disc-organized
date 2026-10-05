"""Complete metric comparison and previously defined failure-group analysis.

No inference, new group thresholds, parameter search, or retraining.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np


def atomic(path, content):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(content,encoding='utf-8');os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.project);study=root/'experiments/mask_boundary_route_20260914';runs=study/'runs'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sources={
        'm':('RUN_e0e0d46defb74e3d9a90094f49ce26fe','RUN_2ae67d6556a14848bceb5783a72e5790'),
        's':('RUN_f9df48632ac742248343184263262429','RUN_2edbec60af4247f4be855834b6ff6382')}
    metric_names=['Mask AP','AP50','AP75','APsmall','APmedium','APlarge','AR1','AR10','AR100','ARsmall','ARmedium','ARlarge']
    models={}
    for model,(evaluation,bank) in sources.items():
        summary=json.loads((runs/evaluation/'SUMMARY.json').read_text(encoding='utf-8'))
        with (runs/evaluation/'decisions.csv').open(encoding='utf-8',newline='') as handle:
            decisions={(int(r['image_id']),int(r['candidate_index']),int(r['annotation_id'])):r for r in csv.DictReader(handle) if r['mode']=='response'}
        items=[]
        with (runs/bank/'instance_records.csv').open(encoding='utf-8',newline='') as handle:
            for r in csv.DictReader(handle):
                if r['variant']!='smooth_gated':continue
                key=(int(r['image_id']),int(r['candidate_index']),int(r['annotation_id']))
                decision=decisions[key]
                b=float(r['baseline_iou']);fixed=float(decision['fixed_smooth_iou']);after=float(decision['gated_iou'])
                assert abs(float(decision['baseline_iou'])-b)<1e-12
                smooth_iou=float(r['iou'])
                restore=abs(smooth_iou-fixed)>1e-12
                if restore:assert abs(fixed-b)<1e-12 and smooth_iou==0
                accepted=int(decision['calibrated'])==1
                assert abs(after-(fixed if accepted else b))<1e-12
                coverage=float(r['baseline_recall']);purity=float(r['baseline_purity'])
                fixed_coverage=coverage if restore else float(r['recall'])
                fixed_purity=purity if restore else float(r['purity'])
                items.append(dict(image_id=key[0],annotation_id=key[2],baseline_iou=b,fixed_iou=fixed,response_iou=after,
                    baseline_coverage=coverage,baseline_purity=purity,fixed_coverage=fixed_coverage,fixed_purity=fixed_purity,
                    response_coverage=fixed_coverage if accepted else coverage,response_purity=fixed_purity if accepted else purity,
                    accepted=accepted,box_good=float(r['box_iou'])>=.75,
                    target=float(r['box_iou'])>=.75 and coverage>=.95 and b<.75))
        assert len(items)==len(decisions)==summary['fixed_slot_outcomes']['response']['matched']
        groups={}
        group_filters={'all_matched':lambda r:True,'predefined_overflow_failure':lambda r:r['target'],
                       'other_baseline_failures':lambda r:r['baseline_iou']<.75 and not r['target'],
                       'baseline_successes':lambda r:r['baseline_iou']>=.75}
        for name,predicate in group_filters.items():
            group=[r for r in items if predicate(r)];n=len(group)
            failures=sum(r['baseline_iou']<.75 for r in group);successes=n-failures
            values={'n':n,'baseline_failures':failures,'baseline_successes':successes}
            for method in ('baseline','fixed','response'):
                repaired=sum(r['baseline_iou']<.75<=r[method+'_iou'] for r in group)
                damaged=sum(r[method+'_iou']<.75<=r['baseline_iou'] for r in group)
                values[method]=dict(repaired=repaired,damaged=damaged,net=repaired-damaged,
                    repair_rate_pct=100*repaired/failures if failures else None,
                    damage_rate_pct=100*damaged/successes if successes else None,
                    mean_iou=float(np.mean([r[method+'_iou'] for r in group])) if n else None,
                    mean_coverage=float(np.mean([r[method+'_coverage'] for r in group])) if n else None,
                    mean_purity=float(np.mean([r[method+'_purity'] for r in group])) if n else None)
            groups[name]=values
        outcomes=groups['all_matched'];official={}
        for index,name in enumerate(metric_names):
            b=summary['metrics']['official_zero'][index]*100;v=summary['metrics']['risk_response'][index]*100
            official[name]=dict(baseline=b,response=v,delta_points=v-b,relative_change_pct=100*(v/b-1) if b else None)
        fixed=outcomes['fixed'];response=outcomes['response'];ordinary=summary['ordinary_gt']
        assert response['repaired']==summary['fixed_slot_outcomes']['response']['repaired']
        assert response['damaged']==summary['fixed_slot_outcomes']['response']['damaged']
        decision_groups={}
        for label,flag in [('accepted',True),('rejected',False)]:
            selected=[r for r in items if r['accepted']==flag];delta=np.array([r['fixed_iou']-r['baseline_iou'] for r in selected])
            decision_groups[label]=dict(n=len(selected),mean_fixed_delta_iou=float(delta.mean()),
                positive_fraction=float((delta>0).mean()),negative_fraction=float((delta<0).mean()))
        models[model]=dict(evaluation_run=evaluation,bank_run=bank,images=summary['images'],ordinary_gt=ordinary,official_metrics=official,
            fixed_slot_r75=dict(baseline_pct=100*outcomes['baseline_successes']/ordinary,
                response_pct=100*(outcomes['baseline_successes']+response['net'])/ordinary,delta_pp=100*response['net']/ordinary,
                unmatched_in_denominator=ordinary-outcomes['n']),groups=groups,decision_groups=decision_groups,
            versus_fixed=dict(avoided_damage=fixed['damaged']-response['damaged'],
                relative_damage_reduction_pct=100*(fixed['damaged']-response['damaged'])/fixed['damaged'],
                lost_repairs=fixed['repaired']-response['repaired'],
                relative_repairs_lost_pct=100*(fixed['repaired']-response['repaired'])/fixed['repaired'],
                net_extra_repairs=response['net']-fixed['net']))
        print(json.dumps(dict(model=model,official=official,r75=models[model]['fixed_slot_r75'],target=groups['predefined_overflow_failure'],versus_fixed=models[model]['versus_fixed'])),flush=True)
    result=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),study_id='STUDY_8fb3468ebb704682a2225ebed0e16206',models=models,
        definitions=dict(target='same pre-existing BoxIoU>=.75 and baseline actual mask coverage>=.95 and baseline MaskIoU<.75',
                         purity='TP / predicted foreground area; coverage=TP / GT area; means are per-instance macro means',
                         official='COCOeval 12 standard segmentation stats; ARsmall averages over IoU .50:.95 at maxDets100',
                         r75='baseline fixed same-class BoxIoU>=.50 one-to-one slots, all ordinary GT denominator; not official COCO AR'),
        limitations=['Existing4500 COCO images; no additional independent data','Post hoc summary of completed frozen methods; no best-bin search',
                     'Target group selected by baseline failures; repair rate is conditional and not whole-dataset gain',
                     'm and s target group membership can differ with their baseline predictions',
                     'Damage reductions are relative to fixed smooth calibration, not reductions of all baseline failures',
                     'No new AP confidence interval or boundary F/Boundary IoU evaluation',
                     'Mean purity improvement must be reported with coverage loss; not proof of neighbor-specific leakage reduction'])
    atomic(out/'SUMMARY.json',json.dumps(result,indent=2))
    lines=['# 已完成方法的指标与失败对象对照','','同一批4500张已探索COCO val2017图片，最终冻结响应门控对比原基线。全部官方指标一并列出，未搜索新分组。',
           '','| 指标 | m基线→方法 | 差值（点） | s基线→方法 | 差值（点） |','|---|---:|---:|---:|---:|']
    for name in metric_names:
        m=models['m']['official_metrics'][name];s=models['s']['official_metrics'][name]
        lines.append(f'| {name} | {m["baseline"]:.3f} → {m["response"]:.3f} | {m["delta_points"]:+.3f} | {s["baseline"]:.3f} → {s["response"]:.3f} | {s["delta_points"]:+.3f} |')
    lines+=['','固定槽位R75为自定义机制诊断指标，分母包含无候选普通GT，与官方AR分别报告。']
    for name,model in models.items():
        t=model['groups']['predefined_overflow_failure'];v=model['versus_fixed'];r=model['fixed_slot_r75']
        lines += ['',f'## YOLO26{name}-seg',f'- 固定槽位R75：{r["baseline_pct"]:.3f}% → {r["response_pct"]:.3f}%（{r["delta_pp"]:+.3f}个百分点）。',
            f'- 预定义外溢失败组 {t["n"]} 个：固定规则修复 {t["fixed"]["repaired"]} 个（{t["fixed"]["repair_rate_pct"]:.2f}%）；响应门控修复 {t["response"]["repaired"]} 个（{t["response"]["repair_rate_pct"]:.2f}%）。',
            f'- 该组平均IoU：{t["baseline"]["mean_iou"]:.4f} → {t["response"]["mean_iou"]:.4f}；纯度：{t["baseline"]["mean_purity"]:.4f} → {t["response"]["mean_purity"]:.4f}；自身覆盖：{t["baseline"]["mean_coverage"]:.4f} → {t["response"]["mean_coverage"]:.4f}。',
            f'- 全部匹配实例相对固定规则少误伤 {v["avoided_damage"]} 个（相对减少 {v["relative_damage_reduction_pct"]:.2f}%），同时少修复 {v["lost_repairs"]} 个（{v["relative_repairs_lost_pct"]:.2f}%），净多修复 {v["net_extra_repairs"]} 个。']
    lines+=['','约30%的条件修复率、相对误伤减少比例、AP点变化有不同分母，不互相替代。','收紧不能凭空恢复漏掉的前景，因此上述指标不能支持“覆盖完全无损”或“全部失败下降相同比例”。']
    atomic(out/'COMPARISON.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
