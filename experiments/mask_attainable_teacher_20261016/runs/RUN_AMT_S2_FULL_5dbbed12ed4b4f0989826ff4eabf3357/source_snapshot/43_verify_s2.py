"""Independent output/COCO recheck; local mode independently recomputes statistics."""
import argparse
import contextlib
import gc
from itertools import zip_longest
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
for d in ('experiments/mask_readout_mechanism_v2_20261012/scripts','shared/evaluation',
          'experiments/official_gt_failure_rebaseline_20261013/scripts'):sys.path.append(str(ROOT/d))
from posttrain_run import read_json,write_new,sha256,require,now
from posttrain_audit import read_jsonl,compare_outputs
from run_s2_evaluation import bootstrap,image_effects
from damage_acceptance import score_damage
from s1_geometry import iter_prediction_array,full_rle_iou


def main(path,local):
    start=time.monotonic()
    run=read_json(path/'run.json');c=read_json(path/'EVAL_CONTRACT.json')
    scope=run['scope'];budget=600 if scope=='smoke' else 1800
    require(run['execution_status']=='completed','incomplete Run')
    require(read_json(path/'CONFIG_AUDIT.json')['status']=='passed','configuration audit failed')
    for rel,digest in read_json(path/'SOURCE_MANIFEST.json')['files'].items():
        require(sha256(ROOT/rel)==digest,'frozen source differs: '+rel)
    panel=read_json(path/'DAMAGE_PANEL.json');receipt=read_json(path/'PANEL_RECEIPT.json')
    require(sha256(path/'DAMAGE_PANEL.json')==receipt['panel_sha256'],'panel SHA differs')
    require(panel['inference_contract_sha256']==sha256(path/'EVAL_CONTRACT.json'),'panel contract binding differs')
    require(panel['checkpoint_sha256']==c['checkpoint_sha256'],'panel checkpoint binding differs')
    ids=read_json(path/'EVALUATION_IMAGES.json')
    union=sorted(set(c['dev_ids'])|set(panel['image_ids']))
    require(ids==(union if scope=='full' else union[:8]),'image denominator differs')
    indices=read_json(path/'ROW_MAPPING.json')['original_global_indices']
    require(indices==sorted(set(indices)),'row mapping not ordered unique')
    base=read_jsonl(path/'BASELINE_SUBSET_GT_LEDGER.jsonl')
    fullbase=read_jsonl(ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs'/c['source_baseline_run']/'ledger/PER_GT_FAILURE.jsonl')
    require(base==[r for r in fullbase if r['image_id'] in set(ids)],'baseline subset differs')
    wanted={(r['image_id'],r['annotation_id']):r for r in base}
    eff={a:read_jsonl(path/(a+'_FIXED_PAIR_EFFECTS.jsonl')) for a in ('U','G','CONTROL')}
    olddir=ROOT/'experiments/mask_readout_large_mask_gate_20261015/runs'/c['control_development_run']
    old=[r for r in read_jsonl(olddir/'FIXED_PAIR_EFFECTS.jsonl') if r['image_id'] in set(ids)]
    key=lambda r:(r['image_id'],r['annotation_id'],r['baseline_output_detection_id'])
    oldkey={key(r):r for r in old}
    for arm,rows in eff.items():
        require(len(rows)==len(oldkey) and {key(r) for r in rows}==set(oldkey),'fixed pair identity differs')
        for r in rows:
            o=oldkey[key(r)]
            require(r['baseline_iou']==o['baseline_iou'] and r['population']==o['population'],'baseline fixed pair drift')
            require(r['delta_iou']==r['method_iou']-r['baseline_iou'],'delta calculation differs')
            require(r['same_output_reaches_075']==(r['method_iou']>=.75),'repair endpoint differs')
            if arm=='CONTROL':require(abs(r['method_iou']-o['method_iou'])<1e-12,'control fixed pair replay differs')
    ledgers={a:read_jsonl(path/(a+'_GT_LEDGER.jsonl')) for a in eff}
    for arm,rows in ledgers.items():
        require(len(rows)==len(wanted) and {(r['image_id'],r['annotation_id']) for r in rows}==set(wanted),'complete GT coverage differs')
        source=read_json(path/'ROW_MAPPING.json')['source_prediction_sha256']
        for r in rows:
            o=wanted[r['image_id'],r['annotation_id']]
            for k in ('is_ignored','gt_category_id','gt_area','task_box75'):require(r[k]==o[k],'GT/box differs')
            oid=r['bbox75_match_output_id']
            mapped=None if oid is None else source+':'+str(indices[int(oid.rsplit(':',1)[1])])
            require(mapped==o['bbox75_match_output_id'],'bbox identity differs')
    summary=read_json(path/'SUMMARY.json');verified={}
    if scope=='full':
        for arm,rows in eff.items():
            require(len(rows)==2925 and sum(r['population']=='T' for r in rows)==655,'effect denominator differs')
            target=bootstrap(image_effects(rows));overall=bootstrap(image_effects(rows,target_only=False))
            damage=score_damage(panel,fullbase,[r for r in ledgers[arm] if r['image_id'] in set(panel['image_ids'])])
            actual=summary[arm]
            require(target==actual['target'] and overall==actual['all_bbox75'] and damage==actual['damage'],'statistics mismatch '+arm)
            keyed={(r['image_id'],r['annotation_id']):r for r in ledgers[arm]}
            t=[r for r in rows if r['population']=='T']
            fixed=sum(r['method_iou']>=.75 for r in t)
            task=sum(keyed[r['image_id'],r['annotation_id']]['task_mask75']=='success' for r in t)
            require(fixed==actual['target_fixed_detection_repaired'] and task==actual['target_task_repaired'],'repair count differs')
            verified[arm]=dict(target=target,all_bbox75=overall,damage=damage,fixed_repaired=fixed,task_repaired=task)
        new=image_effects(eff['G']);control=image_effects(eff['CONTROL'])
        contrast=bootstrap({i:new[i]-control[i] for i in new})
        require(contrast==summary['G_minus_CONTROL'],'paired contrast differs')
        g=verified['G']
        gates=dict(target_effect=g['target']['mean']>=.005 and g['target']['ci95'][0]>0,
            all_bbox75_guardrail=g['all_bbox75']['mean']>=-.001,
            success_damage=g['damage']['gate']=='damage_guardrail_passed',improvement_over_original_G=contrast['ci95'][0]>0)
        status='passed' if all(gates.values()) else ('blocked' if all(gates[k] for k in ('target_effect','all_bbox75_guardrail','improvement_over_original_G')) and g['damage']['gate']=='blocked_insufficient_precision' else 'failed')
        require(gates==summary['gates'] and status==read_json(path/'DECISION.json')['status'],'decision differs')
    if not local:
        from pycocotools.coco import COCO
        from pycocotools import mask as masks
        from build_native_ledger import evaluate_ledger
        gt=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
        coco=COCO(str(gt));byindex={}
        keyed={a:{key(r):r for r in rows} for a,rows in eff.items()}
        for r in old:byindex.setdefault(int(r['baseline_output_detection_id'].rsplit(':',1)[1]),[]).append(r)
        streams=[iter_prediction_array(path/(a+'_predictions.json')) for a in ('NATIVE','U','G','CONTROL')]
        gate_rows=iter(read_jsonl(path/'GATE_ROWS.jsonl'));count=0;selected=changed=0
        for count,rows in enumerate(zip_longest(*streams),1):
            require(all(r is not None for r in rows),'output lengths differ')
            native,u,g,control=rows
            for other in (u,g,control):compare_outputs([native],[other])
            area=int(masks.area(native['segmentation']));apply=area>=9216
            expected=u['segmentation'] if apply else native['segmentation']
            require(g['segmentation']==expected,'gating RLE differs')
            gr=next(gate_rows)
            require(gr['native_mask_area']==area and gr['apply_delta']==apply and gr['original_global_index']==indices[count-1],'gate evidence differs')
            require(gr['mask_changed']==(g['segmentation']!=native['segmentation']),'mask change flag differs')
            selected+=int(apply);changed+=int(gr['mask_changed'])
            for oldrow in byindex.get(indices[count-1],[]):
                target=coco.annToRLE(coco.anns[oldrow['annotation_id']])
                require(abs(full_rle_iou(native['segmentation'],target)-oldrow['baseline_iou'])<1e-12,'native pair RLE differs')
                for arm,pred in [('U',u),('G',g),('CONTROL',control)]:
                    value=full_rle_iou(pred['segmentation'],target)
                    require(abs(value-keyed[arm][key(oldrow)]['method_iou'])<1e-12,'independent fixed pair RLE differs')
            if count%25000==0:
                require(time.monotonic()-start<budget,'independent verification budget')
                print('INDEPENDENT_ROWS',count,flush=True)
        require(count==len(indices) and next(gate_rows,None) is None,'complete rows differ')
        export=read_json(path/'EXPORT.json')
        require(export['outputs']==count==export['zero_delta_RLE_exact'] and selected==export['gate_on'] and changed==export['G_masks_changed'],'export counters differ')
        # Independently verify subset rows against complete frozen source JSONs.
        for arm,source in [('NATIVE',ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs'/c['source_baseline_run']/'trace/native/predictions.json'),
                           ('CONTROL',ROOT/'experiments/mask_gate_s3r_20261015/runs'/c['control_full_run']/'predictions_mixed.json')]:
            local_rows=iter(iter_prediction_array(path/(arm+'_predictions.json')));pos=0
            for index,row in enumerate(iter_prediction_array(source)):
                if pos==len(indices):break
                if index==indices[pos]:
                    require(row==next(local_rows),'frozen source subset differs '+arm);pos+=1
            require(pos==len(indices) and next(local_rows,None) is None,'source subset coverage differs')
        del coco;gc.collect()
        with (path/'INDEPENDENT_G_COCO.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
            fresh,ap=evaluate_ledger(gt,read_json(path/'G_predictions.json'),ids,sha256(path/'G_predictions.json'))
        require(fresh==ledgers['G'],'independent complete G COCO ledger differs')
        if scope=='full':require(ap==summary['G']['secondary_AP_on_union'],'COCO AP receipt differs')
        for arm in eff:
            require({r['source_prediction_sha'] for r in ledgers[arm]}=={sha256(path/(arm+'_predictions.json'))},'ledger/prediction binding differs')
        for rel,digest in c['inputs_sha256'].items():require(sha256(ROOT/rel)==digest,'input changed after evaluation')
    require(time.monotonic()-start<budget,'independent verification budget')
    name='INDEPENDENT_LOCAL.json' if local else 'INDEPENDENT_LAPTOP.json'
    write_new(path/name,dict(status='passed',at=now(),scope='statistics_and_source_identity' if local else 'all_output_identity_gating_RLE_effects_statistics_and_independent_G_COCO',
        images=len(ids),outputs=len(indices),fixed_pairs=len(old),contract_sha256=sha256(path/'EVAL_CONTRACT.json'),
        panel_sha256=sha256(path/'DAMAGE_PANEL.json'),summary_sha256=sha256(path/'SUMMARY.json'),
        statistics=verified,wall_seconds=time.monotonic()-start,python=sys.version))
    print('INDEPENDENT_S2_PASSED',path.name,'local',local,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
    main(a.run,a.statistics_only)
