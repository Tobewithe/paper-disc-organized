"""Independent integer-RLE geometry, identities, denominators and direction verification."""
from common import ROOT, STUDY, S1, S3, BASE, GT, read_json, read_jsonl, require, sha256, write_new, inspect_inputs
import argparse
import json
from pathlib import Path


def foreground_intervals(rle):
    counts=[]
    code=rle['counts']
    cursor=0
    while cursor<len(code):
        value=0
        shift=0
        while True:
            token=ord(code[cursor])-48
            require(0<=token<64, 'invalid compressed RLE token')
            cursor+=1
            value|=(token&31)<<(5*shift)
            shift+=1
            if not token&32:
                if token&16:
                    value|=-1<<(5*shift)
                break
        if len(counts)>2:
            value+=counts[-2]
        require(value>=0,'negative RLE length')
        counts.append(value)
    require(sum(counts)==rle['size'][0]*rle['size'][1],'RLE length mismatch')
    start=0
    intervals=[]
    for n,length in enumerate(counts):
        if n%2 and length:
            intervals.append((start,start+length))
        start+=length
    return intervals


def area(rle):
    return sum(b-a for a,b in foreground_intervals(rle))


def iou(a,b):
    require(a['size']==b['size'],'RLE dimensions differ')
    x,y=foreground_intervals(a),foreground_intervals(b)
    i=j=intersection=0
    while i<len(x) and j<len(y):
        intersection+=max(0,min(x[i][1],y[j][1])-max(x[i][0],y[j][0]))
        if x[i][1]<=y[j][1]:
            i+=1
        else:
            j+=1
    union=sum(v-u for u,v in x)+sum(v-u for u,v in y)-intersection
    return intersection/union if union else 0.0


def box_iou(a,b):
    intersection=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)


def self_test():
    x=dict(size=[2,3],counts='0220')
    y=dict(size=[2,3],counts='123')
    empty=dict(size=[2,3],counts='6')
    full=dict(size=[2,3],counts='06')
    require(area(x)==4 and area(y)==2 and area(empty)==0 and area(full)==6,'RLE self-test area')
    require(iou(x,y)==.2 and iou(x,x)==1 and iou(empty,empty)==0 and iou(full,x)==4/6,'RLE self-test IoU')


def verify(run, smoke, laptop):
    self_test()
    for path in (run,smoke):
        require(read_json(path/'run.json')['execution_status']=='completed','Run not completed')
        require(read_json(path/'CONFIG_AUDIT.json')['status']=='passed','configuration audit failed')
        for record in read_json(path/'ARTIFACT_MANIFEST.json')['artifacts']:
            require(sha256(path/record['relative_path'])==record['sha256'],'artifact hash mismatch')
        snapshots=read_json(path/'SOURCE_SNAPSHOT_MAP.json')
        manifest=read_json(path/'SOURCE_MANIFEST.json')
        require(manifest['files']==read_json(STUDY/'FREEZE_MANIFEST.json')['files'],'Run source freeze mismatch')
        for relative,digest in manifest['files'].items():
            require(sha256(path/snapshots[relative])==digest,'source snapshot mismatch')
        require(manifest['frozen_at']<read_json(path/'run.json')['started_at'],'freeze must precede Run')
    panel=read_json(run/'SOURCE_TARGET_PANEL.json')['targets']
    require(len(panel)==256,'wrong panel denominator')
    contract=read_json(run/'EVAL_CONTRACT.json')
    require(sha256(run/'SOURCE_TARGET_PANEL.json')==contract['original_panel_sha256'],'panel copy changed')
    oracle_path=S1.relative_to(ROOT).as_posix()+'/ORACLE_RESULTS.jsonl'
    require(sha256(run/'SOURCE_ORACLE_RESULTS.jsonl')==contract['input_files'][oracle_path],'oracle copy hash mismatch')
    key=lambda r:(r['image_id'],r['annotation_id'],r['output_detection_id'])
    oracle={key(r):r for r in read_jsonl(run/'SOURCE_ORACLE_RESULTS.jsonl')}
    pairs=read_jsonl(run/'PAIR_RESULTS.jsonl')
    evidence=read_jsonl(run/'PAIRED_MASKS.jsonl')
    require([key(r) for r in pairs]==[key(r) for r in panel]==[key(r) for r in evidence],'panel order/identity changed')
    require(len({key(r) for r in pairs})==256,'duplicate pairs')
    annotations={r['id']:r for r in read_json(run/'SOURCE_GT_ANNOTATIONS.json')}
    ledger={(r['image_id'],r['annotation_id']):r for r in read_json(run/'SOURCE_LEDGER_ROWS.json')}
    tasks={(r['image_id'],r['annotation_id']):r for r in read_json(run/'SOURCE_S3_CENSUS_ROWS.json')}
    counts={'L':0,'G':0,'R':0}
    oracle_count=ungated_count=gated_count=0
    max_error=0.0
    for row,proof,target in zip(pairs,evidence,panel):
        ann=annotations[row['annotation_id']]
        pred=proof['native_prediction']
        source_oracle=oracle[key(row)]
        prior=ledger[(row['image_id'],row['annotation_id'])]
        task=tasks[(row['image_id'],row['annotation_id'])]
        require(row['global_index']==proof['global_index']==int(row['output_detection_id'].rsplit(':',1)[1]),'wrong global index')
        require(ann['image_id']==pred['image_id']==row['image_id'] and ann['category_id']==pred['category_id']==row['category_id'],'image/class identity')
        require(not ann.get('iscrowd',0) and not ann.get('ignore',0) and not prior['is_ignored'],'not ordinary GT')
        require(prior['bbox75_match_output_id']==row['output_detection_id'] and prior['task_mask75']=='failure','original Box75 pair changed')
        require(abs(box_iou(pred['bbox'],ann['bbox'])-row['box_iou'])<1e-12 and row['box_iou']>=.75,'independent Box IoU differs')
        require(proof['native_rle']==pred['segmentation'] and proof['oracle_rle']==source_oracle['oracle_rle'],'source RLE differs')
        values={name:iou(proof[name+'_rle'],proof['GT_rle']) for name in ('native','oracle','ungated','gated')}
        for name,value in values.items():
            error=abs(value-row[name+'_iou'])
            max_error=max(max_error,error)
            require(error<1e-12,'independent RLE IoU differs')
        require(abs(values['native']-target['baseline_mask_iou'])<1e-12 and values['native']<.75,'native failed status changed')
        require(abs(values['oracle']-source_oracle['oracle_mask_iou'])<1e-12,'source oracle IoU differs')
        pixels=area(proof['native_rle'])
        gate=pixels>=9216
        require(pixels==row['native_area'] and gate==row['gate_enabled'],'area gate incorrect')
        require(row['native_area_group']==('S' if pixels<1024 else 'M' if pixels<9216 else 'L'),'area group incorrect')
        require(proof['gated_rle']==proof['ungated_rle' if gate else 'native_rle'],'gate pixels incorrect')
        o,a,g=values['oracle']>=.75,values['ungated']>=.75,values['gated']>=.75
        require((o,a,g)==(row['oracle_found'],row['ungated_repaired'],row['gated_repaired']),'success flags incorrect')
        if o:
            label='L' if values['ungated']<.75 else ('R' if pixels>=9216 else 'G')
            counts[label]+=1
        else:
            label='O_unfound'
        require(label==row['decomposition'],'decomposition incorrect')
        oracle_count+=o
        ungated_count+=a
        gated_count+=g
        require(not task['baseline_success'] and not task['ignored'],'task baseline identity')
        require(row['s3_task_success']==task['method_success'] and row['s3_method_dt_id']==task['method_dt_id'],'COCO task status changed')
        require(row['ungated_COCO_task_success'] is None,'unmeasured COCO state filled')
    summary=read_json(run/'SUMMARY.json')
    require(oracle_count==122 and sum(counts.values())==122 and counts==summary['decomposition'],'O+ partition incorrect')
    require(summary['all']['ungated_repaired']==ungated_count and summary['all']['gated_repaired']==gated_count,'repair denominator incorrect')
    direction=('conditional_repair_learning' if counts['L']>counts['G'] else
               'damage_constrained_benefit_selector' if counts['G']>counts['L'] else
               'joint_conditional_repair_and_selection' if counts['L'] else 'coverage_and_COCO_task_conversion')
    require(summary['priority_direction']==direction==read_json(run/'DECISION.json')['next_action'],'direction differs from preregistration')
    groups={'all':pairs,'oracle_found_group':[r for r in pairs if r['oracle_found']],
            'oracle_unfound_group':[r for r in pairs if not r['oracle_found']],
            'L_group':[r for r in pairs if r['decomposition']=='L']}
    for group,part in groups.items():
        stats=summary[group]
        require(stats['n']==len(part),'group denominator incorrect')
        for name in ('ungated','gated'):
            value=sum(r[name+'_iou']-r['native_iou'] for r in part)/len(part) if part else None
            require(value==stats['mean_'+name+'_delta'],'group mean incorrect')
        for name,predicate in [('improved',lambda r:r['ungated_iou']>r['native_iou']),
                               ('worsened',lambda r:r['ungated_iou']<r['native_iou']),
                               ('unchanged',lambda r:r['ungated_iou']==r['native_iou'])]:
            require(sum(predicate(r) for r in part)==stats[name],'IoU sign counts incorrect')
    full_by_key={key(r):r for r in pairs}
    smoke_pairs=read_jsonl(smoke/'PAIR_RESULTS.jsonl')
    for row in smoke_pairs:
        require(full_by_key[key(row)]==row,'smoke/full same pair differs')
    result=dict(status='passed', pairs=256,images=len({r['image_id'] for r in pairs}),oracle_positive=122,
                independent_integer_RLE_iou_max_error=max_error, decomposition=counts,
                priority_direction=direction, artifact_hashes_and_snapshots=True,
                smoke_overlap_pairs=len(smoke_pairs), raw_source_check='not_requested_on_this_host')
    if laptop:
        inspect_inputs()
        from s1_geometry import iter_prediction_array
        from pycocotools.coco import COCO
        wanted={r['global_index']:r for r in evidence}
        checked=0
        for index,prediction in enumerate(iter_prediction_array(BASE/'trace/native/predictions.json')):
            if index in wanted:
                require(prediction==wanted[index]['native_prediction'],'native source prediction changed')
                checked+=1
        require(checked==256,'source prediction missing')
        for index,prediction in enumerate(iter_prediction_array(S3/'predictions_mixed.json')):
            if index in wanted:
                proof=wanted[index]
                require(prediction=={**proof['native_prediction'],'segmentation':proof['gated_rle']},'S3 source mixed prediction changed')
        coco=COCO(str(GT))
        for proof in evidence:
            ann=coco.anns[proof['annotation_id']]
            require(ann==annotations[proof['annotation_id']],'GT annotation snapshot differs')
            rle=coco.annToRLE(ann)
            if isinstance(rle['counts'],bytes):
                rle['counts']=rle['counts'].decode('ascii')
            require(rle==proof['GT_rle'],'GT RLE snapshot differs')
        result['raw_source_check']='original GT/native/S3 and all frozen file hashes verified on laptop'
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path)
    p.add_argument('--smoke-run',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--laptop-input-audit',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
        print('PASS independent RLE interval geometry self-test')
        return
    require(a.run is not None and a.smoke_run is not None and a.out is not None,'Run/smoke/output required')
    result=verify(a.run,a.smoke_run,a.laptop_input_audit)
    write_new(a.out,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
