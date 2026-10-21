"""Controlled deletion of appearance at fixed OC-SORT association stages."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import time
import numpy as np
from .common import Run,read_json,write_json,setup_paths,iou_matrix,load_gt,sha256
from .tracking import frame_input
from .learn import load_cache,load_identity,embed_array
from .evaluate import official_metrics,metric_data,compact


def tracker_for(mode,config):
    from ultralytics.trackers.deep_oc_sort import DeepOCSORT
    from ultralytics.trackers.utils import matching
    class Controlled(DeepOCSORT):
        def __init__(self):
            super().__init__(SimpleNamespace(**deepcopy(config)))
            self.audit=[]
        def _fuse_appearance(self,dists,tracks,detections,iou_dists=None):
            stage='recovery' if iou_dists is None else 'first'
            fused=super()._fuse_appearance(dists,tracks,detections,iou_dists)
            actual=dists if mode=='motion_only' or (mode=='first_only' and stage=='recovery') or (mode=='recovery_only' and stage=='first') else fused
            if tracks and detections:
                bm,_,_=matching.linear_assignment(dists.copy(),thresh=self.args.match_thresh)
                am,_,_=matching.linear_assignment(actual.copy(),thresh=self.args.match_thresh)
                base=set(map(tuple,bm))
                for i,j in am:
                    if (i,j) not in base:
                        a,b=tracks[i].xyxy,detections[j].xyxy
                        scale=max(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]),1)
                        self.audit.append(dict(frame=self.frame_id,stage=stage,track=int(tracks[i].track_id),det=int(detections[j].idx),motion_cost=float(dists[i,j]),fused_cost=float(actual[i,j]),box_iou=float(iou_matrix(a[None],b[None])[0,0]),center_distance=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/scale)))
            return actual
    return Controlled()


def run_control(mode,records,z,config):
    tracker=tracker_for(mode,config['tracker']);out=[]
    for rec in records:
        boxes,features=frame_input(rec,z)
        rows=tracker.update(boxes,img=None,feats=features)
        tracks=[]
        for row in rows:
            bid,idx=int(row[4]),int(row[-1])
            tracks.append(dict(base_id=bid,det_id=idx,box=rec['boxes'][idx],score=float(rec['scores'][idx]),write=True,identity=bid,status='tracked'))
        out.append(dict(frame=rec['frame'],tracks=tracks))
    return out,tracker.audit


def diagnose(root):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json')
    model,ck=load_identity(root);metrics=official_metrics(root)
    seqs=[s for s in manifest['sequences'] if s['split'] in {'dev','test'}]
    modes=['full','motion_only','first_only','recovery_only']
    contract=dict(operation='same tracker appearance stage deletion',modes=modes,config=cfg,
                  GT='offline evaluation only',new_parameters=0,training_steps=0,sequences=[s['name'] for s in seqs],
                  test_scope='previously observed retrospective set, not new holdout',
                  identity=read_json(root/'identity_pointer.json'),features=read_json(root/'features_pointer.json'))
    with Run(root,'association_stage_diagnosis',contract) as run:
        raw={};table=[];parity=[]
        for seq in seqs:
            rec,rawx=load_cache(root,seq);z=embed_array(model,ck,rawx);gt=load_gt(root,seq)
            for mode in modes:
                start=time.perf_counter();out,events=run_control(mode,rec,z,cfg)
                dest=run.path/mode/seq['name'];dest.mkdir(parents=True)
                write_json(dest/'tracks.json',out);write_json(dest/'association_changes.json',events)
                data=metric_data(gt,out);rs={k:m.eval_sequence(data) for k,m in metrics.items()}
                raw[mode,seq['name']]=rs;write_json(dest/'metric_details.json',rs)
                stat=dict(sequence=seq['name'],split=seq['split'],arm=mode,frames=seq['frames'],seconds=time.perf_counter()-start,**compact(rs),
                          changed_matches=len(events),changed_ocr=sum(e['stage']=='recovery' for e in events),changed_ocr_zero_iou=sum(e['stage']=='recovery' and e['box_iou']==0 for e in events))
                table.append(stat);print('DIAG',stat,flush=True)
                if mode=='full':
                    old=Path(read_json(root/'v1_reference.json')[seq['split']])/'deepocsort'/seq['name']/'tracks.json'
                    assert read_json(old)==out, f'v1 output mismatch: {seq["name"]}'
                    parity.append(seq['name'])
        combined={}
        for split in ['dev','test']:
            names=[s['name'] for s in seqs if s['split']==split]
            combined[split]={mode:compact({k:m.combine_sequences({s:raw[mode,s][k] for s in names}) for k,m in metrics.items()}) for mode in modes}
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,combined=combined,full_v1_exact_parity=parity))
        d=combined['dev'];gain=d['first_only']['IDF1']-d['full']['IDF1'];guard=d['first_only']['FN']<=d['full']['FN']*1.01
        if gain>=1 and guard: focus='constrain_recovery_appearance_with_spatiotemporal_evidence'
        elif d['motion_only']['IDF1']-d['full']['IDF1']>=1: focus='appearance_reliability_before_association'
        else:focus='motion_or_detection_limit_before_further_appearance_learning'
        decision=dict(status='diagnostic_completed',focus=focus,first_only_dev_delta_IDF1=gain,first_only_FN_guard=guard,scientific_scope='stage deletion under fixed tracker; no new method efficacy yet')
        write_json(run.path/'DECISION.json',decision);run.meta['scientific_status']='diagnostic_complete'
        write_json(root/'diagnosis_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        print('DECISION',decision,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();diagnose(a.root)
