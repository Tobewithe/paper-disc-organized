"""Frozen candidate policies on every baseline prediction, then full COCO AP."""
import argparse
import csv
import json
import time
import contextlib
import io
import gc
from pathlib import Path
from collections import Counter
from common import setup,atomic,progress


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();p,out=setup(args.protocol,args.output);evp=p['evaluation']
    import numpy as np
    import torch
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    from data_access import load_bank
    from mask_calibration import input_logits,export_masks
    from local_features import local_features
    from portable_risk import PortableRisk
    variants=evp['variants'];root=Path(evp['bank']);bank=load_bank(root)
    modelroot=root.parent/evp['model_run'];selection=json.loads((modelroot/'SUMMARY.json').read_text())
    estimators={name:PortableRisk(modelroot/(name+'.json')) for name in variants}
    policies=selection['policies'];assert all(selection['advance'][name] for name in variants)
    # Confirm numeric model export reproduces every selected training decision.
    trainfeatures=np.load(root.parent/p['runs']['extract']/'features.npz')
    trainkeys=trainfeatures['keys'];trainlookup={(int(a),int(b)):i for i,(a,b,_) in enumerate(trainkeys)}
    expected=list(csv.DictReader((modelroot/'selection_decisions.csv').open(newline='',encoding='utf-8')))
    indices=np.asarray([trainlookup[(int(r['image_id']),int(r['candidate_index']))] for r in expected])
    for name in variants:
        multi=name.startswith('multi');xx=trainfeatures['shape'][indices]
        if multi:xx=np.concatenate([xx,trainfeatures['tau'][indices,:,None]],2)
        xx=np.concatenate([xx,trainfeatures['local'][indices]],2)
        pred=estimators[name].predict(xx.reshape(-1,xx.shape[-1])).reshape(len(indices),-1) if multi else estimators[name].predict(xx[:,0])[:,None]
        eligible=trainfeatures['eligible'][indices] if multi else trainfeatures['eligible'][indices,:1]
        pred=np.where(eligible,pred,-np.inf);chosen=np.where(pred.max(1)>policies[name]['cutoff'],pred.argmax(1),-1)
        assert np.array_equal(chosen,[int(r['action_'+name]) for r in expected]),'Numeric export changed policy'
    del trainfeatures,expected,xx,pred;gc.collect()
    reference=json.loads((Path(evp['reference_evaluation'])/'SUMMARY.json').read_text())
    originals=json.loads((root/'predictions_official_zero.json').read_text())
    base_lookup={(r['image_id'],r['candidate_index']):r for r in originals}
    assert len(base_lookup)==len(originals)
    matched={(r['image_id'],r['candidate_index']):r for r in bank['matched']}
    coco=COCO(evp['annotations']);ids=sorted(bank['ids']);assert len(ids)==4500
    source=out/'image_paths.txt';source.write_text('\n'.join(str(Path(evp['images'])/coco.imgs[i]['file_name']) for i in ids),encoding='utf-8')
    handles={name:(out/('predictions_'+name+'.jsonl')).open('w',encoding='utf-8') for name in variants}
    decisionfile=(out/'instance_decisions.csv').open('w',newline='',encoding='utf-8');writer=None
    decisions=[];actioncounts={name:Counter() for name in variants};done=0;predcount=0;checked=0;started=time.perf_counter()
    model=YOLO(p['weights']);assert model.model.model[-1].end2end is True
    catbyname={r['name']:r['id'] for r in coco.loadCats(coco.getCatIds())};catmap={i:catbyname[n] for i,n in model.names.items()}
    def encode(masks):
        rles=mu.encode(np.asfortranarray(masks.cpu().numpy().transpose(1,2,0)))
        for r in rles:r['counts']=r['counts'].decode('ascii')
        return rles
    class Evaluator(SegmentationPredictor):
        def construct_result(self,pred,img,orig_img,img_path,proto):
            nonlocal done,predcount,checked,writer
            iid=int(Path(img_path).stem);orig=orig_img.shape[:2];shape=img.shape[2:]
            boxes=ops.scale_boxes(shape,pred[:,:4].clone(),orig).cpu().numpy()
            # Preserve original candidate indices and GEMM batch size including empty candidates.
            for off in range(0,len(pred),24):
                one=pred[off:off+24];logits=input_logits(proto,one[:,6:],one[:,:4],shape)
                binary=(logits>0).byte();base=export_masks(binary,orig);area=base.sum((1,2)).float()
                rles=encode(base);m=len(one);actions=[];features=[];locals_=[];taus=[];eligibility=[]
                bankidx=[bank['lookup'][(iid,j)] for j in range(off,off+m)]
                bx=bank['features'][bankidx]
                assert np.max(np.abs(area.cpu().numpy()-bank['area'][bankidx]))<.5
                for j,a in enumerate(p['actions']):
                    tau=.75/(1+(area/2304).square()) if j==0 else torch.full_like(area,float(a['tau']))
                    trial=(logits>tau[:,None,None]).byte();exp=export_masks(trial,orig)
                    ta=exp.sum((1,2)).cpu().numpy();xf=bx.copy();xf[:,4]=(area.cpu().numpy()-ta)/np.maximum(area.cpu().numpy(),1)
                    features.append(xf);locals_.append(local_features(logits,binary,trial));taus.append(tau.cpu().numpy())
                    eligibility.append(ta>0);actions.append(trial)
                features=np.stack(features,1);lf=np.stack(locals_,1);taus=np.stack(taus,1);eligible=np.stack(eligibility,1)
                chosen={};outputs={};decision_iou={}
                for name in variants:
                    multi=name.startswith('multi');xx=features
                    if multi:xx=np.concatenate([xx,taus[:,:,None]],2)
                    xx=np.concatenate([xx,lf],2)
                    scores=estimators[name].predict(xx.reshape(-1,xx.shape[-1])).reshape(m,-1) if multi else estimators[name].predict(xx[:,0])[:,None]
                    scores=np.where(eligible if multi else eligible[:,:1],scores,-np.inf)
                    choice=np.where(scores.max(1)>policies[name]['cutoff'],scores.argmax(1),-1);chosen[name]=choice
                    masks=binary.clone()
                    for action in range(len(actions)):
                        use=torch.as_tensor(choice==action,device=masks.device);masks[use]=actions[action][use]
                    outputs[name]=encode(export_masks(masks,orig))
                for j in range(m):
                    key=(iid,off+j);old=base_lookup.get(key);row=bank['rows'][bankidx[j]]
                    assert np.max(np.abs(boxes[off+j]-[float(row[z]) for z in ['box_x1','box_y1','box_x2','box_y2']]))<1e-3
                    assert catmap[int(one[j,5])]==int(row['category_id']) and abs(float(one[j,4])-float(row['score']))<1e-6
                    if old is None:
                        assert float(mu.area(rles[j]))==0
                        continue
                    assert old['segmentation']==rles[j],('Baseline RLE mismatch',key)
                    checked+=1;predcount+=1
                    for name in variants:
                        output={z:old[z] for z in ['image_id','category_id','score']};output['segmentation']=outputs[name][j]
                        assert float(mu.area(output['segmentation']))>0
                        handles[name].write(json.dumps(output,separators=(',',':'))+'\n')
                        actioncounts[name][int(chosen[name][j])]+=1
                    if key in matched:
                        r=matched[key];truth=coco.annToRLE(coco.anns[r['annotation_id']]);line=dict(r)
                        for name in variants:
                            rr=outputs[name][j];a=float(mu.area(rr));g=r['gt_area'];ji=float(mu.iou([rr],[truth],[0])[0,0]);tp=ji*(a+g)/(1+ji)
                            line['action_'+name]=int(chosen[name][j]);line['iou_'+name]=ji;line['coverage_'+name]=tp/g
                        if writer is None:writer=csv.DictWriter(decisionfile,fieldnames=list(line));writer.writeheader()
                        writer.writerow(line);decisions.append(line)
            done+=1
            if done%100==0 or done==len(ids):
                for f in handles.values():f.flush()
                decisionfile.flush();progress(out,'export',images=done,total=len(ids),predictions=predcount,elapsed_seconds=time.perf_counter()-started)
            return Results(orig_img,path=img_path,names=self.model.names,boxes=None,masks=None)
    try:
        for _ in model.predict(source=str(source),predictor=Evaluator,imgsz=640,conf=.001,max_det=300,batch=1,
            half=False,retina_masks=False,device=0,verbose=False,stream=True,save=False):pass
    finally:
        for f in handles.values():f.close()
        decisionfile.close()
    assert predcount==506966 and len(decisions)==30426
    stats={}
    for name in variants:
        target=[r for r in decisions if r['target']]
        stats[name]=dict(target_repairs=sum(r['iou_'+name]>=.75 for r in target),
            all_repairs=sum(r['baseline_iou']<.75<=r['iou_'+name] for r in decisions),
            damages=sum(r['iou_'+name]<.75<=r['baseline_iou'] for r in decisions),
            target_mean_iou_gain_pp=float(np.mean([100*(r['iou_'+name]-r['baseline_iou']) for r in target])),
            target_mean_coverage_loss_pp=float(np.mean([100*(r['baseline_coverage']-r['coverage_'+name]) for r in target])))
        stats[name]['net_repairs']=stats[name]['all_repairs']-stats[name]['damages']
    atomic(out/'outcomes.json',stats)
    # Release inference banks before loading per-variant full COCO predictions.
    del originals,base_lookup,bank,model;gc.collect();torch.cuda.empty_cache()
    metrics={}
    for name in variants:
        progress(out,'cocoeval',variant=name,outcomes=stats[name])
        records=[json.loads(line) for line in (out/('predictions_'+name+'.jsonl')).open(encoding='utf-8')]
        with contextlib.redirect_stdout(io.StringIO()) as log:
            dt=coco.loadRes(records);ce=COCOeval(coco,dt,'segm');ce.params.imgIds=ids;ce.evaluate();ce.accumulate();ce.summarize()
        metrics[name]=dict(ap=float(ce.stats[0])*100,ap75=float(ce.stats[2])*100,stats=ce.stats.tolist(),predictions=len(records))
        (out/('cocoeval_'+name+'.log')).write_text(log.getvalue(),encoding='utf-8')
        atomic(out/'metrics_partial.json',metrics);del records,dt,ce;gc.collect()
    verdict={}
    for name in variants:
        checks={}
        for ref in ['frozen_rcmc','direct']:
            s=stats[name];r=reference['outcomes'][ref]
            checks[ref]=dict(target_repair_delta=s['target_repairs']-r['target_repairs'],damage_delta=s['damages']-r['damages'],
                ap_delta=metrics[name]['ap']-reference['metrics'][ref]['ap'])
            checks[ref]['passes']=(s['target_repairs']>=r['target_repairs'] and s['damages']<=r['damages'] and
                (s['target_repairs']>r['target_repairs'] or s['damages']<r['damages']) and checks[ref]['ap_delta']>=-1e-10)
        verdict[name]=dict(checks=checks,passes=all(c['passes'] for c in checks.values()))
    result=dict(images=len(ids),ordinary_gt=sum(not a.get('iscrowd',0) for i in ids for a in coco.imgToAnns[i]),
        matched=len(decisions),targets=1701,original_successes=19065,outcomes=stats,metrics=metrics,verdict=verdict,
        full_baseline_rle_parity=checked,action_counts={n:dict(c) for n,c in actioncounts.items()},policies=policies,
        numeric_policy_matches_selection=True,elapsed_seconds=time.perf_counter()-started,
        decision='needs_independent_confirmation' if any(v['passes'] for v in verdict.values()) else 'stop_no_retuning',
        limitations=['Historical explored COCO val; not a fresh blind test.','No changes in boxes, scores, categories or prediction counts.',
                    'One seed; frozen after train selection; AP point estimates only.','Timing includes research diagnostics and RLE, not a deployment benchmark.'])
    atomic(out/'SUMMARY.json',result);progress(out,'completed',**result)


if __name__=='__main__':main()
