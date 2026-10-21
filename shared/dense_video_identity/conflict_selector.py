"""Counterfactual net-benefit selection of complete association conflict groups."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .common import Run,read_json,write_json,setup_paths,unit,load_gt,sha256
from .learn import load_identity,embed_array,load_cache,match_gt,set_seed
from .tracking import frame_input
from .evaluate import official_metrics,metric_data,compact

FEATURES=['size','base_count','appearance_count','motion_base_mean','motion_app_mean',
          'cosine_base_mean','cosine_app_mean','cosine_app_min','row_margin_mean','row_margin_min',
          'col_margin_mean','col_margin_min','app_iou_mean','app_iou_min','score_mean','score_min',
          'gap_mean','gap_max','age_mean','memory_consistency_mean','memory_consistency_min']


def components(base,alt):
    b=set(map(tuple,base));a=set(map(tuple,alt));edges=b|a
    adjacent={}
    for i,j in edges:
        adjacent.setdefault(('t',i),set()).add(('d',j));adjacent.setdefault(('d',j),set()).add(('t',i))
    result=[];seen=set()
    for start in sorted(adjacent):
        if start in seen:continue
        stack=[start];vertices=set()
        while stack:
            v=stack.pop()
            if v in vertices:continue
            vertices.add(v);seen.add(v);stack.extend(adjacent[v]-vertices)
        rows={i for t,i in vertices if t=='t'};bb=sorted((i,j) for i,j in b if i in rows);aa=sorted((i,j) for i,j in a if i in rows)
        if bb!=aa:result.append((bb,aa))
    return result


def group_features(bb,aa,motion,cos,iou,tracks,dets,frame):
    edges=aa or bb;rows=sorted({i for i,j in bb+aa});cols=sorted({j for i,j in bb+aa})
    mean=lambda vals:float(np.mean(vals)) if len(vals) else 0.
    minimum=lambda vals:float(min(vals)) if len(vals) else 0.
    bc=[cos[i,j] for i,j in bb];ac=[cos[i,j] for i,j in aa]
    rm=[cos[i,j]-np.max(np.delete(cos[i],j),initial=-1) for i,j in edges]
    cm=[cos[i,j]-np.max(np.delete(cos[:,j],i),initial=-1) for i,j in edges]
    io=[iou[i,j] for i,j in edges];scores=[dets[j].score for j in cols]
    gaps=[min((frame-tracks[i].end_frame)/60,1) for i in rows]
    ages=[min(np.log1p(frame-tracks[i].start_frame)/6,1) for i in rows]
    consistency=[float(tracks[i].smooth_feat@tracks[i].curr_feat) for i in rows]
    return np.asarray([len(rows)/10,len(bb)/10,len(aa)/10,mean([motion[i,j] for i,j in bb]),mean([motion[i,j] for i,j in aa]),
        mean(bc),mean(ac),minimum(ac),mean(rm),minimum(rm),mean(cm),minimum(cm),mean(io),minimum(io),mean(scores),minimum(scores),
        mean(gaps),max(gaps),mean(ages),mean(consistency),minimum(consistency)],np.float32)


def build_tracker(mode,config,predict=None,threshold=1.1,collector=None):
    from ultralytics.trackers.byte_tracker import BYTETracker,STrack
    from ultralytics.trackers.utils.stracks import parse_bboxes
    from ultralytics.trackers.utils import matching
    class MemoryTrack(STrack):
        def __init__(self,xywh,score,cls,feat):
            super().__init__(xywh,score,cls);self.curr_feat=unit(feat);self.smooth_feat=self.curr_feat.copy()
        def update_memory(self,new):
            self.curr_feat=new.curr_feat.copy();beta=.05*max(0.,(new.score-.3)/.7)
            self.smooth_feat=unit((1-beta)*self.smooth_feat+beta*self.curr_feat)
        def update(self,new,frame):
            self.update_memory(new);super().update(new,frame)
        def re_activate(self,new,frame,new_id=False):
            self.update_memory(new);super().re_activate(new,frame,new_id)
    class Selective(BYTETracker):
        def __init__(self):
            super().__init__(SimpleNamespace(**deepcopy(config)));self.events=[]
        def init_track(self,results,img=None):
            if not len(results):return []
            return [MemoryTrack(b,s,c,f) for b,s,c,f in zip(parse_bboxes(results),results.conf,results.cls,img)]
        def _first_association(self,pool,dets,activated,refind):
            motion=self.get_dists(pool,dets)
            base,ut,ud=matching.linear_assignment(motion.copy(),thresh=self.args.match_thresh)
            if not pool or not dets:return ut,ud
            cos=np.clip(np.asarray([t.smooth_feat for t in pool])@np.asarray([d.curr_feat for d in dets]).T,-1,1)
            iou=1-matching.iou_distance(pool,dets)
            app=(1-cos)/2;app[(iou<.1)|(cos<.8)]=1.
            alt,_,_=matching.linear_assignment(np.minimum(motion,app),thresh=self.args.match_thresh)
            selected=set(map(tuple,base))
            for bb,aa in components(base,alt):
                x=group_features(bb,aa,motion,cos,iou,pool,dets,self.frame_id)
                prob=float(predict(x)) if predict else None
                choose= mode=='all_appearance' or (mode=='margin_rule' and min(x[9],x[11])>=.05) or (mode in {'net_benefit','correctness'} and prob>=threshold)
                if mode=='motion':choose=False
                if mode in {'extension','extension_trajectory'}:
                    choose=not bb and bool(aa) and (mode=='extension' or prob>=threshold)
                # Offline counterfactual labels may force exactly one group. Normal inference has no such attribute.
                forced=getattr(self,'force_group',None)
                if forced and forced['frame']==self.frame_id:
                    current=sorted((int(pool[i].track_id),int(dets[j].idx)) for i,j in bb)
                    if current==[tuple(e) for e in forced['base']]:choose=True
                if choose:selected.difference_update(bb);selected.update(aa)
                event=dict(frame=self.frame_id,base=[(int(pool[i].track_id),int(dets[j].idx)) for i,j in bb],
                    alternative=[(int(pool[i].track_id),int(dets[j].idx)) for i,j in aa],features=x.tolist(),selected_appearance=choose,probability=prob)
                self.events.append(event)
                if collector is not None:collector(event)
            if mode in {'extension','extension_trajectory'}:
                assert set(map(tuple,base)).issubset(selected),'Protected high-confidence motion edge was overwritten'
            chosen=sorted(selected);assert len({i for i,j in chosen})==len(chosen)==len({j for i,j in chosen})
            self._apply_matches(np.asarray(chosen,dtype=int).reshape(-1,2),pool,dets,activated,refind)
            return [i for i in range(len(pool)) if i not in {a for a,b in chosen}],[j for j in range(len(dets)) if j not in {b for a,b in chosen}]
    return Selective()


def sequence(mode,records,z,config,predict=None,threshold=1.1,collector=None,on_after=None):
    tracker=build_tracker(mode,config['tracker'],predict,threshold,collector);outputs=[]
    for rec in records:
        boxes,features=frame_input(rec,z);rows=tracker.update(boxes,img=None,feats=features)
        ts=[dict(base_id=int(t[4]),det_id=int(t[-1]),box=rec['boxes'][int(t[-1])],score=float(rec['scores'][int(t[-1])]),write=True,identity=int(t[4]),status='tracked') for t in rows]
        outputs.append(dict(frame=rec['frame'],tracks=ts))
        if on_after:on_after(rec['frame'],ts)
    return outputs,tracker.events


class Selector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(len(FEATURES),32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)


def collect(root):
    setup_paths(root);config=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root)
    with Run(root,'conflict_opportunity',dict(features=FEATURES,training_sequences=['pig_3','pig_5','horse_4','horse_6'],GT='offline label only',new_training=0)) as run:
        examples=[]
        for seq in [s for s in manifest['sequences'] if s['split']=='train']:
            records,raw=load_cache(root,seq);z=embed_array(model,ck,raw);mapping=match_gt(records,load_gt(root,seq));last_gt={}
            def label(e):
                mm=mapping[e['frame']]
                # Skip unknown track/detection correspondence rather than inventing a negative label.
                if any(t not in last_gt or d not in mm for t,d in e['base']+e['alternative']):return
                b=sum(last_gt[t]==mm[d] for t,d in e['base']);a=sum(last_gt[t]==mm[d] for t,d in e['alternative'])
                examples.append(dict(e,sequence=seq['name'],gain=a-b,base_correct=b,alt_correct=a,
                                     alt_all_correct=bool(e['alternative']) and a==len(e['alternative'])))
            def update(f,tracks):
                for t in tracks:
                    gid=mapping[f].get(t['det_id'])
                    if gid is not None:last_gt[t['base_id']]=gid
            sequence('motion',records,z,config,collector=label,on_after=update)
        write_json(run.path/'examples.json',examples)
        counts={k:sum((e['gain']>0 if k=='beneficial' else e['gain']<0 if k=='harmful' else e['gain']==0) for e in examples) for k in ['beneficial','harmful','neutral']}
        passed=counts['beneficial']>=20 and counts['harmful']>=20
        write_json(run.path/'SUMMARY.json',dict(counts=counts,examples=len(examples),by_sequence={s:sum(e['sequence']==s for e in examples) for s in sorted({e['sequence'] for e in examples})},gate=passed))
        write_json(run.path/'DECISION.json',dict(status='passed' if passed else 'failed',next_action='train frozen net-benefit and same-capacity target control' if passed else 'stop selector training; prioritize detection and motion observability'))
        run.meta['scientific_status']='opportunity_passed' if passed else 'opportunity_failed'
        write_json(root/'opportunity_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        print('OPPORTUNITY',counts,passed,flush=True)


def fit(root):
    setup_paths(root);ptr=read_json(root/'opportunity_pointer.json');folder=root/ptr['path']
    assert read_json(folder/'SUMMARY.json')['gate'],'Frozen opportunity gate failed'
    examples=read_json(folder/'examples.json');tr=[e for e in examples if e['sequence'] in {'pig_3','horse_4'}];ca=[e for e in examples if e['sequence'] in {'pig_5','horse_6'}]
    x=np.asarray([e['features'] for e in tr],np.float32);xx=np.asarray([e['features'] for e in ca],np.float32)
    mean=x.mean(0);std=np.maximum(x.std(0),.03)
    with Run(root,'conflict_selector_fit',dict(features=FEATURES,architecture=[len(FEATURES),32,3],steps_per_head=400,seed=20261018,train=['pig_3','horse_4'],calibration=['pig_5','horse_6'],parent=ptr,threshold_grid=[.5,.6,.7,.8,.9,.95],min_selected=10,max_harm_fraction=.1)) as run:
        models={}
        for target in ['net_benefit','correctness']:
            set_seed(20261018);model=Selector().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            labels=np.asarray([0 if e['gain']<0 else 2 if e['gain']>0 else 1 for e in tr]) if target=='net_benefit' else np.asarray([2 if e['alt_all_correct'] else 0 for e in tr])
            tx=torch.from_numpy((x-mean)/std).cuda();ty=torch.from_numpy(labels).cuda();rng=np.random.default_rng(20261018);curve=[]
            for step in range(400):
                inds=rng.integers(0,len(tr),128);loss=nn.functional.cross_entropy(model(tx[inds]),ty[inds]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();curve.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():prob=model(torch.from_numpy((xx-mean)/std).cuda()).softmax(-1)[:,2].cpu().numpy()
            grid=[]
            for th in [.5,.6,.7,.8,.9,.95]:
                selected=[e for e,p in zip(ca,prob) if p>=th];n=len(selected);harm=sum(e['gain']<0 for e in selected);gain=sum(e['gain'] for e in selected)
                grid.append(dict(threshold=th,count=n,harm=harm,net_gain=gain,eligible=n>=10 and harm/max(n,1)<=.1))
            eligible=[r for r in grid if r['eligible']];choice=max(eligible,key=lambda e:(e['net_gain'],e['threshold'])) if eligible else dict(threshold=1.1,count=0,harm=0,net_gain=0)
            ck=dict(state={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,std=std,features=FEATURES,threshold=choice['threshold'],target=target)
            torch.save(ck,run.path/(target+'.pt'));write_json(run.path/(target+'_calibration.json'),grid);write_json(run.path/(target+'_loss.json'),curve)
            models[target]=dict(checkpoint=(run.path/(target+'.pt')).relative_to(root).as_posix(),sha256=sha256(run.path/(target+'.pt')),selected=choice,parameters=sum(p.numel() for p in model.parameters()))
            print('FIT',target,models[target],flush=True)
        write_json(run.path/'SUMMARY.json',dict(training_examples=len(tr),calibration_examples=len(ca),models=models,steps=800))
        write_json(root/'selector_pointer.json',dict(run_id=run.id,models=models))


def load_selector(root,target):
    ptr=read_json(root/'selector_pointer.json');ck=torch.load(root/ptr['models'][target]['checkpoint'],map_location='cpu',weights_only=False);model=Selector();model.load_state_dict(ck['state']);model.eval()
    def predict(x):
        with torch.inference_mode():return model(torch.from_numpy((x-ck['mean'])/ck['std'])).softmax(-1)[2].item()
    return predict,ck['threshold']


def evaluate(root,split):
    setup_paths(root);config=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root);metrics=official_metrics(root)
    modes=['motion','all_appearance','margin_rule','net_benefit','correctness'];predictors={m:load_selector(root,m) for m in ['net_benefit','correctness']}
    seqs=[s for s in manifest['sequences'] if s['split']==split]
    with Run(root,'conflict_evaluate_'+split,dict(split=split,modes=modes,selector=read_json(root/'selector_pointer.json'),sequences=[s['name'] for s in seqs],GT='evaluation only',no_history_rewrite=True)) as run:
        raw={m:{} for m in modes};table=[]
        for seq in seqs:
            records,rawx=load_cache(root,seq);z=embed_array(model,ck,rawx);gt=load_gt(root,seq) if seq.get('gt') else None
            for mode in modes:
                pred,th=predictors.get(mode,(None,1.1));out,events=sequence(mode,records,z,config,pred,th)
                dest=run.path/mode/seq['name'];dest.mkdir(parents=True);write_json(dest/'tracks.json',out);write_json(dest/'decisions.json',events)
                stat=dict(sequence=seq['name'],arm=mode,groups=len(events),interventions=sum(e['selected_appearance'] for e in events),frames=seq['frames'])
                if gt is not None:
                    data=metric_data(gt,out);rs={k:m.eval_sequence(data) for k,m in metrics.items()};raw[mode][seq['name']]=rs;stat.update(compact(rs));write_json(dest/'metric_details.json',rs)
                if mode=='motion' and split in {'dev','test','demo'}:
                    v1=Path(read_json(root/'v1_reference.json')[split])/'bytetrack'/seq['name']/'tracks.json'
                    assert read_json(v1)==out,f'ByteTrack parity failed {seq["name"]}'
                table.append(stat);print('CGS_EVAL',stat,flush=True)
        combined={m:compact({k:metric.combine_sequences({s:rs[k] for s,rs in raw[m].items()}) for k,metric in metrics.items()}) for m in modes} if split!='demo' else {}
        write_json(run.path/'SUMMARY.json',dict(split=split,by_sequence=table,combined=combined))
        write_json(root/('conflict_'+split+'_pointer.json'),dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','fit','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--split',default='dev');a=p.parse_args()
    {'collect':lambda:collect(a.root),'fit':lambda:fit(a.root),'evaluate':lambda:evaluate(a.root,a.split)}[a.stage]()
