"""Training-only counterfactual rollout; deployment reads no future frames or GT."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from .common import Run,read_json,write_json,setup_paths,load_gt,sha256
from .learn import load_identity,load_cache,embed_array,match_gt,set_seed
from .tracking import frame_input
from .conflict_selector import build_tracker,sequence,Selector,FEATURES,load_selector
from .evaluate import official_metrics,metric_data,compact


def update(tracker,rec,z):
    boxes,features=frame_input(rec,z)
    return tracker.update(boxes,img=None,feats=features)


def collect(root):
    setup_paths(root)
    from ultralytics.trackers.basetrack import BaseTrack
    cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root)
    with Run(root,'trajectory_opportunity',dict(horizon=12,GT='offline labels only',force='one conflict group at current frame',following_policy='motion',sign_disagreement_gate=.1)) as run:
        examples=[];parity=0
        for seq in [s for s in manifest['sequences'] if s['split']=='train']:
            records,raw=load_cache(root,seq);z=embed_array(model,ck,raw);mapping=match_gt(records,load_gt(root,seq))
            reference,_=sequence('motion',records,z,cfg)
            tracker=build_tracker('motion',cfg['tracker']);last_gt={}
            for idx,rec in enumerate(records):
                before=deepcopy(tracker);prior_count=BaseTrack._count;anchors=last_gt.copy()
                rows=update(tracker,rec,z);actual_count=BaseTrack._count;events=tracker.events;tracker.events=[]
                expected=[(t['identity'],t['det_id']) for t in reference[idx]['tracks']]
                assert [(int(r[4]),int(r[-1])) for r in rows]==expected,'main trajectory parity'
                valid=[e for e in events if idx+12<=len(records) and all(t in anchors and d in mapping[rec['frame']] for t,d in e['base']+e['alternative'])]
                for e in valid:
                    ids=sorted({t for t,d in e['base']+e['alternative']});rewards=[]
                    for arm in ['base','alternative']:
                        branch=deepcopy(before);BaseTrack._count=prior_count
                        if arm=='alternative':branch.force_group=dict(frame=rec['frame'],base=e['base'])
                        curve=[]
                        for offset in range(12):
                            rr=update(branch,records[idx+offset],z);frame=records[idx+offset]['frame']
                            if arm=='base':
                                want=[(t['identity'],t['det_id']) for t in reference[idx+offset]['tracks']]
                                assert [(int(r[4]),int(r[-1])) for r in rr]==want,'counterfactual no-op parity'
                                parity+=1
                            correct=sum(int(r[4]) in ids and mapping[frame].get(int(r[-1]))==anchors[int(r[4])] for r in rr)
                            curve.append(int(correct));branch.events=[]
                        rewards.append(curve)
                    BaseTrack._count=actual_count
                    immediate=rewards[1][0]-rewards[0][0];future=sum(rewards[1])-sum(rewards[0])
                    examples.append(dict(e,sequence=seq['name'],gain=future,immediate_gain=immediate,base_curve=rewards[0],alternative_curve=rewards[1]))
                BaseTrack._count=actual_count
                for r in rows:
                    gid=mapping[rec['frame']].get(int(r[-1]))
                    if gid is not None:last_gt[int(r[4])]=gid
                if (idx+1)%100==0:print('ROLLOUT',seq['name'],idx+1,'examples',len(examples),flush=True)
        counts=dict(beneficial=sum(e['gain']>0 for e in examples),harmful=sum(e['gain']<0 for e in examples),neutral=sum(e['gain']==0 for e in examples))
        disagreement=sum(np.sign(e['gain'])!=np.sign(e['immediate_gain']) for e in examples)
        gate=counts['beneficial']>=20 and counts['harmful']>=20 and disagreement/max(len(examples),1)>=.1
        summary=dict(examples=len(examples),counts=counts,sign_disagreement=int(disagreement),disagreement_fraction=disagreement/max(len(examples),1),no_op_replay_frames=parity,gate=bool(gate))
        write_json(run.path/'examples.json',examples);write_json(run.path/'SUMMARY.json',summary)
        write_json(run.path/'DECISION.json',dict(status='passed' if gate else 'failed',next_action='fit frozen trajectory head' if gate else 'close trajectory target; prioritize cross-scene observability'))
        write_json(root/'trajectory_opportunity_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        print('TRAJECTORY_OPPORTUNITY',summary,flush=True)


def fit(root):
    setup_paths(root);ptr=read_json(root/'trajectory_opportunity_pointer.json');folder=root/ptr['path']
    assert read_json(folder/'SUMMARY.json')['gate'],'Frozen temporal opportunity gate failed'
    es=read_json(folder/'examples.json');tr=[e for e in es if e['sequence'] in {'pig_3','horse_4'}];ca=[e for e in es if e['sequence'] in {'pig_5','horse_6'}]
    oldptr=read_json(root/'selector_pointer.json');oldck=torch.load(root/oldptr['models']['net_benefit']['checkpoint'],map_location='cpu',weights_only=False)
    mean,std=oldck['mean'],oldck['std']
    with Run(root,'trajectory_selector_fit',dict(features=FEATURES,architecture=[21,32,3],steps=400,seed=20261018,parent=ptr,threshold_grid=[.5,.6,.7,.8,.9,.95],min_selected=10,max_harm_fraction=.1)) as run:
        set_seed(20261018);model=Selector().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        tx=torch.from_numpy((np.asarray([e['features'] for e in tr],np.float32)-mean)/std).cuda()
        ty=torch.tensor([0 if e['gain']<0 else 2 if e['gain']>0 else 1 for e in tr],device='cuda')
        rng=np.random.default_rng(20261018);curve=[]
        for step in range(400):
            ids=rng.integers(0,len(tr),128);loss=torch.nn.functional.cross_entropy(model(tx[ids]),ty[ids]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();curve.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():prob=model(torch.from_numpy((np.asarray([e['features'] for e in ca],np.float32)-mean)/std).cuda()).softmax(-1)[:,2].cpu().numpy()
        grid=[]
        for th in [.5,.6,.7,.8,.9,.95]:
            selected=[e for e,p in zip(ca,prob) if p>=th];n=len(selected);harm=sum(e['gain']<0 for e in selected)
            grid.append(dict(threshold=th,count=n,harm=harm,net_gain=sum(e['gain'] for e in selected),eligible=n>=10 and harm/max(n,1)<=.1))
        eligible=[r for r in grid if r['eligible']];choice=max(eligible,key=lambda e:(e['net_gain'],e['threshold'])) if eligible else dict(threshold=1.1,count=0,harm=0,net_gain=0)
        ck=dict(state={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,std=std,features=FEATURES,threshold=choice['threshold'],target='trajectory_benefit')
        torch.save(ck,run.path/'trajectory.pt');write_json(run.path/'loss.json',curve);write_json(run.path/'calibration.json',grid)
        modelmeta=dict(checkpoint=(run.path/'trajectory.pt').relative_to(root).as_posix(),sha256=sha256(run.path/'trajectory.pt'),selected=choice,parameters=803)
        write_json(run.path/'SUMMARY.json',dict(model=modelmeta,training_examples=len(tr),calibration_examples=len(ca),steps=400))
        write_json(root/'trajectory_selector_pointer.json',dict(run_id=run.id,model=modelmeta));print('TRAJECTORY_FIT',modelmeta,flush=True)


def load_trajectory(root):
    ptr=read_json(root/'trajectory_selector_pointer.json');ck=torch.load(root/ptr['model']['checkpoint'],map_location='cpu',weights_only=False);model=Selector();model.load_state_dict(ck['state']);model.eval()
    def predict(x):
        with torch.inference_mode():return model(torch.from_numpy((x-ck['mean'])/ck['std'])).softmax(-1)[2].item()
    return predict,ck['threshold']


def evaluate(root,split):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root);metrics=official_metrics(root)
    modes=['trajectory'] if split!='transfer' else ['motion','all_appearance','margin_rule','net_benefit','correctness','trajectory']
    predictors={m:load_selector(root,m) for m in ['net_benefit','correctness']};predictors['trajectory']=load_trajectory(root)
    with Run(root,'trajectory_evaluate_'+split,dict(split=split,modes=modes,selector=read_json(root/'trajectory_selector_pointer.json'),GT='evaluation only',future_input=False)) as run:
        raw={m:{} for m in modes};table=[]
        for seq in [s for s in manifest['sequences'] if s['split']==split]:
            records,rawx=load_cache(root,seq);z=embed_array(model,ck,rawx);gt=load_gt(root,seq) if seq.get('gt') else None
            for mode in modes:
                pred,th=predictors.get(mode,(None,1.1));out,events=sequence('net_benefit' if mode=='trajectory' else mode,records,z,cfg,pred,th)
                dest=run.path/mode/seq['name'];dest.mkdir(parents=True);write_json(dest/'tracks.json',out);write_json(dest/'decisions.json',events)
                stat=dict(sequence=seq['name'],arm=mode,groups=len(events),interventions=sum(e['selected_appearance'] for e in events),frames=seq['frames'])
                if gt is not None:
                    data=metric_data(gt,out);rs={k:m.eval_sequence(data) for k,m in metrics.items()};raw[mode][seq['name']]=rs;stat.update(compact(rs));write_json(dest/'metric_details.json',rs)
                table.append(stat);print('TRAJECTORY_EVAL',stat,flush=True)
        combined={m:compact({k:metric.combine_sequences({s:rs[k] for s,rs in raw[m].items()}) for k,metric in metrics.items()}) for m in modes} if split!='demo' else {}
        write_json(run.path/'SUMMARY.json',dict(split=split,by_sequence=table,combined=combined))
        write_json(root/('trajectory_'+split+'_pointer.json'),dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','fit','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--split',default='dev');a=p.parse_args()
    {'collect':lambda:collect(a.root),'fit':lambda:fit(a.root),'evaluate':lambda:evaluate(a.root,a.split)}[a.stage]()
