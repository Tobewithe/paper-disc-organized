"""One bounded diagnostic; no training, parameter sweep or new oracle."""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,r'D:\coco_wire\py')
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from pycocotools.coco import COCO
from ultralytics.utils import ops
import frozen_evaluation as ev
from ogps_solver import pooled_design_matrix, solve_ogps
from error_geometry import analyze_error_geometry, math_self_checks


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def key(row):return tuple(int(row[k]) for k in ('image_id','annotation_id','raw_id'))


def plain(x):
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def box_slice(box):
    # All formal masks are GPU crop_mask: x>=x1 and x<x2, likewise y.
    left=max(0,min(640,math.ceil(float(box[0])))); right=max(0,min(640,math.ceil(float(box[2]))))
    top=max(0,min(640,math.ceil(float(box[1])))); bottom=max(0,min(640,math.ceil(float(box[3]))))
    return top,max(top,bottom),left,max(left,right)


def region_stats(zT,zG,truth,neighbors):
    # Called on entire 640 map before restricting to support: morphology must not truncate at box.
    dil=F.max_pool2d(truth.float()[None,None],5,1,2)[0,0].bool()
    ero=(1-F.max_pool2d((~truth).float()[None,None],5,1,2)[0,0]).bool()
    boundary=dil & ~ero
    regions={'boundary':boundary,'interior':truth & ~boundary,
             'neighbor':neighbors & ~truth & ~boundary,
             'remaining_negative':~neighbors & ~truth & ~boundary}
    return regions


def plot_example(path,row,gt8,pred8,e,selected,truth,maskT,maskH,maskG,box):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top,bottom,left,right=box_slice(box)
    if top==bottom or left==right:return
    select=np.zeros(64);select[np.asarray(selected,dtype=int)]=1
    fig,axs=plt.subplots(2,4,figsize=(12,6))
    tiles=[(gt8,'GT soft 8x8','viridis'),(pred8,'7O probability','viridis'),
           (e,'clipped-logit error','coolwarm'),(select.reshape(8,8),'H8 selected cells','gray'),
           (truth[top:bottom,left:right],'COCO GT','gray'),
           (maskT[top:bottom,left:right],'7O -> solver','gray'),
           (maskH[top:bottom,left:right],'8 GT cells replaced','gray'),
           (maskG[top:bottom,left:right],'GT8 -> solver','gray')]
    for ax,(arr,title,cmap) in zip(axs.flat,tiles):
        if torch.is_tensor(arr):arr=arr.detach().cpu().numpy()
        ax.imshow(np.asarray(arr).squeeze(),cmap=cmap);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(f"image {row['image_id']} ann {row['annotation_id']} raw {row['raw_id']} (identity-selected example)")
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');started=time.time()
    def budget_end():
        save(out/'BUDGET_EXHAUSTED.json',dict(elapsed_seconds=time.time()-started,budget_seconds=cfg['budget_seconds'],status='incomplete',automatic_followup=False))
        print('Fixed run budget exhausted; terminating only this diagnostic process.',flush=True)
        os._exit(124)
    timer=threading.Timer(cfg['budget_seconds'],budget_end);timer.daemon=True;timer.start()
    save(out/'ENVIRONMENT.json',dict(host=platform.node(),python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0),training_updates=0,module_paths={name:__import__(name).__file__ for name in ('frozen_evaluation','ogps_solver','error_geometry')}))
    save(out/'MATH_CHECKS.json',math_self_checks())
    val=torch.load(cfg['ownership'],map_location='cpu',weights_only=False)
    pred=torch.load(cfg['predictions'],map_location='cpu',weights_only=False)
    oldrun=Path(cfg['previous_run'])
    previous=[json.loads(s) for s in (oldrun/'PER_CANDIDATE.jsonl').read_text(encoding='utf-8').splitlines()]
    prev={key(r):r for r in previous}
    oldaudit=json.loads((oldrun/'IDENTITY_DECODE_SOLVER_AUDIT.json').read_text(encoding='utf-8'))
    keys=[tuple(map(int,k)) for k in val['keys']]
    if keys!=[tuple(map(int,k)) for k in pred['keys']] or len(set(keys))!=1346 or set(keys)!=set(prev):
        raise AssertionError('Frozen identities mismatch')
    ix={k:j for j,k in enumerate(keys)};by=defaultdict(list)
    for k in keys:by[k[0]].append(k)
    example_keys=set(sorted(k for k in keys if prev[k]['box_iou']>=.75 and prev[k]['iou_A']<.75)[:6])
    save(out/'INPUT_MANIFEST.json',{p:sha(p) for p in (cfg['ownership'],cfg['predictions'],str(oldrun/'PER_CANDIDATE.jsonl'),str(oldrun/'IDENTITY_DECODE_SOLVER_AUDIT.json'),cfg['annotations'])})
    coco=COCO(cfg['annotations']);allrows=[];saved=[];audit=dict(images=0,candidates=0,max_prior_iou_error=0.,max_coefficient_relation_error=0.,max_AR_spectral_norm=0.,empty_support=0,passed=False)
    with (out/'PER_CANDIDATE.jsonl').open('w',encoding='utf-8') as stream:
        for number,iid in enumerate(sorted(by),1):
            if time.time()-started>cfg['budget_seconds']:raise TimeoutError('Fixed diagnostic budget exhausted; do not auto extend')
            p=Path(cfg['prepared'])/f'{iid:012d}.pt'
            if sha(p)!=oldaudit['prepared_sha256'][str(iid)]:raise AssertionError('Prepared cache changed')
            x=torch.load(p,map_location='cpu',weights_only=False)
            ks=[key(r) for r in x['rows']];js=[ix[k] for k in ks]
            if set(ks)!=set(by[iid]):raise AssertionError('Compact identity mismatch')
            proto=x['proto'].float().to(device);boxes=x['boxes'].float().to(device);c0=x['c0'].float().to(device)
            x['proto']=proto;x['boxes']=boxes
            A=pooled_design_matrix(proto,boxes).double()
            qp=pred['logits']['true_roi'][js].to(device).double().sigmoid()
            qg=val['label'][js].to(device).double()
            st=solve_ogps(A,c0,qp);sg=solve_ogps(A,c0,qg)
            e=st['target_logits']-sg['target_logits']
            fullP=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
            coeff={'A':c0,'T':st['coeff'].float(),'G':sg['coeff'].float()}
            store={a:[] for a in ('H8','E8','R8','EM8')};stats=[];regionrows=[]
            ann_ids=coco.getAnnIds(imgIds=[iid],iscrowd=False)
            full_gt={aid:ev._padded_gt(torch.as_tensor(coco.annToMask(coco.anns[aid]).astype(bool),device=device),x['ratio_pad'],(640,640)) for aid in ann_ids}
            union=torch.zeros((640,640),dtype=torch.bool,device=device)
            for gg in full_gt.values():union|=gg
            for i,k in enumerate(ks):
                t,b,l,r=box_slice(boxes[i]);ps=fullP[:,t:b,l:r].flatten(1).T.double();m=ps.shape[0]
                G=ps.T@ps/max(m,1)
                result=analyze_error_geometry(A[i],G,e[i],k)
                K=result['K']
                rel=(sg['coeff'][i]+K@e[i]-st['coeff'][i]).abs().max().item()
                audit['max_coefficient_relation_error']=max(audit['max_coefficient_relation_error'],rel)
                if rel>1e-8:raise AssertionError('Exact linear difference relation failed')
                for arm in store:store[arm].append(sg['coeff'][i]+K@result['arms'][arm]['e_new'])
                permutation_mean=float(np.mean(result['permutation']['full_energy']))
                geometry=dict(input_energy=result['original_input_energy'],full_mse_T=result['original_full_energy'],
                    energy_match_relative_error=result['energy_matching']['relative_mismatch'],energy_match_pass=result['energy_matching']['within_5pct'],
                    perm_mse_mean=permutation_mean,actual_to_permutation_mse_ratio=result['original_full_energy']/permutation_mean if permutation_mean>0 else None,
                    AR_spectral_norm=result['AR_spectral_norm'],n_support=m,empty_support=m==0,
                    matching=result['energy_matching'],permutation=result['permutation'],seeds=result['seeds'],numerical_audit=result['numerical_audit'])
                for arm in store:
                    geometry['input_energy_removed_'+arm]=result['arms'][arm]['removed_input_energy']
                    geometry['full_mse_'+arm]=result['arms'][arm]['full_output_energy']
                    geometry['selected_'+arm]=result['arms'][arm]['selected_idx']
                geometry['H8_greedy_trace']=result['arms']['H8']['greedy_trace']
                audit['max_AR_spectral_norm']=max(audit['max_AR_spectral_norm'],geometry['AR_spectral_norm']);audit['empty_support']+=int(m==0)
                if geometry['AR_spectral_norm']>1+1e-10:raise AssertionError('Coarse-grid contraction check failed')
                stats.append(geometry)
                truth=full_gt[k[1]];neighbors=union & ~truth
                regions=region_stats(None,None,truth,neighbors)
                zg=(ps@sg['coeff'][i]).reshape(b-t,r-l);zt=(ps@st['coeff'][i]).reshape(b-t,r-l)
                diff=zt-zg;total=float(diff.square().sum());reg={}
                for name,mask in regions.items():
                    select=mask[t:b,l:r];n=int(select.sum());yy=truth[t:b,l:r][select];aa=zt[select]>0;gg=zg[select]>0;flips=aa!=gg
                    reg[name]=dict(pixels=n,error_energy=float(diff[select].square().sum()),energy_fraction=float(diff[select].square().sum())/max(total,1e-30),signed_error_mean=float(diff[select].mean()) if n else None,flip_count=int(flips.sum()),T_wrong_G_right=int(((aa!=yy)&(gg==yy)).sum()),T_right_G_wrong=int(((aa==yy)&(gg!=yy)).sum()),flip_near_G_margin1=int((flips&(zg[select].abs()<=1)).sum()))
                regionrows.append(reg)
                if k in example_keys:
                    cs=torch.stack([st['coeff'][i],store['H8'][-1],sg['coeff'][i]]).float()
                    masks=ops.process_mask(proto,cs,boxes[i:i+1].expand(3,-1),(640,640),upsample=True)
                    plot_example(out/'figures'/f'{k[0]}_{k[1]}_{k[2]}.png',x['rows'][i],qg[i],qp[i],e[i].reshape(8,8),result['arms']['H8']['selected_idx'],truth,masks[0],masks[1],masks[2],boxes[i])
            for arm,values in store.items():coeff[arm]=torch.stack(values).float()
            rows=ev.evaluate_image(x,coeff,coco,chunk_size=4)
            for i,(row,k) in enumerate(zip(rows,ks)):
                row.update(stats[i]);row['regions']=regionrows[i];row['box_iou']=prev[k]['box_iou']
                for current,old in (('A','A'),('T','T_TRUE'),('G','G_GTsolver')):
                    err=abs(row['iou_'+current]-prev[k]['iou_'+old]);audit['max_prior_iou_error']=max(audit['max_prior_iou_error'],err)
                    if err>1e-6:raise AssertionError(f'Prior frozen decoder replay changed {k} {current}: {err}')
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=True)+'\n');allrows.append(row)
            stream.flush();saved.append(dict(keys=ks,coeff={k:v.cpu() for k,v in coeff.items()},geometry=stats))
            audit.update(images=number,candidates=len(allrows))
            if number%10==0 or number==1:print(json.dumps(dict(images=number,candidates=len(allrows),elapsed_seconds=round(time.time()-started,1))),flush=True)
    audit['passed']=True;save(out/'REPLAY_GEOMETRY_AUDIT.json',audit)
    torch.save(saved,out/'FIXED_INTERVENTIONS.pt')
    region_summary={}
    for name in ('boundary','interior','neighbor','remaining_negative'):
        rr=[r['regions'][name] for r in allrows]
        region_summary[name]={field:sum(r[field] for r in rr) for field in ('pixels','error_energy','flip_count','T_wrong_G_right','T_right_G_wrong','flip_near_G_margin1')}
        region_summary[name]['candidate_mean_energy_fraction']=float(np.mean([r['energy_fraction'] for r in rr]))
        region_summary[name]['candidate_mean_signed_error_defined_only']=float(np.mean([r['signed_error_mean'] for r in rr if r['signed_error_mean'] is not None]))
    save(out/'REGION_SUMMARY.json',dict(regions=region_summary,note='Descriptive 640-grid continuous propagation calculation. Boundary has priority; remaining negatives may include crowd/unannotated content. These are not original-image binary-mask attribution counts.'))
    from summarize_error_geometry import summarize
    summarize(allrows,out,seed=20261003,bootstrap=5000)
    save(out/'COMPLETE.json',dict(images=len(by),candidates=len(allrows),elapsed_seconds=time.time()-started,training_updates=0,automatic_followup=False))
    timer.cancel()
    print('COMPLETE; no automatic follow-up',flush=True)


if __name__=='__main__':main()
