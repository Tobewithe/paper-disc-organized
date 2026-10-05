"""S042 pure teacher fitting and finite optimizer control. FIT ONLY, no AP.

Same7811fit instances, all3S032states, same head and512positions. This isolates
mixed-loss and optimizer explanations; it is not a deployment candidate search.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,json,shutil,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from run_rich_pixel_readout import read_np,cuda,norm,csv_save
from readout_input_probe import sha,write_json
from train_solver_response import soft_kl


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'source').mkdir();prior=a.prior;start=time.monotonic()
    prev=json.loads((prior/'protocol.json').read_text());src=Path(prev['source']);cache=Path(prev['cache']);fit=prev['fit_images']
    protocol=dict(experiment='S042_FIT_ONLY_OBJECTIVE_OPTIMIZER_ISOLATION',prior=str(prior.resolve()),fit_images=fit,seeds=[0,1,2],
        data='Same7811fitinstances first512coordinates, savedS041GTteacherresponses; no newGT/oracle/transfer/val queries.',
        network='SameGlobalHead73->128->128->32effectiveoutput andoriginalnormalizer. Onlyheadunfrozen; same threeS032states.',
        adam='PureteacherBernoulliKL weightedbyoriginalfactor, nohardloss. Same restoredS032Adam/RNG,lr1e-4,32targets,15epochs,3675updates,45checkpoints.',
        fullbatch='PureteacherKL, same initialS032head. LBFGS lr1 maxiter100 maxeval125 strongwolfe history10 tolgrad1e-7 tolchange1e-10. '
            'Full7811samples objective aggregated by512target chunks. Finite nonconvex optimization, budgetdifferent fromAdam, diagnostic not fairmethodcomparison.',
        gradient='Fullfitparametergradient ofhardBCE+Dice andteacherKL atS032start; cosine/norms/directionalderivative forteacherunderH+T. '
            'Localparametergradient only, not historicalcausalproof. No perimage gradient retention.',
        measurement='Samefit hardloss,teacherKL,MSE,sampledIoU plusresponsecapture=1-KLafter/KLinitial. '
            'Training capture not new-imagegeneralization; failuretofit withinbudget doesnot prove missinginputinformation or capacitylimit. '
            'Do not run taskAP for stronger arm or selectlambda based on fit. New fitting success only licenses later planned transfer test.',
        hashes=dict(prior_receipt=sha(prior/'COMPLETE.json'),teacher_receipt=sha(prior/'TEACHERS_COMPLETE.json'),script=sha(__file__)))
    write_json(out/'protocol.json',protocol)
    for name in ['diagnose_teacher_fitting.py','train_solver_response.py','rich_pixel_readout.py']:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    def progress(stage,**kw):
        r=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',r);print(json.dumps(r),flush=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    norm0=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True);records=[];identities=[]
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    if sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Normalizer')
    for num,iid in enumerate(fit,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Cache')
        item=read_np(path);idx=item['prediction_indices']
        if not len(idx):continue
        lab=src/'labels'/f'{iid}.npz'
        if sha(lab)!=sr[f'labels/{iid}.npz']:raise RuntimeError('Labels')
        ll=read_np(lab)
        if not np.array_equal(ll['annotation_ids'],item['annotation_ids']):raise RuntimeError('Identity')
        xx=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),cuda(item['boxes']).float(),tuple(item['input_shape'])),norm0)
        records.append(dict(x=xx[idx].cpu(),c=torch.tensor(item['coeff'][idx]).float(),p=torch.tensor(item['sample_p'][:,:512]).float(),
            y=torch.tensor(ll['raw_coco']).float(),factor=torch.tensor(item['loss_factor']).float()))
        identities.extend((iid,int(aid),int(item['source_index'][j])) for aid,j in zip(item['annotation_ids'],idx))
        if num%400==0:progress('load',images=num)
    data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]};del records;count=len(identities)
    if count!=7811:raise RuntimeError('Denominator')
    history=[];metrics=[];gradrows=[];lbinfo=[];tcr=json.loads((prior/'TEACHERS_COMPLETE.json').read_text())['hashes']
    def zfor(model,sl):return (data['p'][sl]*(data['c'][sl]+model(data['x'][sl]))[:,None]).sum(-1)
    def measure(model,teacher):
        totals=dict(hard=0.,teacher_kl=0.,teacher_mse=0.,sample_iou=0.)
        with torch.no_grad():
            for first in range(0,count,256):
                sl=slice(first,first+256);z=zfor(model,sl);y=data['y'][sl];ff=data['factor'][sl];n=len(y)
                totals['hard']+=float(loss_value(z,y,ff))*n;totals['teacher_kl']+=float(soft_kl(z,teacher[sl],ff))*n
                totals['teacher_mse']+=float((z-teacher[sl]).square().mean())*n
                totals['sample_iou']+=float((((z>0)&y.bool()).sum(1)/((z>0)|y.bool()).sum(1).clamp_min(1)).sum())
        return {k:v/count for k,v in totals.items()}
    def gradient(model,teacher,kind):
        model.zero_grad(set_to_none=True)
        for first in range(0,count,512):
            sl=slice(first,first+512);z=zfor(model,sl)
            loss=loss_value(z,data['y'][sl],data['factor'][sl]) if kind=='hard' else soft_kl(z,teacher[sl],data['factor'][sl])
            (loss*len(z)/count).backward()
        return torch.cat([p.grad.detach().flatten().double() for p in model.parameters()])
    for seed in range(3):
        tp=prior/'teachers'/f'seed{seed}.pt'
        if sha(tp)!=tcr[tp.name]:raise RuntimeError('Teacher')
        teacherfile=torch.load(tp,map_location='cuda',weights_only=False)
        if [tuple(t) for t in teacherfile['identities']]!=identities:raise RuntimeError('Teacherorder')
        teacher=teacherfile['logits'];ck=torch.load(src/f'raw_coco_s{seed}/checkpoints/epoch015.pt',map_location='cuda',weights_only=False)
        model=GlobalHead().cuda();model.load_state_dict(ck['model']);initial=measure(model,teacher);metrics.append(dict(mode='initial',seed=seed,**initial,response_capture=0.))
        gh=gradient(model,teacher,'hard');gt=gradient(model,teacher,'teacher');gn=gt+gh
        gradrows.append(dict(seed=seed,hard_norm=float(gh.norm()),teacher_norm=float(gt.norm()),cosine=float(gh@gt/(gh.norm()*gt.norm()).clamp_min(1e-30)),
            teacher_under_joint_derivative=float(-gt@gn),teacher_under_pure_derivative=float(-gt@gt)))
        model.zero_grad(set_to_none=True)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4);opt.load_state_dict(deepcopy(ck['optimizer']))
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']]);rng=np.random.default_rng();rng.bit_generator.state=deepcopy(ck['numpy_rng'])
        dest=out/f'pure_adam_s{seed}';(dest/'checkpoints').mkdir(parents=True);updates=0
        for epoch in range(15):
            order=rng.permutation(count);total=0.
            for first in range(0,count,32):
                ix=cuda(order[first:first+32]).long();z=zfor(model,ix);loss=soft_kl(z,teacher[ix],data['factor'][ix])
                if not torch.isfinite(loss):raise RuntimeError('NonfiniteAdam')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                total+=float(loss.detach())*len(ix);updates+=1
            history.append(dict(mode='pure_adam',seed=seed,epoch=epoch+1,updates=updates,loss=total/count))
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,updates=updates,seed=seed),dest/'checkpoints'/f'epoch{epoch+1:03d}.pt')
        measured=measure(model,teacher);metrics.append(dict(mode='pure_adam',seed=seed,**measured,response_capture=1-measured['teacher_kl']/initial['teacher_kl']));progress('pure_adam_done',seed=seed,**measured)
        # Stronger optimizer is a separate bounded fitting reference, never a selected method arm.
        model=GlobalHead().cuda();model.load_state_dict(ck['model']);model.train()
        opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=125,history_size=10,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
        calls=[0];values=[]
        def closure():
            opt.zero_grad(set_to_none=True);total=0.
            for first in range(0,count,512):
                sl=slice(first,first+512);z=zfor(model,sl);loss=soft_kl(z,teacher[sl],data['factor'][sl])*len(z)/count
                if not torch.isfinite(loss):raise RuntimeError('NonfiniteLBFGS')
                loss.backward();total+=float(loss.detach())
            calls[0]+=1;values.append(total)
            if calls[0]%20==0:progress('lbfgs',seed=seed,evaluations=calls[0],loss=total)
            return torch.tensor(total,device='cuda')
        opt.step(closure);gg=gradient(model,teacher,'teacher');measured=measure(model,teacher)
        state=next(iter(opt.state.values()));info=dict(seed=seed,iterations=int(state.get('n_iter',0)),evaluations=calls[0],gradient_inf=float(gg.abs().max()),
            loss_trace=values,budget_limited=int(state.get('n_iter',0))>=100 or calls[0]>=125)
        lbinfo.append(info);metrics.append(dict(mode='pure_lbfgs',seed=seed,**measured,response_capture=1-measured['teacher_kl']/initial['teacher_kl']))
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),seed=seed,fit_only=True),out/f'pure_lbfgs_s{seed}.pt')
        write_json(out/'metrics.json',metrics);write_json(out/'lbfgs.json',lbinfo);csv_save(out/'gradient.csv',gradrows);write_json(out/'history.json',history)
        progress('seed_done',seed=seed,**measured)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,fit_targets=count,seeds=3,adam_checkpoints=45,
        scope='FITONLY no transfer/no taskmetrics. Finite optimization not capacity or information lowerbound.',
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        p=Path(sys.argv[sys.argv.index('--out')+1])
        if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
