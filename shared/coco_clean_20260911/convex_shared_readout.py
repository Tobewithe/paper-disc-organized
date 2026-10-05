"""S043 fixed-representation convex response fitting, FIT ONLY.

Regularized quadratic in the shared final-layer matrix. Matrix-free PCG,
FP64, true residual and strong-convexity objective-gap bound recorded.
No model selection, no transfer/validation input, no claim of feature sufficiency.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from run_rich_pixel_readout import read_np,cuda,norm,csv_save
from readout_input_probe import sha,write_json
from train_solver_response import soft_kl

MODES=['constant32','input73','saved_hidden128','adapted_hidden128']


def whiten(x,weight):
    cov=x.T@(x*weight[:,None])/len(x)
    eig,vec=torch.linalg.eigh((cov+cov.T)/2)
    floor=eig[-1].clamp_min(1e-12)*1e-4
    transform=(vec/eig.clamp_min(floor).sqrt()[None])@vec.T
    return x@transform,transform,dict(dim=x.shape[1],eig_min=float(eig[0]),eig_max=float(eig[-1]),
        eig_floor=float(floor),floored=int((eig<floor).sum()))


def pcg(f,g,b,weight,progress):
    n=len(f);ridge=1e-4
    def mat(w):
        u=f@w;v=torch.bmm(g,u[:,:,None]).squeeze(-1)*weight[:,None]
        return f.T@v/n+ridge*w
    rhs=f.T@(b*weight[:,None])/n
    diag=f.square().T@(torch.diagonal(g,dim1=1,dim2=2)*weight[:,None])/n+ridge
    w=torch.zeros_like(rhs);res=rhs.clone();z=res/diag;direction=z.clone();rz=(res*z).sum()
    normb=rhs.norm().clamp_min(1e-30);trace=[];converged=False;begin=time.monotonic()
    for iteration in range(1,801):
        hd=mat(direction);den=(direction*hd).sum()
        if not torch.isfinite(den) or float(den)<=0:raise RuntimeError('NonSPD PCG operator')
        alpha=rz/den;w+=alpha*direction;res-=alpha*hd
        recurrent=float(res.norm()/normb)
        if iteration%25==0 or recurrent<1e-7 or iteration==800:
            true=rhs-mat(w);relative=float(true.norm()/normb)
            trace.append(dict(iteration=iteration,relative_residual=relative,seconds=time.monotonic()-begin))
            if iteration%100==0:progress(iteration=iteration,relative_residual=relative)
            if relative<=1e-7:converged=True;break
            # Replacing the residual restarts PCG; avoids accumulation drift.
            if iteration%100==0:
                res=true;z=res/diag;direction=z.clone();rz=(res*z).sum();continue
        z=res/diag;next_rz=(res*z).sum();direction=z+(next_rz/rz)*direction;rz=next_rz
    grad=mat(w)-rhs;gap=float(grad.square().sum()/(2*ridge))
    return w,dict(iterations=iteration,converged=converged,relative_residual=float(grad.norm()/normb),
        gradient_norm=float(grad.norm()),gradient_inf=float(grad.abs().max()),ridge=ridge,
        objective_gap_upper_bound=gap,solution_norm=float(w.norm()),seconds=time.monotonic()-begin,trace=trace)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teachers',type=Path,required=True);ap.add_argument('--adapted',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    prior=a.teachers;adapted=a.adapted;out=a.out;out.mkdir(exist_ok=False);(out/'source').mkdir();start=time.monotonic()
    cfg=json.loads((prior/'protocol.json').read_text());src=Path(cfg['source']);cache=Path(cfg['cache']);fit=cfg['fit_images']
    protocol=dict(experiment='S043_CONVEX_FIXED_REPRESENTATION',teachers=str(prior.resolve()),adapted=str(adapted.resolve()),
        source=str(src.resolve()),cache=str(cache.resolve()),fit_images=fit,seeds=[0,1,2],modes=MODES,
        freeze='Same7811fitinstances/512originalpositions/rawGT, originalpredictioncoeff+sharedcorrection. '
            'Input73 is existing normalizedh/level/box; hidden128 is aftersecondSiLU in S032 or S042pureLBFGShead. '
            'All hiddenweights/prototypes/boxes frozen; only sharedlinear finalmatrix solved. No transfer/val queries.',
        objective='0.5 * mean_i[(factor_i/mean_factor) * mean_pixels((original_logit + f_i W Pwhite - teacher_logit)^2)] '
            '+ 0.5e-4*||W||_F^2. Responses are logits, not S041/S042probabilityKL. No coefficient Euclidean target.',
        conditioning='P and appended-bias features whitened using fullfit weightedsecondmoments; eigenfloor=1e-4*maxeig. '
            'Invertible basis change; ridge geometry depends on whitening. Samefunctionclass for eachfixedfeature, '
            'but no claim ridge-independent representationalupperbound. Constantarm same Pwhitening.',
        solve='FP64matrixfreepreconditionedCG;diagpreconditioner,max800,tolerance1e-7 relative TRUE normal-equation residual. '
            'Checktrue residual every25 andatcandidateconvergence;restartresidualevery100. '
            'Final objectivegap bound ||gradient||^2/(2ridge). No tolerance relaxation or result-based iteration extension.',
        comparisons='Same matched teacher perseed; constant32 vs input73 vs savedhidden128 vs adaptedhidden128. '
            'Report raw/weighted teacherMSE,teacherKL,fit512IoU,hardloss and convergence. '
            'Original/S032/S042heads remeasured with same tensor order. Final-layer metric may worsen KL despite quadratic convergence.',
        interpretation='FITONLY conditional regularized projection, not AP/Recall,not missing-information theorem,not novelty. '
            'If convergedyet residuallarge, only limits thisfixedrepresentation/linearmap/ridge. '
            'Fullnetwork capacity/nonlinear learnability/generalization not settled. No new largerhead or iteration search.',
        hashes=dict(script=sha(__file__),teacher_receipt=sha(prior/'TEACHERS_COMPLETE.json'),adapted_receipt=sha(adapted/'COMPLETE.json')))
    write_json(out/'protocol.json',protocol)
    for name in ['convex_shared_readout.py','rich_pixel_readout.py','train_solver_response.py']:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    ar={k.replace('\\','/'):v for k,v in json.loads((adapted/'COMPLETE.json').read_text())['hashes'].items()}
    tr=json.loads((prior/'TEACHERS_COMPLETE.json').read_text())['hashes']
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    if sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Normalizer')
    records=[];identities=[]
    for num,iid in enumerate(fit,1):
        p=cache/'images'/f'{iid}.npz'
        if sha(p)!=cr[f'images/{iid}.npz']:raise RuntimeError('Fitcache')
        item=read_np(p);idx=item['prediction_indices']
        if not len(idx):continue
        lab=src/'labels'/f'{iid}.npz'
        if sha(lab)!=sr[f'labels/{iid}.npz']:raise RuntimeError('Labels')
        label=read_np(lab)
        if not np.array_equal(label['annotation_ids'],item['annotation_ids']):raise RuntimeError('Annotation identity')
        x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),cuda(item['boxes']).float(),tuple(item['input_shape'])),normalization)
        records.append(dict(x=x[idx].cpu(),c=torch.tensor(item['coeff'][idx]).float(),p=torch.tensor(item['sample_p'][:,:512]).float(),
            y=torch.tensor(label['raw_coco']).float(),factor=torch.tensor(item['loss_factor']).float()))
        identities.extend((iid,int(aid),int(item['source_index'][j])) for aid,j in zip(item['annotation_ids'],idx))
        if num%400==0:progress('load',images=num)
    data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]};del records;count=len(identities)
    if count!=7811:raise RuntimeError('Target denominator')
    factor=data['factor'].double();weight=factor/factor.mean();orig=(data['p']*data['c'][:,None]).sum(-1)
    grams=[]
    for first in range(0,count,256):
        pp=data['p'][first:first+256].double();grams.append(pp.transpose(1,2)@pp/512)
    gram=torch.cat(grams);cov=(gram*weight[:,None,None]).mean(0);eig,vec=torch.linalg.eigh((cov+cov.T)/2)
    floor=eig[-1]*1e-4;pt=(vec/eig.clamp_min(floor).sqrt()[None])@vec.T
    g=pt.T[None]@gram@pt[None];conditioning=dict(prototype=dict(eig_min=float(eig[0]),eig_max=float(eig[-1]),floor=float(floor),floored=int((eig<floor).sum())))
    del gram
    torch.save(dict(identities=identities,prototype_transform=pt.cpu(),normalized_factors=weight.cpu()),out/'common.pt')
    metrics=[];solvers=[];replays=[]
    def measure(z,teacher):
        mse=(z.double()-teacher.double()).square().mean(1)
        return dict(teacher_mse=float(mse.mean()),weighted_teacher_mse=float((mse*weight).mean()),
            teacher_kl=float(soft_kl(z,teacher,data['factor'])),hard=float(loss_value(z,data['y'],data['factor'])),
            sample_iou=float((((z>0)&data['y'].bool()).sum(1)/((z>0)|data['y'].bool()).sum(1).clamp_min(1)).mean()))
    for seed in range(3):
        tp=prior/'teachers'/f'seed{seed}.pt'
        if sha(tp)!=tr[tp.name]:raise RuntimeError('Teacher hash')
        teach=torch.load(tp,map_location='cuda',weights_only=False)
        if [tuple(v) for v in teach['identities']]!=identities:raise RuntimeError('Teacher identities')
        teacher=teach['logits'];target=teacher.double()-orig.double();bparts=[]
        for first in range(0,count,256):
            pp=data['p'][first:first+256].double();tt=target[first:first+256]
            bparts.append(((pp.transpose(1,2)@tt[:,:,None]).squeeze(-1)/512)@pt)
        bvec=torch.cat(bparts);representations={'constant32':data['x'].new_ones((count,1)),'input73':data['x']}
        metrics.append(dict(seed=seed,mode='original',**measure(orig,teacher)))
        for typ in ['saved','adapted']:
            path=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt' if typ=='saved' else adapted/f'pure_lbfgs_s{seed}.pt'
            expected=json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256'] if typ=='saved' else ar[path.name]
            if sha(path)!=expected:raise RuntimeError('Fixedhead hash')
            model=GlobalHead().cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);model.eval().requires_grad_(False)
            with torch.no_grad():
                hf=model.net[:4](data['x']);dc=model(data['x']);zz=(data['p']*(data['c']+dc)[:,None]).sum(-1)
                final=model.net[-1];we=final.weight.reshape(4,32,128).mean(0);be=final.bias.reshape(4,32).mean(0)
                error=float((hf@we.T+be-dc).abs().max())
            if error>1e-5:raise RuntimeError('Effective final layer replay')
            representations[typ+'_hidden128']=hf;metrics.append(dict(seed=seed,mode=typ+'_head',**measure(zz,teacher)))
            replays.append(dict(seed=seed,mode=typ,coefficient_replay_error=error))
        for mode in MODES:
            f0=representations[mode].double()
            if mode!='constant32':f0=torch.cat([f0,torch.ones((count,1),device='cuda',dtype=torch.float64)],1)
            f,ft,cond=whiten(f0,weight);conditioning[f'{mode}_s{seed}']=cond
            progress('solve_start',seed=seed,mode=mode,parameters=f.shape[1]*32)
            w,info=pcg(f,g,bvec,weight,lambda **kw:progress('pcg',seed=seed,mode=mode,**kw))
            dc=(f@w)@pt.T;zz=orig.double()+(data['p'].double()*dc[:,None]).sum(-1)
            # Common metric reductions use float32 like previous scripts; quadratic objective is FP64.
            measured=measure(zz.float(),teacher);fitmse=((zz-teacher.double()).square().mean(1)*weight).mean()
            info.update(seed=seed,mode=mode,parameters=w.numel(),objective=float(.5*fitmse+.5e-4*w.square().sum()),
                fit_weighted_mse_fp64=float(fitmse),objective_zero=float(.5*(target.square().mean(1)*weight).mean()))
            if info['objective']>info['objective_zero']+1e-10:raise RuntimeError('Quadratic fit worse than zero')
            metrics.append(dict(seed=seed,mode=mode,**measured));solvers.append(info)
            torch.save(dict(w=w.cpu(),feature_transform=ft.cpu(),prototype_transform=pt.cpu(),
                delta_coefficients=dc.cpu(),seed=seed,mode=mode,fit_only=True,identities=identities),out/f'{mode}_s{seed}.pt')
            write_json(out/'metrics.json',metrics);write_json(out/'solvers.json',solvers);write_json(out/'conditioning.json',conditioning);write_json(out/'replay.json',replays)
            progress('solve_complete',seed=seed,mode=mode,iterations=info['iterations'],converged=info['converged'],relative_residual=info['relative_residual'],**measured)
    progress('COMPLETE')
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,targets=count,seeds=3,solves=len(solvers),
        converged=sum(s['converged'] for s in solvers),scope='Fit-only, no AP/newimageevaluation.',
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        p=Path(sys.argv[sys.argv.index('--out')+1])
        if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
