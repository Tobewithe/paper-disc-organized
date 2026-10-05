"""S045: image residual sharing and cross-fitted prototype-conditioned readout.

Existing FIT images only. Frozen heads already saw these images; cross-fitting
concerns the NEW linear maps, not end-to-end unseen-image generalization.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from rich_pixel_readout import GlobalHead,instance_features
from run_rich_pixel_readout import read_np,cuda,norm,csv_save
from readout_input_probe import sha,write_json,stable_seed
from eval_readout_input_pilot import ici
from convex_shared_readout import whiten,pcg


def pca32(train,test):
    mean=train.mean(0);sd=train.std(0).clamp_min(.001)
    a=(train-mean)/sd;b=(test-mean)/sd
    cov=a.T@a/len(a);eig,vec=torch.linalg.eigh((cov+cov.T)/2)
    transform=vec[:,-32:]/eig[-32:].clamp_min(eig[-1]*1e-4).sqrt()[None]
    return a@transform,b@transform,dict(mean=mean.cpu(),sd=sd.cpu(),transform=transform.cpu())


def add_context(hidden,context,hproj):
    # 128 hidden + 32 context + four 32-dimensional bilinear blocks + bias.
    interaction=(hproj[:,:,None]*context[:,None,:]).flatten(1)
    return torch.cat([hidden,context,interaction,hidden.new_ones((len(hidden),1))],1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    src=a.source;cfg=json.loads((src/'protocol.json').read_text());cache=Path(cfg['cache']);headsdir=Path(cfg['source']);teachers=Path(cfg['teachers'])
    out=a.out;out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fit=cfg['fit_images'];ordered=sorted(fit,key=lambda i:stable_seed('S045-image-fold',i));folds={int(i):int(k>=600) for k,i in enumerate(ordered)}
    protocol=dict(experiment='S045_PROTOTYPE_RESIDUAL_CONDITION',source=str(src.resolve()),cache=str(cache.resolve()),heads=str(headsdir.resolve()),
        fit_images=fit,image_folds=folds,seeds=[0,1,2],
        residual_sharing='S043 saved_hidden128 residual only. Fit common32D correction from OTHER targets in SAME image, '
            'evaluate target own512 logit residual. Exclude singleton images explicitly. Weighted leave-one-target-out '
            'regularized quadratic with ridge1e-4 in S043 global prototype whitening. Diagnostic uses GT of other instances, not deployable. '
            'Compare20 deterministic target permutations stratified by category x GTarea(small32^2/medium96^2/large) x instanceICI(0,<=.5,>.5). '
            'Move residual and Gram as a joint pair, preserving target slots/counts/strata. Exclude query itself even if shuffled into another slot in its pseudoimage; '
            'if this leaves zero donors use zero correction, not drop target. Report moved-across-image fraction; some singletonstrata immovable. '
            'Permutation is descriptive conditional null, not randomized causal evidence.',
        new_maps='Two fixed600/600 IMAGE folds within old1200fit. Train new shared linear map only on onefold, evaluate otherfold, reverse. '
            'Frozen S032 hidden and teachers already trained on full1200; this is conditional crossfit diagnosis, not independent-image confirmation. '
            'Target original-logit+newdelta = same S041teacherlogits, same512pixels/factor. No transfer or val queries.',
        features='Base129=hidden128+bias. All three augmented289 have hidden128+context32+4x32 bilinear blocks+bias, 9248 learned parameters. '
            'instance context32=PCA of hidden128; prototype context32=PCA of [mean32, symmetric secondmoment528] from image P over valid letterbox region. '
            'Bilinear uses four fixed random hidden projections (seed20260912), standardized on trainfold. PCA/scales learned trainfold only; '
            'prototype PCA one row per trainIMAGE, not GT. Pureinstance PCA one row per traintarget. Fixed bases differ, so equal parameters is not identical functionclass. '
            'Shuffled arm permutes true prototype contexts among images WITHIN each fold, one predetermined no-fixed-point cyclic shift per seed. '
            'PCA shared real/shuffled; no imageGT or imageID used as input.',
        solve='Same S043 FP64 matrixfreePCG,ridge1e-4,max800,true residual1e-7; all whitening fit only on currenttrainfold. '
            '24 solves:3seeds*2folds*(base,instance,prototype,shuffled). No budget/recipe changes based on results. '
            'If a solve misses convergence, report incomplete conditional comparison, do not claim feature limits.',
        metrics='Out-of-fold pertarget teacher logitMSE and512sampleIoU. Original/saved reference included; noAP. '
            'Imagecluster2000 bootstrap after seedmeans, all/high/low/middle/nonhigh. Primary prototype-minus-instance and prototype-minus-shuffled. '
            'One prespecified prototype summary/bilinear basis; negative limits this probe only, not all prototypeconditioning.',
        decision='Proceed to method prototype only if both controls beaten in heldfold teacherMSE with CI excluding zero and sampleIoU not harmed; '
            'densityspecificity requires more evidence. Otherwise stop this summaryrecipe, no tuning dimensions/regularization on thisdata.',
        hashes=dict(script=sha(__file__),source_receipt=sha(src/'COMPLETE.json'),teacher_receipt=sha(teachers/'TEACHERS_COMPLETE.json')))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    def progress(stage,**kw):
        r=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',r);print(json.dumps(r),flush=True)
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    normal=torch.load(headsdir/'normalizer.pt',map_location='cuda',weights_only=True)
    obj=json.loads((cache/'conversion_input/instances_probe.json').read_text());anns={v['id']:v for v in obj['annotations']};imganns={}
    for an in anns.values():
        if not an.get('iscrowd',0):imganns.setdefault(an['image_id'],[]).append(an)
    records=[];ids=[];meta=[];summaries=[];images=[];tri=torch.triu_indices(32,32,device='cuda')
    for number,iid in enumerate(fit,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Cache changed')
        it=read_np(path);idx=it['prediction_indices']
        if not len(idx):continue
        x=norm(instance_features(cuda(it['h']).float(),cuda(it['level']).long(),cuda(it['boxes']).float(),tuple(it['input_shape'])),normal)
        lab=read_np(headsdir/'labels'/f'{iid}.npz')
        if not np.array_equal(it['annotation_ids'],lab['annotation_ids']):raise RuntimeError('Label identity')
        pp=cuda(it['proto']).double();ih,iw=map(int,it['input_shape']);sh,sw=map(int,it['shape']);gain=min(ih/sh,iw/sw)
        rh,rw=round(sh*gain),round(sw*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        yy=(torch.arange(pp.shape[1],device='cuda')+.5)*ih/pp.shape[1];xx=(torch.arange(pp.shape[2],device='cuda')+.5)*iw/pp.shape[2]
        valid=(yy[:,None]>=top)&(yy[:,None]<top+rh)&(xx[None]>=left)&(xx[None]<left+rw)
        pv=pp[:,valid];gm=pv@pv.T/pv.shape[1];summary=torch.cat([pv.mean(1),gm[tri[0],tri[1]]]);summaries.append(summary.cpu());images.append(iid)
        records.append(dict(x=x[idx].cpu(),c=torch.tensor(it['coeff'][idx]).float(),p=torch.tensor(it['sample_p'][:,:512]).float(),
            y=torch.tensor(lab['raw_coco']).float(),factor=torch.tensor(it['loss_factor']).float()))
        for aid,j in zip(it['annotation_ids'],idx):
            an=anns[int(aid)];v=ici(an,imganns[iid]);group='high' if v>.5+1e-10 else ('low' if v<=1e-10 else 'middle')
            area=an['area'];size=0 if area<32**2 else (1 if area<96**2 else 2)
            ids.append((iid,int(aid),int(it['source_index'][j])));meta.append(dict(image_id=iid,annotation_id=int(aid),category=an['category_id'],size=size,ici=v,group=group,fold=folds[iid]))
        if number%400==0:progress('load',images=number)
    d={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]};del records;n=len(ids)
    if n!=7811:raise RuntimeError('Target count')
    image_lookup={v:i for i,v in enumerate(images)};image_index=cuda(np.array([image_lookup[v[0]] for v in ids])).long();nf=len(images)
    image_fold=cuda(np.array([folds[v] for v in images])).long();target_fold=cuda(np.array([v['fold'] for v in meta])).long();ps=torch.stack(summaries).cuda()
    rawg=[]
    for first in range(0,n,256):
        p=d['p'][first:first+256].double();rawg.append(p.transpose(1,2)@p/512)
    gram=torch.cat(rawg);weight=d['factor'].double()/d['factor'].double().mean();orig=(d['p']*d['c'][:,None]).sum(-1)
    common=torch.load(src/'common.pt',map_location='cuda',weights_only=False)
    if [tuple(v) for v in common['identities']]!=ids:raise RuntimeError('S043 ordering')
    pt=common['prototype_transform'];g=pt.T[None]@gram@pt[None]
    count=torch.bincount(image_index,minlength=nf);eligible=count[image_index]>1;I=torch.eye(32,device='cuda',dtype=torch.float64)
    # Reused null permutations preserve strata; no outcome-based matching or filtering.
    strata={}
    for k,m in enumerate(meta):strata.setdefault((m['category'],m['size'],m['group']),[]).append(k)
    perms=[];moved=[]
    for rep in range(20):
        rng=np.random.default_rng(stable_seed('S045-residual-null',rep));order=np.arange(n)
        for inds in strata.values():order[inds]=rng.permutation(inds)
        perms.append(cuda(order).long());moved.append(float((image_index[perms[-1]]!=image_index).double().mean()))
    def other_correction(g0,b0,permutation=None):
        ga=g0 if permutation is None else g0[permutation];ba=b0 if permutation is None else b0[permutation]
        sg=torch.zeros((nf,32,32),device='cuda',dtype=torch.float64).index_add_(0,image_index,ga)
        sb=torch.zeros((nf,32),device='cuda',dtype=torch.float64).index_add_(0,image_index,ba)
        ga_sum=sg[image_index]-ga;ba_sum=sb[image_index]-ba;den=count[image_index]-1
        if permutation is not None:
            inverse=torch.argsort(permutation);self_present=(image_index[inverse]==image_index)&(inverse!=torch.arange(n,device='cuda'))
            ga_sum-=g0*self_present[:,None,None];ba_sum-=b0*self_present[:,None];den=den-self_present.long()
            empty=den==0;ga_sum[empty]=0;ba_sum[empty]=0
        den=den.clamp_min(1).double()
        A=ga_sum/den[:,None,None]+1e-4*I
        B=ba_sum/den[:,None]
        return torch.linalg.solve(A,B[:,:,None]).squeeze(-1)
    rows=[];sharing=[];null=[];solvers=[];witness=[]
    for seed in range(3):
        teach=torch.load(teachers/'teachers'/f'seed{seed}.pt',map_location='cuda',weights_only=False)
        if [tuple(v) for v in teach['identities']]!=ids:raise RuntimeError('Teacher ordering')
        teacher=teach['logits'];target=(teacher-orig).double();head=GlobalHead().cuda()
        head.load_state_dict(torch.load(headsdir/f'raw_coco_s{seed}/checkpoints/epoch015.pt',map_location='cuda',weights_only=False)['model']);head.eval().requires_grad_(False)
        with torch.no_grad():hidden=head.net[:4](d['x']).double();saved=(d['p']*(d['c']+head(d['x']))[:,None]).sum(-1)
        old=torch.load(src/f'saved_hidden128_s{seed}.pt',map_location='cuda',weights_only=False)
        pred=orig.double()+(d['p'].double()*old['delta_coefficients'][:,None]).sum(-1);residual=teacher.double()-pred
        bb=[]
        for first in range(0,n,256):
            p=d['p'][first:first+256].double();rr=residual[first:first+256]
            bb.append(((p.transpose(1,2)@rr[:,:,None]).squeeze(-1)/512)@pt)
        b=torch.cat(bb);g0=g*weight[:,None,None];b0=b*weight[:,None];base=residual.square().mean(1)
        actual=other_correction(g0,b0)
        changed=base-2*(actual*b).sum(1)+torch.einsum('ni,nij,nj->n',actual,g,actual)
        for k,m in enumerate(meta):
            if bool(eligible[k]):sharing.append(dict(**m,seed=seed,before=float(base[k]),after=float(changed[k]),other_targets=int(count[image_index[k]]-1)))
        for rep,perm in enumerate(perms):
            u=other_correction(g0,b0,perm);new=base-2*(u*b).sum(1)+torch.einsum('ni,nij,nj->n',u,g,u)
            for group in ['all','high','nonhigh']:
                mask=eligible.clone()
                if group!='all':mask &= cuda(np.array([(v['group']=='high')==(group=='high') for v in meta]))
                null.append(dict(seed=seed,rep=rep,group=group,n=int(mask.sum()),before=float(base[mask].mean()),after=float(new[mask].mean()),moved_image_fraction=moved[rep]))
        progress('residual_sharing',seed=seed,targets=int(eligible.sum()))
        for mode,z in [('original',orig),('saved',saved)]:
            mse=(z.double()-teacher.double()).square().mean(1);y=d['y'].bool();bi=z>0;iou=(bi&y).sum(1)/(bi|y).sum(1).clamp_min(1)
            rows.extend(dict(**m,seed=seed,mode=mode,mse=float(mse[k]),iou=float(iou[k]),factor=float(weight[k])) for k,m in enumerate(meta))
        for fold in [0,1]:
            ti=torch.where(target_fold!=fold)[0];ei=torch.where(target_fold==fold)[0];itr=torch.where(image_fold!=fold)[0];iev=torch.where(image_fold==fold)[0]
            ww=weight[ti];ww=ww/ww.mean();cov=(gram[ti]*ww[:,None,None]).mean(0);ev,vec=torch.linalg.eigh((cov+cov.T)/2)
            transform=(vec/ev.clamp_min(ev[-1]*1e-4).sqrt()[None])@vec.T;train_g=transform.T[None]@gram[ti]@transform[None]
            bparts=[]
            for first in range(0,len(ti),256):
                ix=ti[first:first+256];pp=d['p'][ix].double();tt=target[ix]
                bparts.append(((pp.transpose(1,2)@tt[:,:,None]).squeeze(-1)/512)@transform)
            train_b=torch.cat(bparts)
            ct,ce,pcap=pca32(ps[itr],ps[iev]);cp=ps.new_empty((nf,32));cp[itr]=ct;cp[iev]=ce
            ht,he,pcah=pca32(hidden[ti],hidden[ei]);ch=hidden.new_empty((n,32));ch[ti]=ht;ch[ei]=he
            mean=hidden[ti].mean(0);sd=hidden[ti].std(0).clamp_min(.001);hh=(hidden-mean)/sd
            rand=np.random.default_rng(20260912).normal(size=(128,4))/np.sqrt(128);hp=hh@cuda(rand).double();hp=hp/hp[ti].std(0).clamp_min(.001)
            shuffle=torch.arange(nf,device='cuda')
            for ix in [itr,iev]:
                # Fixed cyclic rotation in deterministic hash order, no matching image.
                ordered_image=sorted(ix.cpu().tolist(),key=lambda j:stable_seed('S045-prototype-permutation',seed,images[j]))
                aa=cuda(np.array(ordered_image)).long();shuffle[aa]=aa.roll(1)
            if bool((shuffle==torch.arange(nf,device='cuda')).any()):raise RuntimeError('Prototype shuffle fixedpoint')
            features=dict(base=torch.cat([hidden,hidden.new_ones((n,1))],1),instance=add_context(hidden,ch,hp),
                prototype=add_context(hidden,cp[image_index],hp),shuffled=add_context(hidden,cp[shuffle[image_index]],hp))
            states=dict(pca_prototype=pcap,pca_hidden=pcah,hidden_mean=mean.cpu(),hidden_sd=sd.cpu(),random_projection=rand,
                image_ids=images,prototype_image_permutation=shuffle.cpu(),prototype_transform=transform.cpu(),seed=seed,eval_fold=fold)
            for mode,ff in features.items():
                f,ft,cond=whiten(ff[ti],ww);progress('solve',seed=seed,fold=fold,mode=mode,parameters=f.shape[1]*32)
                w,info=pcg(f,train_g,train_b,ww,lambda **kw:progress('pcg',seed=seed,fold=fold,mode=mode,**kw))
                dc=((ff[ei]@ft)@w)@transform.T;z=orig[ei].double()+(d['p'][ei].double()*dc[:,None]).sum(-1)
                mse=(z-teacher[ei].double()).square().mean(1);yy=d['y'][ei].bool();binary=z>0;iou=(binary&yy).sum(1)/(binary|yy).sum(1).clamp_min(1)
                train_dc=(f@w)@transform.T;train_z=orig[ti].double()+(d['p'][ti].double()*train_dc[:,None]).sum(-1)
                info.update(seed=seed,eval_fold=fold,mode=mode,parameters=w.numel(),training_targets=len(ti),eval_targets=len(ei),
                    train_mse=float((train_z-teacher[ti].double()).square().mean()),eval_mse=float(mse.mean()),whitening=cond)
                solvers.append(info);rows.extend(dict(**meta[int(k)],seed=seed,mode=mode,mse=float(mse[j]),iou=float(iou[j]),factor=float(weight[k])) for j,k in enumerate(ei.cpu().tolist()))
                torch.save(dict(**states,w=w.cpu(),feature_transform=ft.cpu(),mode=mode,eval_indices=ei.cpu(),delta_coefficients=dc.cpu()),out/f'{mode}_s{seed}_f{fold}.pt')
                write_json(out/'solvers.json',solvers);progress('solved',seed=seed,fold=fold,mode=mode,converged=info['converged'],eval_mse=info['eval_mse'])
        witness.append(dict(seed=seed,targets=n,identity_match=True,prototype_shuffle_no_fixed_images=True))
        csv_save(out/'crossfit.csv',rows);csv_save(out/'image_sharing.csv',sharing);csv_save(out/'permutation_null.csv',null)
    write_json(out/'witness.json',dict(seed_witness=witness,fit_images_with_targets=nf,targets=n,
        singleton_targets=int((~eligible).sum()),same_image_eligible_targets=int(eligible.sum()),residual_null_moved_image_fractions=moved,
        existing_head_scope='Alreadyfitall1200images;newlinearreadoutsalonecrossfitted. No heldouttaskperformance.'))
    progress('COMPLETE')
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,targets=n,seeds=3,solves=len(solvers),
        converged=sum(v['converged'] for v in solvers),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        p=Path(sys.argv[sys.argv.index('--out')+1])
        if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
