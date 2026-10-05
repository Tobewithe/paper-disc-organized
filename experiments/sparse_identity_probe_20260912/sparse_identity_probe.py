"""S046 fixed-budget identity cues, GT-assisted mechanism diagnostic only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,io,json,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from readout_selective_direction import full_metrics
from rich_pixel_readout import GlobalHead,instance_features
from run_rich_pixel_readout import read_np,cuda,norm,csv_save
from readout_input_probe import sha,write_json,stable_seed
from eval_readout_input_pilot import ici

ARMS=['uniform8','balanced8','own_neighbor8','own_background8','wrong_identity8','dense_pool']


def solve_cues(x,z,y,gram):
    x=x.double();z=z.double();y=y.double();gram=gram.double();n,d=x.shape
    w=torch.zeros(d,device=x.device,dtype=x.dtype);eye=torch.eye(d,device=x.device,dtype=x.dtype);trace=[]
    def value(a):return F.binary_cross_entropy_with_logits(z+x@a,y)+.01*(a@gram@a)+.0001*a.square().sum()
    for it in range(30):
        prob=(z+x@w).sigmoid();grad=x.T@(prob-y)/n+.02*(gram@w)+.0002*w
        gi=float(grad.abs().max());v=value(w);trace.append([it,float(v),gi])
        if gi<1e-7:break
        hess=x.T@((prob*(1-prob)/n)[:,None]*x)+.02*gram+.0002*eye
        step=torch.linalg.solve(hess,grad);descent=grad@step;accepted=False
        for back in range(20):
            rate=.5**back;trial=w-rate*step
            if float(value(trial))<=float(v-1e-4*rate*descent):w=trial;accepted=True;break
        if not accepted:break
    grad=x.T@((z+x@w).sigmoid()-y)/n+.02*(gram@w)+.0002*w
    return w.float(),dict(iterations=len(trace),gradient_inf=float(grad.abs().max()),converged=float(grad.abs().max())<1e-7,
        objective_before=float(value(torch.zeros_like(w))),objective_after=float(value(w)),norm=float(w.norm()),trace=trace)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads((a.source/'protocol.json').read_text());src=Path(cfg['source']);cache=Path(cfg['cache']);targets=[tuple(v) for v in cfg['targets']]
    out=a.out;out.mkdir(exist_ok=False);(out/'fits').mkdir();start=time.monotonic();grouped=defaultdict(list)
    for iid,aid in targets:grouped[iid].append(aid)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=dict(experiment='S046_SPARSE_INSTANCE_IDENTITY_CUES',source=str(src.resolve()),cache=str(cache.resolve()),prior=str(a.source.resolve()),
        targets=targets,seeds=[0,1,2],arms=ARMS,training=False,
        selection='Same64preselectedFIT targets fromS044,32high/32nonhigh,nofailureororaclequalityfilter. Deduplicate cached2048positions; '
            'restrict to originalpredictioncrop and valid noncrowd pixels. Stablehashsplit half cuepool C and half common unused E before reading regionlabels. '
            'All compared arms require C own>=4,exclusive sameclassneighbor>=4,background>=4 and E>=16. '
            'If ineligible keep unchanged across ALLarms, explicit reason and denominator. Report all64 intent-to-probe and sharedeligible cohort separately. '
            'No adaptive expansion or dropping hardtargets. This cue-feasibility selection usesGTgeometry and is not a population claim.',
        cues='One predetermined sampling pertarget allseeds. Four ownpositive points shared by balanced8/own_neighbor8/own_background8. '
            'Four negatives are respectively uniformallnegative/exclusive sameclassneighbor/background. uniform8 draws8unconditional C points. '
            'wrong_identity8 uses EXACT own_neighbor coordinates but flips all eight labels, preserving4/4 balance; corruptioncontrol notalternativeplausiblemethod. '
            'dense_pool labelsall C points only as unequalbudget reference. Selected points not picked by currenterrors or oraclegain.',
        fitting='FrozenP,originalbox/scores/classes andsavedS032heads. Each arm solves32 coefficientcorrection: meanBCE(cuepoints) '
            '+ .01 mean_C(delta_logit^2)+.0001||w||^2. ALL arms share normalization from C prototypes and same unlabelled CGram regularizer; '
            'E never used in fitting/normalizing. RMSfloor.001. FP64Newtonmax30/gradInf1e-7/backtrack20; no tunedparameters, allresultskept.',
        metrics='Common E binaryIoU/BCE,owncoverage/neighborFPR/backgroundFPR; normal originalbox complete rawCOCO IoU/coverage/exclusive neighbor/background. '
            'Decode using officialprocess_mask; raw fixed attribution not AP/R75. OutsideC/E fullmask used onlyfinalmetrics. '
            'Three savedstate means, pairedimagebootstrap2000 pointwiseCI. Primary own_neighbor8 minusbalanced8 andown_background8, '
            'must improveIoU without owncoverage loss before considering identitycue specificity. GTrequired forcue selection/labels; not deployable.',
        stopping='One8pointrecipe no count/step sweep. If no benefit vs ordinarysamebudget cues stopthisimplementation. '
            'If benefits first test prediction-only cues underfrozen newprotocol, notclaim mechanismorinnovation directly.',
        hashes=dict(script=sha(__file__),prior=sha(a.source/'COMPLETE.json'),cache=sha(cache/'COMPLETE.json')))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(cache/'conversion_input/instances_probe.json'))
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    normal=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True);heads={}
    for seed in range(3):
        p=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt';h=GlobalHead().cuda();h.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);heads[seed]=h.eval().requires_grad_(False)
    witness=[];rows=[];fits=[]
    for number,(iid,aids) in enumerate(sorted(grouped.items()),1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Cache hash')
        item=read_np(path);shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        mapping={int(aid):(k,int(j)) for k,(aid,j) in enumerate(zip(item['annotation_ids'],item['prediction_indices']))}
        ordinary=[an for an in gt.imgToAnns[iid] if not an.get('iscrowd',0)];masks={an['id']:gt.annToMask(an).astype(bool) for an in gt.imgToAnns[iid]}
        crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for an in gt.imgToAnns[iid]:
            if an.get('iscrowd',0):crowd|=masks[an['id']]
            else:union|=masks[an['id']]
        with torch.no_grad():
            p=cuda(item['proto']).float();c=cuda(item['coeff']).float();boxes=cuda(item['boxes']).float()
            xx=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normal);bases={s:c+h(xx) for s,h in heads.items()}
        gain=min(ih/shape[0],iw/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        def resize(mask):
            mm=torch.zeros(ishape,device='cuda',dtype=torch.bool);mm[top:top+rh,left:left+rw]=F.interpolate(cuda(mask).float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool();return mm
        crowd640=resize(crowd);union640=resize(union)
        for aid in aids:
            k,j=mapping[aid];an=gt.anns[aid];v=ici(an,ordinary);group='high' if v>.5+1e-10 else 'nonhigh'
            own=masks[aid];same=np.zeros(shape,bool)
            for other in ordinary:
                if other['id']!=aid and other['category_id']==an['category_id']:same|=masks[other['id']]
            own640=resize(own);same640=resize(same)&~own640&~crowd640;bg640=~union640&~crowd640
            unique,first=np.unique(item['sample_positions'][k],return_index=True);pos=cuda(unique).long();box=boxes[j]
            valid=(pos%iw>=box[0])&(pos%iw<box[2])&(pos//iw>=box[1])&(pos//iw<box[3])&~crowd640.flatten()[pos]
            allowed=torch.where(valid)[0].cpu().numpy();rng=np.random.default_rng(stable_seed('S046',iid,aid));allowed=rng.permutation(allowed)
            ci=allowed[:len(allowed)//2];ei=allowed[len(allowed)//2:];yc=own640.flatten()[pos].float()
            labels=yc.cpu().numpy();sn=same640.flatten()[pos].cpu().numpy();bg=bg640.flatten()[pos].cpu().numpy()
            positives=ci[labels[ci]>0];neighbors=ci[sn[ci]];backgrounds=ci[bg[ci]];negatives=ci[labels[ci]==0]
            eligible=len(positives)>=4 and len(neighbors)>=4 and len(backgrounds)>=4 and len(ei)>=16
            reasons=[name for name,ok in [('own4',len(positives)>=4),('neighbor4',len(neighbors)>=4),('background4',len(backgrounds)>=4),('eval16',len(ei)>=16)] if not ok]
            meta=dict(image_id=iid,annotation_id=aid,group=group,ici=v,eligible=eligible)
            witness.append(dict(**meta,total_unique=len(unique),in_crop_valid=len(allowed),cuepool=len(ci),evalpool=len(ei),own_available=len(positives),
                neighbor_available=len(neighbors),background_available=len(backgrounds),reason=','.join(reasons)))
            selected={};archive=dict(C=unique[ci],E=unique[ei],eligible=np.array(eligible))
            if eligible:
                pp=rng.choice(positives,4,replace=False);nn=rng.choice(neighbors,4,replace=False)
                selected=dict(uniform8=rng.choice(ci,8,replace=False),balanced8=np.r_[pp,rng.choice(negatives,4,replace=False)],
                    own_neighbor8=np.r_[pp,nn],own_background8=np.r_[pp,rng.choice(backgrounds,4,replace=False)],wrong_identity8=np.r_[pp,nn],dense_pool=ci)
                for arm,inds in selected.items():
                    if np.intersect1d(unique[inds],unique[ei]).size:raise RuntimeError('Cue/E intersection')
                    archive[arm+'_coords']=unique[inds];archive[arm+'_labels']=(1-labels[inds]) if arm=='wrong_identity8' else labels[inds]
            pp=cuda(item['sample_p'][k,first]).float();C=cuda(ci).long();E=cuda(ei).long()
            for seed in range(3):
                coeff=bases[seed][j];basez=(pp*coeff).sum(-1)
                for arm in ['baseline']+ARMS:
                    dc=torch.zeros(32,device='cuda');info=None
                    if eligible and arm!='baseline':
                        rms=pp[C].square().mean(0).sqrt().clamp_min(.001);allx=pp/rms;gram=allx[C].double().T@allx[C].double()/len(C)
                        inds=cuda(selected[arm]).long();ys=yc[inds] if arm!='wrong_identity8' else 1-yc[inds]
                        w,info=solve_cues(allx[inds],basez[inds],ys,gram);dc=w/rms;trace=info.pop('trace');archive[f'{arm}_s{seed}_trace']=np.array(trace)
                        fits.append(dict(**meta,seed=seed,arm=arm,prompts=len(inds),**info));archive[f'{arm}_s{seed}_dc']=dc.cpu().numpy()
                    z=basez+(pp*dc).sum(-1);pred=ops.process_mask(p,(coeff+dc)[None],boxes[j:j+1],ishape,upsample=True)
                    metrics=full_metrics((ops.scale_masks(pred[:,None],shape)[0,0]>.5).cpu().numpy(),own,same,union,crowd)
                    ez=z[E];ey=yc[E].bool();ep=ez>0;den=(ep|ey).sum()
                    em=dict(E_iou=float((ep&ey).sum()/den) if int(den) else None,E_bce=float(F.binary_cross_entropy_with_logits(ez,yc[E])) if len(E) else None)
                    for name,mask in [('own',ey),('neighbor',cuda(sn)[E]),('background',cuda(bg)[E])]:
                        em[f'E_{name}_n']=int(mask.sum());em[f'E_{name}_positive']=float(ep[mask].float().mean()) if bool(mask.any()) else None
                    rows.append(dict(**meta,seed=seed,arm=arm,**metrics,**em))
            np.savez_compressed(out/'fits'/f'{iid}_{aid}.npz',**archive)
        if number%15==0 or number==len(grouped):
            row=dict(images=number,total=len(grouped),seconds=time.monotonic()-start);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    for name,value in [('witness',witness),('metrics',rows),('solvers',fits)]:csv_save(out/f'{name}.csv',value)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,targets=len(targets),eligible=sum(v['eligible'] for v in witness),
        fits=len(fits),converged=sum(v['converged'] for v in fits),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        p=Path(sys.argv[sys.argv.index('--out')+1])
        if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
