"""S044 equal-budget disjoint-pixel teacher repeatability, fit subset only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,contextlib,io,json,shutil,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from readout_spatial_control import solve
from readout_selective_direction import full_metrics
from rich_pixel_readout import GlobalHead,instance_features
from run_rich_pixel_readout import read_np,cuda,norm,csv_save
from readout_input_probe import sha,write_json,stable_seed
from eval_readout_input_pilot import ici


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target-source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    old=json.loads((a.target_source/'protocol.json').read_text());src=Path(old['source']);cache=Path(old['cache'])
    fitset=set(json.loads((cache/'selection.json').read_text())['fit']);targets=[tuple(v) for v in old['targets'] if v[0] in fitset]
    out=a.out;out.mkdir(exist_ok=False);(out/'fits').mkdir();start=time.monotonic();grouped=defaultdict(list)
    for iid,aid in targets:grouped[iid].append(aid)
    protocol=dict(experiment='S044_TEACHER_REPEATABILITY',targets=targets,source=str(src.resolve()),cache=str(cache.resolve()),seeds=[0,1,2],
        sampling='Use64fit targets frompreselectedS039 cohort,32high32nonhigh,no failure/teacher-quality selection. '
            'Deduplicate cached2048coordinates; permute bystable_seed(S044,iid,aid); k=min(512,floor(unique/3)). '
            'A firstk,B nextk,E remainingcoords. A/B/E strictlydisjoint, A/B equalbudgets, E common. '
            'Same A/B/E all3savedheadseeds. Ifk<1 retain skipped witness, nofit; smallk reported, not silentlyremoved.',
        teachers='A/B eachfit originalrawCOCO labels with sameS04032Dquadraticregularized BCE solver. '
            'Eachnormalize P onits ownfitfoldonly, which ispartofsampledependence. No GTfeatures/candidatechoice. '
            'AllGT-assistedsameimage, repeatabilitynot deployable method or independentimagegeneralization.',
        metrics='Common E: deltaA/deltaB normalizedsquared_discrepancy=sum((dA-dB)^2)/(sum(dA^2)+sum(dB^2)), cosine, '
            'teacherbinarydisagreement, teacherIoU versus rawCOCOY. Completeoriginalcrop masks: eachIoU,coverage/neighbor/background. '
            'Denominatorzero returnsNone; no noisevariance decomposition claim, no bootstrap for independent pixels.',
        stop='Onepairedsamplingdiagnosis, no choosingbetterteacher/samplefold. Thisdoesnot prove allteachers stableorunstable. '
            'If theyvary materially, prioritize deterministicteacher target/readoutobjectiveclarity before structurehypothesis. '
            'No teacherensembletraining or transferAP inthisrun.',hashes=dict(script=sha(__file__),targetprotocol=sha(a.target_source/'protocol.json')))
    write_json(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=cr['conversion_input/instances_probe.json'] or sha(src/'normalizer.pt')!=sr['normalizer.pt']:raise RuntimeError('Inputs')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True);heads={}
    for seed in range(3):
        p=src/f'raw_coco_s{seed}/checkpoints/epoch015.pt'
        if sha(p)!=json.loads((p.parent.parent/'COMPLETE.json').read_text())['final_sha256']:raise RuntimeError('Head')
        model=GlobalHead().cuda();model.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);heads[seed]=model.eval().requires_grad_(False)
    rows=[];full=[];witness=[];fitrows=[]
    for num,(iid,aids) in enumerate(sorted(grouped.items()),1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Cache')
        item=read_np(path);shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));ih,iw=ishape
        mapping={int(aid):(k,int(j)) for k,(aid,j) in enumerate(zip(item['annotation_ids'],item['prediction_indices']))}
        ordinary=[ann for ann in gt.imgToAnns[iid] if not ann.get('iscrowd',0)];masks={ann['id']:gt.annToMask(ann).astype(bool) for ann in gt.imgToAnns[iid]}
        crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']]
            else:union|=masks[ann['id']]
        with torch.no_grad():
            c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalization)
            base={s:c+h(x) for s,h in heads.items()}
        gain=min(ih/shape[0],iw/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
        for aid in aids:
            k,j=mapping[aid];ann=gt.anns[aid];density=ici(ann,ordinary)
            unique,first=np.unique(item['sample_positions'][k],return_index=True)
            order=np.random.default_rng(stable_seed('S044',iid,aid)).permutation(len(unique));kk=min(512,len(unique)//3)
            meta=dict(image_id=iid,annotation_id=aid,high=density>.5+1e-10,ici=density)
            witness.append(dict(**meta,unique=len(unique),fit_each=kk,eval_unique=len(unique)-2*kk,skipped=kk<1))
            if kk<1:continue
            ia=order[:kk];ib=order[kk:2*kk];ie=order[2*kk:]
            assert not np.intersect1d(unique[ia],unique[ib]).size and not np.intersect1d(unique[ie],np.r_[unique[ia],unique[ib]]).size
            pp=cuda(item['sample_p'][k,first]).float();mask640=torch.zeros(ishape,device='cuda',dtype=torch.bool)
            mask640[top:top+rh,left:left+rw]=F.interpolate(cuda(masks[aid]).float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool()
            yy=mask640.flatten()[cuda(unique).long()].float();same=np.zeros(shape,bool)
            for other in ordinary:
                if other['id']!=aid and other['category_id']==ann['category_id']:same|=masks[other['id']]
            saved=dict(A=unique[ia],B=unique[ib],E=unique[ie])
            for seed in range(3):
                coeff=base[seed][j];initial=(pp*coeff).sum(-1);ee=cuda(ie).long();values={};deltas={}
                original=ops.process_mask(p,coeff[None],boxes[j:j+1],ishape,upsample=True)
                pred=(ops.scale_masks(original[:,None],shape)[0,0]>.5).cpu().numpy();originalmetrics=full_metrics(pred,masks[aid],same,union,crowd)
                for arm,ids in [('A',ia),('B',ib)]:
                    ii=cuda(ids).long();fitp=pp[ii];rms=fitp.square().mean(0).sqrt().clamp_min(.001);phi=fitp/rms
                    scale=phi.square().mean(0).sqrt().clamp_min(.001);w,info=solve(phi/scale,initial[ii],yy[ii]);dc=w/(rms*scale)
                    z=initial+(pp*dc).sum(-1);values[arm]=z[ee];deltas[arm]=z[ee]-initial[ee];info.pop('trace')
                    fitrows.append(dict(**meta,seed=seed,arm=arm,**info));saved[f'{arm}_s{seed}_dc']=dc.cpu().numpy()
                    mask=ops.process_mask(p,(coeff+dc)[None],boxes[j:j+1],ishape,upsample=True)
                    after=full_metrics((ops.scale_masks(mask[:,None],shape)[0,0]>.5).cpu().numpy(),masks[aid],same,union,crowd)
                    full.append(dict(**meta,seed=seed,arm=arm,**{k+'_before':v for k,v in originalmetrics.items()},**{k+'_after':v for k,v in after.items()}))
                da=deltas['A'].double();db=deltas['B'].double();en=float((da.square()+db.square()).sum());den=float(da.norm()*db.norm())
                def iou(z):
                    binary=z>0;y=yy[ee].bool();return float((binary&y).sum()/(binary|y).sum().clamp_min(1))
                rows.append(dict(**meta,seed=seed,fit_each=kk,eval_unique=len(ie),
                    discrepancy=float((da-db).square().sum())/en if en else None,cosine=float(da@db)/den if den else None,
                    binary_disagreement=float(((values['A']>0)!=(values['B']>0)).float().mean()),
                    original_iou=iou(initial[ee]),A_iou=iou(values['A']),B_iou=iou(values['B']),
                    A_bce=float(F.binary_cross_entropy_with_logits(values['A'],yy[ee])),B_bce=float(F.binary_cross_entropy_with_logits(values['B'],yy[ee])),
                    original_bce=float(F.binary_cross_entropy_with_logits(initial[ee],yy[ee]))))
            np.savez_compressed(out/'fits'/f'{iid}_{aid}.npz',**saved)
        if num%15==0 or num==len(grouped):
            r=dict(images=num,total=len(grouped),seconds=time.monotonic()-start);write_json(out/'progress.json',r);print(json.dumps(r),flush=True)
    for name,rr in [('pixels',rows),('full_masks',full),('witness',witness),('teacher_fit',fitrows)]:csv_save(out/f'{name}.csv',rr)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,targets=len(targets),fits=len(fitrows),
        converged=sum(r['converged'] for r in fitrows),scope='SamefitimageGTdependentteacherrepeatability,nosharedtraining/noAP.',
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        p=Path(sys.argv[sys.argv.index('--out')+1])
        if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
