"""One bounded frozen-cache experiment: point vs predicted-box residual."""
from __future__ import annotations
import argparse, hashlib, json, platform, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, r'D:\coco_wire\py')
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import v8SegmentationLoss
from pycocotools.coco import COCO
from bgcr_models import ResidualReadout, extract_box_features
from frozen_evaluation import evaluate_image


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def load(p): return torch.load(p, map_location='cpu', weights_only=False)
def key(r): return (int(r['image_id']),int(r['annotation_id']),int(r['raw_id']))
def onto(x): return {k:v.cuda() if torch.is_tensor(v) else v for k,v in x.items()}
def log(**x): print(json.dumps(x,ensure_ascii=False),flush=True)


def loss_sum(x,c):
    # The existing official reference, without a teacher or extra regularizer.
    proto=F.interpolate(x['proto'].float()[None],x['masks'].shape[-2:],mode='bilinear',align_corners=False)[0]
    gt=(x['masks'][None]==(x['owners']+1)[:,None,None]).float()
    area=((x['target_boxes'][:,2:]-x['target_boxes'][:,:2])/640).prod(1)
    return v8SegmentationLoss.single_mask_loss(gt,c,proto,x['target_boxes'],area)*float(x['segmentation_gain'])


def normfit(features,levels):
    mu=torch.zeros(3,64);sd=torch.ones(3,64)
    counts=[]
    for lev in range(3):
        f=features[levels==lev].double();counts.append(len(f))
        if len(f):
            mu[lev]=f.mean(0).float();sd[lev]=f.std(0,unbiased=False).clamp_min(.01).float()
    return dict(mean=mu,std=sd,counts=counts)


def normalized(f, levels, norm):
    return (f-norm['mean'].to(f.device)[levels])/norm['std'].to(f.device)[levels]


def prepare(cfg,out):
    cache=Path(cfg['cache']); prep=Path(cfg['prepared'])
    if sha(cache/'INDEX.json')!=cfg['index_sha256']: raise AssertionError('Official identity index changed')
    if sha(cfg['weights'])!=cfg['weight_sha256']: raise AssertionError('Original weight changed')
    index=json.loads((cache/'INDEX.json').read_text(encoding='utf-8-sig'))
    counts={s:[len(v),sum(int(r['n']) for r in v)] for s,v in index.items()}
    assert counts=={'fit':[796,6058],'dev':[197,1402],'val':[196,1346]},counts
    model=YOLO(cfg['weights']).model.eval().requires_grad_(False)
    branches=model.model[-1].one2one_cv4
    audit=dict(counts=counts,h_error=0.,coefficient_error=0.,native_readout_error=0.,
               roi_audits=[],files={},native_shapes=[list(b[-1].weight.shape) for b in branches])
    features={}; identity=[];t=time.monotonic();image_num=0
    for split,items in index.items():
        for item in items:
            iid=int(item['image_id']);p=prep/f'{iid:012d}.pt'; source=cache/'images'/p.name
            x=load(p);raw=load(source)
            ids=torch.tensor([r['raw_id'] for r in x['rows']],dtype=torch.long)
            assert [key(r) for r in x['rows']]==[key(r) for r in raw['rows']]
            assert len(ids)==int(item['n'])
            for name,compact,full in [('h',x['h'],raw['h'][ids]),('coefficient',x['c0'],raw['coeff'][ids])]:
                err=float((compact-full).abs().max());audit[name+'_error']=max(audit[name+'_error'],err)
                torch.testing.assert_close(compact,full,atol=0,rtol=0)
            torch.testing.assert_close(x['boxes'],raw['boxes'][ids],atol=0,rtol=0)
            torch.testing.assert_close(x['levels'],raw['levels'][ids],atol=0,rtol=0)
            for i,row in enumerate(x['rows']):
                assert int(raw['all_annotation_ids'][int(row['gt_index'])])==int(row['annotation_id'])
                identity.append(dict(split=split,branch='one2one',**{k:int(row[k]) for k in ('image_id','annotation_id','raw_id','level','gt_index')}))
            for lev,b in enumerate(branches):
                take=x['levels']==lev
                if take.any():
                    c=F.linear(x['h'][take],b[-1].weight[:,:,0,0],b[-1].bias)
                    err=float((c-x['c0'][take]).abs().max());audit['native_readout_error']=max(audit['native_readout_error'],err)
                    torch.testing.assert_close(c,x['c0'][take],atol=1e-5,rtol=1e-5)
            with torch.no_grad():
                q=extract_box_features({'h':raw['h'].cuda(),'levels':raw['levels'].cuda()},onto(x))
            f={k:v.cpu() if torch.is_tensor(v) else v for k,v in q.items()}
            f.update(keys=[key(r) for r in x['rows']],split=split)
            features[iid]=f
            audit['files'][str(iid)]={'prepared':str(p),'prepared_sha256':sha(p),'full_raw':str(source),'full_raw_sha256':sha(source)}
            audit['roi_audits'].append(dict(image_id=iid,**q['audit']))
            image_num+=1
            if image_num%100==0: log(stage='prepare',images=image_num,elapsed_s=time.monotonic()-t)
    # Standardization belongs to fit only and is frozen before any training.
    hs=[]; rs=[]; ls=[]
    for it in index['fit']:
        iid=int(it['image_id']);x=load(prep/f'{iid:012d}.pt')
        hs.append(x['h']);rs.append(features[iid]['residual']);ls.append(x['levels'])
    levels=torch.cat(ls).long()
    norms={'B':normfit(torch.cat(hs),levels),'C':normfit(torch.cat(rs),levels)}
    torch.save(dict(features=features,norms=norms),out/'ROI_FEATURES.pt')
    dump(out/'OFFICIAL_MANIFEST.json',identity)
    audit['passed']=True;dump(out/'INPUT_ROI_AUDIT.json',audit)
    dump(out/'PREPARE_COMPLETE.json',dict(images=image_num,counts=counts,elapsed_s=time.monotonic()-t))
    del model,hs,rs,ls,raw,x
    return index,features,norms


def train(cfg,out,index,features,norms,started):
    prep=Path(cfg['prepared']); fit=[]
    for it in index['fit']:
        iid=int(it['image_id']);x=load(prep/f'{iid:012d}.pt')
        x.pop('condition',None)
        x['region_residual']=features[iid]['residual'];fit.append(x)
    models={};history=[]
    for arm,input_name in [('B','h'),('C','region_residual')]:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        m=ResidualReadout().cuda()
        assert sum(p.numel() for p in m.parameters())==6240
        opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['eta_min'])
        # This new check does not make extra training updates.
        smoke=onto(fit[0]);d=m(normalized(smoke[input_name],smoke['levels'],norms[arm]),smoke['levels'])
        assert torch.count_nonzero(d)==0
        zero_loss=loss_sum(smoke,smoke['c0']+d)
        direct_loss=loss_sum(smoke,smoke['c0'])
        torch.testing.assert_close(zero_loss,direct_loss,atol=0,rtol=0)
        zero_loss.backward()
        assert any(p.grad is not None and p.grad.norm()>0 for p in m.parameters())
        opt.zero_grad(set_to_none=True);del smoke,d,zero_loss,direct_loss
        for epoch in range(1,cfg['epochs']+1):
            te=time.monotonic();total=0.;n=0
            order=np.random.default_rng(cfg['seed']+epoch).permutation(len(fit))
            for start in range(0,len(order),cfg['batch_images']):
                if time.monotonic()-started>60*cfg['max_minutes']: raise RuntimeError('Fixed wall-time budget reached')
                batch=[fit[int(j)] for j in order[start:start+cfg['batch_images']]]
                nc=sum(len(x['rows']) for x in batch);opt.zero_grad(set_to_none=True)
                for raw in batch:
                    x=onto(raw);delta=m(normalized(x[input_name],x['levels'],norms[arm]),x['levels'])
                    loss=loss_sum(x,x['c0']+delta)
                    assert torch.isfinite(loss)
                    (loss/nc).backward();total+=float(loss.detach());n+=len(x['rows'])
                    assert all(not t.requires_grad for t in x.values() if torch.is_tensor(t))
                opt.step()
            rec=dict(arm=arm,epoch=epoch,online_train_bce_candidate_mean=total/n,lr=opt.param_groups[0]['lr'],elapsed_s=time.monotonic()-te)
            history.append(rec);dump(out/'TRAIN_HISTORY.json',history);log(stage='train',**rec)
            if epoch in (5,10):
                torch.save(dict(arm=arm,epoch=epoch,state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},normalization=norms[arm],input=input_name),out/f'{arm}_epoch{epoch}.pt')
            scheduler.step()
        models[arm]=m.eval()
    del fit
    dump(out/'TRAIN_COMPLETE.json',dict(epochs=cfg['epochs'],arms=list(models),checkpoint_rule=cfg['checkpoint_rule'],parameters_each=6240,teacher=False,regularizer=False))
    return models


@torch.no_grad()
def evaluate(cfg,out,index,features,norms,models):
    coco={}; allrows=[]; saved=[]; prep=Path(cfg['prepared']);summary={}
    historical={}
    for line in Path(cfg['historical_baseline']).read_text(encoding='utf-8-sig').splitlines():
        r=json.loads(line);historical[(r['split'],*key(r))]=r
    audit={'baseline_iou_max_error':0.,'zero_bias_pixel_differences':0,'n':0}
    with (out/'PER_CANDIDATE.jsonl').open('w',encoding='utf-8') as fh:
        for split,items in index.items():
            path=cfg['annotations_val'] if split=='val' else cfg['annotations_train']
            if path not in coco: coco[path]=COCO(path)
            total={a:0. for a in ('A','B','C')};n=0;te=time.monotonic()
            for number,it in enumerate(items,1):
                iid=int(it['image_id']);x=onto(load(prep/f'{iid:012d}.pt'));q=features[iid]
                levels=x['levels'];c={'A':x['c0']}
                for arm,rawinput in [('B',x['h']),('C',q['residual'].cuda()),('W',q['wrong_residual'].cuda())]:
                    ma='C' if arm=='W' else arm
                    c[arm]=x['c0']+models[ma](normalized(rawinput,levels,norms[ma]),levels)
                rr=evaluate_image(x,c,coco[path],chunk_size=4)
                for a in total: total[a]+=float(loss_sum(x,c[a]))
                n+=len(rr)
                for k,r in enumerate(rr):
                    h=historical[(split,*key(r))]
                    error=abs(r['iou_A']-float(h['iou_A']))
                    audit['baseline_iou_max_error']=max(audit['baseline_iou_max_error'],error)
                    if error>1e-6: raise AssertionError(('Baseline changed',key(r),error))
                    r['box_iou']=float(h['box_iou'])
                    r['box_good_original_failure']=r['box_iou']>=.75 and r['iou_A']<.75
                    meta=q['donor_metadata'][k]
                    r['wrong_roi_available']=bool(meta['donor_valid'])
                    r['wrong_roi_donor']=meta
                    for a in ('B','C','W'): r['delta_norm_'+a]=float((c[a][k]-c['A'][k]).norm())
                    fh.write(json.dumps(r,ensure_ascii=False,allow_nan=True)+'\n');allrows.append(r)
                fh.flush()
                saved.append(dict(split=split,image_id=iid,keys=[key(r) for r in rr],coeffs={a:v.cpu() for a,v in c.items()}))
                audit['n']+=len(rr)
                if number%50==0: log(stage='evaluate',split=split,images=number,total=len(items),elapsed_s=time.monotonic()-te)
            summary[split]={a:total[a]/n for a in total}
    audit['passed']=True;dump(out/'BASELINE_DECODE_AUDIT.json',audit)
    dump(out/'FINAL_BCE.json',summary);torch.save(saved,out/'FINAL_COEFFICIENTS.pt')
    from summarize_bgcr import summarize
    result=summarize(allrows,out,seed=cfg['bootstrap_seed'],bootstrap=cfg['bootstrap'])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert ultralytics.__version__=='8.4.100'
    dump(out/'ENVIRONMENT.json',dict(host=platform.node(),python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,ultralytics_import=ultralytics.__file__,gpu=torch.cuda.get_device_name(0),train_scope='cached frozen native features; residual layers only',source_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    index,features,norms=prepare(cfg,out)
    models=train(cfg,out,index,features,norms,started)
    evaluate(cfg,out,index,features,norms,models)
    assert sha(cfg['weights'])==cfg['weight_sha256']
    dump(out/'COMPLETE.json',dict(completed=True,elapsed_s=time.monotonic()-started,automatic_next_stage=False,weights_unchanged=True))
    log(stage='complete',elapsed_s=time.monotonic()-started)


if __name__=='__main__':main()
