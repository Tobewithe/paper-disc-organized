"""Normal original-image mask replay for the frozen stage-1 affine solutions."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from ultralytics.utils import ops
from pycocotools.coco import COCO

def load_arm(path):
    x=json.loads(path.read_text()); A=[torch.tensor(v,dtype=torch.float32) for v in x['A']]
    st=None if x.get('stats') is None else {int(k):(torch.tensor(v[0],dtype=torch.float32),torch.tensor(v[1],dtype=torch.float32)) for k,v in x['stats'].items()}
    return A,st
def apply_delta(h,level,A,st,bias):
    if bias: return A[level]
    mu,sd=st[level]; return torch.cat(((h.float()-mu)/sd,torch.ones(1)))@A[level]
def iou(a,b):
    inter=(a&b).sum().item(); union=(a|b).sum().item(); return inter/max(union,1)
def run_group(group, cache, split, arm_name, arm_json, coco, out):
    A,st=load_arm(arm_json); bias=arm_name=='G-BIAS'; rows=[]
    for iid in split:
        p=cache/'images'/f'{int(iid):012d}.pt'; im=torch.load(p,map_location='cpu',weights_only=False)
        proto=im['proto'].float(); coeff=im['coeff'].float(); boxes=im['boxes'].float(); h=im['h'].float()
        rlist=im['rows']; ids=[]; anns=[]
        for r in rlist:
            raw=int(r['raw_id']); lev=int(r.get('level',im['levels'][raw] if 'levels' in im else 0)); ids.append(raw); anns.append(int(r['annotation_id']))
        if not ids: continue
        base=coeff[ids].clone(); new=base.clone()
        for k,(raw,ann) in enumerate(zip(ids,anns)):
            lev=int(rlist[k].get('level',int(im['levels'][raw])))
            new[k]+=apply_delta(h[raw],lev,A,st,bias)
        mb=ops.process_mask(proto,base,boxes[ids],(640,640),upsample=True)
        mn=ops.process_mask(proto,new,boxes[ids],(640,640),upsample=True)
        shape=tuple(im['original_shape'])
        if 'ratio_pad' in im: rp=im['ratio_pad']
        else: rp=((float(im['gain']),float(im['gain'])),(float(im['left']),float(im['top'])))
        mb=ops.scale_masks(mb[:,None].float(),shape,ratio_pad=rp)[:,0]>0.5
        mn=ops.scale_masks(mn[:,None].float(),shape,ratio_pad=rp)[:,0]>0.5
        vals=[]; vals0=[]; rep=dam=0; cov0=cov1=0
        for k,annid in enumerate(anns):
            gt=torch.from_numpy(coco.annToMask(coco.anns[annid])).bool()
            b=iou(mb[k],gt); n=iou(mn[k],gt); vals0.append(b); vals.append(n); rep+=int(b<.75 and n>=.75); dam+=int(b>=.75 and n<.75); cov0+=int((mb[k]&gt).sum()); cov1+=int((mn[k]&gt).sum())
        rows.append({'image_id':int(iid),'n':len(vals),'macro_iou_A':float(np.mean(vals0)),'macro_iou_B':float(np.mean(vals)),'delta':float(np.mean(vals)-np.mean(vals0)),'repairs':rep,'damage':dam,'coverage_A':float(cov0/max(sum(int(x.sum()) for x in [torch.from_numpy(coco.annToMask(coco.anns[a])).bool() for a in anns]),1)),'coverage_B':float(cov1/max(sum(int(x.sum()) for x in [torch.from_numpy(coco.annToMask(coco.anns[a])).bool() for a in anns]),1))})
    out.write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf-8'); return rows
def summary(allrows,out):
    rng=np.random.default_rng(20261004); ids=np.arange(len(allrows)); d=np.array([x['delta'] for x in allrows],float); draws=rng.integers(0,len(ids),(5000,len(ids))); boot=d[draws].mean(1)
    n=sum(x['n'] for x in allrows); da=sum(x['delta']*x['n'] for x in allrows)/max(n,1)
    return {'images':len(allrows),'candidates':n,'image_macro_A':float(np.mean([x['macro_iou_A'] for x in allrows])),'image_macro_B':float(np.mean([x['macro_iou_B'] for x in allrows])),'image_macro_delta':float(d.mean()),'image_macro_ci95':np.quantile(boot,[.025,.975]).tolist(),'candidate_delta':float(da),'repairs':sum(x['repairs'] for x in allrows),'damage':sum(x['damage'] for x in allrows),'coverage_A':float(np.mean([x['coverage_A'] for x in allrows])),'coverage_B':float(np.mean([x['coverage_B'] for x in allrows]))}
def main(a):
    root=a.root; out=root/'evaluation'; out.mkdir(exist_ok=True); split=json.loads((root/'assets'/'VAL_ONLY_SPLIT.json').read_text())['val']; coco=COCO(str(root/'data/annotations/instances_val2017.json')); allsum={}
    for name,cache in [('O-GB',root/'official_cache'),('G-OFF',root/'geometry_val'),('G-BIAS',root/'geometry_val')]:
        path=out/f'{name}.jsonl'; rows=run_group('val',cache,split,name,root/'stage1'/f'{name}.json',coco,path); allsum[name]=summary(rows,path)
    (out/'SUMMARY.json').write_text(json.dumps(allsum,indent=2),encoding='utf-8')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); main(ap.parse_args())
