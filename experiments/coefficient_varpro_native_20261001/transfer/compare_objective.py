import sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,r'D:\coco_wire\varpro\scripts')
from official_tal_solve_fast_ref import load_rows, stats, Bank
from ultralytics.utils import ops
cache=Path(r'D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache')
rows=load_rows(cache,'fit'); st=stats(rows); device=torch.device('cuda')
bank=Bank(rows,st,device,9.83241)
zero=[torch.zeros((bank.h.shape[1],bank.c.shape[1]),dtype=torch.float64,device=device) for _ in range(3)]
with torch.no_grad(): v,b,r,_=bank.shared_value_grad(zero,False)
print('reference',float(v),float(b),float(r),len(rows))
# cplus style exact loop
sumb=sumreg=0.0; n=0
for item in __import__('json').loads((cache/'INDEX.json').read_text())['fit']:
    iid=int(item['image_id']); x=torch.load(cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
    proto=F.interpolate(x['proto'].float()[None].to(device),(640,640),mode='bilinear',align_corners=False)[0]; masks=x['masks'].to(device)
    for k,row in enumerate(x['rows']):
        c=x['coeff'][int(row['raw_id'])].float().to(device); box=x['target_boxes'][k].float(); support=ops.crop_mask(torch.ones((1,640,640),device=device),box[None].to(device))[0].bool(); p=proto[:,support].T; y=(masks[support]==int(x['owners'][k])+1).float(); z=p@c; area=float(row.get('area',(((box[2:]-box[:2])/640.0).prod()*640.0*640.0))); bce=float((float(x['segmentation_gain'])*(F.softplus(z)-y*z).sum()/area).detach().cpu()); sumb+=bce; n+=1
print('cplus',sumb/n,n,'delta',sumb/n-float(b))
# first 10 compare row values
for j,(rr,item) in enumerate(zip(rows, __import__('json').loads((cache/'INDEX.json').read_text())['fit'])):
    if j>=10: break
    pass
