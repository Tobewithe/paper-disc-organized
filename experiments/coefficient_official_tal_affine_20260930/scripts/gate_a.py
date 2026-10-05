"""Freeze official identities and compare official, ROI and analytic gradients."""
import argparse, hashlib, inspect, json, platform
from pathlib import Path
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils import ops

def dump(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for k in ('cache','old_ids','weights','out'): ap.add_argument('--'+k, type=Path, required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(20260930)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert ultralytics.__version__=='8.4.100'
    w=YOLO(str(a.weights)); gain=float(w.ckpt['train_args']['box'])
    env={'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),'weight_sha256':sha(a.weights),'segmentation_gain':gain,'train_args':w.ckpt['train_args'],'native_last_shapes':[list(b[-1].weight.shape) for b in w.model.model[-1].one2one_cv4]}
    dump(a.out/'ENVIRONMENT.json',env)
    index=json.loads((a.cache/'INDEX.json').read_text()); old=json.loads(a.old_ids.read_text())
    identities={}; counts={}; selected=[]
    key=lambda r:(int(r['image_id']),int(r['annotation_id']),int(r['raw_id']),int(r.get('pyramid_level',r.get('level',-1))))
    for g,items in index.items():
        rr=[]; no_pos=[]; lev={str(i):0 for i in range(3)}
        for pos,it in enumerate(items):
            x=torch.load(a.cache/'images'/f"{int(it['image_id']):012d}.pt",map_location='cpu',weights_only=False)
            if not x['rows']: no_pos.append(int(it['image_id']))
            for k,r in enumerate(x['rows']):
                assert int(x['owners'][k])==int(r['gt_index'])
                assert x['all_annotation_ids'][int(r['gt_index'])]==int(r['annotation_id'])
                rid=int(r['raw_id']); assert int(x['levels'][rid])==int(r['level'])
                rr.append({'split':g,'image_id':int(r['image_id']),'annotation_id':int(r['annotation_id']),'branch':'one2one','raw_id':rid,'pyramid_level':int(r['level']),'target_gt_idx':int(r['gt_index'])})
                lev[str(int(r['level']))]+=1
            # 6 deterministic images per split, all their positives.
            if pos<6: selected.append((g,x))
        assert {key(r) for r in rr}=={key(r) for r in old[g]}, f'identity mismatch: {g}'
        identities[g]=rr
        counts[g]={'planned_images':len(items),'effective_images':len(items)-len(no_pos),'candidates':len(rr),'no_positive_images':no_pos,'level_counts':lev}
    dump(a.out/'OFFICIAL_MANIFEST.json',{'source':'native YOLODataset + frozen one2one TAL','counts':counts,'identity_list':identities,'upstream_list_scope':'existing official effective-image list; earlier planned images outside this list are not restored or invented'})
    dump(a.out/'IDENTITY_AUDIT.json',{'passed':True,'historical_identity_match':True,'counts':counts,'keys':['split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx'],'geometric_assignment_used':False})
    audits=[]; atol=rtol=3e-5
    for group,x in selected:
        proto=F.interpolate(x['proto'].cuda()[None],x['masks'].shape[-2:],mode='bilinear',align_corners=False)[0]
        masks=x['masks'].cuda(); boxes=x['target_boxes'].cuda(); owners=x['owners'].cuda()
        ids=torch.tensor([r['raw_id'] for r in x['rows']],device='cuda')
        base=x['coeff'].cuda()[ids]; n=len(ids)
        gt=(masks[None]==(owners+1)[:,None,None]).float()
        areas=((boxes[:,2:]-boxes[:,:2])/640).prod(1)
        for mode in ('original','random_delta'):
            c=(base+(torch.randn_like(base)*.03 if mode=='random_delta' else 0)).detach().requires_grad_()
            ref=v8SegmentationLoss.single_mask_loss(gt,c,proto,boxes,areas)*gain/n
            gr=torch.autograd.grad(ref,c)[0]
            replay=c.sum()*0
            analytic=torch.zeros_like(c,dtype=torch.float64)
            for k in range(n):
                support=ops.crop_mask(torch.ones((1,640,640),device='cuda'),boxes[k:k+1])[0].bool()
                p=proto[:,support].T; y=gt[k][support]; denom=areas[k]*640*640
                z=p@c[k]
                replay=replay+gain*F.binary_cross_entropy_with_logits(z,y,reduction='sum')/denom/n
                pd=p.double(); yd=y.double(); zd=pd@c[k].double()
                analytic[k]=(gain*(torch.sigmoid(zd)-yd)@pd/denom.double()/n)
            ga=torch.autograd.grad(replay,c)[0]
            torch.testing.assert_close(ref,replay,atol=atol,rtol=rtol)
            torch.testing.assert_close(gr,ga,atol=atol,rtol=rtol)
            torch.testing.assert_close(gr.double(),analytic,atol=atol,rtol=rtol)
            audits.append({'split':group,'image_id':x['rows'][0]['image_id'],'mode':mode,'n':n,'levels':sorted({r['level'] for r in x['rows']}),'value_error':float((ref-replay).abs()),'gradient_error':float((gr-ga).abs().max()),'analytic_double_gradient_error':float((gr.double()-analytic).abs().max())})
    dump(a.out/'LOSS_EQUIVALENCE.json',{'passed':True,'atol':atol,'rtol':rtol,'audits':audits,'reference_source':inspect.getsourcefile(v8SegmentationLoss),'mask_shape':[640,640],'proto_shape':[32,160,160],'interpolation':'bilinear align_corners=False to GT resolution','labels':'official overlap raster; owner+1 following official dedup and area order','crop':'native crop_mask fresh tensor per candidate; >= lower bound and < upper bound','normalization':'full 640x640 mean divided by normalized GT box area, official segmentation gain included','solver_aggregation':'equal candidate mean; no batch, scale or confidence reweighting','old_7D_numeric_equivalence':'not assumed: lambda retained, official gain now explicit; old objective values must not be reused'})
    print(json.dumps({'gate_a':'passed','counts':counts,'max_value_error':max(t['value_error'] for t in audits),'max_gradient_error':max(t['gradient_error'] for t in audits)}),flush=True)

if __name__=='__main__': main()
