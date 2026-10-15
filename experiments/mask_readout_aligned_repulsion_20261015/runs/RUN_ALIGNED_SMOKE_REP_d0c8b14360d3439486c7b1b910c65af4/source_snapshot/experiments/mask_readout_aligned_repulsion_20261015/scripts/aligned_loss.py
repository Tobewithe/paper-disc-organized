"""Native-input-resolution BCE; original proto preservation; GT-masked neighbor repulsion.

Chunking changes memory use only. Per-instance then per-image denominators remain fixed.
GT is used exclusively here during training, never in adapter inference.
"""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
sys.path.insert(0,str(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/scripts'))
sys.path.insert(0,str(ROOT/'shared/evaluation'))
from adapter_model import loss_grids,vector_grouped_loss
from s1_geometry import geometry
from repulsion import choose_neighbors

MAX_CHUNK_PIXELS=4_000_000
REPULSION_WEIGHT={'R':0.,'REP':.1}


def target_at_native_input(coco,cache,annotation_id,device):
    from pycocotools import mask as masks
    info=geometry(cache)
    target=torch.tensor(masks.decode(coco.annToRLE(coco.anns[annotation_id])).copy(),dtype=torch.float32,device=device)
    if list(target.shape)!=info['original_hw']:raise ValueError('GT/native geometry differs')
    resized=F.interpolate(target[None,None],info['resized_hw'],mode='nearest')
    left,top,right,bottom=info['ltrb_padding']
    padded=F.pad(resized,(left,right,top,bottom),value=0)[0,0]
    if list(padded.shape)!=list(cache['native_decode_hw']) or list(padded.shape)!=list(cache['input_hw']):
        raise ValueError('unexpected native-input decoder geometry')
    return padded


def boxes_roi(boxes,shape):
    h,w=shape;x=torch.arange(w,device=boxes.device)[None,None,:];y=torch.arange(h,device=boxes.device)[None,:,None]
    return (x>=boxes[:,0,None,None])&(x<boxes[:,2,None,None])&(y>=boxes[:,1,None,None])&(y<boxes[:,3,None,None])


def terms(logits,targets,roi,neighbor_probability,overlap):
    """Return per-instance means. No-neighbor rows stay in the caller's denominator."""
    counts=roi.sum((1,2))
    if not (counts>0).all().item():raise ValueError('empty aligned ROI for proto-valid supervised row')
    gt=(F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*roi).sum((1,2))/counts
    rep=(logits.sigmoid()*neighbor_probability.detach()*overlap*roi*(1-targets)).sum((1,2))/counts
    return gt,rep


def backward_image(cache,coefficients,coco,join,arm,image_divisor):
    from ultralytics.utils import ops
    logits,baseline,roi=loss_grids(cache,coefficients)
    preservation_weighted,accounting=vector_grouped_loss(logits,baseline,roi,{})
    valid=roi.sum((1,2))>0;indices=[i for i in join if bool(valid[i])]
    boxes=cache['boxes_input'].detach().to(coefficients);shape=cache['native_decode_hw']
    neighbors,_=choose_neighbors(cache['boxes_input'].tolist(),cache['scores'].tolist(),cache['classes'].tolist(),cache['raw_ids']) if len(coefficients) else ([],[])
    chunk_size=max(1,MAX_CHUNK_PIXELS//(shape[0]*shape[1]));gt_total=0.;rep_total=0.;neighbor_rows=0
    for start in range(0,len(indices),chunk_size):
        selected=indices[start:start+chunk_size]
        up=ops.scale_masks(logits[selected][None],shape)[0]
        targets=torch.stack([target_at_native_input(coco,cache,join[i],coefficients.device) for i in selected])
        target_roi=boxes_roi(boxes[selected],shape)
        other=[max(0,neighbors[i]) for i in selected]
        with torch.no_grad():
            probability=ops.scale_masks(baseline[other][None],shape)[0].sigmoid()
            overlap=boxes_roi(boxes[other],shape)&torch.tensor([neighbors[i]>=0 for i in selected],device=coefficients.device)[:,None,None]
        gt,rep=terms(up,targets,target_roi,probability,overlap)
        part=(gt.sum()+REPULSION_WEIGHT[arm]*rep.sum())/len(indices)
        if not torch.isfinite(part).item():raise ValueError('nonfinite aligned loss')
        (part/image_divisor).backward(retain_graph=True)
        gt_total+=float(gt.detach().sum())/len(indices);rep_total+=float(rep.detach().sum())/len(indices)
        neighbor_rows+=sum(neighbors[i]>=0 for i in selected)
        del up,targets,target_roi,probability,overlap,gt,rep,part
    (preservation_weighted/image_divisor).backward()
    total=gt_total+float(preservation_weighted.detach())+REPULSION_WEIGHT[arm]*rep_total
    accounting.update(gt_loss=gt_total,repulsion_loss=rep_total,repulsion_weight=REPULSION_WEIGHT[arm],
        matched_supervised_instances=len(indices),matched_with_eligible_neighbor=neighbor_rows,
        supervision_scale='native_input_prethreshold',native_decode_hw=shape,chunk_pixel_budget=MAX_CHUNK_PIXELS)
    return total,accounting


def train(run,model,source,coco,image_orders,optimizer,whitelist,arm):
    import json,time
    from adapter_data import training_join
    from posttrain_run import require,write_new
    model.train();started=time.monotonic();steps=0;image_count=0;no_matches=0;empty_rois=0;joins={};gradient_names=set()
    with (run.path/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch,sequence in enumerate(image_orders):
            losses=[]
            for offset in range(0,len(sequence),4):
                group=sequence[offset:offset+4];optimizer.zero_grad(set_to_none=True)
                for image_id in group:
                    run.check_budget();cache,rows=source.load(image_id)
                    if image_id not in joins:joins[image_id]=training_join(rows,coco.imgToAnns.get(image_id,[]))
                    delta=model(cache,'train:'+str(epoch))
                    loss,accounting=backward_image(cache,cache['c0'].to(delta)+delta,coco,joins[image_id],arm,len(group))
                    image_count+=1;no_matches+=int(accounting['matched_supervised_instances']==0);empty_rois+=len(accounting['empty_roi_rows'])
                    log.write(json.dumps(dict(epoch=epoch,image_id=image_id,loss=loss,**accounting),allow_nan=False)+'\n');losses.append(loss)
                    del cache,rows,delta
                for name,parameter in model.named_parameters():
                    if parameter.grad is not None:
                        require(torch.isfinite(parameter.grad).all().item(),'nonfinite gradient')
                        if torch.count_nonzero(parameter.grad).item():gradient_names.add(name)
                optimizer.step();steps+=1
                if steps%25==0 or offset+len(group)==len(sequence):
                    log.flush();run.update(phase='adapter_training',experiment_arm=arm,epoch=epoch+1,optimizer_steps=steps,images_seen=image_count)
                    print('TRAIN',arm,'epoch',epoch+1,'steps',steps,'images',image_count,'loss',round(sum(losses)/len(losses),6),'seconds',round(time.monotonic()-started,1),flush=True)
            require(len(losses)==len(sequence),'training image omitted')
    require(set(n for n,p in model.named_parameters() if p.requires_grad)==set(whitelist),'whitelist changed')
    require(any(n.startswith('readout.2') for n in gradient_names),'readout received no gradient')
    require(steps==sum((len(v)+3)//4 for v in image_orders) and image_count==sum(map(len,image_orders)),'training budget mismatch')
    write_new(run.path/'TRAINING_JOIN.json',dict(policy='class_box50_score_order_one_to_one_ordinary_GT',
        images={str(i):{'output_rank_to_annotation_id':{str(k):v for k,v in j.items()}} for i,j in joins.items()}))
    return dict(optimizer_steps=steps,images_seen=image_count,epochs=len(image_orders),images_without_matched_supervision=no_matches,
        empty_roi_instances_across_epochs=empty_rois,parameters_with_nonzero_gradient=sorted(gradient_names),elapsed_training_seconds=time.monotonic()-started)
