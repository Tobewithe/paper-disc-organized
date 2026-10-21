from lrp_common import *
import torch
from pycocotools.coco import COCO
from pycocotools import mask as masks
from adapter_data import FrozenCache,training_join
from lrp_math import PixelResidual,image_context,points

def main():
    data=inspect_inputs()
    with Run('PREPARE') as run:
        ids=data['split']['fit_image_ids']
        run.freeze(dict(stage='fit_only_preparation',images=1600,heldout_supervision=0,environment=data['environment']))
        source=FrozenCache(CACHE/'trace',ids)
        coco=COCO(str(GT))
        (run.path/'data').mkdir()
        records=[]; counts=dict(images=0,selected_outputs=0,editable_outputs=0,matched_editable=0,sampled_pixels=0,native_rows=0)
        with torch.inference_mode():
            for pos,iid in enumerate(ids,1):
                run.check_budget()
                cache,rows=source.load(iid)
                join=training_join(rows,coco.loadAnns(coco.getAnnIds(imgIds=[iid])))
                ranks,selected,basis=image_context(cache,rows)
                instances=[]
                for rank in ranks:
                    rec=points(cache,rows[rank],rank,selected[rank],basis)
                    take=rec['sample']; ann=join.get(rank)
                    target=None
                    if ann is not None:
                        y=masks.decode(coco.annToRLE(coco.anns[ann]))
                        target=torch.as_tensor(y[rec['yy'][take],rec['xx'][take]].copy(),dtype=torch.float32)
                    instances.append(dict(rank=rank,global_index=cache['prediction_indices'][rank],annotation_id=ann,
                        x=rec['x'][take].cpu().clone(),a=rec['a'][take].cpu().clone(),target=target,
                        yy=torch.as_tensor(rec['yy'][take].copy()),xx=torch.as_tensor(rec['xx'][take].copy()),editable_count=len(rec['yy'])))
                    counts['sampled_pixels']+=len(take)
                    counts['editable_outputs']+=int(len(take)>0)
                    counts['matched_editable']+=int(ann is not None and len(take)>0)
                    del rec
                dst=run.path/'data'/f'{iid}.pt'
                torch.save(dict(image_id=iid,instances=instances,cache_sha256=source.records[iid]['cache_sha256']),dst)
                records.append(dict(image_id=iid,path=dst.relative_to(run.path).as_posix(),sha256=sha256(dst),instances=len(instances),join=join))
                counts['images']+=1; counts['selected_outputs']+=len(ranks); counts['native_rows']+=len(rows)
                del cache,rows,ranks,selected,basis,instances
                if pos%50==0 or pos==len(ids):
                    run.update(completed_images=pos,counts=counts)
                    print('PREPARE',pos,'/1600',flush=True)
        torch.manual_seed(0)
        model=PixelResidual()
        require(sum(p.numel() for p in model.parameters())==3297,'parameter count')
        torch.save(dict(state_dict=model.state_dict(),seed=0),run.path/'initial.pt')
        write_new(run.path/'DATA_MANIFEST.json',dict(records=records,counts=counts,split_sha256=sha256(CLD/'SPLIT.json'),initial_sha256=sha256(run.path/'initial.pt'),initial_state_sha256=tensor_state_hash(model)))
        source.verify()
        config_audit(run,fit_images=1600,heldout_images_prepared=0,new_optimizer_steps=0,all_native_rows_replayed=counts['native_rows'])
        run.finish(counts,scientific='engineering_preparation_only')
    save(STUDY/'PREPARE_POINTER.json',dict(run_id=run.run_id))

if __name__=='__main__': main()
