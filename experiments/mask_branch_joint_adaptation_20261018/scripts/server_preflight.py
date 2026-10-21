"""One-time 16-val-image integration check while train ZIP is downloading.

No optimizer, no checkpoint or scientific comparison. Formal prepare/smoke remain mandatory.
"""
from pathlib import Path
import argparse
import json
import time
from joint_data import read, write, sha, build_dataset, dataset_plan, NativeBatches, batch_identity, require
from joint_runner import Run, environment, branch, ARMS

def main():
    import torch
    from pycocotools.coco import COCO
    from adapter_data import training_join, projected_targets
    from joint_model import snapshot_state
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    with Run(args.config,'preflight') as run:
        environment(run)
        gt=read(run.config['val_annotations'])
        files=[Path(run.config['val_images'])/r['file_name'] for r in sorted(gt['images'],key=lambda r:r['id'])]
        require(len(files)==5000 and all(p.is_file() for p in files),'full val image archive required')
        image_list=run.path/'val.txt'
        image_list.write_text(''.join(str(p)+'\n' for p in files))
        _,dataset=build_dataset(run.config,image_list,run.path/'plan')
        plan=dataset_plan(dataset)
        first=plan[0]['image_ids']
        hashes={str(i):sha(Path(run.config['val_images'])/f'{i:012d}.jpg') for i in first}
        inputs={key:{'path':run.config[key],'sha256':sha(run.config[key])} for key in
                ('val_annotations','profile','weights','adapter_initial','existing_G','train_annotations','original_split')}
        write(run.path/'DATA_CONTRACT.json',{'val_batches':plan,'val_image_hashes':hashes,'val_ids':[int(p.stem) for p in files],
              'inputs':inputs,
              'scope':'engineering_only_first_16_val_source_images_not_formal_training_prepare'})
        del dataset
        native=NativeBatches(run.config,run.path,'val',run.path/'native')
        snapshot=snapshot_state(native.model)
        caches,rows,features=native.batch(0,masks=True)
        write(run.path/'NATIVE_IDENTITY.json',batch_identity(caches,rows,features))
        coco=COCO(run.config['val_annotations'])
        receipts={}
        for arm in ARMS:
            model=branch(run,native,arm)
            for i,cache in enumerate(caches): model.verify_initial_replay(cache,features,batch_index=i)
            selected=list(range(4))
            outs=model.forward_batch(caches[:4],features,selected,phase='train:0')
            losses=[]
            for i,out in zip(selected,outs):
                targets=projected_targets(coco,caches[i],training_join(rows[i],coco.imgToAnns.get(caches[i]['image_id'],[])))
                loss,_=model.loss(out,targets)
                losses.append(loss)
            loss=torch.stack(losses).mean()
            require(bool(torch.isfinite(loss)), 'nonfinite integration loss')
            loss.backward()
            grad={n:{'present':p.grad is not None,'nonzero':p.grad is not None and bool(torch.count_nonzero(p.grad)),
                     'finite':p.grad is None or bool(torch.isfinite(p.grad).all())} for n,p in model.named_parameters()}
            require(all(v['finite'] for v in grad.values()),'nonfinite integration gradient')
            if arm=='PC_native': require(any(n.startswith('proto.') and v['nonzero'] for n,v in grad.items()),'detached Proto')
            model.verify_frozen_state(native.model,snapshot)
            receipts[arm]={'initial_P_c_logits_exact':True,'loss':float(loss.detach()),'gradient':grad,
                'optimizer_steps':0,'parameters':model.parameter_manifest()}
            del model,outs,losses,loss,targets
            torch.cuda.empty_cache()
        run.finish({'images':16,'image_ids':first,'native_zero_delta_RLE_exact':True,'arms':receipts,
                    'optimizer_steps':0,'scientific_result':None,'does_not_replace_formal_32_image_smoke':True},
                   'engineering_integration_only')

if __name__=='__main__': main()
