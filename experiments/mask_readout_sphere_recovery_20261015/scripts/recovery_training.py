import json,time
from common import ROOT,require,write_new,sha256
def train(run, model, source, coco, image_orders, optimizer, whitelist):
    import torch
    from adapter_model import loss_grids, vector_grouped_loss
    from adapter_data import training_join, projected_targets
    model.train()
    old=ROOT/'experiments/mask_readout_spherical_adapter_20261015/runs/RUN_SPHERICAL_TRAIN_efde25d2082348c69c765ce70d5d5550'
    prior=[json.loads(line) for line in (old/'training.jsonl').read_text(encoding='utf8').splitlines()] if len(image_orders)==3 else []
    require(not prior or len(prior)==315,'failed prefix changed')
    started = time.monotonic()
    steps, image_count, no_matches, empty_rois = 0, 0, 0, 0
    joins = {}
    gradient_names = set()
    with (run.path/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch, sequence in enumerate(image_orders):
            epoch_losses = []
            for offset in range(0,len(sequence),4):
                group = sequence[offset:offset+4]
                optimizer.zero_grad(set_to_none=True)
                for image_id in group:
                    run.check_budget()
                    cache, rows = source.load(image_id)
                    if image_id not in joins:
                        joins[image_id] = training_join(rows,coco.imgToAnns.get(image_id,[]))
                    targets = projected_targets(coco,cache,joins[image_id])
                    delta = model(cache,'train:'+str(epoch))
                    logits,baseline,roi = loss_grids(cache,cache['c0'].to(delta)+delta)
                    loss, accounting = vector_grouped_loss(logits,baseline,roi,targets)
                    require(torch.isfinite(loss).item(), 'non-finite training loss')
                    (loss / len(group)).backward()
                    image_count += 1
                    no_matches += int(accounting['matched_supervised_instances']==0)
                    empty_rois += len(accounting['empty_roi_rows'])
                    row={'epoch':epoch,'image_id':image_id,'loss':float(loss.detach()),**accounting}
                    if prior and image_count<=len(prior):
                        require(row==prior[image_count-1],'reconstruction prefix differs at image '+str(image_count))
                        if image_count==len(prior):
                            write_new(run.path/'PREFIX_RECONSTRUCTION.json',dict(status='passed_exact_logged_prefix',
                                source_run=old.name,source_training_log_sha256=sha256(old/'training.jsonl'),
                                repeated_image_visits=315,repeated_optimizer_steps=78,partial_gradient_images=3,
                                original_intermediate_checkpoint_available=False,
                                scope='same declared algorithm and exact scalar loss/accounting prefix; lost state cannot be bitwise compared'))
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    epoch_losses.append(row['loss'])
                    del cache,rows,targets,delta,logits,baseline,roi,loss
                for name,parameter in model.named_parameters():
                    if parameter.grad is not None:
                        require(torch.isfinite(parameter.grad).all().item(),'non-finite adapter gradient')
                        if torch.count_nonzero(parameter.grad).item():
                            gradient_names.add(name)
                optimizer.step()
                steps += 1
                if steps%25==0:
                    torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                        completed_optimizer_steps=steps,images_seen=image_count,contract_sha256=sha256(run.path/'EVAL_CONTRACT.json')),
                        run.path/'recovery_latest.pt')
                if steps % 25 == 0 or offset+len(group)==len(sequence):
                    log.flush()
                    run.update(phase='adapter_training',arm=model.arm,epoch=epoch+1,optimizer_steps=steps,images_seen=image_count)
                    print('TRAIN',model.arm,'epoch',epoch+1,'steps',steps,'images',image_count,
                          'loss',round(sum(epoch_losses)/len(epoch_losses),6),'seconds',round(time.monotonic()-started,1),flush=True)
            require(len(epoch_losses)==len(sequence),'training image omitted')
    require(set(n for n,p in model.named_parameters() if p.requires_grad)==set(whitelist),'trainable whitelist drift')
    require(gradient_names and any(n.startswith('readout.2') for n in gradient_names),'new module received no gradient')
    require(steps==sum((len(seq)+3)//4 for seq in image_orders),'optimizer step budget mismatch')
    require(image_count==sum(map(len,image_orders)),'image budget mismatch')
    write_new(run.path/'TRAINING_JOIN.json',{'policy':'class_box50_score_order_one_to_one_ordinary_GT',
             'images':{str(i):{'output_rank_to_annotation_id':{str(k):v for k,v in j.items()}} for i,j in joins.items()}})
    return {'optimizer_steps':steps,'images_seen':image_count,'epochs':len(image_orders),
            'images_without_matched_supervision':no_matches,'empty_roi_instances_across_epochs':empty_rois,
            'parameters_with_nonzero_gradient':sorted(gradient_names),'elapsed_training_seconds':time.monotonic()-started}

