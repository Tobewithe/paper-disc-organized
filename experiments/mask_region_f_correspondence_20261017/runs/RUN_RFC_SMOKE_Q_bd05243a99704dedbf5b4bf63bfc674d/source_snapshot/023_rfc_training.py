"""Original soft-IoU training loop, original initialization and budget, regional-F input only."""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import torch

from rfc_runtime import *
from rfc_model import RegionFAdapter, parameter_receipt
from rfc_inputs import prepare_inputs
from rfc_controls import native_area_gate
from adapter_data import FrozenCache
from native_output_bridge import decode_native, encode_rle
from soft_iou_loss import soft_iou_grouped_loss, LOSS_SPEC
import adapter_model
import run_s2_training as original_training


class Sidecars:
    def __init__(self, export_run, split):
        self.run=export_run
        self.manifest_path=export_run/'REGION_MANIFEST.json'
        self.manifest_sha=sha256(self.manifest_path)
        self.spec=read_json(self.manifest_path)[split]
        self.records={r['image_id']:r for r in self.spec['records']}

    def get(self, cache, phase, gate=None):
        record=self.records[cache['image_id']]
        path=(self.run/record['path']).resolve()
        require(path.is_relative_to(self.run.resolve()), 'sidecar escaped Run')
        require(sha256(path)==record['sha256'], 'region F changed before read')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        return prepare_inputs(cache,payload,record['source_cache_sha256'],self.spec['source_predictions_sha256'],phase,gate)

    def verify(self):
        require(sha256(self.manifest_path)==self.manifest_sha,'region manifest changed')
        for record in self.records.values():
            require(sha256(self.run/record['path'])==record['sha256'],'region F changed after execution')


class TrainingAdapter(RegionFAdapter):
    def __init__(self,arm,sidecars):
        super().__init__(arm)
        self.sidecars=sidecars

    def forward(self, cache, phase):
        return super().forward(self.sidecars.get(cache,phase),phase)


def smoke_controls(run,data,source,fit_ids,export_run):
    """No optimizer updates here. Check initialized RLE and nonzero reference readout on real tensors."""
    from pycocotools import mask as masks
    initial=torch.load(B/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
    nonzero=torch.load(G/'last.pt',map_location='cpu',weights_only=True)['state_dict']
    dev_ids=data['split']['dev_ids'][:8]
    dev=FrozenCache(BASE/'trace',dev_ids)
    fit_regions=Sidecars(export_run,'fit'); dev_regions=Sidecars(export_run,'dev')
    A=RegionFAdapter('A').cuda(); Q=RegionFAdapter('Q').cuda()
    old=adapter_model.CoefficientAdapter('B').cuda()
    checks=[]; changed=0; gradient_names=set()
    with torch.inference_mode():
        for name,cache_source,ids,regions in [('fit',source,fit_ids,fit_regions),('dev',dev,dev_ids,dev_regions)]:
            for iid in ids:
                run.check_budget(); cache,rows=cache_source.load(iid)
                gate=native_area_gate(torch.tensor([int(masks.area(r['segmentation'])) for r in rows],dtype=torch.int64))
                phase='train:0' if name=='fit' else 'eval:0'
                inputs=regions.get(cache,phase,None if name=='fit' else gate)
                for model in (A,Q):
                    model.load_state_dict(initial,strict=True)
                    delta=model(inputs,phase)
                    require(torch.count_nonzero(delta)==0,'nonzero original initialization')
                    decoded=decode_native(cache,cache['c0'].to(delta)+delta)
                    require(all(encode_rle(decoded[j])==row['segmentation'] for j,row in enumerate(rows)), 'initialized adapter native RLE mismatch')
                    del decoded,delta
                A.load_state_dict(nonzero,strict=True); Q.load_state_dict(nonzero,strict=True)
                old.load_state_dict(nonzero,strict=True)
                a=A(inputs,phase); q=Q(inputs,phase)
                changed += int(not torch.equal(a,q))
                # J belongs to evaluation only; construct the declared eval-gated representation.
                if name=='dev':
                    require(torch.equal(A(inputs,'eval:0',intervention='J'),q),'same-weight J and Q disagree')
                from dataclasses import replace
                broadcast=replace(inputs,regions=tuple(f[:,None,None].expand(-1,8,8) for f in cache['features']))
                require(torch.equal(A(broadcast,phase),old(cache,phase)),'broadcast F does not exactly recover original G')
                checks.append(dict(split=name,image_id=iid,rows=len(rows),zero_delta_RLE_exact=True,
                                   broadcast_old_G_exact=True,layout_changes_nonzero_output=not torch.equal(a,q)))
                del cache,rows,inputs,broadcast,a,q
    require(changed>0,'regional layout had no effect on real nonzero reference outputs')
    dev.verify(); fit_regions.verify(); dev_regions.verify()
    write_new(run.path/'REAL_FORWARD_CONTROLS.json',dict(images=checks,layout_effect_images=changed,
              extra_optimizer_steps=0,existing_G_checkpoint_unchanged=sha256(G/'last.pt')==data['spec']['inference']['G_checkpoint_sha256']))
    del dev,A,Q,old,initial,nonzero
    torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--scope',choices=('smoke','full'),required=True)
    p.add_argument('--arm',choices=('A','Q'),required=True); p.add_argument('--smoke-deadline',type=float)
    args=p.parse_args(); full=args.scope=='full'
    export=pointer('INPUT_FULL' if full else 'INPUT_SMOKE')
    if full:
        for arm in ('A','Q'): completed(pointer('SMOKE_'+arm))
        prior=[r for r in (STUDY/'runs').glob('RUN_RFC_TRAIN_'+args.arm+'_*/run.json') if read_json(r).get('optimization_started')]
        require(not prior,'formal arm already consumed budget; no automatic restart')
        if args.arm=='Q': completed(pointer('TRAIN_A'))
    with Run(('TRAIN_' if full else 'SMOKE_')+args.arm,export,7200 if full else 1800,args.smoke_deadline) as run:
        data=inspect_inputs(); resources=initial_resources()
        fit_ids=data['split']['fit_ids'] if full else data['split']['fit_ids'][:8]
        orders=read_json(G/'IMAGE_ORDERS.json') if full else original_training.orders(fit_ids,1)
        if full: shutil.copyfile(G/'IMAGE_ORDERS.json',run.path/'IMAGE_ORDERS.json')
        else: write_new(run.path/'IMAGE_ORDERS.json',orders)
        regions=Sidecars(export,'fit')
        require(set(regions.records)==set(fit_ids),'training F coverage differs from all original fit images')
        source=FrozenCache(CACHE/'trace',fit_ids)
        require(regions.spec['source_manifest_sha256']==source.manifest_sha,'native trace identity drift')
        for iid,record in regions.records.items():
            require(record['source_cache_sha256']==source.records[iid]['cache_sha256'],'F source cache SHA changed')
            require(record['region_rows']==record['rows'],'training rows were gated')
        torch.manual_seed(0)
        model=TrainingAdapter(args.arm,regions).cuda()
        model.load_state_dict(torch.load(B/'initial.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        initial_hash=tensor_state_hash(model)
        require(initial_hash==data['spec']['training']['initial_tensor_state_sha256'],'original initialization mismatch')
        whitelist=[n for n,p in model.named_parameters() if p.requires_grad]
        require(whitelist==data['old_contract']['trainable_whitelist'],'parameter whitelist changed')
        require(parameter_receipt(model)['parameters']==167872 and len(whitelist)==17,'parameter budget changed')
        optimizer=torch.optim.AdamW(model.parameters(),**original_training.OPTIMIZER)
        actual={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        require_no_drift(data['old_contract']['optimizer'],json.loads(json.dumps(actual)),'optimizer')
        require(not optimizer.state,'optimizer state must be fresh')
        contract=copy.deepcopy(data['old_contract'])
        contract.update(arm=args.arm,scope=args.scope,fit_images=len(fit_ids),epochs=len(orders),
                        expected_optimizer_steps=1500 if full else 2,
                        image_orders_sha256=sha256(run.path/'IMAGE_ORDERS.json'),
                        environment=data['environment'],region_manifest_sha256=regions.manifest_sha,
                        region_export_run=export.name,initialization_file_sha256=sha256(B/'initial.pt'),
                        gt_loss=LOSS_SPEC,resources=resources,smoke_deadline=args.smoke_deadline,
                        design_sha256=sha256(STUDY/'DESIGN_SPEC.json'),
                        only_intended_changes=['shared regional F injection','A/Q regional F ordering'],
                        original_training_loop_sha256=sha256(Path(original_training.__file__)))
        run.freeze(contract)
        if full:
            for arm in ('A','Q'):
                require(read_json(pointer('SMOKE_'+arm)/'SOURCE_MANIFEST.json')['files']==run.sources,'source changed after smoke')
            if args.arm=='Q':
                paired=pointer('TRAIN_A')
                require(read_json(paired/'SOURCE_MANIFEST.json')['files']==run.sources,'A/Q source drift')
                other=read_json(paired/'EVAL_CONTRACT.json')
                for key in ('fit_images','epochs','expected_optimizer_steps','image_orders_sha256','environment',
                            'region_manifest_sha256','initialization_file_sha256','optimizer','gt_loss','design_sha256'):
                    require_no_drift(other[key],contract[key],'A/Q matched '+key)
                write_new(run.path/'PAIRED_A.json',dict(run_id=paired.name,contract_sha256=sha256(paired/'EVAL_CONTRACT.json')))
        shutil.copyfile(B/'initial.pt',run.path/'initial.pt')
        write_new(run.path/'ACTUAL_OPTIMIZER.json',actual)
        write_new(run.path/'ENVIRONMENT.json',data['environment'])
        if not full: smoke_controls(run,data,source,fit_ids,export)
        from pycocotools.coco import COCO
        coco=COCO(str(CACHE/'dataset/training_annotations.json'))
        previous=adapter_model.vector_grouped_loss
        adapter_model.vector_grouped_loss=soft_iou_grouped_loss
        try:
            run.update(optimization_started=True,phase='adapter_training',arm=args.arm)
            trained=original_training.train(run,model,source,coco,orders,optimizer,whitelist)
        finally:
            adapter_model.vector_grouped_loss=previous
        require(trained['optimizer_steps']==contract['expected_optimizer_steps'],'step budget drift')
        require(trained['images_seen']==len(fit_ids)*len(orders),'image budget drift')
        require(set(trained['parameters_with_nonzero_gradient'])==set(whitelist),'not all 17 parameter tensors received nonzero gradients')
        require(all(torch.isfinite(p).all().item() for p in model.parameters()),'nonfinite final weights')
        require(tensor_state_hash(model)!=initial_hash,'no learned update')
        actual_joins=read_json(run.path/'TRAINING_JOIN.json')['images']
        old_joins=read_json(G/'TRAINING_JOIN.json')['images']
        require(actual_joins=={str(i):old_joins[str(i)] for i in fit_ids},'original GT pairing drift')
        if full: require(sha256(run.path/'TRAINING_JOIN.json')==sha256(G/'TRAINING_JOIN.json'),'formal GT join bytes drift')
        require_no_drift(json.loads(json.dumps(actual)),json.loads(json.dumps({k:v for k,v in optimizer.param_groups[0].items() if k!='params'})),'final optimizer')
        torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),arm=args.arm,seed=0,
                        completed_epochs=len(orders),steps=trained['optimizer_steps'],
                        contract_sha256=sha256(run.path/'EVAL_CONTRACT.json')),run.path/'last.pt')
        source.verify(); regions.verify()
        audit(run,data,parameter_and_gradient_checks='passed',parameters=167872,trainable_tensors=17,
              optimizer_steps=trained['optimizer_steps'],initial_state_sha256=initial_hash,
              final_state_sha256=tensor_state_hash(model),checkpoint_sha256=sha256(run.path/'last.pt'),
              same_original_training_loop=True,same_original_GT_join=True,training_gate_restriction=False)
        run.finish(dict(**trained,arm=args.arm,initial_state_sha256=initial_hash,
                        final_state_sha256=tensor_state_hash(model),checkpoint_sha256=sha256(run.path/'last.pt')),
                   scientific='trained_not_evaluated' if full else 'engineering_only',
                   next_action='complete paired formal arm then freeze evaluation inputs' if full else 'paired smoke then original 2000-image F export')
    set_pointer(('TRAIN_' if full else 'SMOKE_')+args.arm,run)

if __name__=='__main__': main()
