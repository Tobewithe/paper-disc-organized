"""Post-run audit from serialized checkpoints and per-image logs, independent of training counters."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import torch
from common import ROOT,B_RUN,read_json,require,sha256,write_new


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        item=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str((list(item.shape),item.dtype)).encode())
        digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit(path):
    contract=read_json(path/'EVAL_CONTRACT.json')
    meta=read_json(path/'run.json')
    require(meta['execution_status']=='completed' and contract['scope']=='full','formal completed training required')
    summary=read_json(path/'SUMMARY.json')
    require(sha256(path/'last.pt')==summary['checkpoint_sha256'],'checkpoint identity changed')
    expected=read_json(path/'IMAGE_ORDERS.json')
    actual=[json.loads(s) for s in (path/'training.jsonl').read_text(encoding='utf8').splitlines()]
    require([(r['epoch'],r['image_id']) for r in actual]==[(e,i) for e,ids in enumerate(expected) for i in ids],
            'actual training order or image coverage differs')
    require(len(actual)==6000 and len(expected)==3 and all(len(ids)==2000 and len(set(ids))==2000 for ids in expected),'training budget changed')
    require(set(expected[0])==set(expected[1])==set(expected[2]),'epoch image membership changed')
    for row in actual:
        require(row['repulsion_weight']==contract['repulsion_weight'] and row['supervision_scale']=='native_input_prethreshold','loss settings changed')
        require(all(math.isfinite(row[k]) for k in ('loss','gt_loss','preservation_loss')),'nonfinite loss log')
        require(abs(row['loss']-row['gt_loss']-.1*row['preservation_loss']-contract['repulsion_weight']*row['repulsion_loss'])<2e-6,'actual loss combination differs')
    for item in read_json(path/'ARTIFACT_MANIFEST.json')['artifacts']:
        require(sha256(path/item['relative_path'])==item['sha256'],'artifact hash mismatch')
    for rel,h in read_json(path/'SOURCE_MANIFEST.json')['files'].items():
        require(sha256(ROOT/rel)==h and sha256(path/'source_snapshot'/rel)==h,'frozen source mismatch')
    require(sha256(path/'IMAGE_ORDERS.json')==sha256(B_RUN/'IMAGE_ORDERS.json'),'original image order changed')
    require(sha256(path/'TRAINING_JOIN.json')==sha256(B_RUN/'TRAINING_JOIN.json'),'original training join changed')
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)
    original=torch.load(B_RUN/'initial.pt',map_location='cpu',weights_only=True)
    require(all(torch.equal(v,initial['state_dict'][k]) for k,v in original['state_dict'].items()),'shared initial weights changed')
    final=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    require(final['completed_epochs']==3 and final['arm']==contract['arm'],'checkpoint selection differs')
    require(final['contract_sha256']==sha256(path/'EVAL_CONTRACT.json'),'checkpoint contract differs')
    require(set(final['state_dict'])==set(contract['trainable_whitelist']),'checkpoint parameter whitelist changed')
    require(all(torch.isfinite(v).all().item() for v in final['state_dict'].values()),'nonfinite checkpoint tensors')
    require(sum(v.numel() for v in final['state_dict'].values())==contract['parameters']==167872,'parameter count changed')
    require(state_hash(initial['state_dict'])==contract['model_state_initial_sha256'],'recorded initialization differs')
    require(state_hash(final['state_dict'])==summary['final_state_sha256'],'final checkpoint tensor identity differs')
    require(torch.count_nonzero(initial['state_dict']['readout.2.weight']).item()==0
            and torch.count_nonzero(initial['state_dict']['readout.2.bias']).item()==0,'initial delta readout not zero')
    groups=final['optimizer_state_dict']['param_groups']
    require(len(groups)==1,'unexpected optimizer parameter group')
    actual_optimizer={k:v for k,v in groups[0].items() if k!='params'}
    require(json.dumps(actual_optimizer,sort_keys=True)==json.dumps(contract['optimizer'],sort_keys=True),'checkpoint optimizer hyperparameters differ')
    steps={contract['trainable_whitelist'][idx]:int(state['step']) for idx,state in final['optimizer_state_dict']['state'].items()}
    require(steps['readout.2.weight']==steps['readout.2.bias']==contract['expected_optimizer_steps']==1500,'actual Adam update budget differs')
    require(max(steps.values())==1500,'unexpected optimizer step count')
    report={'status':'passed','scope':'independent post-run checkpoint and complete training-log inspection',
        'arm':contract['arm'],'experiment_arm':contract['experiment_arm'],'exact_image_order_and_epoch_coverage':True,'actual_image_visits':len(actual),
        'image_microbatch':contract['image_microbatch'],'gradient_accumulation':contract['gradient_accumulation'],
        'effective_image_batch':contract['effective_image_batch'],'optimizer_steps_from_checkpoint':steps,
        'actual_optimizer_from_checkpoint':actual_optimizer,'new_parameters':contract['parameters'],
        'all_checkpoint_tensors_finite':True,'last_epoch_checkpoint':True,'zero_initial_readout_verified':True,
        'per_epoch_mean_training_loss':{str(e):sum(r['loss'] for r in actual if r['epoch']==e)/2000 for e in range(3)},
        'checkpoint_sha256':sha256(path/'last.pt'),'training_log_sha256':sha256(path/'training.jsonl'),
        'image_orders_sha256':sha256(path/'IMAGE_ORDERS.json'),'contract_sha256':sha256(path/'EVAL_CONTRACT.json')}
    output=path.parent.parent/('AUDIT_'+path.name+'.json')
    if output.exists():require(json.dumps(read_json(output),sort_keys=True)==json.dumps(report,sort_keys=True),'prior audit differs')
    else:write_new(output,report)
    print(path.name,'INDEPENDENT_CONFIG_AUDIT passed; 6000 image visits; 1500 Adam steps; 167872 parameters',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runs',nargs='+',type=Path)
    for path in parser.parse_args().runs:audit(path)
