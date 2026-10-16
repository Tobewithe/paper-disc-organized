"""Serialized training audit, including every actual target package visit."""
import argparse
from common import *

def verify(path):
    data=inspect_inputs()
    from audit_s2_completed import audit
    audit(path)
    contract=read_json(path/'EVAL_CONTRACT.json')
    require(contract['target_policy']==SPEC,'teacher policy differs')
    teacher=STUDY/'runs'/contract['teacher_run_id']
    require(sha256(teacher/'TEACHER_MANIFEST.json')==contract['teacher_manifest_sha256'],'teacher identity differs')
    packages=read_json(teacher/'TEACHER_MANIFEST.json')['images']
    logs=read_jsonl(path/'training.jsonl')
    visits=read_json(path/'TARGET_USAGE.json')['visits']
    require(len(visits)==len(logs)==6000,'target visits budget differs')
    for row,visit in zip(logs,visits):
        pack=packages[str(row['image_id'])]
        require(visit['image_id']==row['image_id'],'target log image differs')
        require(row['teacher_replaced']==visit['teacher_replaced']==pack['accepted'],'actual teacher replacement count differs')
        require(row['teacher_package_sha256']==visit['teacher_package_sha256']==pack['sha256'],'actual teacher target hash differs')
    for name in ('initial.pt','IMAGE_ORDERS.json','TRAINING_JOIN.json'):
        require(sha256(path/name)==sha256(INITIAL/name),'original file identity differs: '+name)
    for key,value in data['original_contract'].items():
        require(contract[key]==value,'original training config differs: '+key)
    require(contract['gt_loss']==read_json(CONTROL/'EVAL_CONTRACT.json')['gt_loss'],'loss formula differs from control')
    expected=3*sum(r['accepted'] for r in packages.values())
    require(sum(r['teacher_replaced'] for r in logs)==expected,'teacher epoch coverage differs')
    verify_after(data)
    write_new(path/'INDEPENDENT_AMT_AUDIT.json',dict(status='passed',at=now(),actual_image_visits=6000,
        actual_optimizer_steps=1500,teacher_replacements=expected,all_actual_targets_verified=True,
        same_original_init_architecture_optimizer_order_and_budget=True,same_soft_IoU_and_protection=True,
        checkpoint_sha256=sha256(path/'last.pt'),training_contract_sha256=sha256(path/'EVAL_CONTRACT.json'),
        teacher_manifest_sha256=sha256(teacher/'TEACHER_MANIFEST.json'),scientific_status='trained_not_evaluated'))
    print('INDEPENDENT_AMT_TRAINING_PASSED',path.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);verify(p.parse_args().run)
