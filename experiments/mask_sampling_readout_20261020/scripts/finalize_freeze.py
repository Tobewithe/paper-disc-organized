"""After verified input delivery, freeze exact runtime data and all reviewed sources."""
from ssr_runtime import *

def main():
    from ssr_runner import prepare_contract
    require(not (STUDY/'FREEZE_MANIFEST.json').exists(),'already frozen; do not replace freeze')
    expectations=read(STUDY/'DESIGN_EXPECTATIONS.json')
    for row in expectations['files']:
        p=ROOT/row['path']
        normalized=__import__('hashlib').sha256(p.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()
        require(normalized==row['normalized_sha256'],'source differs from reviewed design: '+row['path'])
    model_test=read(STUDY/'preflight/MODEL_ENGINEERING.json')
    require(model_test['status']=='passed','model engineering not passed')
    require(model_test['source_sha256']==sha(STUDY/'scripts/test_model.py'),'model test source changed')
    require(read(pointer('prepare')/'SUMMARY.json')['all_parent_image_hashes_exact'],'input preparation incomplete')
    prepare_contract()
    config=read(STUDY/'CONFIG.json')
    profile,env=environment()
    require(sha(config['weights'])==profile['model_sha256'],'weight identity mismatch')
    require(sha(config['adapter_initial'])=='ce2de59a316eef3cea6cfe6d1763221931fd10693377c8a91ce321843604bacd','initial checkpoint mismatch')
    require(sha(config['existing_G'])=='15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce','old G checkpoint mismatch')
    require(sha(config['existing_G_predictions'])=='15d8cc2884777854a7120099e4ee218b8750ee59f63d2b3fcc4571fa2a191590','historical G predictions mismatch')
    trace=read(config['historical_native_trace'])
    require(sha(trace['predictions_path'])==trace['predictions_sha256'],'historical N prediction SHA mismatch')
    files=[ROOT/r['path'] for r in expectations['files']]
    files += [STUDY/'inputs'/n for n in ('DATA_CONTRACT.json','IMAGE_ORDERS.json','IMAGE_PATHS.json','train.txt','val.txt')]
    rows=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
    value={'study_id':config['study_id'],'code_commit':'bc0507c5400b61935fe93f981a4ff5026964c271',
           'frozen_at':now(),'protocol_sha256':sha(STUDY/'PROTOCOL.md'),'config_sha256':sha(STUDY/'CONFIG.json'),
           'design_expectations_sha256':sha(STUDY/'DESIGN_EXPECTATIONS.json'),'files':rows,
           'authorization':config['authorization'],'environment':env,
           'dirty_patch_sha256':sha(STUDY/'TRACKED_DIRTY.patch'),
           'input_files':[{'path':config[k],'sha256':sha(config[k]),'role':k} for k in
             ('weights','adapter_initial','existing_G','train_annotations','val_annotations','historical_native_trace','historical_native_ledger','existing_G_predictions')]
             +[{'path':trace['predictions_path'],'sha256':trace['predictions_sha256'],'role':'historical_native_predictions'}]}
    write(STUDY/'FREEZE_MANIFEST.json',value)
    write(STUDY/'preflight/FREEZE_VERIFIED.json',{'status':'passed','freeze_sha256':sha(STUDY/'FREEZE_MANIFEST.json'),
          'source_entries':len(rows),'model_engineering_passed':True,'inputs_complete':True,'formal_training_steps':0})
    s=read(STUDY/'study.json');s.update(plan_status='frozen',execution_status='prepared',current_phase='awaiting_smoke')
    write(STUDY/'study.json',s)
    print('EXECUTION_FREEZE_READY',sha(STUDY/'FREEZE_MANIFEST.json'),flush=True)

if __name__=='__main__':main()
