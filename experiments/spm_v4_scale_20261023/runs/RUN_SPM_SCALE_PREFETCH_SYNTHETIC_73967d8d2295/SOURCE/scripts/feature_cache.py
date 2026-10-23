import shutil

import torch

from common import dump, read, sha


SOURCE_FILES = ['scripts/proto_data.py', 'scripts/task_model.py', 'scripts/base_model.py',
                'scripts/data.py', 'scripts/data_base.py', 'TASK_CONFIG_V3.json']


def signatures(root):
    return {name: sha(root/name) for name in SOURCE_FILES}


def reuse(root, run, upstream, stage, dataset):
    pointer = root/('PREFETCH_'+stage.upper()+'_POINTER.json')
    reason = 'prefetch_not_completed'
    if pointer.exists():
        parent = root/'runs'/read(pointer)['run_id']; receipt = read(parent/'CACHE_RECEIPT.json')
        conditions = {'completed': read(parent/'run.json')['execution_status'] == 'completed',
                      'final_selected_Bstar_matches': receipt['upstream_checkpoint_sha256'] == sha(upstream),
                      'dataset_matches': receipt['dataset_sha256'] == sha(root/dataset),
                      'feature_source_matches': receipt['feature_sources'] == signatures(root)}
        if all(conditions.values()):
            expected = receipt['features_sha256']; assert sha(parent/'features.pt') == expected
            shutil.copyfile(parent/'features.pt', run/'features.pt'); assert sha(run/'features.pt') == expected
            payload = torch.load(run/'features.pt', map_location='cpu', weights_only=False)
            expected_domains = ['real', 'cutout'] if stage == 'warm' else ['synthetic']
            assert payload['source_domains'] == expected_domains
            dump(run/'FEATURE_RECEIPT.json', {**read(parent/'FEATURE_RECEIPT.json'), 'source_prefetch_run_id': parent.name})
            dump(run/'FEATURE_CACHE_REUSE.json', {'reused': True, 'parent_run_id': parent.name,
                                                  'checks': conditions, 'features_sha256': expected,
                                                  'training_steps_added': 0})
            print('VERIFIED_FEATURE_CACHE_REUSED', stage, parent.name, flush=True)
            return payload
        reason = 'cache_mismatch'; dump(run/'FEATURE_CACHE_REJECTED.json', {'parent_run_id': parent.name, 'checks': conditions})
    dump(run/'FEATURE_CACHE_REUSE.json', {'reused': False, 'reason': reason, 'action': 'original_frozen_extract',
                                          'training_steps_added': 0})
    return None
