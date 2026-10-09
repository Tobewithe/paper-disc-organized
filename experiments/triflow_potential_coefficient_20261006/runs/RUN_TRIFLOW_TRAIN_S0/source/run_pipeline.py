"""Fixed TriFlow cache -> eight-epoch module train -> full COCO evaluation."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--cache-run', default='RUN_TRIFLOW_FIT_CACHE_S0')
    parser.add_argument('--train-run', default='RUN_TRIFLOW_TRAIN_S0')
    parser.add_argument('--eval-run', default='RUN_TRIFLOW_COCO5000_EVAL_S0')
    a = parser.parse_args()
    root = a.root.resolve()
    preflight = read(root/'runs/RUN_TRIFLOW_PREFLIGHT_S0_R1/PREFLIGHT.json')
    if preflight['status'] != 'PASS':
        raise RuntimeError('Input/environment preflight has not passed')
    diagnostic = read(root/'runs/RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1/TRIFLOW_VERIFICATION.json')
    if diagnostic.get('passed') is not True or diagnostic.get('check_count') != 20 or diagnostic.get('numerical_only'):
        raise RuntimeError('Full mathematical/gradient/actual-model diagnostic required')
    for filename in ('triflow_model.py', 'frozen_io.py'):
        if diagnostic['source_files'][filename]['sha256'] != sha(root/'scripts'/filename):
            raise RuntimeError('Method changed since actual diagnostic: ' + filename)
    smoke = read(root/'runs/RUN_TRIFLOW_SMOKE_TRAIN_S0/TRAINING_AUDIT.json')
    if smoke.get('passed') is not True or smoke.get('epochs') != 1 or not smoke.get('compiler_task_gradient_observed_nonzero'):
        raise RuntimeError('Actual smoke training audit required')
    for filename in ('triflow_model.py', 'frozen_io.py', 'train_triflow.py'):
        if smoke['sources'][filename] != sha(root/'scripts'/filename):
            raise RuntimeError('Training implementation changed since smoke: ' + filename)
    contract = read(root/'runs/RUN_TRIFLOW_EVAL_CONTRACT_S0/EVALUATOR_VERIFICATION.json')
    if contract.get('passed') is not True or contract.get('successful_images') != 4 or not contract.get('strict_loader_smoke_rejection', {}).get('passed'):
        raise RuntimeError('Actual sample native-decoder and smoke rejection contract required')
    for filename in ('evaluate_triflow.py', 'readout_support.py', 'triflow_model.py', 'frozen_io.py'):
        if contract['sources'][filename]['executed_sha256'] != sha(root/'scripts'/filename):
            raise RuntimeError('Evaluator changed since actual sample contract: ' + filename)
    inputs = {'protocol_sha256': sha(root/'PROTOCOL.md'),
              'source_sha256': {p.name:sha(p) for p in sorted((root/'scripts').glob('*.py'))},
              'cache_run':a.cache_run, 'train_run':a.train_run, 'eval_run':a.eval_run,
              'annotation_path':preflight['train_annotations'],
              'formal_epochs':8, 'seed':0, 'automatic_extension':False}
    pipeline = root/'runs'/a.run_id
    (pipeline/'PIPELINE_INPUTS.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')

    def stage(run_id, purpose, script, args):
        directory = root/'runs'/run_id
        if (directory/'run.json').is_file():
            metadata = read(directory/'run.json')
            if metadata.get('execution_status') == 'completed' and metadata.get('exit_code') == 0:
                archived = read(directory/'SOURCE.json')['sha256']
                for name in (script, 'triflow_model.py', 'frozen_io.py'):
                    if archived.get(name) != inputs['source_sha256'][name]:
                        raise RuntimeError('Existing stage has another source; use a new Run ID: '+run_id)
                print('REUSE_VERIFIED_STAGE '+run_id,flush=True)
                return
            raise RuntimeError('Stage already ran/failed/running; distinct retry Run required: '+run_id)
        command = [sys.executable,'-X','utf8',str(root/'scripts/run_stage.py'),
                   '--root',str(root),'--run-id',run_id,'--purpose',purpose,'--script',script,'--']+args
        print('START '+run_id,flush=True)
        code = subprocess.call(command,cwd=root)
        if code:
            raise RuntimeError(f'Stage {run_id} failed exit={code}; partial outputs preserved')
        print('FINISHED '+run_id,flush=True)

    common = ['--root',str(root)]
    weights = 'D:/coco_wire/models/yolo26m-seg.pt'
    vendor = 'D:/coco_wire/vendor_8.4.100'
    stage(a.cache_run,'formal_frozen_features_and_original_gt','prepare_cache.py',
          common+['--run-id',a.cache_run,'--weights',weights,'--vendor',vendor,
                  '--images-list',str(root/'data/fit.txt'),'--annotations',preflight['train_annotations'],
                  '--max-instances','12','--device','cuda'])
    cache = read(root/'runs'/a.cache_run/'CACHE_RECEIPT.json')
    if cache.get('images') != 796 or not cache['frozen_integrity']['passed'] or not cache['all_native_replay_exact']:
        raise RuntimeError('Formal 796-image cache identity/frozen contract failed')
    if cache['images_list_sha256'] != sha(root/'data/fit.txt'):
        raise RuntimeError('Formal fit list changed')
    stage(a.train_run,'fixed_eight_epoch_triflow_module_training','train_triflow.py',
          common+['--run-id',a.train_run,'--cache-run',a.cache_run,'--epochs','8',
                  '--seed','0','--instance-chunk','4','--device','cuda'])
    audit = read(root/'runs'/a.train_run/'TRAINING_AUDIT.json')
    if audit.get('passed') is not True or audit.get('epochs') != 8:
        raise RuntimeError('Formal training audit failed')
    stage(a.eval_run,'full_original_coco5000_triflow_readout','evaluate_triflow.py',
          common+['--run-id',a.eval_run,'--head',str(root/'runs'/a.train_run/'head_final.pt'),
                  '--images-list',str(root/'data/val_full.txt'),
                  '--annotations',preflight['val_annotations'],'--weights',weights,'--vendor',vendor,
                  '--baseline-cache','D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/official',
                  '--device','cuda'])
    summary = read(root/'runs'/a.eval_run/'SUMMARY.json')
    if summary.get('status') != 'complete' or summary.get('image_count') != 5000:
        raise RuntimeError('Full evaluation not complete')
    (pipeline/'PIPELINE_COMPLETE.json').write_text(json.dumps(
        {'completed':True,'cache_run':a.cache_run,'train_run':a.train_run,'evaluation_run':a.eval_run,
         'fixed_budget_finished':True,'automatic_extension':False},indent=2),encoding='utf-8')
    print('FIXED_SCREEN_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
