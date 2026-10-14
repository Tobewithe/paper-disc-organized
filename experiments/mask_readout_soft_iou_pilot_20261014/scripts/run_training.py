"""Reuse the original training loop and initialization; replace only the GT term."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from pilot_common import *
from soft_iou_loss import LOSS_SPEC, soft_iou_grouped_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('smoke', 'full'))
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        data = inspect_inputs()
        print('PREFLIGHT_PASSED', len(data['inputs_sha256']), 'inputs;', len(data['split']['fit_ids']), 'fit images')
        return
    require(args.scope, 'scope required')
    with PilotRun('SMOKE' if args.scope == 'smoke' else 'TRAIN', B_RUN, 900 if args.scope == 'smoke' else 7200) as run:
        data = inspect_inputs()
        import torch
        import adapter_model
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        full = args.scope == 'full'
        fit_ids = data['split']['fit_ids'] if full else data['split']['fit_ids'][:8]
        image_orders = read_json(B_RUN/'IMAGE_ORDERS.json') if full else original_training.orders(fit_ids, 1)
        if full:
            shutil.copyfile(B_RUN/'IMAGE_ORDERS.json', run.path/'IMAGE_ORDERS.json')
        else:
            write_new(run.path/'IMAGE_ORDERS.json', image_orders)
        torch.manual_seed(0)
        model = adapter_model.CoefficientAdapter('B').cuda()
        saved = torch.load(B_RUN/'initial.pt', map_location='cpu', weights_only=True)
        require(saved['arm'] == 'B' and saved['seed'] == 0, 'wrong original initialization')
        model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        initial_hash = tensor_state_hash(model)
        require(initial_hash == data['original_contract']['model_state_initial_sha256'], 'initial state mismatch')
        whitelist = [n for n, p in model.named_parameters() if p.requires_grad]
        require(whitelist == data['original_contract']['trainable_whitelist'], 'trainable whitelist drift')
        optimizer = torch.optim.AdamW(model.parameters(), **original_training.OPTIMIZER)
        require(not optimizer.state, 'optimizer resumed instead of fresh initialization')
        actual_optimizer = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        require(json.dumps(actual_optimizer, sort_keys=True) == json.dumps(data['original_contract']['optimizer'], sort_keys=True), 'optimizer drift')
        optimizer_parameters = [p for g in optimizer.param_groups for p in g['params']]
        require([id(p) for p in optimizer_parameters] == [id(p) for p in model.parameters()], 'optimizer whitelist mismatch')
        contract = copy.deepcopy(data['original_contract'])
        contract.update(scope=args.scope, epochs=3 if full else 1, fit_images=len(fit_ids),
            expected_optimizer_steps=1500 if full else 2, image_orders_sha256=sha256(run.path/'IMAGE_ORDERS.json'),
            gt_loss=LOSS_SPEC, original_B_run=B_RUN.name, initialization_file_sha256=sha256(B_RUN/'initial.pt'),
            inputs_sha256=data['inputs_sha256'], external_evaluator_sources=data['external_evaluator_sources'],
            single_scientific_variable='GT_BCE_to_same_ROI_soft_IoU', original_training_loop_sha256=sha256(Path(original_training.__file__)),
            wall_seconds=run.wall_seconds)
        if full:
            require(args.smoke_run is not None, 'completed smoke required')
            for name in ('DECISION.json', 'CONFIG_AUDIT.json'):
                require(read_json(args.smoke_run/name)['status'] == 'passed', 'smoke failed')
            require(read_json(args.smoke_run/'EVAL_CONTRACT.json')['model_state_initial_sha256'] == initial_hash, 'smoke initialization differs')
            for previous in (STUDY/'runs').glob('RUN_SOFT_IOU_TRAIN_*/run.json'):
                if previous.parent != run.path:
                    require(not read_json(previous).get('optimization_started'), 'formal training budget already consumed')
        run.freeze(contract)
        if full:
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files'] == run.sources, 'source changed after smoke')
        unchanged = {k: contract[k] == v for k, v in data['original_contract'].items()}
        allowed = set() if full else {'scope', 'epochs', 'fit_images', 'expected_optimizer_steps', 'image_orders_sha256'}
        require(all(v or k in allowed for k, v in unchanged.items()), 'non-GT original configuration changed')
        write_new(run.path/'ORIGINAL_CONFIG_COMPARE.json', {'status': 'passed', 'original_field_equality': unchanged,
            'smoke_budget_only_exceptions': sorted(allowed), 'only_scientific_change': LOSS_SPEC,
            'native_argument_differences': data['native_argument_differences']})
        write_new(run.path/'ENVIRONMENT.json', data['environment'])
        write_new(run.path/'ACTUAL_OPTIMIZER.json', actual_optimizer)
        shutil.copyfile(B_RUN/'initial.pt', run.path/'initial.pt')
        source = FrozenCache(CACHE_RUN/'trace', fit_ids)
        coco = COCO(str(TRAIN_GT))
        if not full:
            with (run.path/'tests.log').open('x', encoding='utf8') as stream:
                result = subprocess.run([sys.executable, '-B', str(STUDY/'scripts/test_soft_iou.py')], stdout=stream, stderr=subprocess.STDOUT)
            require(result.returncode == 0, 'loss behavioral tests failed')
            dev_ids = data['split']['dev_ids'][:8]
            dev = FrozenCache(BASE/'trace', dev_ids)
            checks = {'dev_zero_delta': original_training.zero_replay(run, model, dev, dev_ids),
                      'fit_zero_delta': original_training.zero_replay(run, model, source, fit_ids)}
            write_new(run.path/'ENGINEERING_CONTROLS.json', checks)
            dev.verify()
            del dev
        # The original loop imports this function at call time. No other training operation changes.
        original_loss = adapter_model.vector_grouped_loss
        adapter_model.vector_grouped_loss = soft_iou_grouped_loss
        try:
            run.update(optimization_started=True, phase='adapter_training', arm='B')
            trained = original_training.train(run, model, source, coco, image_orders, optimizer, whitelist)
        finally:
            adapter_model.vector_grouped_loss = original_loss
        require(tensor_state_hash(model) != initial_hash, 'adapter did not update')
        require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'nonfinite trained state')
        new_join = read_json(run.path/'TRAINING_JOIN.json')['images']
        old_join = read_json(B_RUN/'TRAINING_JOIN.json')['images']
        require(new_join == {str(i): old_join[str(i)] for i in fit_ids}, 'actual GT joins changed')
        if full:
            require(sha256(run.path/'TRAINING_JOIN.json') == sha256(B_RUN/'TRAINING_JOIN.json'), 'formal join bytes changed')
        require({k: v for k, v in optimizer.param_groups[0].items() if k != 'params'} == actual_optimizer, 'optimizer changed during training')
        torch.save({'state_dict': model.state_dict(), 'arm': 'B', 'seed': 0,
                    'contract_sha256': sha256(run.path/'EVAL_CONTRACT.json'),
                    'optimizer_state_dict': optimizer.state_dict(), 'completed_epochs': len(image_orders)}, run.path/'last.pt')
        source.verify()
        verify_after(data)
        write_new(run.path/'CONFIG_AUDIT.json', {'status': 'passed', 'scope': 'single-variable soft-IoU training',
            'checks': {'all_original_configuration_fields_equal_except_smoke_budget': True,
                'original_initial_pt_and_exact_tensor_state': True, 'fresh_optimizer': True,
                'original_training_loop_and_only_loss_replacement': True, 'same_training_join': True,
                'effective_batch4_and_exact_steps': trained['optimizer_steps'] == contract['expected_optimizer_steps'],
                'only_new_adapter_parameters': True, 'frozen_cache_inputs_environment_unchanged': True},
            'actual_steps': trained['optimizer_steps'], 'actual_images_seen': trained['images_seen'],
            'checkpoint_sha256': sha256(run.path/'last.pt'), 'contract_sha256': sha256(run.path/'EVAL_CONTRACT.json')})
        run.check_budget()
        run.finish({**trained, 'arm': 'B', 'checkpoint_sha256': sha256(run.path/'last.pt'),
                    'initial_state_sha256': initial_hash, 'final_state_sha256': tensor_state_hash(model)},
                   scientific='trained_not_evaluated' if full else 'engineering_only',
                   next_action='Audit serialized checkpoint/logs; evaluate original S2 gates' if full else 'One formal B training from original initial.pt')
        print('SOFT_IOU_TRAIN_COMPLETE', run.run_id, trained['optimizer_steps'], flush=True)


if __name__ == '__main__':
    main()
