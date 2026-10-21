"""Future full CLD evaluation. Explicitly blocked by the current smoke-only authorization."""
import argparse
from collections import defaultdict
import contextlib
import numpy as np
import torch
from cld_common import *
from cld_math import target, context, remap, deploy_coefficients, probe_coefficients, functional_direction
from cld_statistics import summarize
from adapter_data import FrozenCache
from adapter_model import CoefficientAdapter
from native_output_bridge import decode_native, encode_rle
from s1_geometry import full_rle_iou, predicted_roi
from pycocotools.coco import COCO
from pycocotools import mask as masks
from build_native_ledger import evaluate_ledger
from posttrain_audit import compare_outputs


def ledger(path, predictions, ids, mapping, native_sha, log_name='coco_matching.log'):
    # Preserve complete prediction fields, COCO's bbox area policy and separate matching.
    with (path / log_name).open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        rows, unused_ap = evaluate_ledger(GT, predictions, ids, native_sha)
    for row in rows:
        for key in ('bbox75_match_output_id', 'segm75_match_output_id'):
            if row[key] is not None:
                local = int(row[key].rsplit(':', 1)[1])
                row[key] = native_sha + ':' + str(mapping[local])
    # AP internally computed by the inherited ledger helper is not an endpoint here.
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--perm', type=Path, required=True)
    args = parser.parse_args()
    authorization = read_json(STUDY / 'OPERATION_AUTHORIZATION.json')
    require(authorization.get('allow_full_evaluation') is True, 'heldout evaluation not authorized')
    paths = dict(REAL=args.real.resolve(), PERM=args.perm.resolve())
    contracts, checkpoints = {}, {}
    for arm, path in paths.items():
        check_completed(path)
        contracts[arm] = read_json(path / 'EVAL_CONTRACT.json')
        require(contracts[arm]['scope'] == 'full' and contracts[arm]['arm'] == arm and contracts[arm]['expected_optimizer_steps'] == 1200, 'evaluation requires both full last checkpoints')
        checkpoints[arm] = sha256(path / 'last.pt')
    require(contracts['REAL']['amplitude_sha256'] == contracts['PERM']['amplitude_sha256'], 'amplitude source differs')
    with Run('EVALUATION', paths['REAL'], 7200) as run:
        data = inspect_inputs()
        run.freeze(dict(stage='CLD_evaluation', environment=data['environment'], checkpoints=checkpoints,
                        checkpoint_runs={k: v.name for k,v in paths.items()}, operation_authorization=authorization,
                        direction_scope='GT_ASSISTED_EQUAL_AMPLITUDE', utility_scope='complete_400_image_mixed_outputs',
                        amplitude=contracts['REAL']['amplitude'], new_optimizer_steps=0))
        models = {}
        for arm, path in paths.items():
            saved = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
            require(saved['steps'] == 1200 and saved['completed_epochs'] == 3 and saved['arm'] == arm, 'wrong checkpoint history')
            model = CoefficientAdapter('B').cuda().eval().requires_grad_(False)
            model.load_state_dict(saved['state_dict'], strict=True)
            require(tensor_state_hash(model) == read_json(path / 'SUMMARY.json')['final_state_sha256'], 'model state differs')
            models[arm] = model
        held = set(data['split']['heldout_image_ids'])
        ids = sorted(data['split']['fit_image_ids'] + data['split']['heldout_image_ids'])
        teachers = teacher_rows(ids)  # Reached only after both full last identities are frozen above.
        source = FrozenCache(CACHE / 'trace', ids)
        coco = COCO(str(GT))
        historic_pairs = defaultdict(list)
        for r in read_jsonl(FIT / 'FIT_FIXED_PAIRS.jsonl'):
            if r['image_id'] in held:
                historic_pairs[r['image_id']].append(r)
        direction_rows, pair_rows, mapping, predictions = [], [], [], {k: [] for k in ('N','M','S')}
        tensors_dir = run.path / 'directions'
        tensors_dir.mkdir()
        forwards = 0
        with torch.inference_mode():
            for pos, iid in enumerate(ids, 1):
                run.check_budget()
                if iid not in held and not teachers[iid]:
                    continue
                cache, rows = source.load(iid)
                native = decode_native(cache)
                require(all(encode_rle(native[j]) == r['segmentation'] for j,r in enumerate(rows)), 'native replay differs')
                del native
                raw = {arm: model(cache, 'evaluate:final') for arm,model in models.items()}
                forwards += 2
                require(all(torch.isfinite(v).all() for v in raw.values()), 'nonfinite direction')
                directions = dict(R=raw['REAL'], P=remap(raw['PERM'], data['maps'][iid], 'PERM'), J=remap(raw['REAL'], data['maps'][iid], 'J'))
                ctx = context(cache, raw['REAL'].device)
                scales = {r['rank']: target(cache, r)[1] for r in teachers[iid]}
                torch.save(dict(image_id=iid, global_indices=cache['prediction_indices'], donor_ranks=data['maps'][iid], raw={k:v.cpu() for k,v in raw.items()}), tensors_dir / f'{iid}.pt')
                probe_rle, cosine, zero = {}, {}, {}
                for arm, d in directions.items():
                    decoded = decode_native(cache, probe_coefficients(d, ctx, scales))
                    probe_rle[arm] = {r['rank']: encode_rle(decoded[r['rank']]) for r in teachers[iid]}
                    response, _, ms, _ = functional_direction(d, ctx)
                    for r in teachers[iid]:
                        rank = r['rank']
                        roi = ctx['roi'][rank]
                        t = target(cache,r)[0].to(d)
                        rms = float(ms[rank].sqrt())
                        zero[(arm,rank)] = rms <= 1e-6
                        cosine[(arm,rank)] = float((response[rank][roi] * t[roi]).mean() / ms[rank].sqrt()) if rms > 1e-6 else 0.
                    del decoded, response
                for t in teachers[iid]:
                    rank = t['rank']
                    gt = coco.annToRLE(coco.anns[t['annotation_id']])
                    n = full_rle_iou(rows[rank]['segmentation'],gt)
                    c = full_rle_iou(t['oracle_rle'],gt)
                    require(abs(n-t['baseline_iou']) < 1e-12 and abs(c-t['oracle_iou']) < 1e-12, 'teacher endpoint identity differs')
                    record = dict(split='heldout' if iid in held else 'fit', image_id=iid, annotation_id=t['annotation_id'], rank=rank,
                                  output_detection_id=t['output_detection_id'], N=n, C=c, gate=int(masks.area(rows[rank]['segmentation']))>=9216,
                                  scale=scales[rank], probe_RLE={arm: probe_rle[arm][rank] for arm in directions})
                    for arm in directions:
                        record[arm] = full_rle_iou(probe_rle[arm][rank],gt)
                        record['cosine_'+arm], record['zero_'+arm] = cosine[(arm,rank)], zero[(arm,rank)]
                    direction_rows.append(record)
                if iid in held:
                    outputs = {'N':rows}
                    for arm, key in (('R','M'), ('P','S')):
                        decoded = decode_native(cache, deploy_coefficients(directions[arm], ctx, contracts['REAL']['amplitude']))
                        outputs[key] = [{**r, 'segmentation':encode_rle(decoded[j]) if int(masks.area(r['segmentation']))>=9216 else r['segmentation']} for j,r in enumerate(rows)]
                        compare_outputs(rows,outputs[key])
                        del decoded
                    rank_of = {g:j for j,g in enumerate(cache['prediction_indices'])}
                    for old in historic_pairs[iid]:
                        rank = rank_of[int(old['baseline_output_detection_id'].rsplit(':',1)[1])]
                        gt = coco.annToRLE(coco.anns[old['annotation_id']])
                        require(abs(full_rle_iou(rows[rank]['segmentation'],gt)-old['baseline_iou'])<1e-12,'fixed pair changed')
                        pair_rows.append(dict(image_id=iid, annotation_id=old['annotation_id'], population=old['population'],
                                              output_detection_id=old['baseline_output_detection_id'],N=old['baseline_iou'],G=old['G_iou'],
                                              M=full_rle_iou(outputs['M'][rank]['segmentation'],gt),S=full_rle_iou(outputs['S'][rank]['segmentation'],gt)))
                    for key in predictions:
                        predictions[key].extend(outputs[key])
                    mapping.extend(cache['prediction_indices'])
                    del outputs
                if pos % 50 == 0 or pos == len(ids):
                    run.update(images_examined=pos, adapter_forward_calls=forwards)
                    print('EVALUATION',pos,'/',len(ids),flush=True)
                del cache, rows, raw, directions, ctx, probe_rle
        require(len(mapping)==42167 and len(set(mapping))==len(mapping), 'heldout output population differs')
        # Restore original JSON ordering, including score ties, before COCO matching.
        order = sorted(range(len(mapping)), key=mapping.__getitem__)
        mapping = [mapping[j] for j in order]
        native_sha = source.manifest['predictions_sha256']
        ledgers = {}
        for key in predictions:
            predictions[key] = [predictions[key][j] for j in order]
            write_new(run.path / f'{key}_predictions.json', predictions[key])
            ledgers[key] = ledger(run.path,predictions[key],sorted(held),mapping,native_sha)
            jsonl(run.path / f'{key}_GT_LEDGER.jsonl',ledgers[key])
            run.check_budget()
        historic = {(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT / 'BASELINE_GT_LEDGER.jsonl') if r['image_id'] in held}
        for r in ledgers['N']:
            h = historic[(r['image_id'],r['annotation_id'])]
            for key in ('is_ignored','task_mask75','task_box75','bbox75_match_output_id','segm75_match_output_id'):
                require(r[key]==h[key], 'native ledger does not reproduce history: '+key)
        lookup = {k:{(r['image_id'],r['annotation_id']):r for r in values} for k,values in ledgers.items()}
        old_tasks = {(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT / 'FIT_TASKS.jsonl') if r['image_id'] in held}
        tasks = []
        for identity, n in lookup['N'].items():
            for k in ('M','S'):
                q = lookup[k][identity]
                require(q['is_ignored']==n['is_ignored'] and q['bbox75_match_output_id']==n['bbox75_match_output_id'] and q['task_box75']==n['task_box75'], 'Box/ignore guard failed')
            if not n['is_ignored']:
                tasks.append(dict(image_id=identity[0],annotation_id=identity[1],G=old_tasks[identity]['G_success'],
                                  **{k:lookup[k][identity]['task_mask75']=='success' for k in lookup}))
        jsonl(run.path / 'DIRECTION_ROWS.jsonl',direction_rows)
        jsonl(run.path / 'FIXED_PAIRS.jsonl',pair_rows)
        jsonl(run.path / 'TASKS.jsonl',tasks)
        write_new(run.path / 'ROW_MAPPING.json',dict(global_indices=mapping,source_prediction_sha256=native_sha))
        summary, draws, image_order = summarize(direction_rows,pair_rows,tasks)
        np.savez(run.path / 'BOOTSTRAP_INDICES.npz', **draws)
        write_new(run.path / 'STATISTICAL_IMAGE_ORDER.json',image_order)
        source.verify()
        inspect_inputs()
        require(all(sha256(paths[k]/'last.pt')==checkpoints[k] for k in paths),'checkpoint changed')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',heldout_images=400,heldout_rows=42167,
                    complete_mixed_predictions=True,nonmask_identity_and_bbox_matching_unchanged=True,native_ledger_reproduced=True,
                    two_full_last_checkpoints_frozen_before_heldout_access=True,new_optimizer_steps=0,base_model_forward_calls=0,
                    adapter_forward_calls=forwards,scientific_scope='internal_only_no_S2_or_full_AP'))
        run.update(adapter_forward_calls=forwards)
        run.finish(summary,status=summary['decision'],scientific='internal_development_screen',next_action='Independent verification before any scientific report; no automatic S2')


if __name__=='__main__':
    main()
