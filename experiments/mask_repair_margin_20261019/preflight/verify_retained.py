"""Verify retained evidence after a protocol stop; no new solving or scientific estimates."""
from pathlib import Path
import sys, time
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
from common import *


def execute(run, parent):
    import collections
    import numpy as np
    import torch
    from pycocotools import mask as mask_api
    from adapter_data import FrozenCache
    from native_ops import Decoder
    from native_output_bridge import decode_native, encode_rle, native_environment
    from margin_math import draw
    inputs_check()
    profile, environment = configure_native()
    require(environment == read_json(STUDY / 'PREPARATION.json')['environment'], 'environment drift')
    cfg = read_json(parent / 'CONFIG.json')
    metadata = read_json(parent / 'run.json')
    require(metadata['execution_status'] == 'error', 'verify only the stopped run')
    rows = lines(parent / 'PAIR_RESULTS.jsonl')
    original = sorted(lines(C_FULL / 'PAIR_RESULTS.jsonl'), key=lambda x: (x['image_id'], x['annotation_id']))
    require(0 < len(rows) < len(original), 'partial denominator')
    require(len(rows) == metadata['completed_pairs'], 'saved progress differs')
    for row, old in zip(rows, original):
        require(all(row[k] == old[k] for k in ['image_id', 'annotation_id', 'output_detection_id', 'rank', 'global_index', 'raw_id', 'cache_sha256', 'GT_rle', 'c0']), 'source identity')
        require(row['C_coefficients'] == old['arms']['C']['coefficients_fp32'], 'source C coefficients')
    run.freeze({'purpose': 'verify_retained_partial_evidence_only', 'source_run': parent.name,
                'environment': environment, 'config': cfg, 'protocol_scientific_gates_evaluated': False,
                'helper_sha256': sha256(Path(__file__)), 'network_training_steps': 0, 'new_coefficient_solves': 0})
    shutil.copyfile(Path(__file__), run.path / 'verification_source.py')
    rle_checks = full_checks = 0
    amplitude_errors = []
    center_errors = []
    first_masks_exact = 0
    library = np.load(FIT / 'FIT_ERRORS.npz')
    source = FrozenCache(BASE / 'trace', sorted({x['image_id'] for x in rows}))
    with torch.inference_mode():
        for iid in source.image_ids:
            resource_check(run)
            cache, native_rows = source.load(iid)
            for row in [r for r in rows if r['image_id'] == iid]:
                rank = row['rank']
                require(cache['prediction_indices'][rank] == row['global_index'], 'final detection identity')
                require(native_rows[rank]['segmentation'] == row['endpoints']['N']['rle'], 'native RLE identity')
                gt = mask_api.decode(row['GT_rle']).astype(bool)
                measured = list(row['endpoints'].values()) + [p[k] for p in row['pressure'] for k in ['C', 'W']]
                for item in measured:
                    if item is None:
                        continue
                    binary = mask_api.decode(item['rle']).astype(bool)
                    intersection = int(np.count_nonzero(binary & gt))
                    union = int(np.count_nonzero(binary | gt))
                    require((intersection, union) == (item['intersection'], item['union']), 'RLE counts')
                    require(abs(intersection / union - item['iou']) < 1e-14, 'RLE IoU')
                    rle_checks += 1
                dec = Decoder(cache, rank)
                c = np.asarray(row['C_coefficients'], dtype=np.float32)
                w = np.asarray(row['W_coefficients'], dtype=np.float32)
                _, cm, _ = dec.evaluate(c)
                _, wm, _ = dec.evaluate(w)
                require(np.array_equal(cm, wm), 'first binary mask differs')
                first_masks_exact += 1
                center_errors.append(abs(dec.rms(w)[0] / dec.rms(c)[0] - 1))
                for name, coeff in [('N', cache['c0'][rank]), ('C', torch.from_numpy(c)), ('W', torch.from_numpy(w))]:
                    full = cache['c0'].clone()
                    full[rank] = coeff
                    require(encode_rle(decode_native(cache, full)[rank]) == row['endpoints'][name]['rle'], 'full endpoint replay')
                    full_checks += 1
                vectors, labels = draw(library['errors'], library['mean'], library['image_ids'], iid, rank, cfg)
                require(len(row['pressure']) == 24, 'pressure denominator')
                for index, pressure in enumerate(row['pressure']):
                    require(all(pressure[k] == v for k, v in labels[index].items()) and np.array_equal(vectors[index], pressure['vector']), 'pressure identity')
                    for center, value in [('C', c), ('W', w)]:
                        coeff = np.asarray(pressure[center + '_coefficients'], dtype=np.float32)
                        factor = pressure[center + '_normalization_scale']
                        expected = (value.astype(float) + vectors[index] * (row['pressure_amplitude'] / dec.rms(vectors[index])[0]) * factor).astype(np.float32)
                        require(np.array_equal(coeff, expected), 'pressure construction')
                        err = abs(dec.rms(coeff.astype(float) - value.astype(float))[0] / row['pressure_amplitude'] - 1)
                        amplitude_errors.append(err)
                        require(err <= cfg['numerics']['rms_relative_tolerance'], 'retained pressure RMS')
                        full = cache['c0'].clone()
                        full[rank] = torch.from_numpy(coeff)
                        require(encode_rle(decode_native(cache, full)[rank]) == pressure[center]['rle'], 'full pressure replay')
                        full_checks += 1
                del dec
            run.update(verified_images=source.image_ids.index(iid) + 1, full_native_replays=full_checks)
            torch.cuda.empty_cache()
    require(max(center_errors) <= cfg['numerics']['rms_relative_tolerance'], 'retained center scale')
    source.verify()
    inputs_check()
    require(native_environment() == environment, 'end environment drift')
    fallback = collections.Counter(reason for row in rows for reason in row['fallback_reasons'])
    earlier = STUDY / 'runs/RUN_RSM_FULL_e4a8254fa12149b489b5bccfb54e4b7a'
    prefix = lines(earlier / 'PAIR_RESULTS.jsonl')
    for old, row in zip(prefix, rows):
        require(old['output_detection_id'] == row['output_detection_id'] and old['endpoints'] == row['endpoints'], 'retained prefix endpoints differ')
        for p, q in zip(old['pressure'], row['pressure']):
            require(all(p[k] == q[k] for k in ['C_coefficients', 'W_coefficients', 'C', 'W']), 'retained prefix pressures differ')
    summary = {'status': 'passed_partial_artifacts_only', 'parent_run': parent.name, 'verified_pairs': len(rows),
               'required_pairs': cfg['panel_pairs'], 'verified_images': len(source.image_ids),
               'independent_RLE_checks': rle_checks, 'full_native_replays': full_checks,
               'first_masks_exact': first_masks_exact, 'max_center_RMS_relative_error': max(center_errors),
               'max_pressure_RMS_relative_error': max(amplitude_errors),
               'accepted_centers': sum(x['accepted'] for x in rows), 'fallback_reasons': dict(fallback),
               'proxy_invalid_pairs': sum(not x['proxy_valid'] for x in rows),
               'earlier_run_exact_prefix_pairs': len(prefix), 'full_panel_comparison_valid': False,
               'full_panel_IoU_gain': None, 'full_panel_interval': None, 'G_budget_comparison': None,
               'new_training_steps': 0, 'new_coefficient_solves': 0,
               'stopped_pair_identity': {k: original[len(rows)][k] for k in ['image_id', 'annotation_id', 'rank', 'output_detection_id']}}
    write_new(run.path / 'INDEPENDENT_LAPTOP.json', summary)
    write_new(run.path / 'CONFIG_AUDIT.json', {'status': 'passed', 'source_environment_unchanged': True,
              'input_files_verified': len(read_json(STUDY / 'INPUT_MANIFEST.json')['files']),
              'partial_artifacts_verified': True, 'full_comparison_passed': False, 'new_training_steps': 0})
    run.finish(summary, status='partial_evidence_verified', scientific='ENGINEERING_VERIFICATION_ONLY',
               next_action='Close the stopped experiment without a full-panel scientific estimate or additional Run')


if __name__ == '__main__':
    parent = STUDY / 'runs/RUN_RSM_FULL_f02b00986ed14331871c00f9564d6426'
    require(not list((STUDY / 'runs').glob('RUN_RSM_VERIFY_*')), 'verification already registered')
    run = MarginRun('verify', parent)
    with run:
        execute(run, parent)
