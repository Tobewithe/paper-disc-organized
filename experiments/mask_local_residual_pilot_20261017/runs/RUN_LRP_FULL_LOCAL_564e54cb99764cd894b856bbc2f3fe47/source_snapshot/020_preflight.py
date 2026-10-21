"""Actual cache/teacher replay and fit-only scale derivation. No student fitting."""
import statistics
import subprocess
import sys
import torch
from cld_common import *
from cld_math import target
from adapter_data import FrozenCache
from native_output_bridge import decode_native, encode_rle


def main():
    require(read_json(STUDY / 'OPERATION_AUTHORIZATION.json').get('allow_tensor_preflight') is True, 'tensor preflight not authorized')
    with Run('PREFLIGHT', CACHE, 3600) as run:
        data = inspect_inputs()
        ids = data['split']['fit_image_ids']
        run.freeze(dict(stage='engineering_tensor_preflight', image_ids=ids, teacher_source=TEACHER.name,
                        environment=data['environment'], model_forward_calls=0, optimizer_steps=0,
                        scope='fit_only_teacher_amplitude_and_native_replay_no_heldout_model_evaluation'))
        tested = subprocess.run([sys.executable, '-B', str(STUDY / 'scripts/test_cld.py')], capture_output=True, text=True, encoding='utf-8', timeout=180)
        write_new(run.path / 'BEHAVIOR_TESTS.json', dict(returncode=tested.returncode, stdout=tested.stdout, stderr=tested.stderr))
        require(tested.returncode == 0, 'functional/derivative tests failed')
        teachers = teacher_rows(ids)
        require(sum(map(len, teachers.values())) == 2510, 'fit teacher count mismatch')
        source = FrozenCache(CACHE / 'trace', ids)
        scale_rows, replay = [], []
        with torch.inference_mode():
            for count, iid in enumerate(ids, 1):
                run.check_budget()
                cache, rows = source.load(iid)
                native = decode_native(cache)
                require(all(encode_rle(native[j]) == row['segmentation'] for j, row in enumerate(rows)), 'native RLE replay differs')
                del native
                matrix = cache['c0'].clone()
                for teacher in teachers[iid]:
                    require(teacher['cache_sha256'] == source.records[iid]['cache_sha256'], 'teacher cache identity mismatch')
                    target_grid, scale = target(cache, teacher)
                    matrix[teacher['rank']] = torch.tensor(teacher['coefficients_fp32'], dtype=torch.float32)
                    scale_rows.append(dict(image_id=iid, annotation_id=teacher['annotation_id'], rank=teacher['rank'],
                                           output_detection_id=teacher['output_detection_id'], scale=scale,
                                           target_norm_error=abs(float(target_grid.double()[__import__('s1_geometry').predicted_roi(cache, teacher['rank'])].square().mean()) - 1.)))
                    del target_grid
                if teachers[iid]:
                    changed = decode_native(cache, matrix)
                    require(all(encode_rle(changed[t['rank']]) == t['oracle_rle'] for t in teachers[iid]), 'accepted teacher native RLE replay differs')
                    del changed
                replay.append(dict(image_id=iid, native_rows=len(rows), teacher_rows=len(teachers[iid]), native_RLE_exact=True, teacher_RLE_exact=True))
                if count % 100 == 0 or count == len(ids):
                    run.update(images_completed=count, teacher_rows=len(scale_rows))
                    print('PREFLIGHT', count, '/', len(ids), 'teachers', len(scale_rows), flush=True)
                del cache, rows, matrix
        require(len(scale_rows) == 2510, 'amplitude denominator changed')
        amplitude = statistics.median(r['scale'] for r in scale_rows)
        require(amplitude > 1e-6, 'invalid fit-only common amplitude')
        jsonl(run.path / 'FIT_AMPLITUDES.jsonl', scale_rows)
        jsonl(run.path / 'REPLAY.jsonl', replay)
        write_new(run.path / 'AMPLITUDE.json', dict(value=amplitude, rule='ordinary median of 2510 fit-only FP64 teacher RMS values',
                                                  images_sha256=sha256(STUDY / 'SPLIT.json'), rows_sha256=sha256(run.path / 'FIT_AMPLITUDES.jsonl'),
                                                  teacher_source_sha256=sha256(TEACHER / 'TEACHER_ROWS.jsonl'), teacher_count=2510, heldout_teachers_used=0))
        # Verify all images/caches, including those not used for tensor fitting.
        trace = read_json(CACHE / 'trace/TRACE_MANIFEST.json')
        for record in trace['images']:
            require(sha256(CACHE / 'trace' / record['cache']) == record['cache_sha256'], 'cache bytes drifted')
        for record in read_json(CACHE / 'IMAGE_MANIFEST.json'):
            require(sha256(Path(record['source_image'])) == record['image_sha256'], 'original image drift')
        inspect_inputs()
        write_new(run.path / 'CONFIG_AUDIT.json', dict(status='passed', fit_images=1600, teachers=2510,
                   original_images_verified=2000, cache_files_verified=2000, current_runtime_matches=True,
                   native_and_teacher_RLE_replay=True, heldout_teacher_amplitude_contributions=0,
                   base_model_forward_calls=0, adapter_forward_calls=0, optimizer_steps=0))
        run.check_budget()
        run.finish(dict(fit_images=1600, accepted_teachers=2510, native_rows=sum(r['native_rows'] for r in replay),
                        common_amplitude=amplitude, new_training_steps=0, new_model_forwards=0),
                   scientific='engineering_only', next_action='Independent fit-only amplitude verification, then authorized smoke only')


if __name__ == '__main__':
    main()
