"""Freeze actual accepted render inputs; called only after visual acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    args = p.parse_args(); root = Path(args.root)
    pointer = json.loads((root/'FORMAL_POINTER.json').read_text())
    render = root/'runs'/pointer['run_id']
    assert json.loads((root/'VISUAL_ACCEPTANCE.json').read_text())['formal_render_authorized_after_smoke_review']
    assert json.loads((render/'run.json').read_text())['execution_status'] == 'completed'
    old = Path('/srv/gpu-lab/users/a25023104/projects/spm-prototype-training-20261022')
    rows = [r for r in json.loads((old/'INPUT_MANIFEST.json').read_text())['rows'] if r['domain'] == 'real']
    synthetic = json.loads((render/'DATA_MANIFEST.json').read_text())['rows']
    for primary in synthetic:
        if primary['appearance_id'] != 0:
            continue
        pair = next(r for r in synthetic if r['group'] == primary['group'] and r['appearance_id'] == 1)
        assert pair['split'] == primary['split'] and pair['label_sha256'] == primary['label_sha256']
        row = dict(primary)
        row.update(paired_rgb_path=pair['rgb_path'], paired_rgb_sha256=pair['rgb_sha256'],
                   source_id='synthetic/'+primary['group'])
        rows.append(row)
    for i, row in enumerate(rows):
        row['ordinal'] = i
    train_groups = {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'train'}
    dev_groups = {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'dev'}
    assert not train_groups & dev_groups
    dump(root/'TRAIN_INPUT_MANIFEST.json', {'parent_manifest_sha256': sha(old/'INPUT_MANIFEST.json'),
                                          'render_run_id': render.name, 'render_manifest_sha256': sha(render/'DATA_MANIFEST.json'),
                                          'same_geometry_pair_counted_once': True, 'rows': rows})
    cfg = json.loads((old/'CONFIG.json').read_text())
    cfg.update(study_id='STUDY_SPM_FARM_SCENES_20261023', tokens_per_class_image=48,
               cluster_method='balanced_fixed_capacity_spherical_kmeans', kmeans_iterations=20,
               parent_projector=str(old/'runs/RUN_SPM_PROTO_TRAIN_2e46d0049349/projector.pt'),
               parent_projector_sha256='4e32f2ca3c1cd8a1998b184a825510a427bd37dc9efa5f68e7377c62034fe939',
               input_manifest_sha256=sha(root/'TRAIN_INPUT_MANIFEST.json'))
    assert sha(cfg['parent_projector']) == cfg['parent_projector_sha256']
    dump(root/'TRAIN_CONFIG.json', cfg)
    files = ['TRAIN_PROTOCOL.md', 'TRAIN_CONFIG.json', 'TRAIN_INPUT_MANIFEST.json',
             'scripts/train_prototypes.py', 'scripts/base_projector.py', 'scripts/memory_query.py', 'scripts/train.sbatch']
    dump(root/'TRAIN_FREEZE_MANIFEST.json', {'files': {name: sha(root/name) for name in files},
                                          'git_head': json.loads((root/'FREEZE_MANIFEST.json').read_text())['git_head'],
                                          'render_run_id': render.name})
    print('TRAIN_INPUTS_FROZEN', len(rows), sum(r['split'] == 'train' for r in rows), flush=True)


if __name__ == '__main__':
    main()
