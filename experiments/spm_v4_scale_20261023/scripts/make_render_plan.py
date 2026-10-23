from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
parent = root.parent/'spm_farm_scenes_20261023'
templates = json.loads((parent/'SCENES.json').read_text(encoding='utf-8'))['formal']
config = json.loads((parent/'CONFIG.json').read_text(encoding='utf-8'))
def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

scenes = []
for i in range(2048):
    rng = random.Random(231023000+i); recipe = deepcopy(templates[i % len(templates)])
    n = [3, 8, 14, 20][i % 4]
    composition = ['active', 'resting', 'feeding_rest', 'mixed'][(i//4) % 4]
    candidates = {'active': ['standing', 'walking', 'feeding'],
                  'resting': ['prone', 'lateral_left', 'lateral_right', 'standing'],
                  'feeding_rest': ['feeding', 'feeding', 'standing', 'prone', 'lateral_left'],
                  'mixed': ['standing', 'walking', 'feeding', 'prone', 'lateral_left', 'lateral_right']}[composition]
    camera = 'overhead' if (i//16) % 2 == 0 else 'oblique'
    framing = ['full_pen', 'group_close', 'group_close', 'contact_close'][(i//32) % 4]
    recipe.update(geometry_id=f'SCALE_G_{i:05d}', index=i, seed=231023000+i,
                  split='dev' if i < 256 else 'train', requested_pigs=n,
                  density_bin=i % 4, composition=composition, poses=[rng.choice(candidates) for _ in range(n)],
                  body_lengths_m=[rng.uniform(1.1, 1.8) for _ in range(n)], pose_frame=[rng.randint(1, 60) for _ in range(n)],
                  layout=['scatter', 'rest_rows', 'feeder_cluster', 'fence_strip'][(i//128) % 4],
                  floor_type=['concrete', 'slatted', 'bedded', 'stained_concrete'][(i//512) % 4],
                  camera_family=camera, camera_elevation_degrees=rng.uniform(68, 87) if camera == 'overhead' else rng.uniform(28, 49),
                  camera_azimuth_degrees=rng.uniform(0, 360), lens_mm=rng.uniform(30, 48),
                  pen_dimensions_m=[9.4, 7.1] if n == 20 else [7.6, 6.2], framing=framing,
                  framing_distance_factor={'full_pen': rng.uniform(.88, .98), 'group_close': rng.uniform(.64, .78), 'contact_close': rng.uniform(.55, .65)}[framing],
                  appearances=[{'id': 0, 'seed': 331023000+i, 'light_type': 'overhead_soft', 'camera_quality': 'clean'},
                               {'id': 1, 'seed': 431023000+i, 'light_type': 'side_window', 'camera_quality': ['compressed', 'mild_blur'][i % 2]}])
    scenes.append(recipe)
dump(root/'SCENES.json', {'formal': scenes, 'smoke': scenes[:8]})
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
for name, selected in [('smoke', scenes[:8])]+[(f'shard_{k}', scenes[k::3]) for k in range(3)]:
    child = root/'render_plans'/name; (child/'scripts').mkdir(parents=True, exist_ok=True)
    cfg = {**config, 'study_id': 'STUDY_SPM_V4_SCALE_20261023', 'formal_geometry_count': len(selected),
           'total_scale_geometry_count': 2048, 'render_shard': name}
    dump(child/'CONFIG.json', cfg); dump(child/'SCENES.json', {'formal': selected, 'smoke': selected[:8]})
    for source in ['PROTOCOL.md', 'ASSET_INSPECTION.json', 'REAL_TRAIN_PROFILE.json', 'scripts/render_farm.py', 'scripts/run_render.py']:
        shutil.copy2(root/source, child/source)
    files = ['PROTOCOL.md', 'CONFIG.json', 'SCENES.json', 'ASSET_INSPECTION.json', 'REAL_TRAIN_PROFILE.json', 'scripts/render_farm.py', 'scripts/run_render.py']
    dump(child/'FREEZE_MANIFEST.json', {'git_head': head, 'files': {name: sha(child/name) for name in files}})
dump(root/'CONFIG.json', {**config, 'study_id': 'STUDY_SPM_V4_SCALE_20261023', 'formal_geometry_count': 2048})
print('SCALE_RENDER_PLAN', 2048, 'geometries', 4096, 'appearances', 'shards', [len(scenes[k::3]) for k in range(3)])
