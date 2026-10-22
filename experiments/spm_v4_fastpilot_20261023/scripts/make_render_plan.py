import hashlib
import json
from pathlib import Path
import random
import subprocess

root = Path(__file__).resolve().parents[1]
parent = root.parent/'spm_farm_scenes_20261023'
def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
all_scenes = json.loads((parent/'SCENES.json').read_text(encoding='utf-8'))['formal']
selected = sorted(all_scenes, key=lambda r: hashlib.sha256(('v4/'+r['geometry_id']).encode()).hexdigest())[:64]
scenes = []
for i, r in enumerate(selected):
    rng = random.Random(231023+i)
    r.update(geometry_id=f'V4G_{i:04d}', index=i, split='dev' if i < 8 else 'train',
             framing=['full_pen', 'group_close', 'group_close', 'contact_close'][i % 4],
             framing_distance_factor=[.92, .72, .64, .57][i % 4])
    if r['camera_family'] == 'oblique':
        r['camera_elevation_degrees'] = rng.uniform(28, 46)
    scenes.append(r)
dump(root/'SCENES.json', {'formal': scenes, 'smoke': scenes[:1]})
cfg = json.loads((parent/'CONFIG.json').read_text(encoding='utf-8'))
cfg.update(study_id='STUDY_SPM_V4_FASTPILOT_20261023', formal_geometry_count=64,
           scenes_sha256=sha(root/'SCENES.json'))
dump(root/'CONFIG.json', cfg)
files = ['PROTOCOL.md', 'CONFIG.json', 'SCENES.json', 'REAL_TRAIN_PROFILE.json', 'ASSET_INSPECTION.json',
         'scripts/render_farm.py', 'scripts/run_render.py', 'scripts/render.sbatch', 'scripts/make_render_plan.py']
dump(root/'FREEZE_MANIFEST.json', {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                 'files': {name: sha(root/name) for name in files}})
print('V4_render_plan',len(scenes),'geometries',len(scenes)*2,'images')
