import json
import shutil
import zipfile
from pathlib import Path

root = Path('/root/position_objective_complement_20261004')
split = json.loads((root / 'assets/OFFICIAL_SPLIT.json').read_text())
data = root / 'data'
(data / 'annotations').mkdir(parents=True, exist_ok=True)
(data / 'images/train2017').mkdir(parents=True, exist_ok=True)
(data / 'images/val2017').mkdir(parents=True, exist_ok=True)
src = Path('/autodl-pub/data/COCO2017')

with zipfile.ZipFile(src / 'annotations_trainval2017.zip') as z:
    for name in ('annotations/instances_train2017.json', 'annotations/instances_val2017.json'):
        out = data / name
        if not out.exists():
            with z.open(name) as f, out.open('wb') as g:
                shutil.copyfileobj(f, g)

groups = {'train2017': set(split['fit'] + split['dev']), 'val2017': set(split['val'])}
for folder, wanted in groups.items():
    outdir = data / 'images' / folder
    archive = src / ('train2017.zip' if folder == 'train2017' else 'val2017.zip')
    with zipfile.ZipFile(archive) as z:
        names = {int(Path(n).stem): n for n in z.namelist() if n.startswith(folder + '/') and n.lower().endswith('.jpg')}
        missing = sorted(wanted - set(names))
        if missing:
            raise RuntimeError(f'{folder} missing {missing[:10]} total={len(missing)}')
        for iid in sorted(wanted):
            out = outdir / f'{iid:012d}.jpg'
            if not out.exists():
                with z.open(names[iid]) as f, out.open('wb') as g:
                    shutil.copyfileobj(f, g)

print(json.dumps({k: len(v) for k, v in groups.items()}))
