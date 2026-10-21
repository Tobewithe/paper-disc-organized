#!/bin/bash
set -euo pipefail
ROOT=/srv/gpu-lab/users/a25023104/projects/yoloseg-blender-v2-20261020
export PYTHONPATH=/srv/gpu-lab/users/a25023104/projects/lcf-server-20261020/_site
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
"$PY" - <<'PY'
import ast,pathlib,importlib.metadata,json
for p in pathlib.Path('.').rglob('*.py'):
    ast.parse(p.read_text())
from scripts.evaluate import associate
g=[{'bbox':[0,0,10,10]},{'bbox':[20,0,30,10]}]
p=[{'bbox':[20,0,10,10]},{'bbox':[0,0,10,10]}]
assert associate(g,p)=={0:1,1:0}
assert associate(g,[])=={}
print(json.dumps({'status':'syntax_and_matching_passed','gpu_work':False,
 'scipy':importlib.metadata.version('scipy')}))
PY
"$PY" scripts/prepare.py
