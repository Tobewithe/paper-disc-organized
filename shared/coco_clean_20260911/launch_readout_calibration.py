"""Launch finite S021 oracle calibration diagnostic, no network training."""
import json,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/coco_clean_20260911');name='readout_calibration887_v2_20260912';out=R/'diagnostics'/name
assert not out.exists(),out
log=R/'logs'/f'{name}.log';status=R/'logs'/f'{name}.exit';script=R/'logs'/f'{name}.sh'
script.write_text(f'''#!/bin/bash
set -o pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/pytorch/bin/python {R}/readout_calibration_probe.py --out {out} 2>&1 | tee {log}
code=${{PIPESTATUS[0]}}
printf '%s\\n' "$code" > {status}
exit "$code"
''')
subprocess.run(['screen','-dmS',name,'bash',str(script)],check=True)
print(json.dumps(dict(screen=name,out=str(out),log=str(log),exit_path=str(status))))
