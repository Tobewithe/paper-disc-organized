"""Archive raw-float census, rebuild stable-boundary tables from original JSON."""
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parent
src=root/'census';archive=root/'census_raw_float_archive'
if archive.exists():raise RuntimeError('Raw archive already exists; inspect before rerunning')
shutil.copytree(src,archive)
subprocess.run([sys.executable,str(root/'coco_crowding_census.py')],check=True)
print('BOUNDARY_CORRECTED_CENSUS_PASS',flush=True)
