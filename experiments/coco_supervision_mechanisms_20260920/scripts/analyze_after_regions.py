import json,subprocess,time
from pathlib import Path
p=Path('/root/coco_supervision_mechanisms_20260920/runs/RUN_a650886c61b9466cadb0a5499ebf9d79/run.json')
while True:
 d=json.loads(p.read_text())
 if d["status"]=="completed":break
 if d["status"]=="failed":raise SystemExit("Probe failed; analysis not launched")
 time.sleep(5)
raise SystemExit(subprocess.call(['/root/miniconda3/bin/python', '/root/coco_supervision_mechanisms_20260920/runner.py', '--study', 'STUDY_3c1ec86ea1ac42fcbe080285570de10d', '--run-id', 'RUN_07d1a7e78f5a4e6e86622a70e0c832e5', '--output', '/root/coco_supervision_mechanisms_20260920/runs/RUN_07d1a7e78f5a4e6e86622a70e0c832e5', '--cwd', '/root/coco_supervision_mechanisms_20260920', '--snapshot', 'scripts/analyze_error_regions.py', '--snapshot', 'scripts/analyze_parameter_probe.py', '--input', 'runs/RUN_a650886c61b9466cadb0a5499ebf9d79/instances.jsonl', '--input', 'runs/RUN_a650886c61b9466cadb0a5499ebf9d79/COMPLETE.json', '--input', 'runs/RUN_ef11b55c15324433bf498805273d3ff4/matched_ids.json', '--expect', '/root/coco_supervision_mechanisms_20260920/runs/RUN_07d1a7e78f5a4e6e86622a70e0c832e5/SUMMARY.json', '--expect', '/root/coco_supervision_mechanisms_20260920/runs/RUN_07d1a7e78f5a4e6e86622a70e0c832e5/error_region_comparison.png', '--', '/root/miniconda3/bin/python', 'scripts/analyze_error_regions.py', '--run', '/root/coco_supervision_mechanisms_20260920/runs/RUN_a650886c61b9466cadb0a5499ebf9d79', '--pairs', '/root/coco_supervision_mechanisms_20260920/runs/RUN_ef11b55c15324433bf498805273d3ff4/matched_ids.json', '--out', '/root/coco_supervision_mechanisms_20260920/runs/RUN_07d1a7e78f5a4e6e86622a70e0c832e5']))
