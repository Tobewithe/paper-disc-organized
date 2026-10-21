"""One-time source freeze using stdlib. Does not access a GPU or execute experiments."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    if (STUDY/'EXECUTION_FREEZE.json').exists():
        raise RuntimeError('Execution already frozen; preserve this revision and record any recovery separately')
    scientific=json.loads((STUDY/'SCIENTIFIC_FREEZE.json').read_text(encoding='utf-8'))
    for row in scientific['files']:
        assert digest(STUDY/row['path'])==row['sha256'],row['path']
    stamp=datetime.now(timezone.utc).isoformat()
    patch=subprocess.check_output(['git','diff','--binary','--','AGENTS.md','project/ACTIVE_PLAN.md','project/PLAN_CATALOG.json','.gitattributes'],cwd=ROOT)
    (STUDY/'EXECUTION_TRACKED_CONTEXT.patch').write_bytes(patch)
    write(STUDY/'EXECUTION_CONTEXT.json',dict(created_at=stamp,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          tracked_scoped_diff_sha256=digest(STUDY/'EXECUTION_TRACKED_CONTEXT.patch'),
          untracked_source_identity='Exact execution source bytes are separately listed and copied into each Run; tracked diff is not their identity'))
    write(STUDY/'OPERATION_AUTHORIZATION.json',dict(recorded_at=stamp,study_id='STUDY_MASK_DIRECTION_LEARNABILITY_20261016',
          source='User continued the selected implementation, tensor preflight and engineering smoke step',
          allow_tensor_preflight=True,allow_smoke=True,allow_formal_training=False,allow_full_evaluation=False,
          maximum_smoke_updates_per_arm=2,maximum_smoke_images_per_arm=8,automatic_next_stage=False))
    files=list((STUDY/'scripts').glob('*.py'))+[STUDY/'scripts/launch.ps1',STUDY/'EXECUTION.md',STUDY/'EXECUTION_CONTEXT.json',STUDY/'EXECUTION_TRACKED_CONTEXT.patch',STUDY/'SCIENTIFIC_FREEZE.json']
    files += [STUDY/row['path'] for row in scientific['files']]
    files += [ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'/n for n in ('adapter_model.py','adapter_data.py','adapter_primitives.py','s1_geometry.py','posttrain_run.py')]
    files += [ROOT/p for p in ('shared/evaluation/native_output_bridge.py','shared/evaluation/native_source_pins.json','shared/evaluation/posttrain_audit.py',
                              'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py','paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json')]
    mapping={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(files))}
    write(STUDY/'EXECUTION_FREEZE.json',dict(version='CLD_EXECUTION_V1',frozen_at=stamp,files=mapping,
          scientific_freeze_sha256=digest(STUDY/'SCIENTIFIC_FREEZE.json'),authorization_policy='Operation authorization is recorded verbatim in each Run contract; it is not scientific protocol'))
    print(json.dumps(dict(frozen_files=len(mapping),execution_freeze_sha256=digest(STUDY/'EXECUTION_FREEZE.json'))))


if __name__=='__main__':
    main()
