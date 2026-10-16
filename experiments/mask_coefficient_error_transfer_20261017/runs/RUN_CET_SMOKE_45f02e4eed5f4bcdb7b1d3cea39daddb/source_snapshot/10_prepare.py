"""Local metadata partition and pre-run freeze. Does not evaluate pressure outcomes."""
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def put(p,v):
    data=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
    if p.exists():assert p.read_bytes()==data, str(p)
    else:p.write_bytes(data)


def main():
    assert not (STUDY/'EXECUTION_FREEZE.json').exists(),'already frozen'
    cld=ROOT/'experiments/mask_direction_learnability_20261016';inputs=read(cld/'INPUT_MANIFEST.json')
    split=read(cld/'SPLIT.json');(STUDY/'SPLIT.json').write_bytes((cld/'SPLIT.json').read_bytes())
    teacher=ROOT/'experiments/mask_attainable_teacher_20261016/runs/RUN_AMT_TEACHER_FULL_24a4d184c8e54f45bf4327b0ab4710ca'
    original=[json.loads(l) for l in (teacher/'TEACHER_ROWS.jsonl').open(encoding='utf8')]
    counts={}
    for key,expected in [('fit',2510),('heldout',581)]:
        ids=set(split[key+'_image_ids']);rows=[r for r in original if r['image_id'] in ids and r['accepted']]
        assert len(rows)==expected
        (STUDY/(key.upper()+'_TEACHERS.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf8')
        counts[key]={'pairs':len(rows),'images':len({r['image_id'] for r in rows})}
    old=ROOT/'experiments/mask_soft_fit_audit_20261016/runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770'
    g=next(x for x in read(old/'ARTIFACT_MANIFEST.json')['artifacts'] if x['relative_path']=='G_predictions.json')
    files=[]
    selected=[x for x in inputs['files'] if Path(x['path']).name in ['TRACE_MANIFEST.json','TEACHER_ROWS.jsonl','TEACHER_MANIFEST.json','SPLIT_CONTRACT.json','baseline_profile.json']]
    for x in selected:
        assert sha(ROOT/x['path'])==x['sha256'];files.append(dict(path=x['path'],sha256=x['sha256'],size_bytes=(ROOT/x['path']).stat().st_size))
    for p in [old/'ROW_MAPPING.json',old/'ARTIFACT_MANIFEST.json',cld/'INHERITED_CONTRACT.json',cld/'SPLIT.json',teacher/'SUMMARY.json']:
        files.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size))
    put(STUDY/'INPUT_MANIFEST.json',dict(files=files,counts=counts,annotations_sha256=inputs['evaluation_annotations']['sha256'],
        g_predictions_sha256=g['sha256'],g_predictions_size_bytes=g['size_bytes'],
        original_annotation_path=inputs['source_annotations']['executor_path'],original_annotation_sha256=inputs['source_annotations']['sha256'],
        note='Tensor availability and byte identity rechecked on authorized laptop before calculation'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    (STUDY/'PREEXISTING_TRACKED_DIFF.patch').write_bytes(diff)
    put(STUDY/'EXECUTION_CONTEXT.json',dict(created_at=datetime.now(timezone.utc).isoformat(),code_commit=commit,tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
        note='Preexisting tracked state retained; exact new and dependency source bytes separately frozen'))
    sources=[p for p in STUDY.rglob('*') if p.is_file() and p.suffix in ['.py','.ps1','.md','.json','.jsonl'] and p.name not in ['study.json','EXECUTION_FREEZE.json']]
    deps=['shared/evaluation/native_output_bridge.py','shared/evaluation/native_source_pins.json','shared/evaluation/posttrain_audit.py',
          'experiments/mask_readout_mechanism_v2_20261012/scripts/posttrain_run.py','experiments/mask_readout_mechanism_v2_20261012/scripts/adapter_data.py',
          'experiments/mask_readout_mechanism_v2_20261012/scripts/s1_geometry.py','paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']
    sources += [ROOT/p for p in deps]
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}
    put(STUDY/'EXECUTION_FREEZE.json',dict(created_at=datetime.now(timezone.utc).isoformat(),files=frozen,code_commit=commit,before_any_new_computation_Run=True))
    put(STUDY/'study.json',dict(study_id='STUDY_MASK_COEFFICIENT_ERROR_TRANSFER_20261017',title='真实系数误差结构的跨图迁移性核验',research_line='neighbor_sensitive_p3',
        kind='zero_training_diagnostic',protocol='PROTOCOL.md',plan_status='frozen',execution_status='not_started',scientific_status='unadjudicated',
        artifact_status='protocol_and_implementation',transfer_status='pending_laptop_execution',authorization='User explicitly authorized this laptop diagnostic, no training',
        new_training_steps=0,automatic_next_stage=False))
    print(json.dumps(dict(status='frozen',counts=counts,source_files=len(frozen),protocol_sha256=sha(STUDY/'PROTOCOL.md')),ensure_ascii=False))

if __name__=='__main__':main()
