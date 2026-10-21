"""Finalize receipts and artifact hashes; does not alter experimental scores or gates."""
from pathlib import Path
import shutil
from .common import Run,read_json,write_json,sha256,now


def main(root):
    with Run(root,'final_receipt_audit',dict(new_training=0,scope='source and contract hashes; recover exactly matching manifest snapshots; final artifacts')) as audit:
        candidates=[root/'input_manifest.json']+list((root/'runs').glob('*/input_manifest_*.json'))
        byhash={sha256(p):p for p in candidates};rows=[]
        for folder in sorted((root/'runs').iterdir()):
            if not folder.is_dir() or folder==audit.path:continue
            meta=read_json(folder/'run.json');assert meta['execution_status'] in {'completed','failed'}
            assert sha256(folder/'EVAL_CONTRACT.json')==meta['contract_sha256']
            assert sha256(folder/'SOURCE_SNAPSHOT.json')==meta['source_snapshot_sha256']
            for item in read_json(folder/'SOURCE_SNAPSHOT.json'):assert sha256(folder/'source'/item['path'])==item['sha256']
            expected=meta.get('input_manifest.json_sha256');snapshot=byhash.get(expected)
            if snapshot:
                dest=folder/'inputs/input_manifest.json';dest.parent.mkdir(exist_ok=True);shutil.copy2(snapshot,dest);assert sha256(dest)==expected
            rows.append(dict(run_id=meta['run_id'],execution_status=meta['execution_status'],source_and_contract_hashes=True,manifest_exact_snapshot_recovered=bool(snapshot),recovery_source=str(snapshot) if snapshot else None))
            if not (folder/'DECISION.json').exists():
                write_json(folder/'DECISION.json',dict(status='engineering_failed' if meta['execution_status']=='failed' else 'completed_see_whole_experiment_decision',adjudicated_at=now(),whole_experiment='REPORT.md',decision_pointer=read_json(root/'evidence_pointer.json'),scientific_scope='no extra efficacy claim from completion alone'))
        artifacts=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted((root/'showcase').rglob('*')) if p.is_file()]
        write_json(root/'SHOWCASE_MANIFEST.json',artifacts);write_json(audit.path/'ARTIFACT_MANIFEST.json',artifacts)
        write_json(audit.path/'SUMMARY.json',dict(run_checks=rows,artifact_count=len(artifacts),all_source_contract_hashes_passed=True,manifest_recovery_is_late_and_hash_exact=True,network_service_started=False))
        write_json(audit.path/'DECISION.json',dict(status='evidence_verified',next_action='deliver measured local gains and limitations; no further training in closed configurations'))
        write_json(root/'final_audit_pointer.json',dict(run_id=audit.id,path=audit.path.relative_to(root).as_posix()))
    study=read_json(root/'study.json');runs=[]
    for p in sorted((root/'runs').glob('*/run.json')):
        r=read_json(p);runs.append(dict(run_id=r['run_id'],stage_id=r['stage_id'],execution_status=r['execution_status'],scientific_status=r['scientific_status'],receipt='experiments/dense_video_association_v2_20261018/runs/'+r['run_id']+'/run.json'))
    study.update(plan_status='fixed_candidates_closed_followup_not_started',execution_status='completed',scientific_status='limited_zebra12_utility_supported_general_superiority_and_supervision_increment_not_established',artifact_status='reports_figures_videos_and_import_pipeline_verified_live_service_not_started',transfer_status='laptop_complete_desktop_mirror_verifying',runs=runs,
                 report='REPORT.md',decisions=read_json(root/'showcase/results.json')['gates'],annotation_scope='143 usable sparse AI-reviewed records; not human dense GT; zero verified full physical occlusion events',
                 next_action='Verify and roll back false identity reattachments on fixed motion trajectories; not started',ui_block='Automatic review rejected new background service/SSH forwarding; browser rejects file URL preview',updated_at=now())
    write_json(root/'study.json',study);shutil.copy2(root/'showcase/REPORT.md',root/'REPORT.md')
    print('FINAL_AUDIT',dict(runs=len(runs),failed=sum(r['execution_status']=='failed' for r in runs),artifacts=len(artifacts),manifest_recovered=sum(r['manifest_exact_snapshot_recovered'] for r in rows)),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
