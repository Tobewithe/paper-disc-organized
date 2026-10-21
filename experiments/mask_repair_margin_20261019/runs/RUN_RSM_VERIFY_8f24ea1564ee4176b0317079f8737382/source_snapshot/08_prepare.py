"""Read-only input verification followed by the immutable execution freeze."""
from common import *
import argparse, importlib.metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--test-report',required=True);args=p.parse_args()
    require(not (STUDY/'FREEZE_MANIFEST.json').exists(),'already frozen')
    test=read_json(args.test_report);require(test['status']=='passed','synthetic tests failed')
    require(test['source_hashes']==science_files(),'tested source bytes changed')
    resources=resource_check();profile,env=configure_native()
    require(env==read_json(RAY/'EVAL_CONTRACT.json')['environment'],'source native environment mismatch')
    panel=read_json(S1/'TARGET_PANEL.json')['targets']; full=lines(C_FULL/'PAIR_RESULTS.jsonl');smoke=lines(C_SMOKE/'PAIR_RESULTS.jsonl')
    require(sha256(S1/'TARGET_PANEL.json')==read_json(ROOT/'experiments/mask_adapter_ray_audit_20261016/CONFIG.json')['panel']['sha256'],'panel hash')
    require(len(full)==256 and len({r['image_id'] for r in full})==243 and len(smoke)==4,'panel counts')
    require({(r['output_detection_id'],r['annotation_id']) for r in full}=={(r['output_detection_id'],r['annotation_id']) for r in panel},'panel identities')
    require(sum(r['arms']['C']['original_iou']>=.75 for r in full)==159,'C witness source changed')
    ids=sorted({r['image_id'] for r in full+smoke})
    require(not ({r['image_id'] for r in full}&{r['image_id'] for r in smoke}),'smoke overlaps main panel')
    trace=read_json(BASE/'trace/TRACE_MANIFEST.json')
    records=[r for r in trace['images'] if r['image_id'] in ids]
    require(len(records)==len(ids),'missing cache')
    files=[S1/'TARGET_PANEL.json',C_FULL/'PAIR_RESULTS.jsonl',C_SMOKE/'PAIR_RESULTS.jsonl',OLD/'PAIRED_MASKS.jsonl',
           FIT/'FIT_ERRORS.npz',FIT/'FIT_MODEL.json',FIT/'FIT_IDENTITIES.json',BASE/'trace/TRACE_MANIFEST.json',
           Path(trace['predictions_path']),GT,ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt']
    for row in records:
        cache=BASE/'trace'/row['cache'];require(sha256(cache)==row['cache_sha256'],'cache hash')
        image=ROOT/'assets/datasets/coco_native_s0_20261013/images/val2017'/f"{row['image_id']:012d}.jpg"
        require(sha256(image)==row['image_sha256'],'image hash');files.extend([cache,image])
    require(sha256(files[-1]) is not None,'readability')
    require(sha256(FIT/'FIT_ERRORS.npz')==read_json(FIT/'FIT_MODEL.json')['library_sha256'],'donor library hash')
    require(sha256(ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt')==profile['model_sha256'],'base weight hash')
    for source in [BASE,S1,C_FULL,C_SMOKE,FIT,OLD]:require(read_json(source/'run.json')['execution_status']=='completed','incomplete source')
    manifest={'files':[{'path':x.relative_to(ROOT).as_posix(),'sha256':sha256(x),'size_bytes':x.stat().st_size} for x in files],
              'cache_records':records,'full_panel_pairs':256,'full_panel_images':243,'G_budget_same_panel_available':False}
    write_new(STUDY/'INPUT_MANIFEST.json',manifest)
    preparation={'status':'passed','at':now(),'environment':env,'profile':profile,'native_args':trace['native_args'],
                 'resources':resources,'additional_packages':{p:importlib.metadata.version(p) for p in ['scipy','clarabel','psutil']},
                 'smoke_image_ids':sorted({r['image_id'] for r in smoke}),'full_image_ids':sorted({r['image_id'] for r in full}),
                 'scientific_computation_started':False,'test_report_sha256':sha256(Path(args.test_report)),
                 'G_budget_reference_status':'historical_only_checkpoint_unavailable'}
    write_new(STUDY/'PREPARATION.json',preparation)
    freeze={'frozen_at':now(),'files':science_files(),'input_manifest_sha256':sha256(STUDY/'INPUT_MANIFEST.json'),
            'preparation_sha256':sha256(STUDY/'PREPARATION.json')}
    write_new(STUDY/'FREEZE_MANIFEST.json',freeze)
    study=read_json(STUDY/'study.json');study.update(plan_status='frozen',execution_status='not_started',current_phase='smoke_ready',
          protocol_sha256=sha256(STUDY/'PROTOCOL.md'),frozen_at=freeze['frozen_at'],artifact_status='frozen_inputs_and_code')
    save(STUDY/'study.json',study)
    print(json.dumps({'status':'frozen','files':len(files),'protocol_sha256':study['protocol_sha256']}))


if __name__=='__main__':main()
