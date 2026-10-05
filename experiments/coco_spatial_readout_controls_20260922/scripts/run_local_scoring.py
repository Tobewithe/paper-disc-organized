"""Evaluate already generated prediction caches on the local CPU, one arm per process."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

root=Path(__file__).resolve().parent.parent
project=root.parents[1]
source=root/'runs/RUN_4a00a5215db34b6b8702d3099ec7d453'
review=root/'runs/RUN_0143c1c41081430dac938912318b5279/run.json'
annotations=project/'assets/datasets/coco/annotations/instances_val2017.json'
reference=project/'experiments/coco_spatial_calibration_mechanism_20260922/runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66/MATCHED_GT75.json'
arms=json.loads((root/'execution_ids.json').read_text())
order=['global_scalar','global_spatial','native_coeff_mlp','native_quad_coeff_mlp']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    assert not (root/'LOCAL_SCORE_IDS.json').exists()
    assert json.loads((source/'progress.json').read_text())['images']==5000
    verified={Path(row['path']).name:row['sha256'] for row in json.loads(review.read_text())['inputs']
              if Path(row['path']).name.startswith('predictions_')}
    assert len(verified)==4
    for name in order:
        path=source/f'predictions_{name}.jsonl.gz'
        assert digest(path)==verified[path.name],path
        assert (root/'runs'/arms[name]/'SELECTION.json').exists()
    ids={name:'RUN_'+uuid.uuid4().hex for name in order}
    (root/'LOCAL_SCORE_IDS.json').write_text(json.dumps(ids,indent=2),encoding='utf-8')
    study=json.loads((root/'study.json').read_text())['study_id']
    for name in order:
        out=root/'runs'/ids[name]
        pred=source/f'predictions_{name}.jsonl.gz'
        selection=root/'runs'/arms[name]/'SELECTION.json'
        script=root/'scripts/review_cached_predictions.py'
        command=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],
                 '--output',str(out),'--cwd',str(root)]
        for path in [pred,source/'progress.json',source/'RESULTS.json',selection,annotations,reference,review]:
            command+=['--input',str(path)]
        for path in [root/'LOCAL_SCORE_PROTOCOL.json',root/'scripts/run_local_scoring.py',script]:
            command+=['--snapshot',str(path)]
        for file in ['RESULTS.json','MATCHED_GT75.json','COMPLETE.json']:
            command+=['--expect',str(out/file)]
        command+=['--',sys.executable,'-u',str(script),'--source',str(source),
                  '--annotations',str(annotations),'--reference',str(reference),
                  '--arm',name,'--selection',str(root/'runs'/arms[name]),'--out',str(out)]
        (root/'LOCAL_SCORE_STATUS.json').write_text(json.dumps({'status':'running','arm':name,'run_id':ids[name]}),encoding='utf-8')
        subprocess.run(command,check=True)
    (root/'LOCAL_SCORE_STATUS.json').write_text(json.dumps({'status':'completed','run_ids':ids}),encoding='utf-8')


if __name__=='__main__':
    main()
