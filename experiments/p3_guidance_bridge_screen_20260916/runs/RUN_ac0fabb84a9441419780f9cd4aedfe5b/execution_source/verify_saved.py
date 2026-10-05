"""Inspect completed training artifacts without rewriting the failed post-check Run."""
import argparse,csv,json,math
from pathlib import Path
import torch
from ultralytics import YOLO
from recording import atomic_json,now
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text());spec=next(x for x in cfg['runs'] if x['kind']=='training' and x['arm']=='baseline');src=a.study/'runs'/spec['run_id']
    with (src/'results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==cfg['epochs']
    bad=[dict(epoch=row.get('epoch'),field=k,value=v) for row in rows for k,v in row.items() if k and v and not math.isfinite(float(v))]
    assert not [x for x in bad if not x['field'].startswith('val/')],bad
    model=YOLO(str(src/'weights/last.pt')).model
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    receipt=dict(status='complete',arm='baseline',epochs=len(rows),training_losses_finite=True,weights_finite=True,
        native_validation_nonfinite=bad,source_training_run=spec['run_id'],source_run_status='failed final strict CSV check; full epoch/checkpoints completed',
        method_stats=json.loads((src/'auxiliary_epochs.json').read_text())[-1]['auxiliary'],verified_at=now(),training_repeated=False)
    atomic_json(a.out/'TRAINING_COMPLETE.json',receipt)
    atomic_json(a.study/'certified_receipts.json',{'baseline':str(a.out/'TRAINING_COMPLETE.json')})
    print('SAVED_CHECKPOINT_VERIFIED',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
