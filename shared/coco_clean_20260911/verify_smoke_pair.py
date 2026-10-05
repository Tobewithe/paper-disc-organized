"""Verify paired short official training and per-epoch checkpoint content."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent

def main():
    dirs=[ROOT/'smoke_runs'/f'{arm}_s0' for arm in ['baseline','ccl01']]
    a,b=[json.loads((p/'launch_receipt.json').read_text()) for p in dirs]
    assert a['parameter_sha256']==b['parameter_sha256']
    assert (dirs[0]/'initial_batches.jsonl').read_text()==(dirs[1]/'initial_batches.jsonl').read_text()
    diff={k for k in a['resolved_args'] if a['resolved_args'][k]!=b['resolved_args'][k]}
    assert diff=={'name','save_dir'},diff
    results=[]
    for p in dirs:
        receipt=json.loads((p/'TRAINING_COMPLETE.json').read_text());assert receipt['status']=='PASS'
        ckpt=torch.load(p/'weights/epoch0.pt',map_location='cpu',weights_only=False)
        recovery=torch.load(p/'weights/recovery_latest.pt',map_location='cpu',weights_only=False)
        assert ckpt['epoch']==0 and ckpt['optimizer'] and ckpt['ema'] is not None
        assert recovery['epoch']==0 and recovery['optimizer'] and recovery['model'] and recovery['ema']
        assert all(torch.isfinite(v).all() for v in recovery['model'].values() if v.is_floating_point())
        assert {v.dtype for v in recovery['model'].values() if v.is_floating_point()}=={torch.float32}
        rows=[json.loads(s) for s in (p/'epoch_integrity.jsonl').read_text().splitlines()]
        assert len(rows)==1 and rows[0]['optimizer_steps']>0
        if p.name.startswith('ccl'):
            assert rows[0]['o2m']['active_pairs']>0 and rows[0]['o2o']['active_pairs']>0
        results.append(dict(name=p.name,epoch0_bytes=(p/'weights/epoch0.pt').stat().st_size,recovery_bytes=(p/'weights/recovery_latest.pt').stat().st_size,
            optimizer_steps=rows[0]['optimizer_steps'],o2m=rows[0]['o2m'],o2o=rows[0]['o2o']))
    assert results[0]['optimizer_steps']==results[1]['optimizer_steps']
    result=dict(status='PASS',initial_parameters_equal=True,initial_two_batches_equal=True,config_differences=sorted(diff),runs=results,
        note='64 train images, 32 val images, 1 epoch: execution-only smoke, not method efficacy. Minimum warmup iterations mean a single optimizer step in this deliberately short smoke.')
    (ROOT/'audits/SMOKE_PAIR_PASS.json').write_text(json.dumps(result,indent=2))
    print('SMOKE_PAIR_PASS',json.dumps(result),flush=True)

if __name__=='__main__':main()
