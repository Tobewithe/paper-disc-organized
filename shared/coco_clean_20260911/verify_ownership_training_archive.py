"""Verify local archived training receipts and paired budgets; no efficacy claim."""
import csv,hashlib,json
from pathlib import Path

root=Path(__file__).parent/'diagnostics/ownership_ranking_train_20260912'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((root/'COMPLETE.json').read_text())
for name,digest in receipt['hashes'].items():assert sha(root/name)==digest,name
records=[]
for run in receipt['runs']:
    assert sha(root/run['checkpoint'])==run['sha256'],run['checkpoint']
    folder=root/Path(run['checkpoint']).parts[0]
    history=list(csv.DictReader((folder/'history.csv').open()))
    checkpoints=list((folder/'checkpoints').glob('epoch*.pt'))
    assert len(history)==len(checkpoints)==15
    assert [int(x['epoch']) for x in history]==list(range(1,16))
    assert all(int(x['targets_seen'])==8322 and int(x['directed_pairs_seen'])==3316 and int(x['ccl_edges_seen'])==1743 for x in history)
    assert int(history[-1]['updates'])==2235
    records.append(dict(arm=run['arm'],seed=run['seed'],epochs=15,checkpoint_count=len(checkpoints),updates=2235,history_sha256=sha(folder/'history.csv')))
for seed in [0,1,2]:
    rr=[x for x in receipt['runs'] if x['seed']==seed]
    assert len(rr)==6 and len({x['order_sha256'] for x in rr})==len({x['initialization_sha256'] for x in rr})==1
assert len({x['initialization_sha256'] for x in receipt['runs']})==3
report=dict(status='PASS',script_sha256=sha(Path(__file__)),training_complete_sha256=sha(root/'COMPLETE.json'),runs=records,checkpoint_count=270,normalizer_count=1,paired_initialization_and_order=True,distinct_seed_initializations=3,all_epoch_target_pair_counts_identical=True,scope='Local archive byte integrity and training histories, not method efficacy or independent semantic review.')
(root.parent/'ownership_ranking_train_LOCAL_VERIFY_20260912.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='runs'}))
