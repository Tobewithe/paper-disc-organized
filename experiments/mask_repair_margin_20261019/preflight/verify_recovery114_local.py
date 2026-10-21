"""Stdlib receipt, source, and acceptance recomputation after transfer."""
from pathlib import Path
import json,hashlib
s=Path(__file__).resolve().parents[1]
names=['RUN_RSM_RECOVER114_f22b016c00f5482eb7551a4acbe3dad9','RUN_RSM_STATES114_02fb6e10ff3e40228f5b044467b245bc','RUN_RSM_VERIFY114_1924036e4cb543169c4701a45efcc76b']
def read(p):return json.loads(p.read_text(encoding='utf8'))
hashes=0
for name in names:
    r=s/'runs'/name;meta=read(r/'run.json');assert meta['execution_status']=='completed' and meta['exit_code']==0
    for item in read(r/'ARTIFACT_MANIFEST.json')['artifacts']:
        p=r/item['relative_path'];assert p.stat().st_size==item['size_bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];hashes+=1
    for filename in ['PROTOCOL.md','CONFIG.json']:
        assert (r/filename).read_bytes()==(s/filename).read_bytes()
r=s/'runs'/names[0];identity=read(r/'REGENERATED_IDENTITY.json');numerics=read(r/'NUMERICS_114.json')
assert len(identity['checks'])==16 and sum(x['rows'] for x in identity['checks'])==1851
assert all(all(x['tensor_checks'].values()) and all(x['metadata_checks'].values()) and x['zero_delta_RLE_exact'] for x in identity['checks'])
last=numerics['old_fixed_point_history'][-1];assert sum(abs(x)>5e-5 for x in last['signed_relative_errors'])==6
rows=[json.loads(x) for x in (s/'runs'/names[1]/'DIRECTION_RESULTS.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==48;counts={k:{'linear':0,'both':0} for k in ['C','W']}
for row in rows:
    linear=any(abs(x['direction_relative_error'])<=5e-5 for x in row['trials'])
    both=any(abs(x['direction_relative_error'])<=5e-5 and abs(x['decoded_difference_relative_error'])<=5e-5 for x in row['trials'])
    assert both==(row['accepted'] is not None)
    counts[row['center']]['linear']+=int(linear);counts[row['center']]['both']+=int(both)
assert counts=={'C':{'linear':24,'both':24},'W':{'linear':22,'both':16}}
report={'status':'passed','artifact_hashes_verified':hashes,'regenerated_images':16,'regenerated_masks':1851,
        'old_fixed_point_failed_directions':6,'acceptance_recomputed':counts,'complete_repair_valid':False,
        'GPU_replayed_locally':False,'GPU_independent_verification_run':names[2],
        'scope':'source/transfer/acceptance verification, not a new scientific result'}
out=s/'preflight/RECOVERY_INDEPENDENT_LOCAL_20261020.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))
