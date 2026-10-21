"""Verify every transferred byte before merging completed Run artifacts into this Study."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import zipfile

study=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--sha256',required=True);parser.add_argument('--receipt',required=True,type=Path)
args=parser.parse_args()
h=hashlib.sha256()
with args.archive.open('rb') as f:
    for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
assert h.hexdigest()==args.sha256,'archive hash mismatch'
checked=0;created=0
with zipfile.ZipFile(args.archive) as archive:
    manifest=json.loads(archive.read('TRANSFER_MANIFEST.json'))
    assert not manifest['live_runs'],'final transfer cannot contain running tasks'
    for rel,record in manifest['files'].items():
        path=(study/rel).resolve()
        assert path.is_relative_to(study.resolve()),'archive path escapes Study'
        data=archive.read(rel)
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],rel
        if path.exists() and path.read_bytes()!=data:
            assert not rel.startswith('runs/'),'completed Run artifact would be overwritten: '+rel
        if not path.exists():created+=1
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);checked+=1
result={'status':'passed','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'archive_sha256':args.sha256,'files_checked':checked,'new_files':created,
        'all_completed_Run_bytes_preserved':True,'excluded_regional_F':True,'live_runs':[]}
with args.receipt.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
