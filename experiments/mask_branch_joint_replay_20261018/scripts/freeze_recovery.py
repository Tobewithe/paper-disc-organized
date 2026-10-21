"""Freeze a new engineering container without rewriting the original frozen study."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'experiments/mask_branch_joint_adaptation_20261018'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target = STUDY/'FROZEN_MANIFEST.json'
    assert not target.exists(), 'Recovery freeze already exists; preserve it'
    parent = json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())
    for item in parent['files']: assert sha(ROOT/item['path']) == item['sha256'], item['path']
    files = {ROOT/x['path'] for x in parent['files']}
    files.update(p for p in STUDY.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.name not in {'study.json','ENGINEERING_TESTS.json'} and 'transfer' not in p.parts)
    records = [{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(files)]
    manifest = {'schema_version':1,'frozen_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'code_commit':parent['code_commit'],'parent_frozen_manifest_sha256':sha(PARENT/'FROZEN_MANIFEST.json'),
        'source_state':'versioned_engineering_recovery_no_scientific_configuration_change',
        'protocol_sha256':sha(STUDY/'PROTOCOL.md'),'config_sha256':sha(STUDY/'CONFIG.json'),'files':records}
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (STUDY/'transfer').mkdir()
    archive=STUDY/'transfer/recovery_v2.tar.gz'
    # Original frozen files already verified on the server; transfer only new paths.
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(files | {target}):
            if p.is_relative_to(STUDY):t.add(p,arcname=p.relative_to(ROOT).as_posix())
    receipt={'files_in_manifest':len(records),'new_files_in_archive':sum(p.is_relative_to(STUDY) for p in files)+1,
             'archive_sha256':sha(archive),'archive_size_bytes':archive.stat().st_size,
             'manifest_sha256':sha(target),'protocol_identical_to_parent':sha(STUDY/'PROTOCOL.md')==parent['protocol_sha256']}
    (STUDY/'BUNDLE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__': main()
