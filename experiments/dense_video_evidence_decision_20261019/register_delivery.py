"""Desktop transfer verification only. All research execution was on 28358lan."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[2]
STAGE=Path('C:/Users/Public/thesis-video-v4-20261019')
STUDY=Path(__file__).resolve().parent
DEST=REPO/'paper/presentations/CETD_竞争反证时序决策实验_20261019'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

export=json.loads((STAGE/'DELIVERY_EXPORT.json').read_text(encoding='utf-8'))
archive=STAGE/'delivery_v4.zip';assert sha(archive)==export['sha256']
out=STAGE/'verified_delivery';out.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        target=(out/member.filename).resolve()
        assert target.is_relative_to(out.resolve()),member.filename
    z.extractall(out)
manifest=json.loads((out/'TRANSFER_MANIFEST.json').read_text(encoding='utf-8'))
for r in manifest:
    p=out/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
assert len(manifest)==export['verified_files']
shutil.copytree(out/'showcase',DEST,dirs_exist_ok=True)
for p in out.iterdir():
    if p.is_file() and p.suffix in ['.md','.json'] and p.name!='TRANSFER_MANIFEST.json':shutil.copy2(p,STUDY/p.name)
for p in (out/'runs').iterdir():
    d=STUDY/'runs'/p.name;d.mkdir(parents=True,exist_ok=True)
    for f in p.iterdir():
        if f.is_file() and f.name in ['run.json','EVAL_CONTRACT.json','SUMMARY.json','DECISION.json','SOURCE_SNAPSHOT.json','VERIFICATION.json','ACTION_AUDIT_VERIFIED.json','ERROR.txt']:shutil.copy2(f,d/f.name)
receipt=dict(archive_sha256=export['sha256'],archive_bytes=archive.stat().st_size,verified_files=len(manifest),all_file_hashes_passed=True,executor='28358lan',desktop_full_evidence=str(out),showcase=str(DEST),git_head_reference_at_delivery='bc0507c5400b61935fe93f981a4ff5026964c271',source_authority='Actual per-Run SOURCE_SNAPSHOT; working sources were uncommitted. Git HEAD is not represented as executed version.')
(STUDY/'TRANSFER_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
study=json.loads((STUDY/'study.json').read_text(encoding='utf-8'));study.update(transfer_status='verified_desktop_and_laptop',desktop_showcase=str(DEST),desktop_evidence=str(out),transfer_archive_sha256=export['sha256']);(STUDY/'study.json').write_text(json.dumps(study,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
