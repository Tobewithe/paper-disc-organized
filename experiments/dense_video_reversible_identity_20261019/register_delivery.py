"""Verify laptop archive locally, then register only this study's artifacts."""
from pathlib import Path
import hashlib,json,zipfile,shutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
STAGE=Path(r'C:\Users\Public\thesis-video-v3-20261019')
SHOW=ROOT/'paper/presentations/密集视频身份保持_确认与回退实验_20261019'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    info=read(STAGE/'DELIVERY_EXPORT.json');assert sha(STAGE/'delivery_v3.zip')==info['sha256'];dest=STAGE/'verified_delivery'
    with zipfile.ZipFile(STAGE/'delivery_v3.zip') as z:
        target=dest.resolve()
        for n in z.namelist():assert (dest/n).resolve().is_relative_to(target),n
        z.extractall(dest)
    entries=read(dest/'TRANSFER_MANIFEST.json')
    for e in entries:
        p=dest/e['path'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],e['path']
    # Complete desktop mirror remains outside Git, including all immutable predictions.
    for folder in ['runs','annotations','showcase']:
        shutil.copytree(dest/folder,STAGE/folder,dirs_exist_ok=True)
    for p in dest.glob('*'):
        if p.is_file() and p.suffix in {'.md','.json','.log'}:shutil.copy2(p,STAGE/p.name)
    shutil.copytree(dest/'showcase',SHOW,dirs_exist_ok=True)
    for p in sorted((dest/'runs').glob('*/run.json')):
        folder=p.parent;out=STUDY/'runs'/folder.name;out.mkdir(parents=True,exist_ok=True)
        for name in ['run.json','SUMMARY.json','DECISION.json','EVAL_CONTRACT.json','SOURCE_SNAPSHOT.json','ERROR.txt']:
            if (folder/name).exists():shutil.copy2(folder/name,out/name)
    for name in ['ENVIRONMENT.json','CURRENT_SOURCE_MANIFEST.json','RUNTIME_SOURCE_HASHES.json','input_manifest.json','identity_pointer.json','config.json']:
        shutil.copy2(dest/name,STUDY/name)
    study=read(dest/'study.json');study['transfer_status']='verified_desktop_and_laptop';study['artifact_status']='offline_exhibit_verified';study['desktop_evidence']=str(STAGE);study['desktop_showcase']=str(SHOW);study['transfer_archive_sha256']=info['sha256'];write(STUDY/'study.json',study)
    receipt=dict(status='verified',archive_sha256=info['sha256'],archive_bytes=info['bytes'],verified_files=len(entries),laptop_root=r'D:\thesis-video-v3-20261019',desktop_evidence=str(STAGE),showcase=str(SHOW),original_failures_preserved=True,scientific_status=study['scientific_status'])
    write(STUDY/'TRANSFER_VERIFICATION.json',receipt);write(STAGE/'TRANSFER_VERIFICATION.json',receipt)
    print(receipt)

if __name__=='__main__':main()
