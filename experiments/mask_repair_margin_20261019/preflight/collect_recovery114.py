from pathlib import Path
import hashlib,json,zipfile,datetime
s=Path(__file__).resolve().parents[1]
names=['RUN_RSM_RECOVER114_f22b016c00f5482eb7551a4acbe3dad9','RUN_RSM_STATES114_02fb6e10ff3e40228f5b044467b245bc','RUN_RSM_VERIFY114_1924036e4cb543169c4701a45efcc76b']
if (s/'.RSM.lock').exists():raise RuntimeError('own Run still active')
files=[]
for name in names:
    root=s/'runs'/name
    if json.loads((root/'run.json').read_text(encoding='utf8'))['execution_status']!='completed':raise RuntimeError('Run not closed')
    files.extend(p for p in root.rglob('*') if p.is_file())
files.extend(s/x for x in ['NUMERICAL_RECOVERY_20261020.md','NUMERICAL_RECOVERY_STATES_20261020.md','RECOVERY_FREEZE_20261020.json','preflight/recover114.py','preflight/quantized_states114.py','preflight/verify_recovery114.py','preflight/collect_recovery114.py'])
records=[{'path':p.relative_to(s).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size} for p in sorted(files)]
m=s/'RECOVERY_TRANSFER_SOURCE_20261020.json';assert not m.exists()
m.write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':names,'files':records},indent=2)+'\n',encoding='utf8')
z=s/'RECOVERY_RECEIPTS_20261020.zip'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(files)+[m]:archive.write(p,p.relative_to(s).as_posix())
print(json.dumps({'files':len(records),'zip_sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'zip_bytes':z.stat().st_size}))
