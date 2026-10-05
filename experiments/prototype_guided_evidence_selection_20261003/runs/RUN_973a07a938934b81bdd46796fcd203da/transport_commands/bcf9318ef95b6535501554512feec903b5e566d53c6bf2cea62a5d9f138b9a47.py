ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='7b730481644b03fbf06313a7b005ee16f874a920e410f36b61fa4ab53a5256b9'
ITEMS=[[33931, '49530d8ef8ccbc37c87fcfbb06b7125211e2dcc83aba9f0b3a84b2c455b85398', 3399607], [33940, '11ffb75173b13690b088f53766188b0d1614167ee86098f43e1a32a2b27bc337', 3951820], [34019, '7dbc110fb8297093f83270bbc0cb3cdf320e65de354966e6f51e39f1124df1d5', 3785932], [34063, '7fbffbec8a00aeb36560db55b8f0f8610d9fce5dcaed1abfd11f9ddd5854f694', 3813270], [34074, '32216a9b2060852aa532e895a4797ee6763c5f666a0450edbb2d28ecf0aa0fac', 4242388], [34128, '09b7213425a7fba413e71a77951705a6dd7f4990eb2c455f5ac7e48d49282690', 3736773], [34181, '32b5b60c9edf5e5b94fcb2abf5f186bbfe498ce69e632958b493c9cab8633ac2', 3692559], [34205, 'd0ec0040cb2a4f61234afb346a902d2edebae9659d7d486b421ef62b1f07156e', 3802792], [34222, '27e78d66d5be4c02065fce7d090c5ad6a3e45f3299e8eaf8abf2423e4e98e58d', 3927844], [34267, 'b067f617420d47056773e41aadc5b79750f842b608fded568d832e7a54ea5720', 4105665], [34389, '16194413c61aecc1358f0a3c3d59f52c2d98c39db27ab4ee945f221508c72183', 3855276], [34430, '002a780d56269007c5937aca997be702ebcf9fa0a9bb17deb281f5577453f424', 4338753], [34452, '5d5dfbc2fa057859d658239d58cbacdfa939d5fdf58f694454143b4d4ee218b8', 3960078], [34471, '7fc51723710116506ef9f66bceb7a33036850477d620a6ce6a02182da12f859d', 4368053], [34474, '662bc50e8eb37399e20d1dcb86931c6eb4208c87e2c59b25086e62dee710310d', 3685376], [34520, 'd189657117e4e375de02c99193b0b03c2cc289b93b3c16b9c003f84d610c658c', 3541944], [34524, '088127a220f38445d974b5fc5c74f89bce80e72ba58c51dcfa130ac6ead311ce', 3739521], [34539, '90d590e50ef3110e7929bcb4b2399248db8b80b98499896da6226f957d2ec9d4', 4339990], [34608, '8b4851dcea89047832f796d4beea013e342c38d81fedb91b276710131e41face', 3571783], [34680, 'fd43696db71f45c373ec1e832f3bd649bf90e99e02f609335c7be29e1814b45c', 3758345], [34757, 'faed337f3c508e94e7c3b2b4352c8987c4bf44ce754aa10262fbdfe3b692a8d8', 4508008], [34761, '0481baa0fca7bd6e16a6caa0fec5d502fd577e365ab3525148dfabf7157e64ab', 3760615], [34785, 'e595ee3a5f4fc9f9edf8e5d74e0761d33d158d2be969955c6687d73c785b723e', 3929616], [34795, '68ec7afca155c10442f9555fa99f044f75120516848452ddd801c6a5b2edae1e', 4379559], [34810, 'f7864353b0ed828ed95e676fdc74ce912aff86b36f089e8e50c58bbe85a2cc8d', 4168658], [34826, '0150e3b9d32eb1c66541dccd420f1e14b02cffbda8224790c5c80d3876d310d2', 3732759], [34874, '22d188b66a1af6ea5bfa728125998a41f78648a94482e2848dcc26bed2590619', 3904814], [34877, '2667a5299ebcaec84a684640f281babff8757a163a0ed0d5b1c4348ef13e0c3b', 3635911], [35018, 'c7c6f5373994856354931dfb45c0e76eb8f79b78dd24f1d34e8f04adeb269d4e', 3626802], [35031, '2f7feb8e22152c71ea4830d7e9e0e42ae357eff451a0336bed8a84c22c598203', 3896583], [35045, '851b2b1db8f23559ff7d3e0ed2aa8c6ed64443e4e499e7d687b6b18cc4842cc9', 4067108], [35190, 'bd2fc697ec7b428dc81a7eb7f331aae3dffad7659f264f74b0dc107820fd2777', 3607405]]
from pathlib import Path
import hashlib,json,os,shutil,zipfile
r=Path(ROOT);directory=r/'.relay_bundles'/RUN;directory.mkdir(parents=True,exist_ok=True)
bundle=directory/(KEY+'.zip');receipt=directory/(KEY+'.json')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if bundle.exists() and receipt.exists():
 old=json.loads(receipt.read_text())
 if old['items']!=ITEMS or bundle.stat().st_size!=old['bytes'] or digest(bundle)!=old['sha256']:
  raise RuntimeError('Existing bundle differs; preserve failed attempt')
 print(json.dumps(old))
else:
 estimate=sum(row[2] for row in ITEMS)+len(ITEMS)*65536+1024**2
 if estimate>2*1024**3 or shutil.disk_usage(directory).free<estimate+1024**3:
  raise RuntimeError('Laptop bundle staging/free-space guard reached')
 source=r/'assets/images';temporary=bundle.with_suffix('.zip.tmp')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
  for iid,want,size in ITEMS:
   data=source/(f'{iid:012d}.pt.gz');meta=source/(f'{iid:012d}.meta.json')
   m=json.loads(meta.read_text(encoding='utf-8-sig'))
   if int(m['image_id'])!=iid or m['compressed_sha256']!=want or m['compressed_bytes']!=size:
    raise RuntimeError('Committed laptop metadata differs')
   if data.stat().st_size!=size or digest(data)!=want:raise RuntimeError('Laptop archive changed')
   archive.write(data,data.name);archive.write(meta,meta.name)
 os.replace(temporary,bundle)
 result={'path':str(bundle).replace('\\','/'),'sha256':digest(bundle),'bytes':bundle.stat().st_size,'items':ITEMS}
 temporary=receipt.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2));os.replace(temporary,receipt)
 print(json.dumps(result))
