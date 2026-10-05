ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='ea8c4c38878d02f5bd7804180b976ef43893307f0d2ae7163a3313a93c19e0c4'
ITEMS=[[69887, '45ea561ba7dc4b5b095b6316ab3d843bb4c9cf4194403d2902cf0e7d6b142010', 3693585], [69969, '571beed525c2187971b41a8708adacc571526707015cf6793c690286981486b4', 5341610], [70164, '734c14688864f1bd410b6090ee7a43c9120ac8a5fbdcad03a267776ee806f526', 3598524], [70187, '5e7e6c65ec3031b331fd7d6adb8758b6c8db4064eadcb73e867ddaed7a95e424', 4094890], [70201, '9a09601b9eac62789f5f93924d60bd6cdc74bb0de861c298e769965e1a731e17', 3891623], [70251, 'abf9f17027f8cc0b20632a196c5c92a74f4fa7c3cd38540addd0b816d6883baa', 3707174], [70254, 'b6087b65275d3e1c8d6e2a873728869fbb061af632aa2b02659e4d377a9510a4', 4193571], [70261, 'ea1d8e2805536237063b38231bcaa8f7cd16bcc9fc921162010034909fc53337', 3561131], [70282, 'a620a4cb0d12682f192d728dee5127f0afdc0517b1373f7130fde26bc8544280', 3802923], [70310, 'a2c79ae6993ef7df2d0cc5fdd0d3fea1faddab6b0e38c8596f0fe51fba84bc2a', 3861937], [70334, '3677e7ef26af07a673caaa6a5966e7039c70e661fb93e24a5193203a656bc656', 4006295], [70339, '9adddc9fe6dde6b4f59784298fa5aeb3b81f46fdbad684c94a176c7afa23b437', 4022749], [70477, 'b52cdf7388d33d317b4d051cc7c1e93cdb42ac9a7b722ecc06fcb1c2a74b1eb9', 3988223], [70487, '63794802512d8407b077275574f5cfae91ede183310116c0bb2c14ea7bddfc86', 3780095], [70527, 'd80fca45f6e7d305590a3816b4c01ec04a6da0dc7752f945d798d7cab7d74045', 4563560], [70600, '0a719a027d067022f9710c596f2e8e980e6d29e24fd50fcbfc736b542bd59f0d', 3840887], [70657, '4f6cc5f58edce91e4327e083d1f7ccfed3ceb3cd180418f762f23bcff4325beb', 3943135], [70749, '750dc085d4d0fdc7f6b0598714de4255f1e5deae3c8a60d7e47f19f341c01dc2', 4288335], [70758, '2a5b3ee49e31f2bcebec25ad1a058d4d12848d1104cb409d6c2ac59fe34df467', 3708807], [70774, 'bc140f7fd22d24cf7f3a81f726082513a2e2ff5ce9ccdd25646ccd0e4889e913', 4128112], [70921, 'cd3dab587b1bf8e92acfee691f17ae94807b3a277b75f207421a9cea6dbd6ea8', 3873624], [70923, 'df3f902c86643f52f79a62cdc93b997beb643347d1c989928d1737e32ca5b670', 4001862], [70935, '828f7b1d6fb47736a158aa6a1ffc26b1b7d06f807d2f7b82a516aaa13b609007', 3794551], [71226, '76ae4e6056e6e509b36f7cd534b315640701be07f1f1f033cf2317aaaee29f86', 3945702], [71255, 'e64bdfc285e909742a466f9d721ea8dfecb470c04f32738e309efe88fcd357a4', 4011089], [71328, '318e15ce709424454473c475fdadc0c47ccd1bc852c6d3ff3715db1826637e91', 3690903], [71347, '191fd6d1dcc85020ad79cb8990f80a179c96f472867f508532cac0adc2ed0a0a', 4154405], [71357, '8e63b454ebf2a389b18c10dc48d27775ca676c4b292850c0b21c11398570615b', 3727989], [71407, '4bd8b8a81ecdc810c3b07aff9e06987c9287a3ebb9ae92cf51e5ed64d681aa21', 4020245], [71499, '80029d5c4ebad7b2c0109559aa8bfa9a46f0992294b25f47dbb416ecf5f38f22', 4059482], [71528, '87079cf39c5b2ffe95ca764aca3f3cc976ff71778b468905c2348c33904dbef3', 3930907], [71608, 'e418e4d4f7f5c652a02d8c22720264c498feb54da338133d0f18f48a1930fdae', 4957678]]
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
