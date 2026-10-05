ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='7ea0a54d164f1ff141df33194cacf6db389abfc37b8c8175214f2cbdc47f87e4'
ITEMS=[[64474, 'd6c11a67cfc518589711db6087ce33f1ab37c957c3e58d6a08e38fe30ee79880', 4012773], [64499, 'd4748ba7a730031bf2797183db99880dfde0fb4d1a0d689fb0ce027d074c838d', 3840450], [64523, '5778c66f1f17c9accf041758186487017fc9a0647ab1bce47148877f3c9b0bd9', 4447212], [64578, '0c473d0542c87b3f240a55c6768ecb71fb84635ea470b33a151c87f90c4a7baf', 3881234], [64593, '5939ea99a169ffc8a8e3d2f33ead9121f13f1780660a2cf936a4b3055bedaa81', 3910446], [64622, '04727ae7015d6e0bc6cfb1b88b4b1043ff69e7b195f05d2af36ecd429a598a3d', 4215334], [64779, '4b64df576c01131839e0c462e6bb0769fc78ff87c84c3b5421b6dc81fcd7a8ed', 3904527], [64799, 'caf3835da19bb83c85b132cf6227fef4b8d4349cc3b6775b7bb2b2bccee37e73', 3926247], [64823, '18ed50a3bc20356c11af55ada45aeb4e6f27c0193a580636cc0867237c54227b', 3813580], [64868, '6f4177da73e52b258923a41d326fa96eaf798fe16e389503ec05196090d2500f', 3896826], [64898, '5995db5da4591425de508d7c8862c8198227918c0fdc0745c8744ffd78142f8b', 3767552], [65012, 'b770508be9dda72fa3dc295572b7ad20d4749797d8084e93188ad2688f92e67c', 3800772], [65016, '8b0d58f07204e959b7c80853e3b0f024962ffb31123c1a6868c45bf14e27e5dd', 3549931], [65035, 'cfa38b6279af66fb92fdf3cf7b16bd35f3958a69e28edd5eb33f0a17fe731fcd', 3666925], [65043, '8dd60af643d8850aa98ad8c82a7ec1cee15a9afee25a6a544e913ab3daf91fbe', 3693366], [65074, '9987efdadec8e040b90c739c1ca8c845ac69a1648370adcc50ffcf825b31470e', 3689457], [65124, 'f93e5cab8648866669ed5ec22892a25e3b6a6f757d764a397ec82c8aeccb71b2', 3721977], [65189, '47744f16c32695013b7ce1a0ee5a1bf6296aa1e6a9d77dc447874edaa54716d1', 3846444], [65207, '7bf0a97b29b1b8333169a622d01473a40bbcbe1cb629d37fafdb38d3162d5cc2', 3883959], [65239, 'fcefc93e1b46d63cb1f829b6b7a114055b61dbb8609be3b2b7cb471b9628dbf9', 3817350], [65270, '93c277e4b342424d39470a07f1bba76d0ed57a756c3fbc4133eb718c1fa907b4', 3827205], [65288, 'd14f721ef903b0ecdb767547811a86c2afaca8753bb9b17bb69b455e517363d0', 4997419], [65306, 'e1cbe80d5497f9466588eaa78d7225cc9a1e916873670f61ebaee1c7304d2f0c', 3859147], [65420, '3b49c61fa765bb9a3fc1400ba225bb50a93e47a2299869103e60b25c2f91cb12', 4340149], [65455, '66fb579ae17d2a664c701be80e97838f0f9a2a6cfc0e781c440df17f4c1858e2', 4371740], [65488, '92cf47a482866a32d2a2564f16cb6319898953135d7b793b4b99101017cf3275', 3874666], [65530, 'cf6765f64d217a137e629838726fffe1e334dd0640b77a622c51b032f97bd8c3', 4174072], [65557, '92a728495336915961bc75f0208d56c2e9d58adbdd97f0fb8cb23d90be6bc41b', 3950151], [65586, '233bd8a83755cde567467149e339134c30d69b535f7b8776bdc42aae9db308e9', 3468013], [65705, '35f3f47ef4806afcfdc92bb2ff9cec3fdfdc4983d1e29b8ab221ddbe4bbd5b59', 3888155], [65736, '622fe8e4f5b3e1825f1f2160bc82a432258d035c11cbd19b07a63fb208aa6e1a', 3571596], [65755, '4220f09954b3e262ed2e061ed939e3166a4e91c7789bd3463f41da99379af7c6', 3563087]]
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
