ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='ae5c824cc2b7b5480773506870fe8f8d3476afb676e3d2bc3c894ae8601e4b09'
ITEMS=[[59622, '444a2c246f0608d46d09c2fe54aead8144ee8157d9193bf28b7f1dd4cae4c4f4', 3744958], [59736, 'ec4e0e7c1a6a5c58d00a17325e0c03b6230cd6a16c5bebcccedbe3d621007a49', 3750008], [59776, '45a1058a7ece5a2c4082e7cc2394dcd96887e5187d01c45c2d6c34d1082e9cdc', 3741050], [59953, '7bad93e635a034f3a55070fac292839f57f5fc6cb6cc9455c3f5efd9995d6c75', 490930], [59971, 'a6c1702ec6b31903b2f5b8abbd68343ab5020e8b11b7cf149c6329464480e787', 3873278], [60010, '1870f2e7f2ea96f0018e09fee65d05c7fdcc207ced5936171bfa6538a0f3a7d5', 3788588], [60041, '8b50cd34806e8224f3b76da48ed88f6fbbaf5d574e742fca95c5ccdaa5982732', 3831634], [60081, 'ffe7bf82c63fd1ae9d436ced0323a2cd9a85e325f162905c8b0fa96270e24754', 4462545], [60090, 'a788463a431e52b2329e4d225fc851ab80dd6721578a10b3ef8e34d157294569', 3963244], [60125, 'f7bc9315297b1cd82c7b8f1c40fef11919d9911158d3fb8f3d1491b2ecea3be1', 4102011], [60132, 'a2cb6442ceb847aa35dce7f9510e347d888f981054729f708b9ae9b529155424', 3890884], [60179, '88a1fabfeeae0836a6636ed6894e64168c4f7251d7be7f82ae00c0f271b53443', 3681220], [60190, 'fb790749d9352e59ec0ee1c520722c4c5e32bacba868432ab66b480fbc4af3f8', 4900623], [60347, '6059035df1ce5d4d34871bfe6e09e43e58360df7907b7db769c41f02739c648f', 3911554], [60364, '099deec200f08d73b9e5d52247fed5e2e3fc3833b6062b7f27b3a0a7545fa6a0', 3752591], [60569, '70b1350b5a853c26e004923a4e37745140469c43c72f9b4cdedd367b856b645b', 3565148], [60730, '37ada0e6462cb3cc9b6d876429e9890b3ac1871644b58bbccc1268f3dd923b54', 4212052], [60828, 'c64fcba51e7f82a03acfcbcc1645c19de7afb7d4b36976a6b93a84770e98c2d7', 4106883], [60859, '34b7d630fa6dbde7b8870af13ea606d8552440de74caebd26d54c43dea2909a2', 3601376], [60874, '4723da985d4450e188977afb951bf571c2dab6b3cb89557d1f3f246b8b3b4875', 4333407], [60899, '89c2f36dbc2f97bd66a8cafc5f59b33b7c67f3b58a6b6a7868864dcfae80aa01', 3847365], [61108, '21d59347a60aab0672781b936001bfd00a5e0e7103af55eb928aeef1072c9af6', 4101491], [61150, 'e9534535664e44cc00ed2bae33cfe29141efc891b67deabc2bafae5c4526a32a', 3860176], [61171, '88324742da613794c678ec5fda5dd78e10386dda2a61bc1a5a4e716ddba85989', 4073934], [61268, 'a00f6cee215029f4cf38549c300226a575c787a9932db2abefb9ca4a25e16250', 3795366], [61288, 'e5088dca5b3157c87f6bae06b4d6e55df5416c3a8dc8d4f61f2a6e1bce59975f', 3869463], [61354, '661b1d8785d4cfea4955a1db0a1e67bd366a215857dfe3fd3b82e59ecfcbe98d', 4637695], [61377, 'd2f275453ccae5922a11d526aec79c0b2d865c71cb04e7dcd5ee7f6fd2da9d6b', 3897953], [61418, 'cb54196be8fcffe44872fef5c299d880586bb2861b3003d4d41271cea4ee393b', 4254905], [61492, 'ff9f85998d36a2403971d597813d0c3dd8d74905df3e20e7d6e8cea8db835c86', 3638581], [61503, 'd6a46382c8cb693b182b2d63a50bf56b5bfd0c90f22718be2001d7c1c7f8733c', 3763225], [61526, '194e9aea8a9e3bc10db666f7f24f518616e2042ff5653c04c4187437a81c9a0b', 3646688]]
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
