ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='070584f8013c3fa5e9c6a60ae5cf5d8ea17a240e487fe18c5ac35c51e312a895'
ITEMS=[[82085, '813cd4ab546ce04014229d08a1ee4eaab87a396fe34c41f2e29b80bcfe29dc10', 4129603], [82198, 'ee1061d78be9a6ce77be1b9b1626e94fb8856a7dc8e6af334308ec0fa05ae62d', 3965906], [82252, '6dd05100f05936c81902082292cb9832039916ed7828a72d8fedadee0b0d0d4a', 3909586], [82287, 'b845616527830c123c86145737928867ee6be9ade093363d977a41271d091af2', 3712677], [82300, '6aef7d84bec5e7f41187f514c8386f6e4b86ce4b270fa8ff53a83c4d336305b2', 3605009], [82324, '20fde39b8733494b8a432c62808d2a73ff000117978ee9f25722e58f6d408e02', 3675100], [82363, 'a58f80907e0c63447f2144191fbea113a30383984167948dc10d3216ef4198f5', 3685072], [82372, '1737456e559bc8208450aa4946a2a72b2d25e1b8aa193d7ffa62e2a6ff359cc4', 4356322], [82405, 'e7b10379413e5539da1359c1201740ef5b556b6e17f6fc65824b2fb163e1a92a', 3746584], [82554, '7f780ed43340012cbdaa92072306afa2d90bdc2ab018c3d537975d39ea7922a7', 4149097], [82597, '52b3ddf717e6298605910d4e77ab0d1df08f7d3d95ec68f9f227bd38817e2aea', 3667372], [82611, 'e743af6fc3cb9561d693d33852f7f1b4b06001ec4f509aefa7f95c51d3bf9360', 3571614], [82631, '717a66ea24249c94f14963698f5066f9794f19f841b2fcc41e9e304c62b8fdb3', 4993660], [82688, '63751ae06205ae29e2aa7a412c5632d5a982315127ab9b759a58f3280e2c672a', 3780537], [82696, 'e8a2d0cf5c4f29a78ef5696c85cfb2ed575cace5c2ebf1b92e2e34fe52134a91', 4070836], [82715, '8c3e05bcbea2a645e45f6a196bdbd03db2f6b245d5d24c502d5d8603101d6b28', 3765621], [82718, '2b305a68b9db9623a210cc514c2412ae153c41a74612de26db604b3bd69abcfb', 3999851], [82735, 'ad41373e866381b83e9bb0c84ecc16e2b0986d42587cadacffbb5d1d0d6728ca', 3557541], [82770, 'be10c95a596c3c7656b20fd97866c78740fce34666eb847b57ff3f7ba6db4165', 3903624], [82778, '8f163de4d633382fd6e37bcb93bac9ec46f32186d5dd98726c9e851cf42fe728', 4100534], [82796, '9b8b8c3748c32824eb3c298b187378efc41f1fe20e6cee176ea06c89c8d88720', 4073655], [82898, '07d49964b2cb51b24a4fe8db4c29c1d27bd533920f30c9ad9d927c2b8297f01a', 3863026], [83085, 'd10e0ed1d6e514e8bd615638c0a791b7d729f6c5e673e7230dbdfda042f9ba59', 4245726], [83256, '74bf6c27e67c6ef66a36c69f10e1433ba5844a427157877bd5c3393fef4ad27d', 3654761], [83348, 'a7d3c68b196e7e6af866197e1970243b88bf302d074c49b00add2ea24da42fab', 4038932], [83352, '0a255b42662262595483fdb82d769aef3fd5c293d32d08f1a7f2caf45d8f296c', 3610685], [83407, '90338f4d38a67040d6a8840a61407535e06ed660f3cfa0f0d96999e8847d77c5', 4207209], [83441, '479d9b3e27361ed7d65bb672d3a8033f64bdc5f1d34e0d211714bcde12f122eb', 3697581], [83452, 'ba730c93f6794ea731cecddf58e303fc09803116a8722f3170301e4336aad998', 3682862], [83454, '1ebf6e53bac8b4cf3ca93c058ec1e7156ed1699af3811cae498c47cc11b2b4f1', 3618759], [83507, '0d63685c0d172da9575971d28c9dc2302df12fba219316059468a66097574d3c', 3765678], [83547, '4c86f65bd444dc2c21f4922bf6046ab10eb415855b4bc6d3d2bcb2574e87c9d0', 3826896]]
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
