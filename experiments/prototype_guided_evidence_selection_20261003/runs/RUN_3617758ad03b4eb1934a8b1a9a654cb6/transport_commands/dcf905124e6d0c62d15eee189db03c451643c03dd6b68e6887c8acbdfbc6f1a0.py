ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='5e7f24efb8fa1c6b06d28a3a4f467ad74476f75d61e65656deb2de87de9e047e'
ITEMS=[[8186, '386836b9f88446526a79c2d34a504285774e8fe0631c2781a11a69804f52d01e', 3902836], [8190, 'f2f38525fbf18e3ba777256eb3851fc3d51b508bf0e98399b312ad6ee70c45b0', 3717476], [8314, '1a407e3143536f7301ca7ce4186f8b8d26b6fcf1528911454cc5a1c04a1d5e1c', 3767610], [8373, '62a109ad1537fe83b0358f4bc10f30473e695c2cb43561a6b9c0223db41793da', 4060135], [8431, '32d96e6a154e2556b0fa5dcc1b7c9fb1731e1a841112bcbfca49e62065fba9f7', 3812342], [8536, '82566da842fa5540e2d5d5ffc10272c8cca1db5cf670cbd4f50b37aaf716b0d6', 3638572], [8553, '580705158ed0bf44412a39885bb711c2d96f86007acd12de00f115e3e2461bb0', 3751016], [8665, '99f279a249dc8b2fa10adc8f0ea7e330755a03a06162899d9462617f3f2ace1b', 3735065], [8781, '52fc71b7f871a53920ae7f1f0a302b9bae8317bbbd8b6e9027e73fae5499061a', 4188722], [8881, '2204e2f3da8cc549a7c3171141ea3acbf199e3324c27da8bd1a7b3b329aa82f9', 3859281], [8920, '9dff77a006025acd33b9fed2c188cec95a938f99ce107c789b555c4ef9159387', 3566916], [8936, '4b999b0e953b329901912f418e8763d50bde60f10bf965c1cc060435d96533f7', 3917163], [9025, '0548e65a8da74530a7a50e8a60127bd9479edd26377675037a1d193561655d38', 4208998], [9029, '5ae7862d65c17208e678e96b1971053f8da58ea377830a1fd3b4d5f8d3249c5e', 4127996], [9060, '2f8e222a8bdde94dfb694191f4be089f168e6ba85a6f038febd190112f0fc263', 3904615], [9091, 'a9bc7e67d270e38e56e3d3af18d2e920a0cef429675e8272b25c7ea290f3f048', 3569010], [9105, '7b6350f0a91428c549da2d4a21ca357635386eef64c29a09571497a6d9ef69ed', 3444696], [9113, '020560c3f47ea20d0d5fcd407cb3c29c91716c445df42b2bf2deb062043c94a0', 4158570], [9321, '5f7c804f8c7a8dc513e4bab35a926b586a53d4e5e586445e691471df02a6d248', 3881589], [9322, '0402bf03c1152f09f9132a86f23470be7d3a68128bf75274a291202e3bfdb002', 3651528], [9381, 'dd4545346bdca58a8ecfc658bffa404c007bbe7a6fa73975d8f0bdecbb030629', 4148822], [9420, '73089e60de024e92c283425a577592039a4a5335cf12f27d35647e70618ad31e', 4961271], [9509, 'ba227e6ce76a408afaf9e0fcc4773138fee41a6fe86c0cefd30d636954560b8e', 3713459], [9647, 'f40525fe80c242c77a5ecf4a79ff9df19725b6f67a1227585411027e8406b411', 4611151], [9760, '22d3b2fa384991ac41c5da6622d16fac7eccfb925f36bfe9da95be54cfa0c559', 4998206], [9800, '80b5e11e6febc3117eec1030e77eb572a02aafe51c285bfffe10540213b5d801', 4258315], [9872, 'a1cf5bcaad1f05f97545993c736c10739e224ad6834595805ce869cbba916547', 3855856], [9935, '3fd87063c6f8f71fed6b7a2ab69e0f0d28ad8fe520424c4e55ab9165dd90312b', 3550989], [10130, 'ebde3518901a66cb8e4f354cf8d9da922997163c8494f422453dfb68ff2c61e2', 4251706], [10216, 'c05195014265181893a804183f0eda9a0541581149c014715d2c85b353ee35db', 4520525], [10256, '66eabcfc170e48452c45eb0d3ec1988133edfcd1998c4c8f1deab53eb08934a3', 3675762], [10263, '2b3d94f5db144c9c3027c708a2c844175dc65676c25f559c54c80a5988ff14f7', 722282]]
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
