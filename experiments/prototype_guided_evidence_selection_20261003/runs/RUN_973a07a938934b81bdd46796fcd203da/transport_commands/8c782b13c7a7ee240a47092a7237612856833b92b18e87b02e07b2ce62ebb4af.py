ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='741916bb0b183ab3d20010fa2483fcfa7960e481a5ca8e29073d4a18253433fd'
ITEMS=[[36957, '4a161371d9df1a387e340168f9963c3b653bc14b71ebfe2306f79ff1892df3cf', 3945100], [36996, '707895fd302a72bcb75b13587facc16d1722e36c1b0fe6e8c30bd7fa46ebde48', 3831274], [37076, '9fa8d8e191879332d342699a4e1bf69dfd03569c5acd212e0dd7f55572f86082', 3588451], [37156, '1a883e8df21f7348da8c1a4597557c87e42f9eee4a62c84a8248b3aeb01de24f', 3683944], [37282, 'f2f1167bd6144e9233dc23e000ca4c2d656c425060545142a7f5b7d8c6556867', 3909429], [37326, '52370a49fea5170e07a3dd62e368c011a5faae03512e3d574d5e56ba2e721f60', 4115251], [37479, '485c4c8823835bfcfea040616f1336a951e167757b0932e061f01f640f51c574', 4681598], [37514, '6c276cc97d36ca7ce14d5ceb945b9b819bf53ea0bbf03b9a01ba2a2c35cc4865', 4488936], [37566, '3e868a0a800829b4463f638b5cddfb23f7823669f78516ce0c48aa5e785033a3', 3813379], [37615, '5e5c200b52cbe9a891d8b0a5da7db8ace5d4b39215efc31a63713b30ba7b4d6e', 3921749], [37670, 'fe7a65725e31d3cbcab23e44c90fbc4d1d84397ee46a56b2408fe420358852db', 3704912], [37675, '4de2f70d5ff59f202a30d955d03723991034de318fe8fa146bb33aa492e0074f', 3768178], [37688, '0990f7b1efaeeb679af7694c5fd14b8c416be392b9e6b4d4fe1ff0a5dff53f17', 4151497], [37727, 'f44affa735eb017c0098fa85ca0cc79dc4fb098dc8143b5ec6909f370213e8fb', 3906556], [37740, '353735c572a84c53269c5d38f2e36466684662d89c7194ee7849a3c0681baa10', 4432197], [37751, 'b6a6d4c496cfd345bb6cac12511b2c579307c2f97b2f9d2c6e50e0ad9387485d', 3855028], [37777, 'a41615691e4301da2d482a8ceff562fd61d2febc7a225abe57b33d511e8dd213', 3934121], [37868, '4c6c1bd363d96671c635ff448bf7e0c88a0f41e7fd64320501318f3219d0b5a0', 3790367], [37925, 'ffe33134cb5b17e9ec6d867248928657bdad0fed104fa76171423d6a8fca1ef0', 4190159], [37945, '5fa0e535fc4aa370dfeb6cfc3e20ba2974c32f523d7a28962a5ff95b5d5e9219', 3631864], [37955, '8d17c6a78e02cee4cf1ba86c38b69ca190e4029424fc806866729c2527a75143', 3690714], [37988, '6c6315f7d7661a02b419e69c155cbb12491a65f70e1d72af6a93885be8e0ebba', 3764449], [38017, '3f33489cfe2d95f2f543404faea342229118af502a23b30ac78ba19a2b8e65c8', 3605555], [38041, '3135676a35455f7b647c1a0e03b6a646270c9d9cce9841c886cf6c1c8b1ce97c', 3707943], [38048, 'e5474e5298694e7f708cd8d5df6fe391e041ed94b332aed061b0cd6e6f12120b', 3710155], [38070, '5eeefba8b6ec89125df196920f9681f14bd14ad229703dd2c67c2c8ee9e6f098', 3603800], [38117, '500651f519a65ecaf88fbf985b9f1126ea28433cf599ac5ca8831c0afba2bc41', 3931295], [38134, '50e01a6674ce804593b658ab6fe829b6c07842ca3f96d268f56a01aa5859210a', 3923327], [38252, '65bac5d33022312e9e9a5dca9d48d080155e61685f262fd8f6a76386ccc6be6c', 3648719], [38277, 'b1be7ae3c32eb468e2fa77ace786aa715e836af6799396eebc62898b2c313c89', 3713590], [38305, 'ad7de0e16aa4d9ae614c4189de5128e89ab2f1b6fe359310715a03e414968a64', 3750420], [38350, '86e83707b291db2a0266d44c0e972e55b0841b7611b1231cb730426188b9acb1', 3624624]]
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
