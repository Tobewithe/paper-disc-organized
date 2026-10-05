ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='14a548e88ae5dc7acf1721b3d0d3fd81bcd493f0c94541a487ca8a85190de7b0'
ITEMS=[[4424, '6ed816f9b579db72af24ad4c0fc8ab17e98d5626ed97f9354699c6b31ab94fbf', 3781059], [4509, '63ba8839508f36bce97465a0d836c86242b52d9ff2cbee79e0be6603ef9b7c3f', 3756906], [4554, '24a17dbdd55a4c936275f296a2e168f4eeab15c541d039becddec85f3ae60529', 3886766], [4700, 'a0c801ba64b45729ce813f0f0cb49ab8234306a88257768217ccce0a459f99e4', 4568045], [4705, '765b738512bde9a0a7105aa69bffd4db877af9ecea7faa51e15cda45117dd3c0', 3850787], [4731, 'eddf35b837b5f8e81664186aa751b8b60a64aa7d0625507146940be40e811ebb', 4224446], [4785, '092750e3300f63c71c5e32fcdeb32fd0bda5935b3f05a68e55dd3249c141ad7c', 4109858], [4823, 'b50087d0e8499b5bc921e5b02b342bca77568067ebdba2ce7b7ae83006066104', 3834580], [4840, 'dd1f68239f7180a02b41de76b2ff3c5bd167e3e818d518617f51a7c9a11f29e9', 3759439], [5016, '4044af3564c8b015f928214ad84fb94a4a7c03f6014472f8a04ea6ca55c76a67', 3830072], [5105, 'a693744c4957a380a3ff609438fb2e5b1f94ef9b9e904d01a609d724c3e8f028', 4319755], [5113, '24d52c4722b3975b84dfc67fdd7c97a1f74131f99745cb8f57e6044f0897286c', 4074094], [5184, '7a496240153f5db027bb65f6e2c71cf3fdeee40ace6a71f0023f9a2347592758', 3955672], [5205, 'b0c8b090eaaa479722e680a361f1a62f2532d67244b22a68c86d2112cf594136', 3909893], [5288, '57a175fc2be2b03a8ec80295ab1caf16125c5ee5e019480a6f6bcf64268d0788', 3971381], [5324, '0c3c1c6846c92f2421d604cee4bf312a07de61db434d5f8385f60cb0f35142c4', 4279629], [5328, 'c707fefc313ae2fb4952ee8eb93c2f7204a93d54f7065f32379747ea139c2495', 3700886], [5336, '179c7b5c53ae085a49e12a8dd52665b7ba0e85f20bba9706dbc837b157f3598e', 4186568], [5476, '8571df8854f1376082d35befa62bbdba55441c11d88f12c006f46c140707d028', 4461516], [5595, 'f8ed5e3d3f1c1828cdecb73b28a72592f33a5c7ac27286b94e76118c98ce3205', 4283245], [5670, '9bf3f0ffe2aa973cc29a0e006bd851e50e121391129867143f854168ad1115d5', 3669065], [5694, '47bfe79831e6db51602dd838d5bf4e641beba839ff02308dc285523a1bdcfc35', 3844549], [5756, '0440aaa41c96443ba41f12b67d9f0575aa627a468530ce72991800adc742cb6f', 4847001], [5757, 'f5c29fbc7c7d311228c459ce33dff5f6218e0e67e515b554eb028430da9ddd09', 3894087], [5809, '9e346e227ee5914f0bae7a9eeea0ba511381c06728ad21ba7963ed2012c81770', 3743148], [5820, '34499dadea501f5d5f7b16e1c201404ed1b2f9c99eb7a984a5d1556548dedc5d', 3698287], [5830, '03efce100ab91b71e19b6a10c91be819f4a3efaf64dd3a158e028f30db130ef8', 3685322], [6066, '7d48f54f06bc4ad91d334894cd771e6e6e5aa3af437fb29c1ee10b59ab527e2e', 4515837], [6075, 'b658e40f099acf09c251091f6de6e35596e515a70b0d8cb981521669ad109004', 4527433], [6148, 'fa1745387e96259b524a386182f99d11f13084ffbf87c5753f3e25b704388864', 3873603], [6151, 'b2fb187e1858ba0c1fddc969ca431e5177a28032291852ead8e876c4f4c5354f', 3895769], [6155, '0725fb707461188c441b5d5982bc1ef717bdc7e6581e7cc957399f076460be43', 4022727]]
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
