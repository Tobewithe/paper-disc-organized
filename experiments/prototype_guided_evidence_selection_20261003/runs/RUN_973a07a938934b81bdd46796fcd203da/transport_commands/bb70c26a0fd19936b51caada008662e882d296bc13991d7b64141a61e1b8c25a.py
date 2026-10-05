ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='c783b1dda86019e06ef6defc5cf45af153dba663cf279eddc85a78221c176a9f'
ITEMS=[[84693, '0f3773f52a6c4a4318e73cdc00f9653ad903795a7eea3e14f83fd11f6f9304c9', 3574219], [84694, '9b08524bc5cc8bbfbece79d2610bae971087e77eb0d51f2a05330f72b3e45b43', 5624909], [84758, 'c00b18390deb9a754e3586cf4e7a882510609e2ec4046eb7509301999aa57ba5', 4240140], [84829, '87b1c68ecf08868ca4314658ec8cc6cdd09745348d29cf73cd06c67d509a5f8a', 3969475], [84840, 'a3f2c9efa96d692e178609ccbb59b08b23bf5ceed209efb2fbf426a099ad672b', 3871079], [84853, '0b4a9107f0368b09f96376f646ccbb83b981c6060e1ce0ecdfc890ec81b1b653', 3727382], [84902, '85aeddec83f081e09d18108deff98ca644bc07826cc4df60c0981803a9326c9e', 3992072], [84980, 'd79dfa45a6381e8226d2ab40b92c4a03b0dc425f604a2cbf035a5191207c3ba5', 4164435], [85092, '0b64d4eac6b0f8b17f6a40dbe45a5012fd0466f64e90c358bfb3d83058ad14b1', 3831465], [85114, 'd5d3c33f2dafbf8248f0eb0238af587af938d90e83498507b059f2b501318f98', 4338934], [85176, 'd6b6865d0554c702879b0b55ed6afdf66fa57d4cb022e09317111f0abbabebe7', 3838885], [85187, '3c8d274a02d9014f62246bf725be2cecbe0cd0d7cfefd99f0cdd1eba646baaba', 4662798], [85218, 'a7dc8b593089aaa54415e90170dfb8bd632bd0a3301db0450a707acabb96531b', 3693572], [85329, '2e3b5baa85b9c7c23b49211cfd4bc084c3a88ba06e7bdd72731bbc4999dc851c', 3377892], [85381, 'd7e0006be7f9167130d08c4d83e106db5413c16368f33c47bc4e5666a45a0218', 4247157], [85423, '4a102a9193d719274b65ff701a1beb758d6a64c5fe0cff20304c4cc95e3922c0', 3981039], [85475, 'bdb7aed7bb3355b9e882bf8b373adaf4cab3a8e22daa36c5356caecc1fd6398e', 3825469], [85478, '54629eb5f831702b90ea697ab068a7d7cccc59fb2a28465de0a621db7e2deef2', 3666403], [85589, '25fa16fa6800b0627614aedeee98c0e9091801707d9b40effc12be91b94d3289', 3844692], [85626, 'aec05a0a7d4a3225ff55832b1c0ac92432bc17b0c2dfa10363dbdf90094f3b0e', 3774054], [85629, '4e4a7d9775b06e82eb68f727c4c65f3f20607694d4b3204ba1682bef024eb083', 3856908], [85665, 'cfb857d5331c4251e307645ede3bb81bd5d98ddf122514dd96f26f68c3a0e191', 3424439], [85682, '54306f67217300c77c1f84f8da41a8cffbcdab1bf30de93eaf2ee34d38eb5278', 4142899], [85750, '9f98e4b8920d48e93b6c967bc1830e0bca08905ded5c83ff5b9c2e9d2dc8a25f', 3652705], [85772, 'e0ce55997b0907da8a50d3237616077fa5157e084f62481e6c71679ab2e7dd83', 3753506], [85781, '9a73d7ecc5eb67bc00111145947c082fc90ee6f401bdfaa942ae10d034d96dcf', 3795745], [85798, '628f2e87ca46589693910caa592df2582c22bb357523b53c2c2a66bac7bac47e', 3964049], [85823, '6b384e07f0c076a51f8efe09d99df9ae6309fd749125df638882f375b57338ff', 3871723], [85846, 'b1198b3d120eeb2ff6b260fea7da651d7c97e3731304e367d35566d59333542e', 3948686], [85865, 'f42f6e98e4dee5576f66bb2128e02a804c0ee96f4641c0d014d26fc8e9d377dc', 3917119], [85888, '484bcf62852dceedf5f324ffbbf2510ee2f7424f765e185d74db685cc484a367', 3983984], [85940, 'cdadb67adfd734b74755f60d6bb1283c96cb78396a57339d22378744dd166910', 3821677]]
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
