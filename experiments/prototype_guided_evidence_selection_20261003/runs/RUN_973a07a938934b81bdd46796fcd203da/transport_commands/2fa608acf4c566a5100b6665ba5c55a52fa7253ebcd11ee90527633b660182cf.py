ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='63372fc9c4309ccbb0b4a5a8b9fb54b8fe6ef4d3e109999f65fd919e72c72934'
ITEMS=[[24861, '47086530f81fb7997a3a95588fbf0877b3ed64777231e43fbe26fc90dbec1297', 3762326], [25017, '1bd936beae265c18973439eaf07380e3309e75f7d55a6495d52051979b57d355', 3922462], [25174, '89e638a02357bb7658a875cd5484a1d01d32e2683ae463b24d0e83aae1a894e8', 3813037], [25181, 'd46cc081831dd38fd8722acaaa239f50eafdb5f276b037d973ace2ffebfbd070', 3956644], [25192, 'f77ebacd4c156c771a5aeaf73a8455de33d498e4777f71b0eac7f18a04041dc0', 4103250], [25245, '4ddbd2ce4eec45769bbeb2d5906a681719a6ee7bf8f77e20e1270ef0ef40fda5', 4188827], [25393, 'eeb5ef4d2aba856e9829225dd282451e4644c8b043f986fd7008ce364bbda9a5', 3963224], [25412, 'a1f3572d67a1b29482c57421b55e9161399701b4b65dbb71b6c25f560c9c4e56', 4084903], [25424, '7e3c43c9f83b925fe0590b0a4516a22407921d5a2c6c3e624ee8efe20e86618e', 3683718], [25457, '412f832c4cef067f99921a3d4ca19b930cda723170c90e46e7d7259424aa0322', 4120014], [25515, 'f8504139eda6d38cf32fb74d6f48af0d54430d8cb86d158ce8c1490001624221', 3806895], [25525, 'aff99b7b04882191bd4634f0e6ca073847e034ad48f5dde1b032ca9e769db10b', 3864231], [25625, '2fe792f226a84a599ee4a12ac2d785b331efe1a2e2cf51be73c8014884e60da3', 4038815], [25710, 'e6d9a4d87d3355491d6f4a1206f76407cd9d873ee99ef1e9f4537dc5ac15f995', 3739049], [25747, '9c2eba000ae059e2172bc6cce03ad3b79b0f50e8533123eb83dc849b847d6366', 3843071], [25854, '74b3ef6eeac26a0ab54aecc47dbff9a29c362675176461f9ca7c7ba31029c21f', 3675070], [25860, '96dc2d32892f2ce3f5271d6830876f8145b2912b4379f60f7be7987f4f1a26e8', 4445129], [25895, '60699d4d773c40d3b20a72e70274e8b52cabbdbf0f68fb6e4813cd01dd3f539e', 3541163], [25974, 'a57f0317c8fd5febbe7a9bb1f3c31101d579f82e72c174f28983fd2fe098b9a9', 3808590], [25986, '8af6ad812ba83a8edf8cfa51c2b62983949a2c8032966ffda289833c0d6355a6', 4397488], [25987, '36c63b1aff95cedf3758d68caacc42366bba3feaed4eebeb3a5602845d3efdc8', 4118400], [26045, '0cfcba22f97a41c92559e13b4620f8540f4752d32b1c9fecb99c9842f1c294a3', 3730486], [26051, '323bfa368306351ca26ef423e7822ea8d01d9a4f3e586b5cabd430827389dc71', 3978446], [26068, '301e0836ef75bfc311f4daddf02115f1d5b3aec9cfebf7e410db830894d66203', 3797149], [26111, '56286419573ba3164e5ed2fee9513fa79673d4350e3d55de9124a53438fcd348', 3707477], [26159, '6f063da5cd0edecc7c85b37308f25a58469b8711a01b9e8c824e3e881642cfe1', 3786190], [26204, 'b53e01f55fa1d49313edaa3d63d45feec70bd7d0d62d42eadfec78cbbb5b8379', 4176832], [26247, '61fd79c08664e06deb78e035a747bf7cd8cc365c5c836e3d272dee7bdd5e9d8a', 3764300], [26274, '9d414dbb1125a708ee23f33da93b0a42f9a35ad789f690b446c8d3cc7b647726', 4353792], [26281, '0ad2dedfe20f966ab9119c27dde6100b4437b14c6c69987a3072a2ef2fba600d', 3559445], [26368, '165abf31d3fd5f536188bf2527b2c272311a03d3260407f97ad455f4329cec77', 3740088], [26409, '748648eee7fa59722d69cb8f39549fc10bd7a2d3fb98d1dcebe7683cd5509e67', 4529507]]
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
