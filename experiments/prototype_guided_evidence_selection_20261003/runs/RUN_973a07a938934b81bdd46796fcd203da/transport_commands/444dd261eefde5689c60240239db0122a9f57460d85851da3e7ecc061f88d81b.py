ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='8364b7ff36c779e0885d81f443af8d9d2d34d89c6b766cdbc014e8475bff5c6a'
ITEMS=[[91006, '8abd2c94b1993496efd1ddb30ffb22c9ea5b464eae4af1da1a7ed828750e4573', 3934941], [91040, '282987ef71816976d0e15e9cd2bd6fd00287249a9add6eba7d53d0b7c156bd7f', 4203241], [91056, 'b2545853adb70705f88fe9427cbe54bb350b587d3e42c9425350a7406e6f58e8', 4403540], [91172, '79c0ac464a2d2252af7dc3a0d4aa1a1c5886ab241519a7026ad78490b8107683', 3718833], [91195, '48fdad24e8a41f0f20dd3c3dce02c360ecd4ddd5bcf81e606a8f46e786ec0faa', 3794615], [91211, '38b00f815183403a2ba0e0d3628c012c5e050f0974b2ce432b45ad45413d65d3', 3717321], [91227, '03f71de7dc037c487a60869a17e1edc8a4758d0a9634ac2b8077910f52b32762', 3613444], [91257, '8dffd5152cca9b57eb897b47e71f7cfe57d74567f964b8aee53fcdfa17fe9049', 4350741], [91307, '3ccb7f09fb5b29f9b0205812319b0077f533daf8602f02a008530eeed0353639', 4147674], [91318, 'c78f66a9ee32d2a075f849731eee1fa673035087403228c070fada6c3e51d0da', 3974167], [91343, 'cd9b574690668d35f149469261ede3ff956bfc10640d8c73aa8ff86f3f70e2ad', 3753062], [91366, '6981680df39f634fd912e69e2b942d212209ed7cc9fc900bd9ccd2695a882b0c', 3689789], [91372, '3ef7055446e9a57b39cbbe9d4523a395eb4f6485ccc9c0084d0c7435d0d32d30', 743835], [91406, '0c7b42a4a1b0a7523545908aac1aca1dc394fabe1a2e4b853a3604d685559abf', 4852794], [91436, '1426f427daa41417e33ade194feb54ffb5702eae3e0c9a1831189f460d91c5e1', 3938613], [91463, '54979192545c54cb684d4c1cbcbef1c639c3bc7aa737d0eb5f722a0242125e52', 4147779], [91495, 'eded7976593d95bc230483c15ca369fe9295f109e0a0c469809cc659e575c357', 3728283], [91500, 'c7850a8ac30898e2436d78173305c10e1ac249f8ea02d01e41fc6f0e9be6a8cc', 4395799], [91520, '56e03d0311f927bc5a2de826c188514cce1e85c681dd6fc90659860a27efcfe1', 3910102], [91541, '4d54455ac947e65ac72cc94a7f2840752fa1a940db175979c60bd2b887d4be91', 4448089], [91546, '999d8c5cc590edc27615895a8618686638a63216c4256d60584d76b18689a1ab', 3692214], [91547, 'b36e7c8c0526426039d7ec8cd52e8169e02475028883e83021bc2918ee77835e', 4029208], [91553, 'a1285dcb19dc9f341e4b3bf1d60cb5539023ea4acfdafd1939074d7e85cf4132', 3767525], [91615, 'c8065c7465edadb7d9df68269cf8f194658d884625280c3dbe323fa299afb33d', 4096460], [91639, 'd4da1ccba991637f1c28e90c233d6d4d307f809c046f2768705d469697398859', 4106479], [91644, '096b9e94abd38f27a1badf366b4222dd20b0729c4fd8f0d6acee069b227481b8', 3647732], [91744, '4962cfdb98b5e56f073594b2585562e41e5e4fbcce1b0f107a1d7cb1c5947ecc', 3971073], [91837, '81721250498da68f5039a9297d1e6dda3979b28ce70e681b13f8f6a634f46576', 3770678], [91883, '88289dc92ae71148340a0f79fc7d06e7a04e784dd8c13062581bedbd42118384', 4518651], [91921, '356352c7a40234f9aa385cc6d2bbbcfc24d4b682dca51eba4f3624ab9452d02b', 3886501], [91942, '63f6d5f2fc200441e64b8bcb6d235a8c26311294a4a0e530a276e84492d4646f', 4039952], [91994, '2acba43339a99286ba9ec9b986d4b44568a663a3b2b5e28c1098bb33a5211918', 3695624]]
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
