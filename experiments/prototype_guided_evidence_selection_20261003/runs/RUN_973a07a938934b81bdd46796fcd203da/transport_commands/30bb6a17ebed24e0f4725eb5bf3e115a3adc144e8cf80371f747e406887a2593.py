ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='0f4518f60b0f672d30f950fb86885f2ffde5b221a4d61da196c54e4b58ee0f88'
ITEMS=[[68331, '6e63636c288403af3c5cb317c1891d4b963bb6754d42f285b7d33558d314fc20', 3892987], [68335, 'a25e39bae66fb5a9c93bcb7b6d202dccdeec513497d526ead06b0609e946f978', 3924391], [68385, '01ea54a2768d01c0d4b2f041308221a9416f4e34d2240b0011859f6025713f49', 3897170], [68401, '76f37e67216860c6b8c7f9aaddc6579be177d53b075287ccd7e249f82258267c', 3690121], [68409, '11d859f3c4976a5d9ce98b9798ba12961810f09c377decf8e001e78daba3434b', 4045247], [68418, 'd57b131daff637b76474730428752ca7aeb9dda0220e0a9674abb0ce65e8753f', 3829558], [68532, '4b37de8afbc90d6185434b0da7ae7f8a7ec86ad86ebb11e089a345849df64218', 3951031], [68609, 'a7ccf9d9d22f186bba3637b743c7a727f945641e1f7ef961f7ebe152ea4ae904', 3802243], [68623, '7e1ae9eb60522f93bc9f1ddbab7760f5ce3c1a94279adde1e91fb9a2bb0fd1fc', 3902523], [68628, '9504e87d6c7b79fdc3f5c6122d8a1d0f2b09d1c74f5dfdaa28de7afa5a767803', 3646488], [68656, '147e68fc1aa3bdff35241407e9b3cac2811258185267a2c81705887523af9de2', 4188399], [68713, '1278aae460da443777473a940b3f2fbc180c14e4e60e13e069cb87924f0aec5c', 3845786], [68752, 'a4a8600d719388f1f00783229313eff726e989f8b69f18510499a965d64d9791', 3791726], [68764, 'e1d9d7d9d5c1917cbcd08d4d16a110b8c066b1030f64e14e6843693dc08e67ff', 3866584], [68765, '2349c7df06089290949ac53b413e3ef155cda21f8ae325e8040d41d361ad3def', 3770288], [68826, '77af529bfbf897ea5cd115923013888b9bdbce84b2192376f69f08fd43b43fb8', 3947445], [68933, '3e7a1c6adbfb4e20adf76fefcd008b7b0cb367641dd0b80b1e30b92e4da65de3', 3844213], [68947, '3c0821da44d14ac692011ec241f59bbf4caa4daa4e7ef5b62a69788168da3364', 4118671], [69015, '1177a36e4afa2bfc5176f3ee180530ddf082ffd5a5b2b5c150326370c5d7d516', 3917231], [69054, '87606aa3fcb6ab3e3851aaa0c547468c0393fe7d0a7485aff5cd71c483a17300', 3561044], [69106, '0ec692644235b75a9e01761cef498c791b38898eea31003c4880757003ad812c', 3797409], [69138, '44d80b9600c67a784966348e47fc73021b067a4a30bd35b28fc03c52ddf412be', 3604601], [69209, '6ae8e89fb7bc79df9221fe6c5c36fc597c361356567c6f6311bf2248c3aaff62', 3781137], [69213, '1a6cba85325cba634b6c85ff45ea8e06efa82a508f86b78812e5e4ee285b6016', 3973528], [69224, '803956e1a4444ce216f78732c3c975e820e21a23dac008552ca289e6ffd86c05', 4099216], [69226, '87497cc7b3d2b1c8a600e41760ff6b3e25520fdfd789e2901d53bbc7cc0cb382', 3642637], [69257, '10db0ad347b39fd6502e7e9023124aa15be3e5fec48bb3e1166294a884c9c91e', 3846285], [69290, '8cdb65f796eb4587054d443745690a8f414e0d63154aad55ef7681ffc797b4a1', 3889443], [69577, '3e5a0c79331e96e95b35943803c3dddc258165ebfe17738430cc9a96c43aaf7b', 4600890], [69625, 'c16a9646cbb5ab6d522eb68a77800cf16685beb9f477320dbb84792e13ad7762', 3829115], [69768, 'ae2035901a71e8e1637a9fc659093eb4f9da785ffe71cb1fa89e34e3f18a85cf', 4272881], [69884, 'f5025f3d796cc5f11d39f5fc2cfd89e65c8d28d00dbfdd2d71b4286b1342e51f', 4133802]]
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
