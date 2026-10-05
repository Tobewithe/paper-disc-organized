ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='889c523d5df8bea9ecbb4c4397e97f983bac0c47a257aca59265f9ba300fcf53'
ITEMS=[[71675, 'f6d804be9334da31b87dc516d63d6c351dc5ab6fb3205ff2c4fe7deddf499922', 3926963], [71723, '8bb57d67ff7bb195b20a383687ae0ff8129772c9d42b06232a358a861ae6c22f', 3717490], [71738, 'bf665ce07c861323988c6d55cf88fc1df0594ba23dfade530ebf50e1040651bc', 4231437], [71756, '5a51c9a8c1b402163d005abce1d849d34d374e8521f33a94c2da67df25c1e130', 3756703], [71855, 'bed0d28ac71ec7a93077ec8ace9e471ff67709dd30e99b73f50662aa8efb6c49', 4320332], [71870, '7c70efa994a37a674e0057aff5e8b8db2386a4cbd57111470d2ec32f5f0df3e8', 3911184], [71877, 'e535158883461c9f7bf20c75ae86625fe9e67ae2fcd513ab55477acc068d5395', 3755591], [71881, '6ea88ff9a0ee830cb1cb10d0abf93ed1f2ed9a5e77a72b10ab3d405d2c79702d', 3991980], [71908, '7ad46b86a383a4eaf60f16eecd85b1c7801bbccfac99fe715cd1634e2abed671', 4112935], [71964, '22b578fa16f2f07bce28fe6679301006f3b5d3a38c918b168f7d73c6a8641e9c', 3805660], [71992, '0965f01e3c5e61950b2ca6505f78dddcf5187dd4d019757cb90442e71b263b91', 4832619], [72016, 'b88e72032f075837b1feb1457aecfcf522e4a1ef8064d1fb116b408c040dd713', 4212021], [72091, '4c9082bd2771495002e9fb9f8c47e6b9c008f46b18d6f05cdcf56a4f26c41c49', 3689943], [72133, '97095259f27175908e85b0f4b00db62474035ae3261177891979405b03cb5fbe', 3779564], [72177, 'a07c6ee126fec86a18712db5840d7f9de99503e5e34ff5c764ee564e579359dd', 4387728], [72281, '41a2a539bb272876bdb84c21f25105f57bfc4cbe7ceb73e1d6601bada90f8e46', 3686675], [72424, '7e79bfd8ddd657e044882607e59622389eee9faad0110db5c452aea88ecc3dd7', 3859833], [72437, '4cd2896546fa8257123fdf41f4f2901f406e967a71ba7f6d4e33be20cae86c28', 4107244], [72525, '2cdb50755a027d6d9ab00ff1bd1cb5fb4886f8db4af1ceb60c0c5a7c6870a61b', 3774241], [72605, 'f1f53b88c47274d0b7d4b2b8e2c9d1e3f60fa129ddc522907e976cf86e287c08', 3934509], [72632, 'cecef65f7993e125dd3610c3ebbed28d8cfda8b9f99a45f7c9b90c2d6e075570', 3769231], [72650, 'e907ac9e12d3718735c67aa5a59fec9c4341f859f5c6ce446ff224e71d7b5f41', 3848103], [72702, 'd3af85a6ea7013445d0f114a078eeeedbde27ceeb3d04f890f38ea68c7cffc1f', 3913344], [72733, '7593a9c738d0a43fd57ecf54e49de918190ac035d8ef7172fca9e13e34515bc2', 3682218], [72792, '8593026a117d652ecb6189ad18fb16bdd643f97360d4c8fc8abafe042ecd2117', 4957077], [72813, '5aaec6e442d39385ff106c60268e3ccf8c8d998afd57a6aa18e64b93deca680a', 3827829], [72843, 'aae34e7290eb76ee551c58cd67ded6fe7b99de75b54da81376a1d83e35a18046', 3699940], [72861, 'c1bae6f2bda80a7d21a12b13849da297a1f6312bf20c605fcdd788f20e47f804', 4085378], [72969, 'd8292f85b25b723a8b74d04ca14e3328af6f8b037a409533e1b63b1d7463a774', 4139012], [72978, 'a8a3377e81192bccdeaa5877b7ccc0e9a2c3d5b46b4a90fe204ad9dc62d6c60f', 643698], [73009, 'd7fdafbf5f19b3fb1fa67b57cfdb18d370a9052dfb590d08880f8e1d0886a0ff', 4027853], [73215, 'dbd1c66a0b9c66c2d490ae773714bdc92ff5f491d331d728be01fbfe1ac08c11', 3791959]]
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
