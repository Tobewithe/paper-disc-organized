ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='b78c088734292828110c649cad0ca52eacf0a80ad65875868a12f5f53b603763'
ITEMS=[[96304, '2ef307674c5eeb52cff31b176fe52bcc3c9a8d465685743cb04972b91e7a85d0', 3742623], [96453, '1d6d60fde8be18b049fdab4bb913005c8f00085f4ba87cdebcbf5fe19cbe5f14', 3882387], [96466, '7c8ec706b6b3c65f22de788768fe226e254b67735461732cbc76a1829ce7a7af', 4077699], [96496, 'c98bbfc9e5bcaa5b7f77cf6a3f25923154675d1cdf33e1954263388c79dbf15d', 3773518], [96505, 'd51b2ccbdef64860d529872be837f5632e1039a2d490ec7a2b1382a65af6b99e', 3792190], [96512, '4694b28cda1450658d89662286ea12899d0082935469a32511570475565142a9', 4205887], [96517, 'f48212fa6ff2155893a75f887c425ab57ba5304bdb8bb421eb060f95fbd150cb', 3732163], [96705, '3b9434a605ef64f1420b1a61f718a02157660a8dd2da6b5209d9491d4ef2e700', 3633591], [96712, 'bcf1881488e31e6aa636c914ea7476285b1d3f36337a92b96c815059c285c682', 4254948], [96732, 'd3aef995cc753007f25ec90e9c2e2a7d10a11f1ec1bd59fd24f4db6d7ee9324f', 3706152], [96826, '1612397e401ce45fa64c334f4bf64b8682f6e8a011c625f89be45dca6984e3dd', 4001410], [96881, 'f48bcefab21831380b0794c19bd516fa387242ddff4aa32f3d93dd88df793e78', 4309867], [96960, 'ad237651c8afe186f59279d59e9e6f3d69e81591097ddb53288b255a307ec8dc', 3761040], [96998, '4e8268e67b950a5e6bf251687c7422c2e1dd8d28ef137a441ffb75e50f072c68', 3906141], [97021, 'bb7ac57b0792552eefd4984f1cf01f8722ea78e3c1da8a1d71631042d16045c0', 3961145], [97049, 'f7b7ef7c9fdfec960af4a7e118423ad5a7377a4ef43435fb0c0356271d7a983e', 3666254], [97094, '1f29a473d30e7b7598ac4e9abac2f65f3f405f7ca14a0ad78ce0445a67343f17', 3766366], [97146, '752d44127e519e949cfc8a2edc67606add6393772b5609cc1cfc29ea46967175', 4009425], [97209, 'ce038baa4ccf03ea96d1857b7bcf3509c8858d9f56dd2efdde92be82268ed13f', 3990599], [97321, 'e932135df1f9f42dd2ce84370cd2dc781df03174c4b59b6d62c1caf21612480c', 4201980], [97334, '6489535c1f3b41c73310181f61829b9f6a8cb5e566ddbac56c68ede539c10d89', 3806818], [97365, '8f5abf94ae4b2f68743e0fdb9d0fe9371e934de4b98cc43b0e0703621542356e', 3864986], [97503, '853fac73280a9599789b8a97de8df65e4244c123bae9a13f51e7e45aa71f5dda', 3708341], [97519, 'da07d33ad0e4c0254e79da042089a8ee8259091624e42bd282456a305deb7523', 3767880], [97543, '7ae8dfbffd564e182b528e923ec97eee8d4c162d28d09dc2e3507b1c9f671c85', 3822933], [97599, 'a7a30380736be04ca4faac796ea59e87a0cac5d25212d78ee9e14288bbc33730', 3632784], [97660, '4c668d30cd4e294368bb23d94fadcb2f0889badff29161765af569eea63ba1af', 5330366], [97662, '3f053a4216e54c0fe1022af9e6eb6a57646654472b49fe2f1b467b6b40b8a27f', 3698615], [97679, '17409c0d02521825840ba5ecf49f3be3abc65af40651a9a3909af42fba8e05ae', 3837519], [97685, '19492fa4e8aa18caa946febac3c0e574f9dcb97d3bf09b93e89d9fcd241eaca5', 3940129], [97733, '23ce1a3f641b63bbb61fd63776d48e50d7928653f98eed6b0023fbe8bc14a4a6', 3987128], [97825, 'dc3a0cd949d28fc632e5c82868dad692b63e1a4ea70c9cfb71f9dfc38fdb2a03', 3531598]]
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
