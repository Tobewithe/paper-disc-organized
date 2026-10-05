ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_91811d001f384d7eb952f512c631dcdd'
KEY='1e8130fa6cf4838a4bebe668429f06daab8404df80b7111226ad2641b3b8c706'
ITEMS=[[2684, '332b3b6730a0f4306630d776643b1928a52d83f5e252caaa624181d656ce32a3', 13831280], [4051, 'a72b7a72d12c9f05a9f8d0afe60e8ef220d49cefd530bb78c49b20057c0c6e2e', 13514378], [6153, '101f4e4856be066571d5156f2cd3e44c59d9d6e3fb7e8fb30991431ba480710b', 13366451], [10040, '29a2610d43f90439c44f8351521a9beea14d5104bf4f65d90996551823ce0180', 13623692], [13159, 'bbd700b7b7a2529e9aa9be8248f7592b2471810ce42ac77ff8a786e7e7965ac6', 13445001], [14781, '15bb9af77022d97ef2cf6dfefcd0a613d24fff3c56752f6afda7033f38ae2267', 13532568], [56288, 'ee14de34e2256f426710118f672748a1f09538ecb19891c84899526d3bfe6806', 13463647], [60835, '1204455af07df523d9c88e9121d71a37eeae3b60f47a02b35b8f9048b8ed649c', 13612458], [142092, 'be745fd2216b0ebb1d760cb04feb359b9671f46defbedcf46099e33d8c57dfb5', 13441645], [345469, '45cf8ed1e6e3776718474e20c879c498559271a29012b9ceec21c7fe6727b101', 13880179], [430073, 'c7e96e347e0f1da989bc02c220c4dabaee7c7eced59f87d1cd56e98f73cd28e1', 14374341], [445846, '2405958ada1d93d490353a22e72815ec3b23870c01dde814722f776eec36221d', 13368023], [94, '2cc55084e9c76eb1ad3873719118e8f43cb192203842405f1efe9e896d61463d', 3791715], [139, '18c6803a00799bbbf12f63f2280e3388c008be9a9be49b2c0c787cbdf9ebf556', 4275960], [387, '0cad9328eec3ea61b92af0efcaf1e69e8cc73002eb8ad302a04485fa0be84798', 3889896], [724, '8be7134d40584d1bb3fd467e02a04ae39eb7219fec73efaf75994cdfcf77dcc0', 3833271], [776, '0848caad696f00100b27949e09257706586baf6be0a4639b079eaafa79fff159', 3883330], [872, '6472d4d7c388c3edefc6b736ae2b0764789f042ac028aa8a9a70fc1bf69975cb', 4164146], [1205, 'f9983e4b732b1edd38104c115d1cb9c0e7baaba385c5a052ed14164216a857f0', 3841185], [1503, '13e2fbe54c6b08ee1cd2a4b7b66b31a7a6ec284e279ac7a252905c02a2b865aa', 3694377], [1818, '01a33f98174f518d0e70df054eab76223a479b7b524574f338e46e0092e249e1', 3807148], [2261, '2b108fdbc108cd707337895b490c16d2c76bfdd6ad26dbf2366a7726af093314', 3717800], [2299, '8ebe3b59244927056c2541e8fe641556385a154070b7bc013567da554376763d', 4456654], [2587, '252761b273a80caa9ba3d342f3477c6036690df618997dd245712b80d0780f00', 3710448], [2592, '51aa30e0c94f1524d2b888deee089e3acf23034034da81d699ebb9a152ccebe8', 3725086], [2923, '3e0e7faf406d7f0d14e5c6336cbdecd9284111dfef82cbc1a1aee9b59d0e7bc3', 3815582], [3255, 'c545baf823711bc07c239dfdf4b4b4874772d44ca07e2b876f92989b58581b3d', 3900980], [3457, 'be31320471c6376d54c3fc592d4b4a364d9f6308803bed89ef7f977eab2bf262', 3934259], [3661, 'e01835d271bcd04e5efcf5e57653b379b9501aa3a0e657e3af2fabc70d4ec54b', 3633558], [4011, '913883a54c3838292e529675329cfaabf5dba976307a5578fa373e54e2314ac0', 4084406], [4032, 'de356b9dc830adb03ce430f7916b75d9af771e344962b589c7871e05c2d05eac', 3401239], [4322, 'fd0b10f61ce58bac848fe987ea88652c7345da52ff02a8393f5ef3b13fd0187d', 3917899]]
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
