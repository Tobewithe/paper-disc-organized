ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='8ed5f6723364258c3283ff3d69ef55c679663ed92b72ce031a62d66b4b05ee95'
ITEMS=[[100187, '7ad0efb1730c0e9df0a9a827426a9cc47fd536fbb602fd68a93beddb0fc667b5', 4040528], [100238, '3ef5d0392a143470e65b5e2a7b0ae25efe52d2d39de94a36a79f0eb59382b463', 4137855], [100331, '7d43a3904808e039f3199031ebf2e0c1a7d0507432c590b906a60b41404f99a4', 4322990], [100386, '9c9932f259da85618a4f261ed3d591407b3d6baa43f7782b4aff49d9498a6558', 4006179], [100395, 'aba55b5463db5c11b8bb5bb159bd24790c39e7c3bbd38b45543dddafca84bbf9', 3706464], [100404, '6fcd302c3f81eab5448cdfc1f33fff6b8edb38bedac7383270b29aedd270c11d', 3380198], [100510, '93649e4c80bd5dea03ba451b7ffbfce4ec21ce2e141e2bc4f40ef7506a0c4f0d', 3917231], [100530, '97e8b46d88c57531e35e34ab2afd81bc46cfb8f56f2f152662b01753bc71cf6f', 3728112], [100582, '20d617f024e66c6baaae9b23a98492c28fa577caa243ddaf3d7b8c4f863fe795', 3794873], [100678, '030b85ec5ae4377ce20c729bc08df6c30faaeb17bd0d2ec40e1c950c92d15623', 4087962], [100698, '3c36bb7c1cd3f85b92f569c077607fa7b21ee55a436c5e637975ec6f2608361d', 4023332], [100723, 'cffa018975bd13c4ca8d7993a3b669998a64c3efecdd21febb647abebbcf7547', 4300849], [100777, 'd6754b7304035ddb5f245a4292ec88592635853ea4f346bd5d40aa5ef0b1de03', 4885036], [100918, 'd1447bf53f16a0d660774b9499c05260ebe0d24400eee3400ffff3f54398a1e5', 3747460], [100973, 'ae4da87daac4e15ad1916c374ca9b5be88d5a707f3eb475273c2222ac1a915f3', 3941831], [100975, '93e573d9365dd4d1af43925c3b67d8f30db0e85aaf91509af7a29f9326c4849a', 4493425], [100978, '72678286753558d6f81b27bc32236aa465a207c2c491544be15b158a98b09aeb', 4409053], [101022, '96e781bf62a1f39f4c66f60021f93d5c1499ce24a66246ede778e157a171b53e', 805100], [101068, 'ea9aa357e5d64956ac8cd6fd0be889b3f52176fcac53b0304189d9e0ebe7a515', 3742883], [101073, '496e4df141fff59a5a279c5981388b10a21b9d8881cd739622dbd384ccd8d0cf', 810615], [101079, '3c21b64158b8c6595906da4a212d141b5a760f86bd6038b7f47034a4d6ec456b', 3828872], [101090, 'ddef9b75fe74185d1041b64bfa12b71d107d3c71d3f498de5b11f00695ca4289', 4426151], [101094, '2e27c2e102544039f47cd8dcf16697dba0c2b358f20c71e84f7e05709ad9c2bb', 4408579], [101223, '88c7ab0acfd7459551e625b56a26a2375cbb394b43c6ba4fd841728922f7f3d6', 3720540], [101243, 'e3b8390f3ac4f7da794f98cf1893b0cc92a3f23c55dce1a947fa7420c8cb7ac8', 3741391], [101369, '2bfd9f2d980bdc1dd396ec8101a1d426b17e4254134bf88fdb8e0dcbad029999', 4268659], [101420, '3871885d8d055ae7dba3d15afeede534e51c3a31d9b5f2bdc69c2b6aa7937170', 3711526], [101491, '49e725939104043d3ff1e4d9a5f011ad6603bedba5f3390b651f9db4180b74c4', 4337024], [101515, '083997d22cd6b1a244d5b25bbbc9397a82f5daee64cb40414e426a3a0cb6e12f', 3565492], [101762, '2c53ddeb6a2dd55b6250e876ff8b621c7d0ae3e6862b0945c2adf6415894640b', 3809455], [101772, 'b6ac45c80ab74be34d468188f09de3834fb771e510b76e9ebbb16b6a6fb59116', 3963506], [101832, 'e9195b2f730743424421ef7b77b1d9f856bdc632cc954acbbfea5b4e37f03116', 3823307]]
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
