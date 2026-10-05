ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='dbc2bb6e0d49886d75af0a19da256df4becdecc363e4edda8ced921f25cbe07b'
ITEMS=[[49217, '5ecaa5977b86763c48edf65b3f7bbb42d48153fe53c0d87654f6e92daa5d8d2b', 3871685], [49338, '02d61baca018d3382659cf4451faa359c0accd150dc0f7cbd55b110a9c1665b8', 4147672], [49429, 'd047709a29b54d0320c3183fa07ab9d435fc6aa9a41ee9fd9cef0f42f77af957', 3914464], [49473, 'dfdedfb9759192e9a89c085148c532c8e4fc3c5efe348b4053f2a14c3e8cd4de', 3466454], [49491, 'bc8e8a2a27f6dcfedc9b63de2ef337c86c84bfa7459bbf418bf392a275bd052c', 3829009], [49508, 'ca84c076d1ebb66826a773e8cea5e36a047e5da4115c6b5df32d36e2fcd7197e', 4099380], [49554, '27509e2929d0ab8b2aef5f5e278597dc264311c50cafafee24c2a33588f97f7b', 3891252], [49573, '146b173a0b33e31261221576367c003164a988cb77f828e5d03f7a5ef1cd622f', 3706494], [49643, 'f0110b939a7582929b47b3b51cf54eba566642dfd6f9c02f74664c94c36706cc', 3803551], [49648, '178b75c56771e5b728a3cddc76d161d56dd7e168faaefeda606442296b273cc1', 3699351], [49713, '331635abe74f75ade6247e963fed2266366924925926e0f81a3d64f59f9b079e', 3685475], [49719, '0c82438b243e3e5623a35da8acf20f936c16b29999968ceb460483338895fa10', 3940093], [49720, 'ed7289f67264920717bc37a8127e442f880e8aea2b289fa30683a1991d6ff602', 3512903], [49746, 'eabab1fe0a03633838fd5f8828244fe766c88159c6a5e7436b22a4b5b1be0811', 4116904], [49810, 'ec3937e8c044a4e59c19115125102f8a9243bc075abaa2e8eba12feb29daa9ab', 3768617], [49942, '6156695581a25bcb971288291108eaed7caa13457c8efa91d4eab4a144dcdede', 3819067], [50025, 'e75d0223640f070041418d4180ab64cb1e96eb35af60dd17b3829a0cf388729f', 3743814], [50145, '317d16ba29aa611dab1412db073898fe83f073708b0ac29b06841bb610cb17d9', 3968533], [50165, '1ef16ff8dd9cebc4b1b5e6a8565c0c651c7583b44498ea92d0213d58affb0c24', 3685260], [50274, '5da24927fb5c9a7340bda02d4b0aaf9739399464f23ca7ddd2d74ad63a783a64', 3818375], [50331, 'b9b89bc7ce288287aa993544a594692e3d9ba1e8a85f036e411c517ba244897e', 3925912], [50340, '6566c7dfcf0a21f23ea5860fba70db5713ac23a876456380c1b2cdf7ddae500b', 3842874], [50380, 'fb49e7df75046dea89445ab6ca81e1453b70b83bc9ff060ed89c4961e9631a25', 3850161], [50407, '71a91bfe89aba2b7acb1298a7768e8604d85a5f343b3fc0b6dc393c810817e32', 3898823], [50409, 'f1a1e92de4336238441b89816e53a84fddf44bad741740cb6b2168b15870715e', 3789588], [50411, '2f3ef6cfa6f0c50fae7f3cc00e547912d6aeb2d0821561be8cc5146a35bf2fd9', 4053116], [50485, '27f546fd17c5d97632b08700d2a9e45dc938b6c685332fa096ed040c8fa2fdc2', 4051125], [50536, 'd0c7be42380cdd0f39a9c4588091dd5cc4ff611b58913ee118baf2d681a8b4d6', 4095095], [50597, '511ef4c09d9deb2600153b58e59f9e04b9d764c77ee24e99c2c34c1733817119', 3939813], [50679, '930c96091fb3df90cbe56dc14b8837cc1adef8618c93416131f0128489658c11', 4033582], [50772, '9b296d2e0e8046eb4e46fba117ab130f1d492759fac89c906f9226af911df220', 4758929], [50819, '8c55a651ec17363533b29067c017f26dfbf13a3080657ae0d005662aa429a219', 3649439]]
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
