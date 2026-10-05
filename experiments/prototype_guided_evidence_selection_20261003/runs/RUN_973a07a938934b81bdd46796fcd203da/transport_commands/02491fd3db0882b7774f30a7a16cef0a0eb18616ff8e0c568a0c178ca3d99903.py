ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='1848e026c3df2398d2623bcfdc76e90c187e6aa9273d3d189ed20b3d09745b27'
ITEMS=[[55135, 'dabc9c16b0608f50d0ca0aa0b3a0c31f928cec089727c5644e4fbda4cf28d801', 4237438], [55150, '34fe3c4b91a6fb613d509214a03f8d33fc937b364e132b5344275d5638db1f23', 4208671], [55155, '9e01a18bbdb0ba134c555132b491197dbd34bbd0d54a42c90b0126a7ddae4bcd', 3960786], [55158, '0d23797a8fa9ba842e388d08ee551243110411e59d5359fc3692cd3c8842e1f1', 4106976], [55169, '6e012e75ade31dbd3ff62d03bcb1d2a005176212305f0685583fa9286a4b9393', 3941197], [55184, 'ecd6f63ee56152f4a673df7b9d2c70994ff6f543df8e5d1fe24390b126cfb770', 3739418], [55199, '4fc98e5ff95d6a25ef1dfcf429f5a80543fa24afc527feabd61e2e59c86f6d5f', 4043263], [55221, '0ff1539f0f974287f1a95ed6cee6dd5bb9d1a6f8cb98bc2d1cf60351dde50e1f', 3778445], [55232, 'a87cbf1e51152896c852630cde7a654f8a0b62ff31d44ab51abc9423ba3cc4f2', 4296507], [55280, '523ff1c2edc251c319d6b863f9cc0638a962e645da1d7fdf9ad2ae80bd519a55', 4172428], [55299, '9d80aa6c662d2c38a068cd4d82787d6b479e9694530f112a2adeae6d31726790', 3701196], [55467, '7fef8d4613cdb4da83a4adf1297741d90d50e18024e42b4a55b24ff5d7466e86', 4649860], [55469, '764b78075a45b36f1c739a5447793fb248158a6ced4c8fefa4b47134de40b65c', 3842510], [55544, '9c6c1eac5b244dbe21e877efa59d20a031a4ca36152840d5b7deacfce9c71e7a', 3744323], [55629, '019f944e85c9fea9c3aa767e1200333da7ebd6144971945e4877747091b90974', 4257235], [55776, '377252c88d85acc49f8dd4159d928071b391b2653dc913fe253e90e915e5b9b6', 760471], [56002, '88864e1820b36cc1754f4dd95771e54164e16fc299ea94a92f4bcaf0bc7b6417', 3990135], [56028, '4b2c5a125bb525fa56d59fa8502fc1fbe79b5317011117dbc1593e057b8832eb', 4223029], [56233, 'f84303da7508759ceb9ceb5d27f17e3d2ab14e8bd5a3071f43dad291cc28cea1', 3916166], [56240, '65cba094b74cd1a7a43067cb6700fc5e34a103de87f49ba47444682d38007ada', 3814002], [56397, 'cade9117e357f51501b9a03cfcc570b11c193097221b60bf4a4af5f497e46ee6', 3622273], [56426, 'f45fa0ec637471275d9df952a60325d01f2512d5d9235f552100aafe566d41c8', 3742513], [56480, '4d7b5739ebe98998679e37c3f86e08eafb6b44f0e4d0c2d720e92c1c4f570212', 3901514], [56504, 'ee36655f86c4c272decd3ab9cb1107c41dc6c0235915f5564d2c6d708ac7df07', 4331592], [56508, '813fec748d284bdbfb384bf076365ae6da7fc84cf6e34fd3b5c99139396dac9c', 4516715], [56580, '00faa32cd38285ceab93083547f27a6737bec6544365d2681c7efc289a752d3f', 3832539], [56608, 'e4ca12e8faa5d539e933fed3ce52581a35f26292e88f842779394ba7e788af02', 4038385], [56646, 'ad2d2c3de462bd21698908bc0d938b772a648032dd36e08b3f5fb8bf408b6cfa', 3527229], [56652, '2e59b5c796af51413ef28c8b34df32182ea609589a09b50cb221783ce87d007a', 3829881], [56695, 'ae6bda4fd49748fce1dbb340bab305d82f4b9f1e8d63b8a6b718a28031dd96b9', 599934], [56716, '01257539c517c653a51c92679d085f9ab1b6b0c380bdec4199c1b3ed9497cecf', 3716247], [56733, '46df7f1163c1c5082edc38755d58706d5fdef2880de91bd1f57a0e4b1d556bc2', 3811460]]
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
