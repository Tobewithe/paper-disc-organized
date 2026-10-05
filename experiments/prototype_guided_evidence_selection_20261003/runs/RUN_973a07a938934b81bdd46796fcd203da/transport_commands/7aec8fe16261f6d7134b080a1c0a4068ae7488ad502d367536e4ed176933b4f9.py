ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='76a188c018bf7e44f4a0109bed02ff947c243d9213ec5d8a3a5521ef90a8bbea'
ITEMS=[[50828, 'b90fa0f6866527f08271b0802f930489b1a6007759ec572c6889fa73363c91d0', 3789910], [50896, '9d7a20b751087d343607d6b2e8636596d033f903c2c40a952a46fb0020e96367', 4504069], [50913, 'aa849a3e24e8745d5901e4f9e9af5f10521e3a57f6fb4014749fdf80ebaa447e', 4415548], [50924, '881f2d81122937cf284630ec743cdf9c4b151fefb0f2deac4e56511993510c71', 4006980], [50926, '3197bdb990618a94b480d63b34afa12d05ffa9135aad8c5caaf23e9e99b6e7af', 3793917], [50943, '1da1c0297a622ff08d05163c5571632c872ee2a14ef27d332744e3c25751f3e1', 3483414], [50961, 'ea6ef73e5691706a556bd48c45487eaf919d0b3645af2bc758f4fdc943837d69', 4071087], [50980, 'b2101cf048ca657d9414093a96cceb37f8254582fcb7fc106e59400eaa672333', 3972982], [50992, 'c11062e2c428ea876d1fc083f40119f2cf44087b2ff8eb86a8fa85f9f9b9e2b9', 3814671], [51008, 'a0e2a01dc16f182261bea322aaa6a2b3d9fc548b5bb53ff705d355f7fdbf71fa', 3797579], [51152, 'da613b9514d01b17d1d89cb9e229fd01b3f36b1631b0a6cb7f3fbc63da617998', 4280914], [51222, '6fe2df398edb6a0a9db832613bb31790a9ab68e1106d631db1ab53d68b6578dd', 3738045], [51291, 'e1f9aa2e13a719b344649dfa6a955ee443edc9454a45914ac9372f9ca1f58154', 3885941], [51304, 'c23ce9d85f6618f30ca37878368b0febd5dea72b0a27c60ea54ad6d26c3fd0c9', 3749606], [51322, '9508d18618077d9a5636f4da3c5593dd131798d37d4e6868ffd3d20b3b5a6ddc', 4467052], [51465, 'f519659febc323acbd2c0a7ae918c5b9f0f42c3332462d0f38d070dcef46dab3', 4434360], [51478, '1340340c07e61ba81996fb6798257bf98d8d7b3b896b50200b40119b2cd39349', 3801129], [51484, '359f2378f0856a13db4ddbed6ec31ca11ac993e23e5ba0120844e474b3d6a080', 3944773], [51598, '1262078a12ac409d8139342704fdc65b276eb44f5451accddd5624511b1eca2d', 3516837], [51610, 'bee8d4ad2116efbcba71b9cb90c2680c061b8429f516a2f28dd35ab548d2256d', 3457916], [51628, 'c8481779da932b806a333e18f78f69169da78e2f6c83bb8ac77e49008211828c', 4128884], [51639, 'd55ce47e9de428b74cdfd0b3c20f8dfd35ed0531cfdadda1899fd15ab4a547bb', 3837521], [51716, '8f95adf428d7cbcd82ae8ca2411b42ff2a6df90f0d1a861b7767717327c137fb', 3870211], [51735, '123449ae6c9a11121cc42dce5af9f409c47456923b583b1b27b54b29865ea516', 4097837], [51738, '48db3489f5da179fcbed2721986543625f8e9bdce5fe17b1b05404d4c0410e4f', 3659857], [51829, '618854f83072a3e786f82ccee214fb1962e60f79521677cf6b28eaa08f626553', 4529963], [51835, '1267175a45c669d66532b522464e53640c313c2a60fd8d9a814484cb1d65b2a3', 3971296], [51880, '6c69b3a6228bd98b1e695fe841fd87fb6939e284b998e5cf00fb11668c7d4b9c', 3643836], [51938, 'e0c69151fe52dbae543b7cada407721bfc451c493d5f463b046fc56a68a4cc29', 4046154], [51976, '6aa928d33123796580f22f12891f04878f7deacebec795abc38d628d72a13c10', 3673890], [52011, 'e653cb14c6da6a09084a39bb085531f647d6c99761ab624f6c6868ffd4fad0e0', 3494710], [52069, '3d1801c0513dea2e9a91b6be95874a07a25abbc562a1bfb69f434eddd82dd2e3', 3605043]]
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
