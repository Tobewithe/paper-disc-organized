ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='4d245a8bc99efaea3ed787ba731091a901467929b9990bd211495132d9d69ef6'
ITEMS=[[41087, '80980b635cd6c576ff66bb3a81cff147c42472330ce76130d11dd93404e8da95', 3882081], [41186, '65743186a32919aecaaa07e7afd5c7bbe2b964e7d99a0db7e639c0e9dddf6e9d', 3852663], [41226, '8733432effe480a8002ccf61a12defd6d0e0d6642fc19737f79a3a0daa9c55f3', 3733631], [41561, '62dfa6e095fe9a46a976ef1edf6089f3a198898ab2710c3241cff228632a33e2', 4138693], [41710, '3a5a6e315e89931f78ab21b503b31f998cc1fe214178ee269380967e50718d8e', 3800739], [41739, '2d01af387177146b6e8c8190c7bf4610764a406d5e51dc2f47560f1e84389261', 4043895], [41753, 'f0e4588b40555769d0e9e771087a5833367808b280407259514aab46dd486a89', 4041876], [41808, '574dd7e849b776ffa00afd1857ef31c687fa0f61b5141a45cb9d52cfed98925c', 3612208], [41872, '33f2760ea21fe127988020ff2a8a7165b3fb90fbaef1e4aef01b2e8de9ae4f67', 3793233], [41888, 'c4430e55c3f00b5e1f1a48501af7f8f8e2b0bb1a4c38e2a83cf7c8fbc4aad1c6', 3866947], [41939, '2e42ba1abf30da8d2e97cf13c19e9508b2052af25b135f6bf72d1c76c31de6b8', 3752411], [41998, '1714e4b9e47ac4ba48a4102c82c497f1ca10390e8f99adf7a31e3eae144e7177', 4329744], [42003, '5794feb89bf37c7293829c9db2051da68d6cbf6bb95abd76c5d43a12f360f3e2', 3676480], [42008, '22dc8076a5f8a0fb62255a873ded47054940de5d90473b6e6ba2c6f621d35f93', 3921299], [42010, 'dde57db01901b5e07377a4dfb1dac0a8a6e161ead91b535413fdc91666839610', 3815658], [42024, '6e1800773376e29d1813645b9666bcd000664254ac8a286932bdc471cdcc33e0', 3941113], [42070, 'edabb82ebaae5953d0e23501dd73146ada5e4dac5103e28440eac1ffb28788ef', 3896631], [42091, '77b24bd1d8b8ceb21f2521f8451a1a3346fa0853114e451e19bf15d49b0fecbc', 3911389], [42137, '66cdb9619e8c8aa79b6c39b781f79306007824ceca2565e61172dfd02d4b9e54', 3892537], [42147, '507388cca0cf6ff134b44541f0aa98ef8039b66eab8676bd27c8f95729e14472', 3474205], [42165, '3f170b7c2e535f14790c8a0134709c28803ce2cab788ca2f5b11cf70e4f4f75f', 4295552], [42225, 'd5855ab9f89eed79d8c17a241d15d7b43be82392dfc1f4b366a1d00261d62744', 3951862], [42279, '5dc2f9f2e236c53b319bea5a39e9f27312a5ea71912042133cb9ceb540714a4e', 3910054], [42296, '13a9a0ef0e90967c1d8117ee09c19bcd2763dbe4e754e5c7310efe6d382ea481', 3863327], [42333, 'ea69e876d84b6940693ee5ccf2b3e2fe5938eb3d13fb1cae7823a7f86fa7e96c', 3640564], [42339, '8a5b02bad27d6af1ed818c6270b83f6b29e0ff89d6662a508e4d24a3fa887a6d', 4054475], [42416, '938554b575cd93d02a123829bd0956b6883ba77cf556fda01780c852f527cbd4', 4143776], [42418, '2e17fa7e32d901f363710c58dfca55a595c15d6041dd5215dc68452c6a0564e8', 3909650], [42422, '55b239ff9b5603c0442650a3b303e7abe0ba66b2887ee34065f6c14cb879eef0', 3756538], [42458, '65a9fe6836b3172dcccd2feb7b2c448ec679b6a831a2940146905c94adc976bb', 3623228], [42471, 'fc7b5150fb42555e1e31702efe714bef74e66a78d0ed699b6ed9fee5bb23b46c', 3761902], [42526, 'a0fe683be6c9444febdc80669d9e45694720f5cca52c8a3cd26aa921920bbf80', 3813966]]
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
