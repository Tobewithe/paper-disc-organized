ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='a7ea6f73a9e55c5c1e17394a974fb308028f41641b0ba1f4f2c43deafd23d881'
ITEMS=[[21639, '44f615702712438f9129962f8e8ec539f35a765498636524e23b310473f369bc', 3968709], [21711, '7950009f4ec67619bfcb04a63e18c5fab0f96204152280a7430dc9d55fb20f47', 4028323], [21740, 'a4786f99e33f980967606581a145d9207eed9d7891cbbec934ff1c9d37c5ed28', 4349136], [21772, 'b0d53d5ba8d414b653994b1e853a5d3fd904fd47882c9b74c95af0e9913fe8db', 3919610], [21776, 'ef677b04e896c3055920766fbf220d28da691be810474d009756c2a36029bef0', 3731938], [21900, 'e89d95c127f9afefc28e5e058afad124405a962107c0b122ff6a47a838d84faa', 3854253], [21969, 'ed3cc2cda53055af0c67318e4b0ce577dbc78863110866659b3d7949aae28943', 3655089], [22002, '79cc7f39050daf8dfccc96e7ce3182616e127a1a77d13231ea23da14fdcc4939', 3783723], [22059, '8798e2dba47cd62356790d0a4bdb6580552d9edcc97aeb62ab2db210929554c9', 3724423], [22090, '9ced2d1aeb8fcd177d7352080181eb3115dc205b6866538b9ca57bd5743a0828', 3801022], [22097, '13e385ba029ca3ac39e1d4bd4c1cc8733a8f3c68ba801890e223c34ad0b35df3', 3684961], [22129, '63138c91c171344e181ef55102b4665ddc1d960d6947a98a4a275b455da83cd7', 4124584], [22142, '0b030880c8933a2bce0d5c53768298878a10e42b7f3aea9f3861f2b4846494c5', 4379663], [22215, '42b003ab935edcb162edc31385c075f128df57e9377a24e6bbf2983c0e378049', 3871111], [22429, '51366de9929144bfb125989409a77205d2dde0e8060f9c761521b27bf894551e', 3826178], [22500, 'bfdb6a4661b700067148d319015897b741c842ed951bd0576253bac1960dc4e6', 4198067], [22593, 'aae789f9598839c9ec35b051a10957afc21c7bdb64e145667f8b403eb34eac3a', 4010258], [22796, '7a0314401ca4e5a5e8135aae06b0dddaeee5c054e7237f4eeaf7fe33dc9a481e', 3951722], [22802, 'd9cac8546c1321ffe97d7b168bdb52eac9634732888b6cfa2aac470c78d547b2', 4161549], [22861, 'a70f319be66a145a0cdccc9458c57d1a0d631402a494415282e96a1895ccae3f', 3906384], [22863, '39756183ca59992c1375cee0fe8706804c881ae4e5ad59d1ed596c26d1d715bb', 4974520], [22929, '4575bb485fbe798a4ff3c2bf425eb3135ea82b694bfe0100ff08ebd5f8ac0387', 4023963], [22940, '97cca18fd09bedd1e4dc114fab0720cb84e1c280c476951a66b70c6aa4c10ab9', 3946195], [23035, 'cb7eed57a68c9172e2b6241d7edc203c643164d12dbc9957030c3d96b80950fb', 3857406], [23173, '4537736be8d12aae4c1e4b63ae5969596a9dc7c7c8f3c507d9bd32d46b01682b', 3751583], [23274, 'f3d80c4e4e78e3bd8637d5273bd56ce5545f4a1597abe30a35b6789c63e21b8d', 3932634], [23355, '97c4c65da20dd560fe0a5a574b3e509ca81c3146ed49b5115b91e995d41ab8eb', 3914250], [23369, '73d25ee09249828c2442039d9ca8e8fb81670d1c5d97ea5ec2c60a99d4e31235', 3700423], [23413, 'f78432e6d4576220ce5ea1e7fb7c717feca6819c16e6b646ad5f3089e2f0b2b0', 3853733], [23673, '5a0ce792c6ee72ce20418fb5791754fc9b297094dbf056e5c3b275f9ef5cd498', 3688811], [23786, '8c83a6bfe4f4a952464f04260fa7a8a097b4c6ff8db82de976285e32e16e7a3a', 3655435], [23821, '24571c4dc63210f90ddfebbd4c1ff77f90b7713b1814d226c17aeb8016cf00e1', 4453573]]
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
