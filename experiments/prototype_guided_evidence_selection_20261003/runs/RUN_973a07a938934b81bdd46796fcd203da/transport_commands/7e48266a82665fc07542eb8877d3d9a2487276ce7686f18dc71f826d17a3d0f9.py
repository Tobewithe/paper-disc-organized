ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='b64d9cbd4b0ad328e1f7b02c18cd95a9aff2926678daec58ac7594e39bac15b0'
ITEMS=[[19926, '29290ef91b9202994df5108a67dcc11e34f30b1affe2843cd367d7f64224a9d1', 4036887], [19994, '2ef50da76b151f994a3d2165cf6cd6ce60ac38f66fbb502a50e330f80ef01708', 3959936], [20177, '223d6121cc13fe1adee3e93aa8daddc9ed471782bd751d9d000f1e3ba69baf7b', 5475578], [20333, '8f1d7b63419673ed8e801dc3c55e7a73479799790ca77b07637657e35e188e8f', 3811336], [20444, 'eedbdb82e5cd8ac71a4226563cf55146087b22a77e673b10d3533dc52891bab5', 4183987], [20553, '0fc66ed2b970269a600edb0715f8119c6bc367709428accc639353711d1d967b', 4112830], [21167, '12a66adbe181f1dce0065231e85105e6fdab3c15df32fe2c7a7a13c9d56b035e', 3651977], [21217, '582e2764b9dbf743f8dad90ba13ede5ad10acf938e6b07d2a6d4341c75e8ff7d', 4281212], [21281, '42839c114eff6f597c34675e6e388e6ed7c2490c0a5b86645d6e7ba68f675b4b', 3836349], [21387, '887faa0a72931e40e4efc117a1061816ecd314aa8f5e797e8eb121953ee90e13', 3841269], [21465, '1d16f5c248cbff1892bbbd961826b7210d7a552a113fff4fd0328ef26c7f8937', 3666503], [21639, '44f615702712438f9129962f8e8ec539f35a765498636524e23b310473f369bc', 3968709], [21711, '7950009f4ec67619bfcb04a63e18c5fab0f96204152280a7430dc9d55fb20f47', 4028323], [21740, 'a4786f99e33f980967606581a145d9207eed9d7891cbbec934ff1c9d37c5ed28', 4349136], [21772, 'b0d53d5ba8d414b653994b1e853a5d3fd904fd47882c9b74c95af0e9913fe8db', 3919610], [21776, 'ef677b04e896c3055920766fbf220d28da691be810474d009756c2a36029bef0', 3731938], [21839, '52ddc2c0f5cc4d1eecdc7744d319437ae4eb917e4b1a13dfe619143af42aff4d', 4283188], [21900, 'e89d95c127f9afefc28e5e058afad124405a962107c0b122ff6a47a838d84faa', 3854253], [21969, 'ed3cc2cda53055af0c67318e4b0ce577dbc78863110866659b3d7949aae28943', 3655089], [22002, '79cc7f39050daf8dfccc96e7ce3182616e127a1a77d13231ea23da14fdcc4939', 3783723], [22059, '8798e2dba47cd62356790d0a4bdb6580552d9edcc97aeb62ab2db210929554c9', 3724423], [22090, '9ced2d1aeb8fcd177d7352080181eb3115dc205b6866538b9ca57bd5743a0828', 3801022], [22097, '13e385ba029ca3ac39e1d4bd4c1cc8733a8f3c68ba801890e223c34ad0b35df3', 3684961], [22129, '63138c91c171344e181ef55102b4665ddc1d960d6947a98a4a275b455da83cd7', 4124584], [22142, '0b030880c8933a2bce0d5c53768298878a10e42b7f3aea9f3861f2b4846494c5', 4379663], [22215, '42b003ab935edcb162edc31385c075f128df57e9377a24e6bbf2983c0e378049', 3871111], [22229, 'e0e1bcae9a8b6968d767046c8218178d17e072c6080e4ea15cb80c1f6418cf2b', 3939335], [22429, '51366de9929144bfb125989409a77205d2dde0e8060f9c761521b27bf894551e', 3826178], [22500, 'bfdb6a4661b700067148d319015897b741c842ed951bd0576253bac1960dc4e6', 4198067], [22589, 'edcb2c785eac0847ce15ac54c9109eb531e79d2308f7203ce12ddf0f27d851be', 3883316], [22593, 'aae789f9598839c9ec35b051a10957afc21c7bdb64e145667f8b403eb34eac3a', 4010258], [22705, '9fb03b476c71c3f9c3c98c4066d385df95243c7b17e591bfcef75ea6668cd869', 3935876]]
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
