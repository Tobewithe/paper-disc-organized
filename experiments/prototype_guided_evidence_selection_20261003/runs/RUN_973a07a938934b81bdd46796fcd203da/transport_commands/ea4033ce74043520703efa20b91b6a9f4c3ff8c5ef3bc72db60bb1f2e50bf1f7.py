ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='c44ce19760940d3931c6ef55553e3cfeaf620cbe8d5ab9bf43957c80e203d812'
ITEMS=[[94753, '404fb93735106d68dcdcfac90c0576e50075c405f445f5ed405db179925a0f8d', 3716406], [94852, 'd6820b305387e6540191110c564f480653495a93506508baaadc44cf1965d6c3', 3907693], [94984, '063e826155e1de2f410bdd3f35f297a37103d92dc7d4f7fd41559a005417f6f3', 5339687], [95003, '415e070bc2f1f4a5f9ba0b3375b1151d23b63ce2149d4312652fd63059802280', 3627831], [95034, '1c1fa16819b5d7f486817d65b0733ac5e8a8ac256c9452155bd137fdad4a04df', 3731191], [95078, '4d9807dd41ec919728d97e43d5615a63e9089634f054272f4c19b95df7b5f7db', 3740250], [95124, '35f1eb4f2af37a8560beb8bc0df6a0a9ad7d299f096b52222765e264b4c86569', 4022379], [95155, '51deeac775e0439eed4c2437295bc039f733d309a2d2e3154a4d052ad12f19c7', 3917445], [95171, '5c68a369ab015f9005d3780ac07cfab431c06bf0453047f7c05973b9d47f5f8e', 3845905], [95337, '7bc11284794be55c410b2788bb848b84331129a8ad79b9f8b8705e0730032b7e', 3793986], [95363, 'e76ee88b813d427386fe8d69f7877b2882dac3bd8453f912bf6ee014da06e6a1', 3795968], [95493, '9aca30e287d56ae7485a2d8fb3b362db1a9773318b4203833cbbde147b37c824', 3845907], [95518, '4e915dabd7d04fe80a330be5846e15b41d68e42227b5cb5cde4eed8f3c1ff417', 4159269], [95569, 'a12b718e68e348739d3801305d4db86877ee5e61a91ee1fb13e0e86dcc938fd9', 4110822], [95667, '60de4af84097bda7d3acbf550b3e0f29205e97925695645340b35794641231f5', 4027614], [95692, '4bff31ed3c334d727a917ea3a856804cdf51c8ede882a0340aa3bd1aa62f86f9', 3588111], [95707, '6a552b6eaeb3ff4e4110e2da12dfda85642a7645e3e4919f4f169ec09426a6e4', 3891875], [95786, '6f20462f5077a47c66d8a1e65ef9b193e920ce8d72cb5576299829cb25156584', 3629616], [95808, 'e3252c6aff37eec02d3efb2d3ab826b7aa6c67d9f7374758f932fc2b7577aa53', 4352451], [95828, '51a7a49beb3cf2d1ada43d90a71b42105e7d533da31d47b73c513a13a82b1dc0', 4051408], [95843, '69d1e509059f1df3d6fa499760ed18b7bb55fa1c13143076e3f2b0bfa26a9ab5', 3638706], [95862, '5b0443e469f7b16b85783389c59d5f2884e4a6b7747a74095e84ffdcc23ae260', 4363461], [95899, '7c5037739aabbbc5278f4b7de9ab6a16f0fe16dafc4f59f78ee2a2d08d21c8bd', 3672763], [95902, 'ce368e12d453657a03101bfb25546b61fad62dd5c1ec3b1a28f4b38d507c9c73', 3908866], [95951, '2d87e0a01a347ea36a579fa48dd585d1bfb86c8759576e4278474746ab610235', 3717110], [95959, 'a9bbd78f2813f8cf4acd8087beb303c07ea46feba2b14d74dab7b06cff7ff774', 3770347], [96043, 'c5a05db70470dc211f9ba554aedeaa631cf59691c52f5d9c60b88e28468416de', 4277226], [96082, '354c7c7efc347d7bede4a868c8ffc3a617646f15fc14b0510d028a9b5a7e126f', 4399562], [96183, '02c39aebe071b2c28ea5e279ff8f277310a6079405e84d860f424c9b1ffab751', 4247717], [96208, 'f7456ff4608a8f4b3f04f82a0b80b6dfc22ee85e2002cf0b15b16764396c5289', 3706230], [96257, '3aa2a026e18ec6ebbfb30c8b3c5ad97aa6c0ea3d7ec38a6c8b23b8385a423a33', 3761148], [96288, '16d400a1d41fbcad2436ef4485e6bb3acdf9852d301730e032f54a574e571135', 4166917]]
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
