ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='c962dae6b94d330443e80adf24e5c4f12df2b73b979675e2d1f419a0e280a95c'
ITEMS=[[26472, '5776584244d4aadb674ba9844aa90605a00d275433e716a3347637d40f5b4b6e', 3773606], [26584, 'e1aac42468b0bd1c69becb6963bb753ecd1ee22f1c72955cf79ad1681ab9f5d6', 3690423], [26617, '7050a234fb46a6914f948f49d68c279d8ae48eca29491c3840d56f3a1b1e4ec4', 4203384], [26624, '09516e2de0c46df8999eb089ce690bff7f299ba2dfee3e516fca15995585413f', 3848975], [26685, 'c3d30f5ca8c6d0152f75c0dfab4de8f5025dde2a050e81bcc4491e9f1b46f370', 3795605], [26747, '026629003d04ea0ad0fbcb470654bbb0d6ee47e2e6d6da6bc06e1e990babefcf', 4487779], [26748, 'e6476f7ceb8aa497bbac5ccca131b8de50aef7eb07597250b963a23628b49922', 4087958], [26784, 'cd02ab67ee9d3132078034babaff8e37cd26236a8c84a81c11eaeeba6ca23215', 4743930], [26893, 'b661957863cad8377491b246acb4633ab156c08a0aa52b8617a0b6b66568d76c', 3863823], [26905, '4bb6e9cc0a5cca28d6c9e51835b71814a7861f2cdc65a4c82546ebab819499c8', 3707397], [26908, '069a76689671ca31aaaffd55f70355f4a1ba484353cb3b26ebf472fde612ee1c', 3767079], [26926, '44acbb457d028fa16259f7c249c6f0be7db83d27d01f6979f8cff44e4c801cfd', 3807274], [26932, 'ea421bb5e0c7ef0c1215280ccbb29a8e943fbd446d30a4b2e771da256f61d51d', 3810291], [26941, '0f82a4fb567dc6ee984ea640a554b164adc211b107a9d4efcf277d60cb688170', 4057868], [26943, '4fbff07204e7c527636247927251420bca061dd7cdfadb1475e861521638d452', 3893964], [26982, 'c1fe5fed5c5e30e0346d8e5721330202a4a7a6998d805e2d8283d3482946be6d', 3838307], [27046, '46164eaf275d2c7f5201f40bf40283b429f78ee163de3cda6bf2b0a7f0a0e3f0', 3794494], [27089, '216ec7f7033fc6759b7236466a145913f7db6be6f8eb365dc7a38af65db526d6', 3751924], [27108, '2dddba59e83d77215659b131b548b1e3e210a2be4314039cd3ebcfc0e61b3290', 3738772], [27109, 'aba40dfcd5aa77679388c01576034255dfb3a8144760a8bb31ac567c6c18aa19', 4059595], [27252, '82a37711cd30ab1af94b9e8c5dc1f99781b7d6a0fd30986762913b815fd8e05f', 4045454], [27340, 'afbf3a4348433e187c31b0cd6e199b72d03dd0ceb289724924a812a838f662ca', 3641237], [27371, '36d37ad947ed8c5b0df0ea2abe83f6b8dcfae84737b8453fa8db995bfab41aff', 4099800], [27390, '17aae47a83e2aefa438bd73a2ae4657bb676bcd5ed1baa29987c90ec7a96f485', 3917701], [27438, '501b9ba99b1a7b99a0df99a99066d92c077b933b48cd47e504e35aa8620bea92', 3638425], [27493, 'ce2ad7698522b3370cd46768609c5ae49999904aaca2815e8e5afea3a9b79c78', 3808609], [27562, '4ba6356d31a6a4f007a34d0cda5b5098df01490168aed9763c47ca349ea8686d', 4042381], [27591, '0d3a4c1a7dcd58172a25814f21af0fac8b75d79e2539b5f86e8a190fbf8a2037', 3803208], [27617, '9f5eb80cc2aeb430243faf0639455f000986ec78a2a85e98ea0550260fdc5036', 3762657], [27620, '2f5890252bbfd28d9b92b7dd73e991c615cd1bfece20154fe6ad2b303ab862b6', 4024787], [27750, '517b9e55a8d4c8636e5ddd0ad40604316826bbf68d69e023676524ec9c179228', 3948122], [27768, '3014f8490ebe6e1ac7c26e40d60791709b1054c984438f8ade3ff5a4bfcef65b', 4219229]]
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
