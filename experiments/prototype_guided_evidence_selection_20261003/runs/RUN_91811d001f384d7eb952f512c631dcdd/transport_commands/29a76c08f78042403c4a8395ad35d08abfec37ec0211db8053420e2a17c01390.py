ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_91811d001f384d7eb952f512c631dcdd'
KEY='e87a8a98655c39184ccf5c9b6c7cd1c0501ea1e46febac383345262ec409344a'
ITEMS=[[15267, 'ee41fe49d4197d58d5df89f4bd25e56a2d6ada8d5a1fbd36b26d38e611cb10c4', 3941953], [15335, '30f80bc92789cebb51df009a23ddefca6c2335e55b1167dd42b7f37a9d8a974e', 4233072], [15338, 'b0726a86e31398b7b067152fa8634b5796176d1724abd4bad4f973f2a4bb9aab', 3878611], [15440, 'd3d212861e6735b79490307424b291c9e0859da444e7dc97f32750755be69ae1', 3750415], [15517, '39ba38b1945e49fb0659fe019824b585827100ba8c8bde022c20e70b9533ac1a', 4173945], [15660, '80abd8044a01d48c6c642b683cbe605744ad7f720784dcb1af179b07325a575d', 3781142], [16308, '29fe07ff92833adcd088fdd3452be046c46e158a58949bf7bac5e1cdd8840a28', 4321502], [16496, '9c8d90041b4ce7a5150d5d10692270be4f414f54810d869a68a9900c6121d9a7', 4826990], [16531, 'd4e44dff0c5e919e54822fabc82d7b298db438a707efd6ffa13b81007871a337', 4139791], [16653, '98a2000d618c1cb930755acf3d3c81ba1efd5c6ea574aa3ef9c0a00216081782', 3706354], [16670, '1a830f49832d144b867f2ccdc5b5876c3dfe141369df486548483404d90b525c', 3862622], [17029, 'f21c058304e55d12ed06e5313249a02313f58ce79c91b8d7b1549454ec724994', 4232015], [17182, '341f177d401abfdd2e5e91dfbe4f5b021ad56e759bd10b6a4658e39ad40a476c', 3926349], [17379, '4fcefeba455fcff4efe5d7350f9dc9bcc3694feaf3b08cfc4606e9420b4a9dc6', 3922332], [17573, '1b08052cc0a47bb5f3bc7322c9365f0ec14fc9b0ce63bced6497a2e6a3bcbb1f', 3672448], [17766, 'ceb290602f85e27cfac1299aed2ec0f55e3085ec2eb41361bb4464a82ee453bb', 4183404], [17959, '01ca96ee41d885df8bc60e13f03e4a755b732a41126a5a53788f4eaf5827e8a3', 4421911], [18157, '947165bc7ee743726c722024af5016d5b78722290a32a9987ef7a29d6d372104', 3742067], [18267, '1681deb8442d6a2c77658b191fb82e769390470329e9b6c369a8e110eb0a67be', 3665420], [18319, '7b06d086058cada6cb7ac70508be68e884ea356b5a932365ce2215b44431e59c', 3875733], [18432, '0186a229f7f444e157413401d507842347acdedd0c292eab59c0b22b22d44d59', 3873158], [18480, 'd0cf6b1c53b756e21cda183f765ee87b17cd90b399fad6d1dbad403195d36f7b', 4290707], [18491, '89bc95d6995bb24f5c17c41b98d117f43e97f3b1aa086765be99b4143a51fd61', 4122702], [18496, '42b93ccf5db5a52a679400f39fa0cc8931d752f5770b76af0553b3908f615101', 3916267], [18575, '4582b2ff2969042a017714a5a9fcb86cbddbc4ee91213ab80fe6d442c4530314', 4104174], [18605, '9a97ef1f9954f47a63ecc6503ebda56fe7e459c2f625a3abfcb5f1f9fab37a7f', 3924997], [18737, '2b6bab3fac103c2cd5752042b2cfa763dab780c45b4159c390202c38569a4c0f', 3673383], [19109, '23c33d2cea7938ce7ed557819220a18681bbfdb91ee374504c172db9d25175bd', 4698552], [19432, '9658fd76316448bf635bf10f4f1417d6b528737745217b0a876696eec9934634', 4220320], [19742, 'e7db5cf61d7c2d3171c671ee87db2cf2da25cec36a7d92b187284eb0be5328b0', 3605328], [19817, '13fa9f288c581a57d180ea547b3b4ba066701430a81b693ad05e2b6b1695b520', 3745318], [19924, '4de896a568a0ae732e3abe4b46dd35e01460b1251d30b4788aa541404a5b2382', 3840634]]
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
