ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='5b075c93cd85a79f8a8d5d58673e2e1165c85f8647c024d3b4394cf645e31e74'
ITEMS=[[47333, 'aed0d83b2089003002b4749cb1876d973839d4174e0c0413f8f4e07e691dd5da', 3897639], [47409, '662da664195b69e66308feec71c781540197078ac79fa324f07d06e272d14f03', 3668653], [47419, 'b97bb4a0befc240939492a2b9cb7abed8fb04db70b7e115db8aafa055b5b0064', 4118866], [47524, 'd7928bc78347f8c5050fde8ebbf61d74a7e03ced0dd6e701d43a9b984f6f8634', 3605474], [47585, 'd527cfdb5b228aabf3f75f32f79f22e71e2345a2115c4d697f1e82488df13f4a', 3901324], [47640, '6fec42bb44e08ead7f2d45b0bd450d61f9cd571e66e15071136fedc864320342', 3615494], [47651, '30237ecd45e1da02d9a8d946a7f4756d554b90f286e412662e4b5dfa92f806bd', 4073394], [47684, '4d32822be2e9b7e1f92b6a6a06c63428967f3f403bef17900915d682c9a755bf', 3953349], [47940, '61d9145a3abd36c26cb5583d48beb12bf69196da35123ce7b380372d7b763983', 3873319], [47953, '47cfee636c63345d1268eb44b538fd4e9d4a512e6b177263c9009ab8189a4b01', 4096236], [47972, 'c083bab6830d4f99fe9182122369c8cd5692e18a773793c541de5538ce6a9b7d', 4248491], [48143, '0a3ccf519155a4ae7dfb738db7f8475e5aa34f346a0cd0340364f4c82641b720', 3991331], [48160, '33fbc4b06f2fb61a23689c2ea1406889f70c0c9d2c91dfc5fb307197cd07c863', 4190129], [48169, '2fc45445a0e0d3cf62755fa95f29e03b19ef0176467c70f9a16a29224570d8a3', 3755961], [48196, '4b8e79de9ce3fbd2e3d2ad35abf7aeaa9dc8edd35a67c636908086d923b8062f', 3718097], [48220, 'd8354ec26e618aa2049573878e1cac6b66bff32ea8028e052937240d7556c12d', 3807691], [48226, 'b826c3637e178135f132472ca95e1e2d8f52211f3e9d7dda73798a2818764110', 4017327], [48340, '0c0d1723f0dec9c776f50014c4c2383df0fd58d39a2e468d9e9783aab53c6344', 4517176], [48358, '0c72bc1608efba9a1e3d1ec1bd56ede51f068366250ac3d4689964b216abee2b', 3894034], [48489, '08c0cde49d8f9799464987fb5e870d8dbfb9fe5b499ffa67e9fee4592109dc84', 3568828], [48555, 'd102c0bef10e77eb8223468412b1fd6111c197a2be312e637cd8a5d87962c7b4', 3684418], [48571, '04c1cba1af10fa23b95fd3a30a95515b34ee8e3d3b98fcb288001663f6bbba3f', 3854742], [48575, '0ae2a7a8f83925adeb823fe31dafa75cf8d6e5b4e82f27d04c2cd3084b6372e5', 3753618], [48630, 'deb4ae8ec7240bf94e34ef6d235cc5b055044986ac4e22e68cb2ec51c742c91c', 4042089], [48685, 'af5c03ffca7d6ec41645e314afb9f7ff89134a00e1abce857876fb766553d4a6', 3894260], [48867, '5a823be11c123f0af4557943ceef710f8def197d1be1400253f3845a063900a0', 3703432], [48917, 'f523f2dbeaaf6210cea58dc69f387ac11264fc96c87ef62a2541e45f699d44d7', 3670714], [49022, '4afc8475c1dc76bbbda16ec73effeece6642b11cd14fa95825b53e31d7c78077', 4012520], [49060, '4481a3efdaa6a72a5834b6842f23ba8869dec58578bfc89de0c31bfa6247604c', 3927952], [49091, 'bcc2e5ea9012b978363c2bdfc407d7b84cb7bc89779350d865aad413cc0168a2', 705972], [49097, '9654912c61c7317ea28330394c890652379db527aacd3acef9495908d2c97e4f', 3844923], [49120, 'e46ff0de5e3bccd94230342d473b6766cd80d568746d7b52e7ddfd1630da79ba', 4686651]]
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
