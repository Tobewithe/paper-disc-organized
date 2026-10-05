ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='a4b3f571d449fdc29e0f3e2bb455409cb550ef860d27d3161a6e9ab507b75fcf'
ITEMS=[[97858, '48353456810efe212ba40f0cda311830936646fe759c429c2e67c58085b133ab', 3810715], [97861, 'fc6241b60902a5dcb318f4df4e7184d81927461ca17b1dead49ea85ad0f9231c', 4384688], [97890, '5b325b01234bb81ef7417a37cdff11ba99abcb0b0601f2d9a7685c5206e07b0e', 4060016], [97902, 'de35d56ba58d96ce215061d764dc29c2ee6e2c2407bb266276a39bcc6ad358e5', 3942818], [97974, '2dbc9ae1b14b62ac9d5b58ab58b73d644e76202c3c3f4654ebbe2c66b4d4ce2f', 3657689], [97994, '150a98edfd7a688ff759597fda36ed4847200cdfcfa57f57bbdc000abab029e1', 4280949], [98018, '2a1a87abd53bcf3de3f679a12f60347647bba7eddecfb5f78932222970770093', 3873947], [98137, '861a3adede1fa4542033e50139260cf8ce901af9846ef00a927c663c76d9214f', 3635738], [98190, '7a3cf9ece4c69b352d8a3248746ff526f17980d65bd0f0cfb693fd108a602b8a', 3862225], [98194, '590a84e7f58075253f352711a763b4fc26e6f93229e3f673d288a67c3e489cb4', 3903910], [98197, '26bf8980c14e70d46759d29223cb4eec2450d7c1fd9bae4bf54c19f83fa28a75', 3789946], [98261, '76717e1d3d93c79b1afa731043f4b089e2e63943fdb2b6facccf7d155d7aea93', 3342087], [98287, '3a608ea4ee5981441fc9cc7b0c9e3dc822f49a44c773a7c5d09906407eed17b8', 4235938], [98302, '34708548c503964759e11244b1abb5d404d092381be8356e2d1cf18c95d6ac91', 3644593], [98385, 'c4096e540fa15352bb2c8ebd3104761f3116345e57f495cb9e563a8d4d8d2f64', 3738369], [98479, 'c9f2c22cf7019604f6f9fcffee757ec9f1c52f31a90c7c549899a7abb4a30fd8', 4088971], [98493, 'd56cb53e8c8f510ae36ec5b57d3ced89034f503323e2ca4622cbd61124382406', 4326330], [98495, 'a2b32b1971327a5504edb6ca54cd19522144120dc9bb723c96c8ea2a0b8efa6a', 3671763], [98497, '868fea13165c9ade963a959d2f5c72d1488f09b8b063ea8061f3ae42f8d357f8', 712606], [98564, '898023b662c274bd015cc7e5d35b157fefbc0b6e1cea8f5fa903dca903794a14', 3939242], [98590, 'ec8b83bc8d733600d7f7c09f503ad3a94b3aa4834a93d73de66c1a125e94cdca', 4071775], [98616, 'fb2717741a77d6ac7fbb20518ca1f1e78be5bc16f2b6b50997aa857b3218af2e', 4204073], [98641, '8f7ce5cffdaa8ba2cf753ee33d07f0245ea6bc5ea6cacef804afa3a24d0e97ef', 4685020], [98674, '0e634ee96b3ffb6d79bb47aa4940141da9e7c2ba68ba9768218d4e0a1716420e', 3882236], [98689, 'de01a0091c5a66db92151284c9ce516e54ec63bd1ed4b6ab581b6bed18a38b41', 3754395], [98739, '5d76976cb187003848850ad8d060565b3711ce7afc20300962ff2105dcc27f34', 3850859], [98752, '3e2664fc3738a67b629bbb984927c756a48f1472ad2ae375457fef519a1ab8bd', 3889980], [98760, '40d0ccb614749809032be468cc4deef10867d9f0708d9838bbb04177128b4fa9', 3711704], [98836, '4dea2573d5fd03059e0b811ac89ca2f55174d40ef9c00f6625e7d4fee36b43f6', 4646422], [98854, 'f50fac5384b0ad222772b10d075051200d14a3cf2acb46eb9719342ea8c97c37', 3717123], [98924, 'b72b9fb5a9e365285ee8b44cf71d6b1697aa88558455609bab77c0d1f56bdfda', 3607369], [98925, '4141adfdbcc179240e9b6e6146b3796dcaaaa7f40803f2e3727f02c9ed714798', 3613392]]
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
