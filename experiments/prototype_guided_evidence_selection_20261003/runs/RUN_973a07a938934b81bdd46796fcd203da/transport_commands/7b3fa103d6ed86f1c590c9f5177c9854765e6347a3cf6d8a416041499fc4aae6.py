ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='92beb71678209e348250b2ebd004e97bb58a3900d952cd3e4362268aaa782c02'
ITEMS=[[79069, 'b1dd1b0d43ea8b1ddf59fca75db06866db83ce3cd5d1a2f1fcc937ab57ab5b5d', 3978360], [79070, '72b567b202b6e3ee4d04e961ce5f4189dd1b53d5eaf62ae00ab949d3d1bed787', 3959838], [79085, '986d0331eb9de2730f030ee161ccb183da480c7d031c20488f032a13a52b0e32', 3788628], [79189, 'b0e20cd883924df37494828ba75332becc6fe2fc1b3fb746de8f7676aa1c7442', 4291220], [79193, 'e3abc515f29963de360713d0123452fcfe3c8beb8f14323b51926321f7c1dd49', 3999293], [79355, 'cdabdf6c032d721f4d1df077b7c9df8f1c55572f64cc2de691185b4d8cd017d0', 3849220], [79357, '2ff8df72378d5a1b0bcbdff02ec798330490e762644bfe7d3fdbc652cd5c9a35', 4334017], [79369, '07493f0a4a9588e6ff172282436b79eb69fa36347ea3721713ad54db1d68635c', 3911462], [79413, '5b3c7a81fc99f0ddd012dcc5a034224e315d01c2525615f13f4a1bda176233a4', 3678046], [79456, '02ecc2c33bb040268f312cb2bac9106936924dfee0df3b555c804cbb76ae1fdb', 4023711], [79462, '91cc439be91a964395748ebae4965d381fd0609315e2c2fa65034a2d8da91823', 3966517], [79552, 'ecddb7adb1db4a46d54b8327dd548bf1123f5abc496fcaedc064cdaa7e05948c', 3654012], [79565, '36a9c99c1bfc73ed452a9c48d5456cc019e24ff1058d4b7cacb3ee7b50cfcd0f', 3663113], [79642, 'a5874afbf5d75a6fd5fd35e977cfc289d981e0314d5837ffd0fbfa1ff88431b5', 3827061], [79654, '9971d8e08689476359e1a461b4f16a626aacfb52f918e71fdf0ba594ea172a55', 4204400], [79671, 'da9ba423bff268607d9a3c4550263e7a42388a5d49cb30d784c32a4a1be329f0', 747115], [79808, '2e48069d17ed44c6851b5859a70b247c9d388d9633f03bc52fb615f3cf48d7c4', 3691346], [79837, 'c57c8a3b1afa0ba246f950c1978fe7f3d063287566238fdeb0ffb3172b6b7438', 3767127], [79957, '01d4cc17f9d72724eb75dbac395aedd9054c40b18ecf4147e9fb99729e43affb', 3930676], [79992, '4bc6eb99c30a25f343ffce6715a9c2b60817e328bd1b33c61663f6d810a80773', 3804833], [80016, '7b7bd79c30b2f7bac68c6a930ff74e6294f4d535f129743c377ced360ff84f24', 3780010], [80085, '937bde4bc2b43a18a0789f8ce8265f0472156000970b6065bd4887aad062899c', 3846732], [80147, '70b1dabcf78b823d5572b295e70f8a10c057b0814128cb5aac102183d27f7e35', 3807653], [80185, '0d2ae27df0ddc9e5869afc29ca4cf2599604ea4f2df227a56117bd0df55f83f6', 4395883], [80187, 'f5cd3518eed7fa3b165ff9a0a145d29c5aff9e7b39cf4a9b51b63c22046cb22f', 4258014], [80215, '3a436ba7badcf9ee704ac9736b210d87874d52497a11535d6bf55d874a7ae25f', 4393498], [80273, 'cd8f85666feb19945d56d8c144edda80fdf6bb4402cb75c487e2ce375e85f711', 3645231], [80279, 'cff78f05b45ee8043b129689fdb662ff0a1e6ec1ac9473e59e807567d8e77caa', 3755869], [80304, '035804d607abfe351f820ab8a0150d15ebfa32a822bd24ceec5942c3d7c381cd', 3659894], [80340, 'efc7bc1086dd7cbfb12e33d596312e2e1ab27c0d6d6e7337418dc4c82fdb4611', 4437909], [80413, '3dd2aefdc994bc370fdfb9b05768ca48dee9bda8ac906d21e940b21b728baa19', 3703671], [80431, '930d6b7ed1c48e45a69ce628df386b1e5d306225abd397be547e7f86212499d7', 3658234]]
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
