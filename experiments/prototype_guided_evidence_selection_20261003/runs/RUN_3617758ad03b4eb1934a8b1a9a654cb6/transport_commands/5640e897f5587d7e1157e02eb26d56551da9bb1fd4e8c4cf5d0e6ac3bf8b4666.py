ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='f2887ae52b062ebfd26c61e41e8341f3f25f1a3cb687ed853d0c7cd14d241108'
ITEMS=[[17870, '088a098e5ac8ecb72eb1b1cda2925ee255da29c8462f73406b07a508e138fa73', 3708692], [17927, 'c26f7599301ac18cee486d32856c6a6272e22425f5179906af663f1bf25347dc', 4094330], [17946, '23977580ff1c35880e040cec63a8785001d4c239a12ea3f82c511b62ef66c1b4', 3537256], [17962, '40b4883a847625fced06787e64e925c4b453601073d97645b91635738139ff74', 3954711], [17997, 'a8034b8bc2329d47898fc94af8557e564f715145551c85da7e92db0edae96450', 3670290], [18045, 'f8ef6a711f878c0e385e48079d0d2c352b4479e747562d1e2ff381a6f2873d5d', 4204637], [18048, 'c7f47ea661c825a9c91f3a6fb7a2ac7365344dcc76537cc9be7c267a3e26b0a8', 4267245], [18058, '228d4cd2ebaa742867dbfb130c3b9fbdec4d08f29e023737ed9ba32d6ee1f9f8', 3860592], [18441, '8988a4c9f2b82fd7239fa95fdf0989fb8cf89905b2c126a357ea9738f066d4ff', 3543401], [18464, '9e1de95bc1c8c7692ebaf5943e72f5bf7d009b1558cc1582c04578be121ce4d9', 4276391], [18485, 'ba290ffd97afddd16b9787b4edbcdd83da4480e9ff116566f2373355d7633b91', 3783535], [18687, 'e1b883b0a86d24dea9a594334e0a3576cc44869397d122a820e68e9fbab7bd16', 4879940], [18723, 'bbdacfb0858d99f623344ccb9c82deab901cb702fb1ac9d0fa8c58653acab8f8', 3607588], [18866, 'abaf7fdec3eb582e1c1acf4961b8867b02796241cb1f55553448e1628d8bc445', 4442225], [18994, '95db1bea12380b0099ccc0444c730f8eeb2f3a24241545093767dbd75856ada8', 3758763], [19002, '2a867254199d25bf0e366a041643360aa456a8484ba1916936c4cebf671a91b8', 4271642], [19093, 'ca2fe438390ae8be17167560704edc9fdde40ae045f92e3c7479332d338a5046', 3965795], [19131, '39d6ef90c3e87727132e655da841c6b19ac36e7c622c8ca643ca4f286068e61c', 3905478], [19333, '5e77138f6e9acc6f7d2065072fd3eb748e258daac1d0b94d6c7817b2abc7f163', 4039431], [19349, '11b2e4b814f21e712774a8007681cd41b781915c974d2abd6111cd9b99e230a6', 4235070], [19397, 'e9cf599a2631996e3da818d1e604eb545a19e79be4be876b58690b477381657f', 4063019], [19491, '3f73213ec299bc6997b26f5d9ce3c4c4b6e92391ff78de4273f30ed6c3c76741', 3926049], [19534, '29c919f7aa8383b40b206021fe468adef373a0670163fffc403f6fc21693b87c', 4220789], [19542, 'e7c9053b9b0e5afbd3aa4c9c0dde65f5ee5fecbd17d60596f0178d12c9e9d46f', 3740477], [19609, 'ed1a0f72dae2208c66c3fdd76f61ec5ccb622d47eb2ccc7478e0f1bfe57f62c5', 4806310], [19622, '42409cbee8342b6d23dbee54d7ee8740dd3429223d418a110a57f70c70a584ed', 3571489], [19711, 'b6f710935fe36357cbd05172afa5ea0a0a774f8d2754c31a0af5b50e760a93cc', 3817434], [19712, '18295c69237556c55837e17f140b3dd8ff44c3df6214b764de3724d829afaf8d', 3950861], [19716, 'd082e942ba91f9b4f7e82fdcb841a56310a2c5a6f81938ce678963fd55dcfbfb', 3663858], [19753, '93a9f86a9d203a4d3b25f0c8d7709662e0dcffad99a1ecc2b9a6a9168712a4c0', 3857514], [19754, 'c4aa58c8a4cd37cd1c47c2e03ff529ae36f77f4dc929cc39ed3849c8e899eb80', 3431786], [19781, 'fccfda7a67636a8303ee290581313351d59afc4be3abb22993a71180d61fe17c', 3746280]]
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
