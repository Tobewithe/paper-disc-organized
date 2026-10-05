ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='2d474c58e300ad0167ae9d0d9f890153e3f9f2388629f3deb217513a39596bca'
ITEMS=[[98956, '1e0c461c937db22f4a2366e103bcc351fc96f6fb5c6b2d5137fa5472a8905d6e', 3862839], [99024, '75e06592751236d8d6a5102cc020b9b6f2c3b3ceb9e0a294db11fa735f1ed46d', 3875893], [99039, 'ea59c9e60a91aafe43a892b29c43f2a28b8458fe7d8c8e7751bfe2318a96a1a9', 4199865], [99053, '66d6b71bf927878381f858df92b1eaa47d47d166397b226303b96a9cb692b9dd', 4181277], [99070, 'fc15cab19a449aa60bc43ab28abe9df65d0e91b7e5967bbfd0449c9fcfd418b9', 3951313], [99115, '63096f56cbdf6c841d605100be0aa66e286b4f91ce21b3f4aeb401917d2455fe', 3849102], [99211, '37b1cbe66af87e9126915f1eaad90e25732ba3f1a4484a2cb00d76730a9dd5bf', 4010308], [99230, 'ce0da0e9b3995c5ee02ecf3d34bf0cfad1493ca06cfcf528c483e1c8371011ee', 3846403], [99299, '522b34a6311caa9b78fe2445b5985f2727b0c49eac13656cc08450ebb7dc81a7', 3809099], [99341, 'fc14b6b9a64a54ea7f0e52834f0380527247021e88447d140000b330c8d88e2a', 4180935], [99342, '00b03ca287912ba79f873e3b7582e62723e6b3524de0cd1e26d62f89c3945326', 4067990], [99348, '61cbf104fc94e9faa169afd917cde3c62362c5d082ffbe38ccc6d1c5971db250', 4512070], [99388, '1c7f57bbec39e030185f479fa5e542aa5b312766d0b44093d646bc653ad83357', 4257481], [99389, 'b79b3044f0841a6e6b18ff08d09d79e731e49e459c8c0e4a3c9678b7c0608fce', 3972828], [99393, '87069b67041a257bf20b9f07a30648ee28ca615fbbf35b6861cfff1b63a934cc', 3726728], [99428, 'ce7a4f4f98d8b6a43113dedc52163452a592b0aaed4f89b6c9e848413801fe84', 3894733], [99443, '6616c503a5dfe3416c71af1c67c9dd073ec14fed45d625066d91d7f8bf239add', 3906680], [99466, 'fd53d4576304b5e86ca7bd4270d6f1192248c936dc3bf1e4c703bbccee828182', 4021351], [99515, 'e6d903229e872f678e9416b9bb5b54407f0b83224a53d4ecf1f06c6eadcb053c', 4709405], [99548, 'ec8034b4b575a252996ac14edbbf74ae8509e49c057aec234d193a58d6cd6fed', 3799282], [99599, '9c2308e8c2df67f2f31f309926dd435e48b708ba5f7c67b4e7ac91ba8d5842ec', 4050437], [99753, '7196b40fee912a22d149658e9cd83e80259dfad512f6fec2aa2cda989b097db9', 3931072], [99785, '9ce9ecaeb62959b7efab3f0ec2c5527ecdd8fcde9d930ede42cf1024883e9e23', 3741055], [99808, 'b4e3257b64624234f2234c653b650fbb7134c628e94355488aeed87483d07e5b', 4199350], [99892, '01e5a42c2d8db448eaee781c755743b73aa88d3f387b1e0b81ccfa49479ed24c', 3925561], [99970, '5f4da49fb3bc06656e736c388a840d2091403a73c184c5ae1dffc669c4f7506f', 3811863], [99984, '8691ddb8778442120f818ae34eb6239953f92757e126997b68840aced3c8fb24', 3706714], [99990, '608314a9b5260aa1f9ed911abf9d20a163ecbaa178c07b59fa7827bb5eb95f17', 3854834], [100014, 'd9384ccfdc09e0d209f5e19ed905f2c0306afdfeda334ff78bb19375af5b9dc8', 3599050], [100020, 'c42ab828c3d6ed7c50fabddbad83d9f9e38c2f9ec394fbb436508287cd60b790', 3933672], [100022, 'f22e8888942710798c3ad8e9f0c49d3c466df3d518311ac88febf1618d9e46af', 4038253], [100037, 'f242f15c784ee30fb4ffac8aaaf70ffe474b8a2c99858e96831b9eff8af04774', 3869841]]
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
