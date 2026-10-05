ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='da13628334fea07aca188706464c3d652cbcee4b1a318756b1196ad0a5f4edcc'
ITEMS=[[38389, 'd32d72cd7454236c5556d05fc3e629d3850335e5120f04d251f19c84a6bc6ddd', 601030], [38584, '95cf9454866acf39c87ec88e2d678eefa0c07905e0001c877a8b48f025650cf9', 4050539], [38601, '3d0792c5fad51e5db90b2b73b1abdb806f11add5eef9e7dab5eb7fb164e59a28', 3562975], [38616, '621f87ddc6436671c3cdfb87b9c9e77a8b14a96db8670245cdbd50cda3a06a8b', 3714802], [38622, '6f2f83eb5ddee4dd692941aeb3ecf88569b41ab53e2c04a29edb052c99e76e62', 3777959], [38652, '0621a86989ea39a95585ddb4b43170cde748cceae7a8f8f955b886a7f81cda2c', 4477112], [38668, '13c7e829c922fb5b4b77dd596d1d6c6cd2a0f463aab8dd40d37f8cfbf5a5625d', 3864842], [38678, '9546d16a11894e68d0af497e8133a614b77485ae0a97f16974a5e242522490d0', 4008494], [38682, 'f209b0600feb039eba2c734767a2e12438f36d18fdba269046926d29fff16f3b', 3674677], [38685, '5a1dba6fdcccdc4eb9a7d412fb31a7d79048bbb221a9376b5c8d3b6925bf2db9', 3761932], [38691, '7e5af55290acb5ca71ce90a5b3851e3d6a43e68468da58b2df222f19bdfbccda', 691969], [38718, '0a611c5d7f8075b355d71fbf23088657ccabd37425e9c8c9601a461df4c5b3a8', 4278441], [38730, '05d5dcf33bde714f61e3d82b9b68a57d5e361474e7b608daa2b9e494c3338db1', 3997068], [38756, '7210e11ef303f4eafd03b9eedf9aad69d32265f479ef3c20a24b5a228efbe637', 4106725], [38919, 'ff555ff08e22fc1f4fc29ac7aeba4f26684977707dd515520b9aaff2c84fbe9d', 3978740], [38933, 'fddec6333853d09c902c6fc2acad1e32c61b6c3013f08b316723a441fbeea76e', 4343552], [38963, '0b6ec6e3a1228e876b77bfc1498196bcd11176cdeb47dbdd7317584aa018bacf', 3993105], [38986, '21437be1e1dc37d15692d59e4e68a9678a162fcab48af308d7dd4c33624467e1', 3457099], [39155, '90708542dad80ee9553725529b8bfb1c8f721834f8afb27990ee8569593d400c', 4012185], [39359, '6a2a227826d368736843c69ca33d323e10f5b9a54dd1e399ba6e1ae6f4ac72fc', 3673012], [39405, '9db81c8d79fa15372c3c828f6ffc6f33b4a9dc7a38ca8cdf8953beff9703b404', 4630076], [39438, '85de3d4308c6af2438516564d8efa6a9c9372f1bd2030f074e3345ffe9f0a6e2', 3928179], [39468, '3296d6a9d2851ed1f42fa912bf04b75ecb1d5f21fa226f748a2040d2e00ba614', 3817934], [39512, '062f0d9653eb462d839581ae976c263571d6cc7feaae58558b641434ccc078a6', 3892186], [39514, 'f994363ace7ade4287289dc39c3b53460ed9dfcbaa78606aa4bf74b895c2448b', 3897349], [39527, 'e1e515dbed3ce53983b29b2846f5f0293b30a1ee86d7e3c0d2613159022abec7', 3888552], [39531, '480415dae45c9f67b071295192e29c6263d9450a547c3b11d1757a2e1b070827', 3821609], [39538, 'bfb8e451b8e44399dc34db0192078484e311f7274a18412de58b518c81460b54', 4172576], [39555, '2dd7de4c40797aa5a39ea905940660ace9589c923cad4de027102d6421caf827', 4017131], [39592, '443ba8b7aef557227f059db7670b7487dcf3d8d8a1993744da3cb81b0b001f1b', 4336173], [39640, 'a5e60e136606b1ebf82be1b3e34288fec0362ecd6585c5f19ede5585337e7b41', 3759065], [39685, '1a9f5cb7b6bdd1e7df131f65ecf52dd53ca9d12c90d03df768bd6667932e7fb3', 3584899]]
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
