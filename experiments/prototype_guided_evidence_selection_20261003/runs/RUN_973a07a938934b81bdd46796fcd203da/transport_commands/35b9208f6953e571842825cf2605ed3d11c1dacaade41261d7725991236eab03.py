ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='f21a91dc8393a0394154f411e42526c2ba7c586dd7e649a01574ddb6a6927604'
ITEMS=[[42551, '8285aa05914e85390500983fde5cb627ecaf65283c11b4cff597e2b72cdff8c4', 3827789], [42621, '414a326a2988d446aaa1f220f5f08829f4a4713495551fab88636dbfd69b9cd4', 3707158], [42667, 'f81d73e86d407d93ca68289c35faceef8039ae4a07399c17b96d473c1aa80f2d', 3877274], [42675, 'c27b3234038b7e4fcab81be476223bec775d6e619427d97e309af29c0307052b', 4118904], [42867, 'b8ef7cc7488164b8bbc6c4b78f83b0673032abba41cad75c29747bc173138883', 3944138], [42876, '808e3764d33e3402708d29be49ad7988a8992a2c69903d099f8073bc7454318f', 3593601], [42889, '1f0b056890ebea7925dbce23200af970f2c8abc8c19f83692a20fbdda38f7917', 3699753], [42960, 'eedc975b4e1825afa7d5d041a2dffb8fb344950810240bcb98b2804c80505a6d', 3874511], [42970, '5f9e033364458eef2ff4956ff667be624112f0017a6fac278c99cd7ccc5105c4', 4952144], [43011, '2b34ef754fb31f3a6f129d9fce7e1e49b20e63206b0edb01ecfaf7fa5b469dec', 3813184], [43024, '61af66a1d507126fcd1a9f336b2fe23bb56ccc42302a649de1302d0581c3b91b', 3754868], [43171, 'f26ae13c3e49082893505ae10e3f158fe779299e822b4b76f205faed1331973b', 3989068], [43251, 'f1d1cc322b5e217d9c10ecb38fe0c27b5216c3e67ac1c4b29eaf8f68f0580a6f', 3835553], [43331, '1e378364d42b47cc75c1e0010f3f01fda0f364447f1975341a31266b39a94247', 4099305], [43344, '3e49dfe475edfc38e978edafa0b698bf39be69c5a688de3119e1ffa066d95d6c', 4442974], [43353, 'd421fb965e15dca66528408e546bb8eebafc6eae605290cf1a7882f60f22c114', 4082659], [43367, '7198de3fe4a8f6fab719a6492df7b9ea432a64311bb78d4ba3d3abfbf27487e0', 3649204], [43417, 'd5b3b13825b00eb34cd7fab7e4d1426c2c0720193544212bec9d9524b5354f09', 3640958], [43435, 'fc03933eac5d19923ebc9d3e997f41f939c7221d42042275ba7e167d5aefa3b9', 4119414], [43555, '95a11dff44b7dc208a0b5129aac9cf6bf9eefead289691b6e9028ac379a855b2', 3790024], [43576, '7ec36242907dc12774fd8655fb012cd276c32c43c4d25cb51f5a282e21822ac3', 3736432], [43579, 'cb214e624933702d2bdab4580111d38fea89609ac82a09ced89fa7fef0fe09f1', 3671338], [43629, '2f4b2a6b2bbaeaa8eb2a47f888a36673a977b98fa4a0031249df6707db46ab5a', 3831628], [43680, '97d774a61ef11487491d31f63befbf70569b050ed2a1356fd2313f6a5562aaec', 3974328], [43692, 'f1f3d54d3cd97906ad2898dd899ed998706573e64eb05e82871b3ad84186548b', 3828263], [43737, 'd4f1e1196ecb7c3d698c11c0522507d96f43b427e1eb1539cd5755396ac76f56', 3848204], [43739, '4e09ec1ede7d0f5d939f6aa9371830c772c9de4862af1e947fccefce8094a9cc', 3898003], [43912, '02fcdf4d708186c428b74c4c34eab6fcf22c5ee8ac44ed3a6dc8aa5f8afd4f6f', 4198527], [43960, '2c563b5a504fc36125557d7984cace0f53731829dbea05753d5ab3fb9fc3ce25', 4462957], [43966, 'fe59234a8315f8fb10d138c9abecd680d68ff265d1d1405b7003901224ab85e2', 3967487], [43971, '51b7f89561a5ca20827ca33d1175924b614247cfecf2c3324986746e7a28b6af', 5386611], [44047, '3cd6106c8e3b4d406c3041609e9711afb486103ef6387efb5b6b5a4d66c60f50', 3760100]]
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
