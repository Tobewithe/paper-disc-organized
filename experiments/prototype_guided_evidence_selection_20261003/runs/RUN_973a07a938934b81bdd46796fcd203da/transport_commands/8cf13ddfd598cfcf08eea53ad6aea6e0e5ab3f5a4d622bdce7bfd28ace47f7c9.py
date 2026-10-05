ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='9664ffd3bccab64e5ec4d1a0e1edf7e9eedeb8e545b57b0dc42a06b58c602639'
ITEMS=[[56752, 'e2fec0ca37e9e69dafdf26d52a5aa2f18760c12183cb2e21bedd7079002efc35', 3934744], [56753, '52e35d0f6204ec7a8b495fb20a40295f4a604574b94fe5ffad8785bc7a34f8d0', 4204649], [56868, 'a126bff49904986047c78d05fa2dc775e1e493f9120f62eb048324e2c166e849', 4478155], [57006, 'bd5672158c6d802a6ce439d94a8c7b8153c5bf61c2ef75bf44535d781d2f909e', 3743805], [57027, '3590cf95940f2af774445a24b44ba104df5dc33073e3691168018ad1924f6ed1', 3851944], [57107, '86a1fc3a079e045473828303348e479730e9ef3e9506c10a5eb0e4b6061f5b0a', 3876125], [57147, 'e55641cd653ab96c49be2d48bbabf39bded7624db9061999e10dbbafbaf0145d', 3864787], [57150, '5bdc60002d46064a999b4430744a786c4c5e1609bb115bb8d7c9649bf7c3fc64', 4463593], [57178, '87c0d9a7768b187fc90efbceab6d4a42debd33ed8cc3cb55813a04cf554e2f7f', 3838464], [57232, '29604ef81abecdc04b5e5424b24448c9041c18110faf25ce9ba5b3dffb4fd864', 4567999], [57264, '7c6a4a4cc3884c79116421fe7588318f964baf0ca15025aba96c174e89561b53', 3705358], [57274, '3b4406301bd6871f6dba9b8014227e8c30321ca921d286a92355996f0028596c', 3709781], [57300, '706aee299ee5638cdcd3b091f327e4d12cac851a20e0021df2452b0176c07768', 3653941], [57415, 'bc3d0ac04f543e3aed72492bb954b9bde78c6a91222a39b74e37f0f1cbee2f7e', 3746025], [57424, '935f477deee29270b0d5bf6fc29309cd09501c8b4f13696436392a14584d6b38', 4209603], [57498, '4081c1e9865c3542908f8e3ea9b83b8f97001012e864aa9bb75bf13070fc5ec2', 4316941], [57523, 'be03c1694ba630dd0a52437fcad7764e4d6a09a46a838583814a8e13122baff8', 3995505], [57579, '8a601cfb0199a2e1a3700ff667ff77e4165340be0906ff717b547822ec0f2610', 4064365], [57586, 'd03ed8bd079b4c412dde475529274d2451beb1a8eca3c7d0a56363d0cbaab258', 3590130], [57597, 'db7621dde73a06c35e32b69fc95b2e957da998182f324ebe33bb4b404d7530dc', 4707345], [57671, 'cf8bbf3c9da6b2e084ac043e4c99037142263c79a93487e16fc2cde82f3c67d5', 3877254], [57672, '6488f76f75a73a4ea695a7ef490b9727aad17a64b458a6565f41a0dcaf65049d', 4334784], [57673, 'aa5d9da4189384ad65196f9d1f54dba5b60c829d63913a2151db5c1888a4bf18', 4124119], [57750, 'caca30d99bc976bd1c7dd27f65d731a047eb355cdc8d5aa06c72b32fd14a8d17', 3909533], [57868, '08f6553754228448f58d75c011bdd9e07424035d8aff61f35295ae31d3e0cdaa', 3975350], [57979, '5d5d7fb4ec6e865a985b2073caaf30acd3b0a400725ce3b45fafcd5f0f87da49', 3615064], [58029, '8d2cf101a4bf43b41cd57147865fb761c6489ad6bbb57ea84dd6931a3307504b', 3741855], [58080, '7f20e80528603107d9f1b0239f934a6690ab61b22e800e45e3ad4691158cf360', 3750844], [58102, 'e58398ba98d8e754bb29da7f4d338dbb3b9eb4614cec2adf2f7d937380464824', 4001046], [58143, 'abaa637073353e10c2c94a1d951772db3bd0eabbdb63a71cb4ee2489665a89d1', 3853032], [58149, 'e98463211e4ab3fb02282230be185207102600a163ce2a328400b502388ae34f', 4083414], [58153, 'ddc258415d86f24a33a25dda85760ab55f61a957bfb2577a18f1dc707b7b3fb1', 4280226]]
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
