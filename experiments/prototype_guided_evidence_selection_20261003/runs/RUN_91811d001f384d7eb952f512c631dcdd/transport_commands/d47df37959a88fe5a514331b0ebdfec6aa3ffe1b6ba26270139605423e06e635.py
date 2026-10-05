ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_91811d001f384d7eb952f512c631dcdd'
KEY='3f438894de1e796a529f6685102bca7978f6ab22337ea04b7f96799cf87739e2'
ITEMS=[[4595, 'b0ec9ef5034ab3a2f256d7b25470b2dbdd1ec60a34d646310b7d8dbdf5ec3dbc', 3771152], [5037, '5ae78ecce2dad6934af3be62c2a1a89f2eab2fc294ecd9aea8ff29db2249af1c', 4028579], [5060, 'c47b402c13df6376c399145ec226a6ac26485e21b0db8ebaa3134aafecd1c903', 3912580], [5477, 'ae7bc5ed6710946a8c792336f752d08a47ab3d78fd632defd39725c58c6d51d5', 3545520], [5586, '2ef3152eb050cba6dc21631aeb2aa69b9f9b442617b8cf495b051c5b305ba437', 3999086], [5915, '572ddf3ae010c8a9af05237bd89fdbf7a8727d159901b289777f017f59e4ed96', 4113937], [6471, 'a4ee541b793cb62f8d6a0df49210de8102891784df91154dfbc812b06add59ab', 4167520], [6954, '2a31b6d08a2e5f22a2ea39733503b1c35488b44371b6fc66c84319bf9e0bd59c', 4041738], [7088, '9404920de61ba32cef6cdaa84a251ec93f1bcdc60da5dcdbda59d6c7c707ddb0', 3946299], [7386, '705cc899e27f6897da3198ff100f428500240f07f18bea6cc074633ba0cfb090', 3868344], [7511, 'ee2b6e995ef069e437f9b31590113cabdb4ce0bb74b6916cd261fefa6432d0c6', 4237909], [7666, '145b692c477c18c6bd3b882d1a9a890f1ea62244c7caedc00e7722c809d7ac71', 3741015], [7713, '7dcee502df0b668c4331d12109e9521668ef1bf15ee44337dd22b751e7793d1f', 3851645], [7931, '09eb6579d0d69c0201c72136ed87e8b6ecc66ba4a547ff3dfd2e435ac7513837', 3780289], [7954, '4cd67f301ae185255723b43a8bcfd1a566f071835b4a0bbd0b07bba557d631bc', 4091071], [7991, '70da05a16e79d01e3d109158c99ea8aa8464b44d156ef5924447151d0ea9388c', 3724412], [8267, '683cbe2ede75e91257f4b58c65d2852e686e53ac3f38b09e1eecef2cbff3faa7', 3611729], [8532, 'd1140d004b582ed2fda844731a3214602f8f7e5b525deb8498ae63d7635ec7c3', 3618680], [8583, '78bf592531f2b2c6bee4144bf5534be928e67ca01021cfa053ff1d85d001589f', 4311414], [8731, 'eb1cccc2efdc6312adfa631c902f6074acc3827733a76399f9bab5009611f4be', 3936443], [8762, '1253fbd4bf47b49be9929897a51d0e16458370b889964dde9f88dd7c565f4663', 3656346], [8786, '1ad4073ea02bc04a3ef8af4d8eabad44c10a7361173bc724addd203cc54d22af', 3854461], [8816, '94e110282b96f0f963fe75b649a56561d9eea4a7bb9d955a92cbce611311a80d', 3606323], [8844, 'e9773c080bd4c611496616f6fdcaa5fbe1e260a38f6bba4dac4ac8558ef742ce', 3855028], [9190, '9170c1f3d7b1f9dbb71335b59598a18e28257ab3c9ff539dc13531b97371f4e9', 4108997], [9214, 'dc6f1d90170c5c7aaf03bd92d76cdea68560f3c90797eb652fc0b206169ddf10', 3811820], [9378, 'fb9740c86ef5ed34724779fee0f375575984d33da86f256ab52f0a2ef2ee5aa0', 3914158], [9448, '8c9b4cca735f6f490b2db9502b1d652caf6f674435fd9839e14d5d8d2f5cc35f', 3962780], [9483, '9a37fced8a5972771d4b033b465d8a31b1b6f9c87a4ea2061296c7e5126f9354', 3954097], [9514, '270923987797aa53c808f2ce035365ff3e6db59f5d93a992abf845836472975c', 3476911], [9769, 'e0543456092290f5d83b3f2b8317cf45aad53eeed668df836fae6da50d11a69e', 4040574], [9891, 'dd4b982d043e04a50d96d013ada9e03de262a5f9564e00501d3db2b75a0247ba', 4221659]]
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
