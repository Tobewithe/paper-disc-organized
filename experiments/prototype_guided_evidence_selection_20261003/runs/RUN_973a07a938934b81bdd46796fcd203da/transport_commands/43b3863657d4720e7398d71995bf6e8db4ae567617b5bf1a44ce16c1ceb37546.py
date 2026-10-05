ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='d354ca51770fdfc780c56e2425b4fcfc9d8c8b2a7204884858c411e4655d8fdb'
ITEMS=[[39697, '735dcc899af962213b7fafedc1ed7557aeebb9b16f3c368f29f0ad784900b68e', 3729488], [39785, '2662f36ba093a8828a43fad3d7b0069bb5884b16051f01a06f3cd21f43f77fc8', 3918415], [39847, '96c6e1b44c9150b0881781858acbf068ec2437f1d121bbe6c275f1c319fac29e', 739094], [39852, '787ea50ac393923b2558e950b6236170e74efccb372f9c6068e33d245732c869', 4018036], [39866, '7999afc8284a5a3d4bfdf4b31c36544199f8cce275468f820bca91b06a3a459e', 4104080], [39881, '1e52d5062a664c6cf4a491c50a87c127bc9180b58cc0ba9a2cf2bfa8d6419f3a', 3689486], [39888, '46f5e1fc008e483d2d3f5cabff9f92021c5f98e73a8c5593e6721e3893639133', 3823172], [39914, 'ce138435ea0977501a91e3af9cf6b29744faffeff0eaf4d5d5482800432e5992', 4171338], [39951, 'ffe138074c747ab158a5734eff92830bfec84e0f017b7e5f66b6549ead58727f', 3764761], [39958, '8f5badbba286ac2a6007939d3827389e1fae0e05bab10848a612d7f6d64dc178', 3579985], [39963, '95baf0f9ff7b687735de9b91494ca6d2d0ea370309f53d63d1806d18df2e03d9', 3859061], [39978, '9372f4a77968d77df754e55f7c2728f272a41c72f958467609d4199502c53266', 3702858], [40083, '47b507d6feb6118c42ebd0ae12a3210cfdf8389df05883e4454fffe97315971a', 4116196], [40085, '8c845d4993514bd29a1039af48ba9f8d9b5e4e9c075e00578be19999acaa2a01', 3796357], [40103, 'ef02d1cb63ef01e053b45045e33edf623d1750f41abefe23e9c858b4a4f5a082', 3606700], [40130, '3df36af0753d6dfa2e5160f8668fdb587d223cedbb977214d9a2c918dee4fa73', 3986163], [40201, 'b614ad8b64c89099f71539b9c5ec915dcbf352deeec8926c23c422b7409c60bf', 3498585], [40210, '7a4d2ecdec6635259735f1dacfbf29e8f529f88e36114fcf6a46d8a8e2575041', 3892540], [40219, 'ad9de0d9ab7cebc877b4a7ea536242c2820af18c637f21bce45bd6c378608927', 3905304], [40248, 'd3f77b254a4a4ff909f6d0a6c695349387e50bfbb3c222ea258fc1dd44adcf38', 5001842], [40259, '94c8197a83d2134e1f5e76ceabc1f57fa634976fbf2584c3721d372678019b8a', 3733880], [40286, '156d1c7b4e93c95c373c2d60d1daf68d2bb74487d71afa2cf1a7c5e2bc1a51e4', 4022425], [40341, '0e7f2dc542bb7656b01cee7c6426afda46dd82d1cbb6bf83811a35db36bc8a66', 4340392], [40348, 'f779b2c05f73635df8994773ab84f77b3db88b6c701ef6a055c418e2498833b6', 3801116], [40428, 'a728f7a314fd636e30cb8d048a6e3f63ac037f7f0a773115056b438252449d05', 4884119], [40729, '6da60817e91f333b233a71cd6240af1d9275c96833c225bca9fcaf2d37873e66', 4475579], [40757, 'b9bfc04e0c9ae130636897d9b187ad28dca5f407dc511cf66a38b7c98c0cd099', 3655038], [40768, '8098c4c2d3d7250d7d1136e5d425ca2b2277a26268ba2376dbee1998d32b645b', 3587320], [40779, 'd973842c5dc8f8fe94a0f6bbf4aff778e4c1c9b2b82b729fa2b2a947471a4d7f', 3789996], [40844, '27faada3fe8946e4010d87d7504d53dc547c5704c49f53342cd6c2fdafc19652', 4379996], [40910, '53e1d4669f109b1960d08b8a8ec34d2e44a6df90b0aeeb4c065681fcdb5346a9', 4089071], [40987, 'b2b744c07337c232850b1496d2404cf779e298b445b880ea365b4ca9f57601a7', 4291135]]
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
