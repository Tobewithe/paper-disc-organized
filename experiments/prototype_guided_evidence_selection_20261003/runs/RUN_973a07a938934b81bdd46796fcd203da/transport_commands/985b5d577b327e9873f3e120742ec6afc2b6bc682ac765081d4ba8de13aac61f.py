ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='6f99d2cfff64e68eb5eac075005777d1b7169640c518be9c58f213d9522d24f2'
ITEMS=[[53674, '867b76ec809870c3cb6500db07b7447b95cbf073ee7e86e32c896363a38bed85', 4358602], [53696, '38a3439a0afd16e7c4f95f9fe48389ee67b8f2b3fe3715b6c2c8903cb429ad30', 3730525], [53725, '8122cd5ab692117d4b02e608379642640c86536c9671def85cea36be77aceacf', 4160934], [53778, 'c237b954ed7c13e09ed1e03a814d0f57b99b2378828191f97b99ec9a4cf5f79f', 3799730], [53870, '9b219ed076e01d811fa7ad338991f4441307e2a4ef6471130372b3d4870d4fb5', 4006483], [53909, 'b4e2c84915130006dad6e76a2efc32c07699b727b43c92ec1a85c1656331c324', 3821851], [53965, 'cb1f4c689992be38ad1b31ad96549639ca5b6b0c248d4e2150e5948103b63a73', 3779751], [54003, 'f008fa15cab065203a81d21bfd31a6e2d8823b88aff24cbe3d9a6a5bf3ee998f', 4298601], [54071, '75da5d5eee61a7749e55c19eb1c46e4415be3f234d74dc8e73c764cbb0e8c36e', 3914291], [54091, '0e8299cf66ae22efc91e19d079876adce226f2a834dee45f691583141ad13343', 4066242], [54121, 'a83c2fcff35f3987c7d063e2918613d0bb80ac0b1d0a2b0fae61dcc7690fe48c', 3897387], [54123, '0e2435e89bc1b83bcf5fb5a4612117c9cb48b1158324df048aa937f529bc15e3', 3957101], [54124, 'cf8eea7039e54179b97eef111a1d2552571522976eee75d709c75690bdf8ca5b', 4143693], [54245, '5863fe4c5ae1e88cf46b0a74f85f55517fe01d58d00c985aa869500fb83e711c', 4070986], [54448, 'aa785d7f683d92b82f1097994678f194d772cb42ae9da560e0aa3f634003b633', 4196691], [54485, 'c685063753b39d96a12a14eb33262a5b10fbada2bb5cc37f6a4168e693d529cd', 4594924], [54513, '0cd6ee84c46b453af165fcbb8ede906403453cc22ca21d01b2b1d6ca4b38e899', 4045701], [54541, '4a2717c355dfda15ed73572180e21f53c7425c09b9a9361a15ca8338bc6c76c3', 3810438], [54560, '58b4ecaebd69ae1f0d6e168aa2f1564765217c40cdf2c2930a740ee53628b36a', 3772188], [54575, '09a07c27ee9dd4fb73448e78279e9fbbe982e9a652b23f2de63cf7044cd763e5', 3512602], [54594, 'e8c2a2ca23cf7efe8f24e6d2e2e75a268f0d40db9b7a7c2d7b5b7e33be54232c', 4488233], [54628, 'b947265a2041cbe10fd8e2c9a822a6db66638896d4b53ac9bcde8cd81835d371', 4540499], [54643, 'cb49c50a579a97f177803a5074d7d8ce98266a3730cf36babadc34ad9bc45881', 3805093], [54666, '0939f01440d9a44514c379e849ea67e71186e1008e1c2849e373182f8e3f7230', 3973821], [54671, '4f9a3db3e153b64b45a4f6ab20d606716e45801bec340ab58ee2af368bd5744f', 4145609], [54679, 'fdc985b2d95c8bb5918df0d713f4d91c020c0206b6af2414a5da0b5fd2456591', 3648990], [54903, 'f5db2e98a6d7692847bf205de7203ea7698557f961974d0c3e17a386484a9fa8', 4007595], [54961, '32fe762cccac0ae783a7c84b86d5dc9716653fbdfdeac7066a0e9e499509ea2e', 4122761], [55017, 'af925dd0fe8d35768dd230d79ae2810d6008c803551f1a4dcd0deb35d133a3b1', 4071593], [55072, 'f99528603e943680033133260eb410bddca0afaf66a2c4efa8fea0232067018c', 3692351], [55077, 'f2c96ef3e6fd77a7317eacf073992124f11ff00b3a78f317e2260e036f8479f0', 3696834], [55109, '90282b54e6b5ef3526b3573db0613043c826d1691c39df82586fcef7fef05ddd', 3679831]]
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
