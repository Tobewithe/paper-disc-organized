ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='3395159bff5c312d8b5a046a167f5805f6300ad9a1aca91372e8f239a3f3fa58'
ITEMS=[[22796, '7a0314401ca4e5a5e8135aae06b0dddaeee5c054e7237f4eeaf7fe33dc9a481e', 3951722], [22802, 'd9cac8546c1321ffe97d7b168bdb52eac9634732888b6cfa2aac470c78d547b2', 4161549], [22861, 'a70f319be66a145a0cdccc9458c57d1a0d631402a494415282e96a1895ccae3f', 3906384], [22863, '39756183ca59992c1375cee0fe8706804c881ae4e5ad59d1ed596c26d1d715bb', 4974520], [22929, '4575bb485fbe798a4ff3c2bf425eb3135ea82b694bfe0100ff08ebd5f8ac0387', 4023963], [22935, '08467544d0129b935bb46ac78d6688c885a98e675d25c7803333dbb6220f526a', 3744375], [22940, '97cca18fd09bedd1e4dc114fab0720cb84e1c280c476951a66b70c6aa4c10ab9', 3946195], [23035, 'cb7eed57a68c9172e2b6241d7edc203c643164d12dbc9957030c3d96b80950fb', 3857406], [23038, 'ef198496a4cdf4398b22362258982a6c73a9bcac18603b24b3e4f7d0a3728142', 3740343], [23173, '4537736be8d12aae4c1e4b63ae5969596a9dc7c7c8f3c507d9bd32d46b01682b', 3751583], [23230, '65b454a3d478fa3082b04bc761fb0f88541aeff661a3752d55ab6478f55ac408', 3961736], [23272, '8e18501f6bfa9ae28a71fdcaedb9c146b5fd77a1e71963c21aa01e5debb145fa', 3838039], [23274, 'f3d80c4e4e78e3bd8637d5273bd56ce5545f4a1597abe30a35b6789c63e21b8d', 3932634], [23355, '97c4c65da20dd560fe0a5a574b3e509ca81c3146ed49b5115b91e995d41ab8eb', 3914250], [23369, '73d25ee09249828c2442039d9ca8e8fb81670d1c5d97ea5ec2c60a99d4e31235', 3700423], [23413, 'f78432e6d4576220ce5ea1e7fb7c717feca6819c16e6b646ad5f3089e2f0b2b0', 3853733], [23673, '5a0ce792c6ee72ce20418fb5791754fc9b297094dbf056e5c3b275f9ef5cd498', 3688811], [23786, '8c83a6bfe4f4a952464f04260fa7a8a097b4c6ff8db82de976285e32e16e7a3a', 3655435], [23821, '24571c4dc63210f90ddfebbd4c1ff77f90b7713b1814d226c17aeb8016cf00e1', 4453573], [23998, '1601efd75a9c5be768784a239cc75280bd307b5144d316df23821cc363611c41', 3772705], [24027, 'bac00d3381d7ccb27d92f3c7cfadd488aaed90560cf5bda2626f94ca8b72e294', 3615875], [24040, '2642abe8d78a2c24a48e7e4f2e86d576a28e26699d681ef37576eab915afb70c', 3746433], [24195, '339a0ec38b263128f0d0a667c6555712120dfa66e72ea681de7f0475f3d10314', 3976885], [24260, '0eed604cef7fbafe39c89e5eff137a8cd2bf15e159e0f965bff2e2354eb7cbf4', 4208264], [24299, 'e6d9d920deda9cf417ad73b3d1d5c5db779df67e2689993288dfc28959b5082f', 3858739], [24385, 'daff6f1110831e8a819a06b54ad3f84f3ff8d8bf819b7ef2525dc47cfdddb21a', 4212815], [24499, '5110b2571f8849a89cbf6a633f1a56dd6b243cc100780b75a5e9d926c174968c', 657858], [24567, 'c3546bbb5fbc2ce56bd31dd65c1d7f8a94c97815e719e66eac1c83cce47410a5', 4138649], [24571, '5155f271e21e67a8fff1366883a9a00e8fe986325fcfeb2e4528032f9217b2bd', 3956811], [24582, '176b22f907e88973e2f69e611116c53b462c6ee611c42f598173a5f815e0c272', 3761697], [24674, '9bee075538f0761ecf4a6f624fda9ef189c8c3b8ee2b06db9cf2e49740eca219', 4652562], [24730, '350c4fa806a1ba72ceb75b78eb610508029fc0c77927e890949fe18f75f78e3a', 4444094]]
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
