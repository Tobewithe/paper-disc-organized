ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='a87e6d6e309c698e7558e7205467096dd422ead481e12e71301b84ca53965218'
ITEMS=[[45479, '04e3c23ca12947d106437ca02c8a2ca9a37b6c43452680cb153a89053e299f4b', 3678988], [45580, '2a51ff42093e32b4ad6142f7ee63900e82d22aa4620ae3e9acd5ad4be0b80c4c', 3872977], [45593, '6b6c2c86625d199f293dc827960043143be519a57170da94deea0338a3d86c79', 4127104], [45627, '05adc38718133ca29d1b0bffa7909a333fb774ce03c675a8acd2e1781bb6736c', 3766441], [45771, '3b14dc8cacc604e288c8b9de505ec5b1b1bf5eafb0976c27f80cc3468cbf0350', 3966360], [45792, '094515c1ac44fc180769de526a77596fc7e602af0d45d92ba56d4245d321b2e5', 5025380], [45808, '3adb09843111652a5165f6cc6773dedf26491c48e6fc5d83077d6fd79b97e834', 4009568], [45859, '59d281e508f5a28934867b86ba4bc504313fdee1d91cf320dcb3665b6f46f6ed', 3742299], [45864, 'c49b3299d2f62b88081b5dfd0e7b7b0c7c79389e973e90f6e06bb9913ce4fa83', 4131380], [46004, '78c2bcef35db1009ecf57aeccda80da186849fd59121e1a007f4ceff8ea4bcc4', 4031299], [46025, '0cfd661aa084ed8199b5e878f176d90db8d8113cb2bfb19f2359a341af2f3e92', 3842858], [46048, '32e4409abe3457a118cdfa91a8956804909d4aa0fa26453e006e3c2da879c87b', 3830329], [46146, '5f21cacde3333dd25e3a36bd129dbd5fb214304ae9faa98308abdb9a86635e33', 3965920], [46252, '80761f5b46c8fbc2ae5db3e5cae00c938c892be94617de23a88497d4d4f407b7', 4041069], [46356, 'adfb4959fde52309a9a13e2218b9f48a85dc75451622f8b17371d97d7b069465', 3737046], [46432, 'e12ed147314d3d4f4774e6ffb668a02afd6827a1102b6ddd9e4e3a49f46ec87c', 3725180], [46503, '9fbbd56255524d1f0be3f4518ce92239eee063725f45a2baf29cbd97dd69894a', 3969465], [46544, '1775730751c0328d59d99480b7607714caf48868bbcec7a55803a013f5c15e55', 3850936], [46591, 'f3663806c358f4a1d91c8ebde27a6b730ef8cb052bd7b0e12cb7b171617f19df', 3683530], [46609, '721cada61e150d0964d4a36b472ca2670742f178562ad01f95ce426a42235683', 3935478], [46630, '08498005a5504a6953aa510cf7e30e831ce15a730303140e75edb5ef1118347d', 4381332], [46754, 'cc7cf059b93e85c741161a1143e78c805970df916fd407c1cb612b9617fcaff8', 3501103], [46786, '79894fa6994a4fb5c237a9e869a87301df9e65cdf041a029fadf6392b207215e', 3660729], [46852, '236b08fab27b8a50d02359c3eb8524da0edadd2cea80b1b893da367d239d51b3', 3859618], [46885, 'ea7cd2bae819fec5871803f38444a36e6941eb15370bf34cef1a88d5c4fbfbb5', 3643341], [46981, '2b263fb65f09a6d02b9b059bab6fbe50149dac553018df1340bfbc9dd62a7dd7', 3985290], [47010, '2d74429a0172e2c24f7fde5e34900e186f672017c513e3ca671fa22839e5d555', 3969428], [47016, 'da282e999d1cc6b454bcc1a3d7d941e0913c7cfba4243f2601040472958d73b0', 3450252], [47066, 'de9271d2c00bc91563a96bca2ea994c4e4549e853b733cdfe0e25abe17a36c3a', 4112972], [47073, 'd81868db513351eaf880097d72037040d2f7ca073407084f31e2aa689efa3fb1', 3752814], [47112, '8a912a9626e94a04180671537038a0ea121ba0670810c82a303b65d8d9666a28', 4604986], [47121, '38179aea1b9f18e69ee4e7f47f7ba27b3fce57612bbbe2c168b51249d0d1e3c2', 3772843]]
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
