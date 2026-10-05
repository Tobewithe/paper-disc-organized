ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='c82b738a911df4122bc08275369a4ac53e7bb60f0d2da98db62de8ed4c128c4d'
ITEMS=[[80521, '6cafa5fb710f2c58a8be1a5f2ed7b8f74cb667c9f2ca30798171e0eed656d6ab', 3868965], [80526, '98bee7dbcaa543f279dec4594c83ad1da27ffceed533e9a98da51687982e79f0', 4146824], [80585, 'd232f776c3dfa07922011981a829dfc775b226b2dc8adfee53443f67b066741f', 4303529], [80609, '91b74aa6ec6ce3639606820342deb957dae184478ab32106369c8252866a60c8', 3548705], [80646, '70e1119f86d970fe02064c30de83519ee48fe869f4f59eb58a31baabf97e925c', 3692876], [80671, '96d347c2e083315e0cd718ba986acc0e12eeb527ac5e8cc01278566fd3c90cf3', 3591589], [80747, 'd33d6252621406638a058d0819197affba9fbe189508153f20ea060e681859d3', 3863413], [80782, '1799af475e73a18f9e7bf063a36952e1c065a392f1e528e5b154d30bb677121f', 3779077], [80821, 'c37b9fb113600f4e1e733ec516a82d9b4472c43d2c87451340e4a3789f5d050a', 4053982], [80916, '347256b82404a0249ab432ce4bfa9aa630345768f943e0c79e71cc08770b7c65', 3866110], [80987, 'b840a86965cd8cc2ca548656a1b6c5c67c78bf8c4e9f0fbb33b82ab2077a4038', 3870114], [81003, '518067cfcfd73aaf549fb08a44196cf11a177e24423c87eead257fe7037be6ec', 3913238], [81054, 'bdf85dc0db2d7fc6cd11965f07dc1890185be034875bb795848a174d37fe8cf7', 3777036], [81057, '116ab4ec46d81b45106128076e39fe6e65b7b53cf4d1977332edfd32975de8b4', 4328662], [81105, '020fbbdcfcb325db9829685b35e2b0bc0bdb2e294678712a68753699b482123a', 3675353], [81228, '6ade571858da6e1144ea5c68c22c39411695b963830981ac86534e286b5be4a0', 3842115], [81239, '80f6115c3c1ad96b4203e624a7695246d6adac59d3d8206c77cbf57a2f3e7a13', 4076837], [81251, 'ec4ed95b19eda0c350d63349db32b4e5f7f7896d20f5a2bb0bcf8ac27e5a3573', 3660515], [81294, '6804aade703baad2b0b736f11f4ee691f31b7259b8a3a00db56bdf5dcb355503', 3928582], [81307, '888f6dbe03229d5ef523fcd02482f9bf240f071127403d86d563c2732b609167', 3620696], [81317, '38e11e4216e29a873aa5ccc5c2097494befca98bb19fe01e2ccc6626b0fa8976', 4265647], [81318, 'f47bb934238cf5345e019ea4e21ec15b7cf9629b3791172249fc9decb4f0ecfe', 3469731], [81357, 'e3f5a95193a163558644fd4a2f09906cedcd6e234618568838b1564ef2e71214', 3826890], [81398, '1fa55a1619234dff8d1e43924d910599daa0e73c0800d4a8187e9185b3e2e057', 3767014], [81461, '6e4a9df8c9ac2a8632e39a6b24fc6facaaa097cc3ce12cb48ec91c5e9cdb8863', 3618547], [81544, 'cdb83f4b206f05793ea84513922e227094318388ce5fd31c59279a049abe039a', 3691051], [81605, '341538e3b80e3abe8870785c9a31ee20fea7225654b87245b6ecbdeb69892d39', 3799039], [81612, '56fea93f5d6a163744c685eeb2a2179db8279ce0319d372cbd976501255c4ad7', 3713677], [81620, 'ff38ff4d0d9ef5e4b96be8e73e379a71013bf8fe230e41e9a774ef05b2decf7f', 3941008], [81738, '5b52def8efbdf2a582c492b713e8857ad31b0cd1b9084068722cb10fef8fcf43', 3960005], [81784, 'ba2ceaba7cbe19004450298516007dd17484b8cb4b25de3a3525aae166731e91', 3999527], [81798, 'be0cbfb1e40dbc6b51c61467eeb703966440fb91358d1d665d8c3e52beec40c2', 4275278]]
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
