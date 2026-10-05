ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='e4e7d34d8f4b34efc569b8ae89a987b4943e3dc56ae1f3a850cf726ee1185f4f'
ITEMS=[[10321, 'e0e36c8ba9d10a8060fa14890a0bce0e3d347d77a6930369b368e1ae21f052ed', 4330796], [10342, 'ea70222e7d1d759c14cf82fd11098f4eb7542deb508696f9d3fbc481e663a698', 4130793], [10414, '2c6a3d4cf1174b15b5151f6e02f0deab4bd978c59eddb28b121e367964b6a590', 4332068], [10430, '40ddc6be612540602a1691f4793d0b762de191e2abf014874fd34a10a42267d6', 3772433], [10613, 'aad6c65cb0cbe7ef8558b967c0005965708c4a8f97f233bcbf463ee3f26defd1', 4192208], [10639, '6dfe00e0850c631d05d72b047c4537366fc3644cf7d8089d4135c8414e786507', 4418687], [10769, '086b4997c47ca6c8857d01adadb1224b7a273d907a85b32c94790016c86e4682', 3776371], [10822, 'edfe0d43f9e0a9a41e109d95bc477cbb09ce2cac534f4ee1285a0b5883ce7f0c', 3670922], [10851, 'ecd3e7e5f554825ad7641bb599df5c8663a9f2afe7a5742a03d6b9d11d7271a1', 3952337], [10871, 'f26558c7cccec19d124c8137c521d7338071ac36d4bd9ed8bdfb27cc29469338', 3767079], [10890, '106e87f63fe6bde338d72dde7475dfb1a5006b0716e7d984c30d6abf7ddc2902', 3981648], [10928, '3c10795e59ff6f4706e261470e4239ebd835ea17f159914a6f59e0261ab9b2b7', 3902285], [10935, '61de8bfbab3bcc369777aa77563cf038367f6a37729d0e7beb5df7d014aea014', 4061193], [10991, 'e5ab185dbe80c2c7c5eecd7dfccd7b4dec5eee8905d7db1401ad44fd4eeb26f3', 3900744], [11168, 'b4b23eb0f88ecba638f5ea4837391da13699ad7ea2d647edba955bfecb5355de', 3692870], [11398, '4497065c678cea2c22e7d8c531e8f1cbf436e362842c655ce64721eb4a021763', 3787445], [11497, 'b7e8b97e78c572fbe5a143c1e45a831b3a1cb5ba67ec01cbcce7a8ae439066b7', 3780096], [11630, '90651e8f483c76112778ea87d4a4d56414a5dd2bee1b4a214bdec9c7955bc4ae', 4584215], [11667, '010783b99089d165718fade600c5696b9f48b3de10065e854577081b6246cdbe', 4146007], [11680, 'cc7bc68a6bf5ba17e6e456d3aa017561d034bb758f1082949b3b07f7cf9f9068', 3781287], [11720, 'f489e7accfc7b8f819f78a5aab3016bb24e8cbc02a453c34c7d360cf6b758a81', 3923612], [11726, '09f27077190cfc1a987831efe24e88256c75cd6b60ffb849a56851455704355e', 4090530], [11877, 'a2de20dc336bf724dda22517a98ec2ebe666d99f15600b3cc8f5ed16ab524745', 3668920], [11926, 'beb54ebf5abf0df31078ad635a4fc5f165d69416f65b75ebcea10f284ad0e41d', 3914828], [12107, 'e7d4b6ed760d95147fa1da317f8b22fba18e5a9f8692cf2391b34e12cb6b2076', 3754590], [12109, '50c2b137e752e43a0d695f28eb0e234d585c549a5936f82ba86a3c126304e995', 4106497], [12169, 'aea4ffe39cda787b0a4c949d74eaa30c36ac3759fb4f75536b4727a42488851e', 3788245], [12179, '2a72b213c83c85a5a8c4d01bc46f6c70ce82ac5eba80208db897dfdd95b0cb93', 3612751], [12377, '9af94fa231497a7f7ad7d39dee17d28dda92c0156e99174392944b9d2f0652a8', 3776823], [12407, '0d089999a5acfa5defc167fa56ceb1c1857f9098f0a733001dd6450fa5ac5800', 3865125], [12443, '754d605f38b33da7fd31b6e1ffdac3aa93394f66bffcfced84d92f3817a7113b', 3892754], [12495, '8d171aa594246978f3529bb28f805fb65c626cf6ad036ee314e78dd5ca9b7543', 3849681]]
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
