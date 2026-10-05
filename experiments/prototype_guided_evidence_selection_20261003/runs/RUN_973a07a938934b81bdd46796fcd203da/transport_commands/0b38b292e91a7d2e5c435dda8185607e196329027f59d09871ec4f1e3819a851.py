ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='d58fcfe7ac958ac01d8afc85d2366bd27675354768e12244cc531e0fe27f999e'
ITEMS=[[44160, '8378cffdc245ef959889388937b6af40131a069e5c70a0a4e665f88f5b38a015', 3781991], [44167, '25df40c24190aa4a0460a13ae274ade233958264646cdfb7e2afd44e3b23d40a', 3770061], [44170, 'f915cfe1cfe71d2a130cfb685c6e7442f561c12fe43bea8d958a92a31a77378b', 4317300], [44227, 'd9f6f795e9f9f7c5692bb61a6f350f899729f0fd0674a8d28aa43d106d561613', 4062819], [44232, '0eb8bf59382e71717cd68b88a7cbecf350cc7ff08ccf60f74f09774faac31451', 3757952], [44240, '935f540412d7687386bd2b1fc1f8c1129f58d984bf14a3001748c4c176363bba', 3875096], [44279, '916be94d18b01e5c4e38749a3808a6868d492b27942e64e5e46f2f2849dfda4d', 4065972], [44298, '744fa8e4c71060713f6464222a6b95bf10845add880c9a80ff9dbf79211b97db', 3922127], [44385, '7a14239e2a81676ce81a11c86f0f988464fad26f7aa8d2cf5cf75168e25ff67b', 3714322], [44388, '42f8a6262a4293ce23a6114c56e69d2f0db63333034fa2f9227e2692cc346770', 3736485], [44404, '328f0de1553a7a90c3d676b9a9fce5f12c527f5b4167baf4a74803e69e9cd587', 3883451], [44509, '1b04ab3031feafe22c306b3a3aa2c1d230012a1490311851eb66790ba5ee8135', 3587908], [44590, '4a87e4036f8ed87860d956dc02689675f24ed0863987de027ba3a4c16494d1bb', 4259915], [44594, '441c5b888547afff0a456d5e6f1d3d5b3b82f8a995f731009f8f4aabf6071dcc', 3970535], [44609, '910d02496db2bf88b0bead576148f4900d8926708fdfe04512ce2a0e2570630a', 3688222], [44612, '3617f6ef9d4423d2d9fb970a766b2daba485c1e33f3a846d31d81d478fa9db3d', 4129137], [44637, '7f8997bf3a54428f1d113063333f6e7e6839071edf37365fcf2c0e694920120e', 3692408], [44663, 'b45885a09c9af851fab31251d483dfdddbdf6a2db69043a4cedf686f731ca719', 3799136], [44671, 'f95180a3717223291130b667cc8f08090a811b11a9ece4e61017b0e9611f96fe', 4089945], [44672, '9941b2741a4c018ef754069541c1eb3a2b23036355df5fbd6afc911d98185467', 3599519], [44958, '9e7eb3c3f4abc1aa2c2e5fc337e5980c9c4b5c67ffb31e735cfd204d98c8e694', 3997279], [44977, '1173d37713f9b89ff88ac4f68576432f66f6e9bc1bbf8006d58637247af692da', 4911036], [45070, '2cfd05b683d7900c2ea51ed2ce5ae85e8416175a5013b9cbaef7024bdd6486ee', 4074795], [45094, 'a13617c2c0554a47be56ae7463e0185771492bb04d5589dad233e92826839b01', 4018170], [45108, '678220b46d89e5392e3dff414398e2f239bf14398088da1e8a95c813d85c9993', 3860277], [45175, 'd8d1da968d0ed817e917bb61a2596714b0a6965c5c2a85a2904831056dc24a1f', 3862283], [45180, '1d31d6521920767b9c02b8f3aed1b34b5ae4a3983304a2cf5885cb541f88c637', 3673839], [45188, '56345464be9bde7d64e6a30c2efaef71bc84e5a337da93178f4f74f26d85689a', 3837942], [45247, '5fddff80cf8f1ca2e6d1efaaae552a90275b041cf863b3f15d014e4c3a48381f', 3808935], [45276, 'ecaad460e6343cb0e5ab10ae79d47f456ff5717b3cecc7daf38a1d4736136f83', 3644549], [45283, 'bb8f1c5771231febb393415d8bdf2ff4b1aa45e974f827c0de60d4753ab55a72', 3748255], [45351, '954416552a706d0732b6bcdb51cc4f61bb190074d104d567d7cb942eab5299a2', 3862895]]
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
