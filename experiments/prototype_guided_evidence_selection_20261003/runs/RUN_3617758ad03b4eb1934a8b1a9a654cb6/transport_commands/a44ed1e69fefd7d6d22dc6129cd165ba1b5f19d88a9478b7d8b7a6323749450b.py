ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='46a377ad2fa765a30f78e718de092dcd069c8dc9bd47ff752d9b1327fcf32455'
ITEMS=[[14483, '09d8605b310d6e4e20169f6ea3f02283f12f5d2f66f94b34480a2f55617c6c9f', 4038670], [14514, '39bf58f6200f4258a2e38aa76b168be1e28e164775174c680b6e0a3079913b34', 3810406], [14537, '04045e749108b29932dfbeff541f6f2c83a31df06794eb41f484d7a0df29a839', 3766101], [14546, '7d8e7cfbae7c65454bef760bdba47221c1841cd8a09470fccf9701d10aba74ed', 3603910], [14619, 'fadb1aaa826c7d1d53a151874ce681a9bb2052cf563918a2ce807f1c08acef26', 3848532], [14622, '6e06ffe353d4883f22bc72e4e05c3229740e25377214b5a7a0f68c25003760f3', 3723357], [14691, 'b3d31f233f0ea3e07d17be55bd32ce06ac8070ff8b955c6869d98ea6e4e3f872', 3569341], [14748, '442cda79e980acaa95569ebdd818ec6a3fa707231de54661e0107203a53443d0', 3927803], [14774, 'c430fbfc92efaab850dbdee0b75e643238e6301fae14b5d8b3fa871c054b5e9b', 3714175], [14783, 'f121e35b24ecceaa4615df4bcbc67c225d1b72c4cd1c0bdbcca5307d09e677fc', 3654280], [14791, '653a9f74cd8f6e893692891289ad84d2a642c565547f23b40e9e4a9e7697e7b1', 4186340], [14819, '29775eeb3d200ca5af1cabdc7b91e9166bc58c30409a5f401961237462c54bc6', 3484721], [14870, '752a6ce56768a0ca74d8ac6ce7a1ede6ba3b8b716066f6bcd62d895938457a30', 4648471], [14874, '3d976140a78b6c25704506219c5323c941c6e22ac85658ad6ad20985265f0ae7', 3662634], [14941, '259fdfa82b2ec29fa8ae2b8def61b6bbfa8b6a3484cc573c21c0a2c21d39e79d', 3973059], [14966, '0b01425b079ff4d0562577aee86a4c0d9796bb3b412d9cd569b815f86306aa6c', 4264163], [14975, 'e684b4d4db06cdb1bffc481464c74534957c2f90abea8d43840f4524cbf443e7', 4027299], [15029, '54e5f852d5036e4feb99f5519245afc2273e89a30d8ee2f31b1edb5329edc56c', 3708252], [15062, 'ac2136b854230f7cc02ad14514862a0b7bb8ffde364885b1824188498fa14950', 3540903], [15073, '8bcd6a32d764c6eaf10fa7c42eae6c90b70b5d816a53cd88115c4b83f1d2ad9a', 3914578], [15117, '865c75206237926cf5293fc86501f1330d28f21d4409dabfbf9d5fee226e6d7a', 4100023], [15140, '339a81c2e2b973447cc7e7ffa3af33fcf8f1f00b727066b66fc67c35f72c84b2', 4016716], [15180, '3bec1b83d46578e2aa032bd2c6521a08553dc0d5e303f0e886112ba3c1d11409', 4498806], [15374, '3a745fe532e23ae429a29469c60545f7bb85ca5d82ba30c79c24e0c452a8fe72', 4134097], [15379, '9d02e2d9eb78474350bea52cd861da905a30bbedda566d81914eec754e835d2c', 4189203], [15391, '68cd5c24e8b13526c4c6d3e1ff8f1d76ad6f995bfe389dd51942c64ad1354bfb', 3836079], [15451, 'ef6d781dea5f95bbca738d9c3bb28bc82220785ad75ac2c14a43cc9c374be66c', 4547393], [15485, '8833e22bc8f913c3c46c4448ea7c707dc08aa8cffee6ccca6fd18f0306c4a63f', 4017551], [15496, '73db3d57e53bc1e84830a776710f679f7fd22164fb02fa2aadc37440bbb2d82e', 3717026], [15544, 'b6ddc58f0e3b9b328b041cc6b4c5d138834f9d33f03f9aa01bc1f8c74b9952a5', 4104620], [15554, 'd004301f2fcb064e8ed38abf359aeaa6572b1132cb33b691f21c32e2c8d7918c', 4091539], [15567, '73a7221793d3922312c67d0bb13dd1b8b5be745238dcbe42044c5f9348745135', 4091840]]
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
