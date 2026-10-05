ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='b0f2eb55b9bbe31e2a07f281e8ddd30a4df26671d875acd30e2e03b029e5f78d'
ITEMS=[[52076, '0f096a594a0192cdda03f184cf5f3b88b010552307e1e9b12ae1754efca312ab', 3778811], [52091, 'e3780d414ff6e457760e26fe97f8ad8bf028ded897ae9da76b112611456ae902', 3735216], [52168, 'ee775861e668bae13374fd6be2579bed12d1ae052922c763f32b6f172b636a45', 3645888], [52192, 'f4a744c23b378207b5d6dc79308c04abb4ac44355a4131f0578eb0eb0f719f2a', 4024325], [52238, '03e3aef7832f2d99a2e11a99605f72129a222321824f1818bdda519a27fdd0d5', 4054901], [52281, '79ffc3722c854f688df95475b07eedb377fe4b96b483a93483668445d268cd58', 3793998], [52324, 'a82a047f336235a001accd89b739d0b670943b63ddfe7eda8be7f34694b44e57', 3872552], [52388, '7f1fea6b31b7a191dd1f6e648af17da70e063e5ed6d0b7afec4075fe7e209438', 3747014], [52412, 'fc6293466a837235f2fce5e4c268df7b37f47bc8154a7ee25bbe9276cf79d196', 4202832], [52462, 'c64f27964ceb7c004d3d5bb036d21e124754c6a97b233d70980e9d4f4b4bd5ec', 3815627], [52472, '1220b569d505f63e44299597c2a128b22ccf386cdea7a489fd2449aab35816e3', 3929784], [52565, '7e0431a81d036f9a295d8e94fc4b6a2f67cf2d6486eb75d151e59ac318b60a37', 3681937], [52729, '672d00dcd09f4713843d78bf5c0aa8362f96879848fd5e466ea4fcc4a0775fa1', 3956620], [52790, '111fc5b20eff357455c0ab6fb2b65740cdc7938d6d9a67f2819daa48106d1fc6', 4148595], [52891, '784a61ec8047e45bb4dbc2d0bbe84016bc64674874992c34f61df86a75b60355', 3822147], [52929, 'c3c9205d90c7fbf608126c2188656ab7529ffe83958092a146f1b8bc2c82d137', 4319341], [52949, 'c59208e9b8593737d8734ae61b8bf874fe4480ba3c905326333c051bb0e3987e', 3836433], [52979, '72a0bd54f4f74911cf94f436cdbe72beb3be8eefcf01189049d3d80ae9705adb', 4074952], [53082, 'c6511750e199ab761dbe6163122578215fb726e1695e6435e9a977aa67ef92a7', 3732907], [53113, '764db6cca43c4877a2ec0383993d5fc59c31a4f1d347d6f0e5c1c5c02b782fe1', 3807607], [53170, 'a086e468f48cfcaf93ca5c58ce86e61ef5f5ac12d6ae435af732be2c9c1ec7bd', 4277756], [53196, 'bc0f13634f1cc8222c0e94c9b9e7e9b553cbd656383372b06faad8ad66d6c515', 4264776], [53208, '0a65c6db097de784fb96f0fd5ee4a48c3c82f75c1980debc59ddb17729285616', 3873331], [53294, 'd2530e3afb443bbcb59d99bed66f97e69ce9aad0e61b7076c40a20cbf11095c0', 4170551], [53370, 'eb99715de91a8ef01e7b08c86b930e0ef0c95f790b8b45c99049cc7e7a31e1f1', 4028212], [53388, '27b4371514efbe628a178477b7bc35ecdcc5fee26b76939b6a8dc1b5c607de2e', 3930947], [53405, '66f0cc62a7095009015e819b932d0fa1728e8e205db97e7f650c96a055455664', 4257454], [53450, '9f235a79770d6e04fb59f202b131947cb73eaec1b89e865f9479436f943531f2', 3938426], [53604, 'b2d2f5c20b0b26239ba224524c018691e91736bb24e0fd5196cce6c7959a52ef', 3831613], [53624, 'e601fc537c06679bda4857e01b18f2e493a8c740affb01e6d8058ed39e823834', 3914028], [53641, '7e006580fba824407c91091ab0e21cd3af77cf2ab8e8b2c39ab19d99a743b542', 3996290], [53672, '3900d795da4ad5c0c7b43ae3c95bebd5bb3fb71a7c40f17b7b737f9f9bc712df', 4649813]]
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
