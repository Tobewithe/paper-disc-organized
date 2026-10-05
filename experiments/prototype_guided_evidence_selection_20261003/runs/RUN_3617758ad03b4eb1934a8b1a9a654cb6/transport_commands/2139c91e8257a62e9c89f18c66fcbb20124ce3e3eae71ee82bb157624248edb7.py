ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='3c01478f32b1ad82b7392777713076dff9286af39fa7600f044eb12625a0b7ef'
ITEMS=[[15944, 'b303b4cd1486ad1e170b86d38c51956502cd5b8be74d80241b2d0b29c4d36f43', 4391847], [15947, '1ba4d7d9c205d01fc26999d4fb18624c1451a7eba9c5d079f523d79483423bfd', 4181205], [15976, 'fc9e968da433d33b43fdef011f4c7646cb9278e05d8da46de45d2eec6dedd008', 4291613], [15994, '774573f06a52eb01b6d308ff9470fa2c954a3eb1c25f7a9874ac0c4ad74b70f6', 3863433], [16015, 'd976b83a22b8b386b600b924a6e0f4291548102eca94062f91c35cf873732764', 4654405], [16060, 'fd593b89f3c01580f6efb0ebca6af2c892b256bbe5638b7a867473762142f5ab', 4549953], [16063, '7ac4d23e9406bd57f6b6981b63ab95cbff4097e2386ef55d2031d9f56201b4a2', 4263513], [16180, 'd46671bd3e43e69f3bfd63a60ab1c892b53b3a688132be966d8a5bfe5d4c7c77', 3962971], [16318, '6fdf4dc830ecaea48c952792274e675bde9c7876c29e59c3bfc14909103c303c', 3686671], [16327, '2f8db729b7749c1f0873e0317f99752da851fe423f82cf5e1fe1fbc58a0df93f', 4319459], [16344, '3d368fcb3ac4fa446267b46d6a6a254d2235d9cf3d083ebef999c59e6abf33fd', 3689996], [16361, 'ab150a43c5f6be105ef17b667024e9335dae1423372504c27805b9580eda4cf3', 3974661], [16414, 'c17374a78f63061aea74d7913935c63e447245684be84ec2bd51662f8cfbd4a1', 3757281], [16491, 'cf5729f1f3e59fbaaf58300e48cfc6d28e0fce56c473685d1b541801d04b3181', 4315248], [16546, '56ccd2549f9dec5cdccd488be600db764b69dc4cf87802140b95ec424c633a3b', 3926801], [16605, '467b595456a36e2e0e16a4035940acf078000238de479d2e504fe9e38e650071', 3784854], [16659, '339d92d5e7ff406398edd28907d7fb0cabbe87eb16c90cba089823beeaa9bb7e', 3666820], [16689, '2a4a6a9e2e474a3c18da0bb24f43429c23a3cb83b06bab7d122c62eae5aeb3fa', 754529], [17183, '9acf058d005a9e68a3b29aa27147b6be8012565dfda2f598bb741654f2022bc0', 3906920], [17244, '377fa36cde8640cf2e86a14163941984a24e8c3ce804ffa6aefca66821077e6a', 3758777], [17301, '393a9b987e5918bc2ddc23298812bd25e9a19c039bfaa399b6628c35894151e7', 3620018], [17413, 'c47cadeab7413725e614deb9d718497d72c9f0c5f1aef4cdca9f8077898e83e0', 3578300], [17425, '23bd76fb520ab09e1a8796914b7ea142975532310f4e332e988d0ccbdb2c4523', 4028401], [17449, '6860ee2ba771eff2f11fcbfa833e2153467e59c8220094984abdc685f1c01e42', 4407753], [17526, '44e3e93a25a2f2a87d77046f01652256e944d969a47b95fb192665b04f894d16', 3829805], [17578, '22eb62d97b63a1e0ec93362831a5e01b5e3ecef4398bc54a6f6e448c5c0aab17', 4142643], [17604, '038601a47656644d76053a3c382838a52a90068359503a19a00d1c74112f4d0b', 4145713], [17651, 'f9207b50087e04cf5edb5096423ae703d771a5b990f0bfbd00d3cd7bf38c8083', 4014656], [17760, 'b63f2733f12056e202530dd12cef2e993ac145c76a53d6c92954642e05e66df4', 3777125], [17778, '2a81dd110021f9a7fc315741c1e04082f88f8eb7daa5671b878cabad2e182903', 4003141], [17839, '9adb80e7b8f55d1a5d4073ceed5493e79a108e77218cc3206e19079ab01131f0', 4339414], [17866, 'e716f79412da54297e8be97f4a4065b8aca691591d5b2933a325f1fad99e47ea', 3856176]]
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
