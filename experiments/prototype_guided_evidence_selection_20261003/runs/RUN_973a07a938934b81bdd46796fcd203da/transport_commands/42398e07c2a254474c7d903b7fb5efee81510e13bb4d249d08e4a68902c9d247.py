ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='6e6489d898a0db56d113a1a56f38b4387f8ca36bc2de796ab83624f188bba71c'
ITEMS=[[61564, 'cc7843c40b68006cc5d6f73941b53d6690bfb1bc70f226dacfd002aeab878cca', 3968075], [61621, 'ec827b4a0d2448a3c4df59ff06d85671132a79ca1522807908eaf98498dbaca2', 3888766], [61658, 'f9f906b1cecae945501f44f627543288e8b89105a1dfd6469b6104b445e03e9e', 4220497], [61698, 'e15aeb9525d79dcd3a4f4528bd3685c736903b44405208d1eb07f42e01dbf211', 3758255], [61755, '4f8ed1e4019bba9de2f0737c4fc04d287cc2616a9a5ca03363ccd60aaabf9f8a', 3812976], [61782, 'baf7ccd780af362c5397a19ad29febf28ce42e42345656ab334f8262506593dc', 3730684], [61814, 'fdd64037a3746c150b2ba02102f171068c66672cea42acdc4c32c6f974e650da', 3711227], [61830, '449de224dd42a6e22f76b3132f0d3873bed8a0ed271542b6d75388b139a19a56', 4178016], [61842, '6f68b2828c2857072cca9e670eadcd02d8c98e9c9aeb8af9ba99998260d23bfd', 3738686], [61849, 'b2c59013caefccdfbf58334f11df01a97d818899855f87f050dd80f9239e6fd2', 3653567], [61865, '4aebd2ac6612963db84c2130d17b8df109d0301145ec9177f74862ae30858eee', 3690754], [61892, 'ad6418674730d5f9aa8e882a58c52de8fca43a8bef17eb2b35a777646e138ded', 4192440], [61925, '9ec19417a346b1a8e515fa3c8adef294cf178aa965e271094995351dde51e0ca', 4012971], [61951, 'eb3aeb6ea3e59818a87c63d8c764085feaa24dced7d757e4241417070157e4c6', 4461454], [61959, '30172deafdb163ed49398b9f8c9d3b122cbf1ff5f168d54cff11052049d0402c', 3975903], [62128, '9a54d0d262e8824140e23ca4fcb33965abfa654d357635ca8e04dcd262b5ff5d', 4137639], [62170, '8ad27723c7d5547e49c5567e990d582cb61d1d8713bf7e22829dd627280931ef', 3750239], [62204, 'f582b8498fc39423f51d7fa7b00807595966194e9538d7792c0031b95e261fe4', 4307105], [62220, '1213f84c1b7b2878a5517c4b72d3d406e7abaa981d663a957be650959a20d7aa', 3829872], [62245, 'd6567f44486a70273b3fcd9fbfd58af7717537f52aab70e4413d8b9dff80535d', 3897830], [62350, '868684eadedbba5f49f8c077ef6ae1e1f100370d6b2b4e5c32534f87d46ae95d', 3856202], [62390, '30b6d45eceae41622298ba42751dac99aa84ec8de6cda1321ebee99fcfe2cbc1', 4384348], [62392, 'c91aa34f82137b40c45e09a8b8df302e772b33012d5fae7439733a06a6c85ba0', 3877740], [62398, '40cd4d81dff0c7829a44b32d0f932d7ae1a9f06b65003163517e761700503465', 3819579], [62459, '7347461ec4a77294e373dd5236e24c78816af7824e05a6b0152d90deb1d42d06', 3818669], [62604, 'ad35dc83cd0c874e46f921d1597dbe0376d690fcc8fa0cd74fc76cea7f7eb67e', 4029164], [62692, '7b6e25c3cf7a2546acfa767e3747badb63845f8353466f8fbd19dca96112f663', 3738049], [62778, 'e78120441f58fb410ed1a5f160155e5938771c2c2e1af4f062962885234bebb8', 4159663], [62815, 'd73f9bdd31e8a5fa13b076a1319745bc56ad05464d2a8d703055d35300fac09a', 4444685], [62821, '96575ffc9663e9dc3434ce4675b61ffd99f4de0a760fc55d55815cb02099733e', 3658180], [62844, 'da6f5cf3027c5ef20cf9606a940967668f7d91361e93b41116b1ba6d00f61071', 3962908], [62864, '3ae9b56a9c42d9b42fb2c5e61e9ceba19c8824b9ac65937f64d0c0b69dc9f51c', 3608736]]
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
