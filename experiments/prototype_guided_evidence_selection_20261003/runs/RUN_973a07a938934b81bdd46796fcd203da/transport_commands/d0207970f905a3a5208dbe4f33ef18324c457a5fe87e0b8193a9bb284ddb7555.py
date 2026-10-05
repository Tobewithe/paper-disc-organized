ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='ef2b10afdfa977251d8b07125458950929cea2c7cd7611399dee430c67173587'
ITEMS=[[58157, '4b76294a000807bb85e12ff1951f573dea56a04bde4763b5021dd80d8e959051', 4002501], [58161, '2832fe8a8eaef12a0f9272d4f3f8c65834d8a04653bd3ed4e4b42c17ce72a749', 3712434], [58225, 'e0bfc8d3da8b5a234f9b3cb6f2ba7d9ff22f9dbe3871bb7e25bfd828739ed19e', 3835616], [58254, 'c1e396dbc1b4e93f92c5ee739b8fb7fc2c47535810aea8de805f71de8baf6644', 3807213], [58268, '42e0897ffcdfdd96fd9f1f420bf6b9567c8c0f83fca8ca5aa4a12b1c384e21ba', 4519970], [58296, '037832dc9f12f7df706199b9cb0dccebf8f8ef8806a15e22bfedc7e76870c762', 3708606], [58393, 'a6eabaedee9711c500e42bb4ef373bc79fde43c03f31bbc6a77030bf65090377', 3597339], [58406, 'ab4834a80e9642754f0984cb8b82841a7091e14c3af1aae0c77829a6f0a2e0b9', 3892691], [58413, '090250bd623f7b3887c8a2a56f4f7eaa839437af4aa37dd7aa29698971bfea46', 3560001], [58522, '748291f0d85952a63bd81aab14afed3432d016f5a8f5aa0b04c9dcfcfdcd9f4b', 3722594], [58541, '44eec6e820fed30b3061f7a0202555dbd8f45585863205d6c07db257af76b48e', 4155003], [58614, 'e930861ecea74aac769088189620c80ffca5bf92d144c7bf6719f0bf4d572c4b', 4465130], [58705, '6ff69b463d89eda500a524ba8bde3fd52c90182c8930695b1575c9d231ea0be1', 3922552], [58738, '2028f5a0c82042fd67493f260854d1df89f10a6f163170c9523093edce8eccd4', 3814058], [58754, '1ed7fef8c35d2e8c2627b3300336e9cc446a85e5938f2ddbb521868e777238b6', 3734954], [58816, '9c0296c61401a77a10d1bebcb91ab0d365def6f3684e92d7ad773dd59f88eb4a', 3701759], [58911, '4247c3a1ce3a8892ef7fe364e7123ea6cfd4938b2a6badd00caa1b67e44fd494', 4483632], [58915, '07e2fd2c9cf0d6795a2253b8f69d7a0e311742d759018d8246d51954629c263a', 3658750], [58928, 'd8434c68d751611a30d8fa9d22363c3afa20be824776ad5e583cb18437a6002f', 4368173], [59012, 'ccb76ad689feac3500cae285e31a07959a20bdd654d06805a667d692c3711ed3', 3808686], [59084, '336eb17d1b1fe4d426ff32a9de1eaee796a8d04b0b81f42f2a0aa9d639872709', 3793587], [59114, '26dfa72b6bd516470f57eb9ecc2afcb97f2f83d3f6e530fd7433f9a7d7db5a53', 3830461], [59141, '542acc7d3466e5c8f89babb22a158f369f010e3106e9881f23cfe28612e48a0a', 4440221], [59151, '95f9d5736505cce3904c8cbf5877a20d920d755c8554e435178f0be7b57b2674', 3739379], [59231, 'fc894e60df0eb34f685c5c37cf29e7b584c9762458d241e27f64c25f6dce9357', 3837072], [59295, '6eba5a0dc39416261e196177e6f58821b89a0f59aa8978ec76260642ad92431f', 3837482], [59321, '3bed586d10f5f5df2350fa1eef4e36d47fee863b5ddd7ed59fdd54308c98eb74', 3648830], [59352, '0f10eeb9203069c2e61097ea10ef2dc31e0ff73e68ba76095f32bef63146d560', 3536510], [59361, '5d3bba28c178ac05d9babec98389d63e3289816167b15fe092c8e4aa1d83de21', 4139144], [59382, 'd0ba28fd616f4bf5e0e330db9bc99515cd9c9a1507d35532c0ec45752092df74', 3561590], [59386, 'a7a72af0fe93dea1932e130dcccdb82692d44e5700b9ce7037e33b02fb7efaff', 4061556], [59540, '8d4aad0174b119163429ae3a8a6de2cde2da0b0fc78f98091abe32285392739e', 3721738]]
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
