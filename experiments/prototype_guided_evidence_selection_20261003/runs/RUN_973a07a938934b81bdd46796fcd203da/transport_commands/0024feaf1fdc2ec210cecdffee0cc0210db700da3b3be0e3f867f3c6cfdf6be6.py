ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='89b9a625d7acf1b794569e4d2327f7a8b3f01293a796a04c02cc77ce563e41b2'
ITEMS=[[62889, '0452a12b3a3ab191a59b1648bd3aa12c0d67ee3a8f97f0c2200b5f029d9774f6', 3842699], [62941, '01720ac7d3cbbe454bd4ac7eb6cbb17bae46fa0826491f33054677ae491a619b', 3410147], [63047, '2dcf53f15c7b7b3fa84a29e295396a17a314937bf9046305fbf38020db864bd5', 4182238], [63050, '1ab94fd049c8648ca65efbad60a81faaff57b5d5430d1dcceb890c308cd65185', 3939979], [63081, 'f39df0983bea6401e3bb3593a2c3339116bb0fdb3962a98b3b7b25e58c8417f3', 4319775], [63154, '8a4cc044389f5da183bb07981efea1ccc2123dd1173655378e7aa4928f43f432', 3683177], [63185, 'fceb26f137afc11f6b03f2efd7de12a3a7c88e768f80f4852dc818700cf4b458', 3634556], [63214, 'c805e0c2cdd8864f849e62df5ff0cffebf2502c65243df4e95c0c8edc9e07444', 3777240], [63238, 'a1b6a652a6b4566881212030c35e77527f691be26880fba35eb681ef57fd1a2c', 4048293], [63263, '9bf9591f9c4523bc1368004a8131ee6b23e19a5298977aa6ee5817b609220a2b', 3940133], [63266, 'a083cfc590efeff359ffaabe2d78019c83c15eaa25fa850e86ace7e1f9cb9e2c', 3863677], [63290, 'f02fff66bb8e4245f2dbbf8c4b8cd5a76d3920c45600f13b73b0608148aab08b', 3738734], [63299, 'c81724cdf98db51ee0a4e8342e68129c273b677e16ec09dfeb70b34ac9af7a64', 3579381], [63309, '2406843d04b7d46a2e3ab88be5dad3a81c62b4ba8f0a2e72d5e1e81a7ceef923', 3763074], [63370, '2ac4d12b63c0371dda3fea21eda4790d2ab71c3f4db20552ecc5353cd981c830', 3765059], [63476, '942b1a7acbc88679d1a9b40e0cbf2149ab35ce4d49aeed37181d1a03017a66b8', 3908850], [63552, '6bb107c4950e896718fb8a2b882942dbd56d0d03ebbf8c3a0b89deddba28f9de', 3931222], [63740, '3dd8888ef1c573b43c666cbe0ca1c608cd3395383ac66e43be1a0afeb25911cd', 4089943], [63857, '26c28dd38be7173d7d253431b18cc68a7bb8861a9373fef2a5754ca3bd7116bb', 3750455], [63881, 'f1ec9152738aed3df314165fec0589480fd4522fb82c510f443f9a2df002635e', 3767010], [63953, 'a96681adffeef6615500d3c8ef91d665de1d0d1cea99aa982984aad16e0c3d1d', 4241397], [63958, 'f74c94162128ce32f6ec5d17aa5639e1dc3f03b1228a049feefaba3011fe148b', 3937223], [63974, '4b984b1d4669768d987d3ee8629eea0c0d521e736d9b0cbc9569ccb4ee7ed62b', 4370211], [64008, '863d188bc1b14f365281fefd7e47c988d6d256af596de811378224a91ad26306', 4532383], [64089, '4ad10988a5a408360d9ef3499aa8229325822e2d523b65d0b8cc6c98882ff15e', 3981382], [64189, '9bcd82fc24923d69fc0b73ade4daf2ae9d1725f59da5be17cd3e120c046d8aed', 3798093], [64300, '798bc36867aa724d9b9c0fc3e51dfda037e38128af4bbedfa3c51ede4875b435', 3612946], [64348, 'c700bbfc2199cb2bff4a1cdd5f24f39a378ef5bb016af702d02bcc58115ec379', 3916508], [64350, '014db6a2d93e44f277c29d5d86752ccb87ced53528aa6a858e3c7a82d6e52962', 4040601], [64417, '13e5aaf37131740186a83d623747565708eb3a553583786b33499232630aaaa8', 3918960], [64423, '0080b4754ba8c7de8b28f2fd460d4afd70ec7f6a7c95401204254ef7763d46b5', 3900629], [64440, '93f5b42205b6e91f113a374b19b5f1ab6c6d67f75ab2260af68152fcc0857b36', 4136321]]
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
