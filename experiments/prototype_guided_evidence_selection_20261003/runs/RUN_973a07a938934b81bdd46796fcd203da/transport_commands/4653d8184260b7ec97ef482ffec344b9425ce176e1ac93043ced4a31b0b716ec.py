ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='9f017715a3d1163fcca262837a7ec27b7291d761fe0f7bc367a1258657eb3d85'
ITEMS=[[35216, '52ba620a63fead3fb0f26673a218b938741bd8bee8f30d4f9adf65ead7568c87', 3567886], [35299, 'c0ea4b44d248b542b9175c945cfa9676e504bd0863cb259d260571d680056568', 4001690], [35367, 'eaba46eb5342a08519e336883b30643b4be0c5c42d1d2777b38e071f0a25efac', 4493798], [35456, '8dafa9b8137cd80e49e896295e34dfee3390c49712e61d164bfa559df5fc78ea', 4223234], [35529, 'd4a8938d1535f1c3ce2fbe88391716759491919ddf55f178551e17ff1d861160', 4122296], [35628, '02f035ea07aa137fd2a5c69d805f29f4cf0c2e1c6e5d90d9f088fb07cf67fb53', 3799439], [35682, '8319035237da7b0577bd8af4b59719c932cdb3593ba1a7d6f0cecbefbb76661c', 4357591], [35770, '81ef5273f6c77a96221c4ca37ac5dcc328824915a39bc064a99d608b4fba5486', 3866701], [35844, 'e16923f8dd05a7bfbffda499332b95b2fdba524d56400fc1519994a9821d28cd', 4245873], [35964, 'f6f6df51a92e36be3bf4c17908ef27d9c5fe96da273b8ebecda634c281951696', 4280073], [35975, '38f7fac47ff108141c39f3856362f6c52690d63c3f55ae473a16d4befa2e49f8', 3980888], [36017, 'e5a7276353311eb4dd52ba0ed473d17f0774202cb6af753d5bc5b761734c4c0b', 4832539], [36098, '0fb726256314169e13d8701eb110f2a38e3cdad063b2d1eeaa3f7fb55b862f1d', 3834609], [36123, '3ddca1534381412bebd0afef0cd0c040c37a0d0cafda89e64dc17b8a6ccdafcc', 4549811], [36289, '3bcba6063b25316fad937def63c65f8cf43e594069cf19ce5aa3edc4448e020a', 3735221], [36333, 'e173f8e0678e03100d7d3426a11c142bad62a03cd36cff45704898b28d178468', 3755183], [36351, '83b1b88b4e9cb6b27d5f3299657be972f21a7e49fb130faf1a2f848651d285e1', 4109948], [36397, '31fd6cd7e0fbce9781dd817e1563f466daae403ce3ea6dbbb22f8a69e110c2ab', 3540035], [36414, '97173bdc7e70990da2a12ab99f5ee1b30503e3c286e0e13b0f3b186c6a2d3c64', 3744290], [36421, '304445bcbdb8e9ca03e211a622a006276b685e8bf8dcf0c408f74de199a7bbe3', 3851408], [36482, '384d5109740bdcc70f7273dc82b1f5e9db737399b6caa470c4b6a328c425b91e', 4197706], [36533, '9b40d6a673b264e17138b13a4e9365b1a6177dd8f24c2c425eece2673ac906be', 3663545], [36539, '00bf608606510d9b471b1287adfaeddc3e6af3909385c08f4754023167609bf8', 3845919], [36605, '84cd060a6c2e5af930f12eb14b97145860f1fccda99d5828373c1cadc430ad60', 3766932], [36678, 'c9059b24c906b740f7feb276435fa6ee6b75565e14e885aaa849bebcb0542910', 3739072], [36713, '0154c8b04fda0608dd9f9a586ffa5e377b13b4216713359bffa71c5dd4d14f7f', 3881408], [36728, '800db88a65f5a87e79b79cfc1dd1001db3dbb33e9b84c53f9dc2dce4f1ebb969', 4472825], [36755, '36358da29d074517ecc1ed50802337bcb77cb21a0c9d6dfe6ce554f63f677d56', 3857423], [36768, '538c5e75858ac9e65d95d797fc5b3487611bf7931b355dea8ce2995fb6dbc584', 4072465], [36861, '571f1b7fa7269aa3117b6c421beb21536053fbcc794bdf73663e13bf610b8359', 3825256], [36914, '4f645f07856891bebae13a5c4e617cb891e983fcf51ea9f7f22cd7c27ed38129', 3899216], [36915, '0522f847bfe25d3f52b8f2896bfe694068bc1631c951a89aee7e0b8f2a5d2954', 3560452]]
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
