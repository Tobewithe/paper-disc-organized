ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='98a89be79b26c30bbe7ad0a91baa9df272647c6cc8ea30d913691d208728e320'
ITEMS=[[83566, '9d569e54e4cca8d999fdc5d2ef5154da8c01599c15ac262dfccd651fc69bbf2d', 3511775], [83605, 'edeb3f87b960ce39b06305d482597cb53ad9ab05e4e3075e5513de4c0abf4ab8', 3699669], [83625, '02f3733af7a5f60ba526712f0bb8b0925b95a0ba8ac3d51921512543d526eaa1', 3937823], [83641, '5ad85157dccc2b434a513173b21a9d16e7321166121b22f2d677c9f9650fd268', 4007258], [83656, 'c6802aa297883a103fecb12baa7b9964ae01d58b5c0109d45d3edd875191e04a', 3583036], [83755, 'd2632c85157ab651d897cbb73aa7c2d61f5ac3fcef09ff861bb501795d9cd682', 3795349], [83768, 'c86635c5702689eb3226aae41ac70720d29e8f8f41fe3cf4f98239988476d9c8', 4085649], [83862, 'd59eab7c593ac1fc2f6fe072548b6a4529e29dcde13003a4595adccb626b305a', 4133026], [83925, '52c747c16b2320610cb5a2f601c9fa84371fe1d8476653ab3c51e9fb3a73af8d', 4001116], [83931, 'a3a8820afc6d9962b3e08761bec5ec1679f8688829eb44609e035a50dc8d126c', 3956454], [84018, '79148d08fb7c1b049cf20edf6148cf4f88d275c49c68937340f51e3a601a28d8', 573217], [84040, '50570603cfa3cc7590f877773d28112981594ac71fdd0d09cf5f76ef626c30eb', 3849673], [84114, 'b26273bcf09a10f032cd1d8517c05015493437b9e0db6739e258e5b2fd184bfe', 3982219], [84120, '190562f68fafc94ba8858613e54d726febaab97dabd2b3fb3d1681d4d089ef0a', 4001456], [84174, '521ac1f5f0bf8f6cb0fe1ee9bf04e48dab4c81af8bda452d77dda5f88dec5850', 3940495], [84200, '894a490d9fa0b9889226ae6b592dcff85d454ce9c2edd88c8a0e9ae61772172a', 3883435], [84270, 'b9d94d4d68ddca02efab2d8e04a7bd80b3b3f3d507c866c4502a4126d4a818be', 4678984], [84277, '5cc8b6a15e9bd0e9a12ca0e35f9221f2e53c34c6571ee80732daa5c9f174cdbf', 4483651], [84284, '0feb37a2d4f35fa1408062f3354a164ef363e03d1056e3711b18d5d2fd5f56d8', 4139465], [84356, 'a0ebc8951f5a2dba90bcc3aabf1d3528a2cc10ec762ec0f8149be5cb7b13e1e9', 3761145], [84362, 'ae7afb8ca01c4d5327f46edc5b3a1757c579631a234b7e279a87eae712a8da86', 3963033], [84391, '29a09ba7c0dcaaa2a835a976fd6eb805a2bd3c5a56acf72fadf8ff20a273d0f6', 4407704], [84440, '3987b043fb824470d27b224bc9dd134de26a420e021cc3baeb14512f911f831d', 4106139], [84463, 'c8784ac03e077b2d9476c6e3fba53eb2d764f49644202d16aed68fae06eb3511', 3762517], [84469, '4ac606740cd3448f5379d91c6ab0a6f5eee3dbe01e940b58c9ee73589d5021bb', 3654031], [84477, '6a190bdb8c13f29d1da2f74994b57a79db0844653f60bbfa6607bd06b64fed59', 3681312], [84493, 'c4d4b0903828e9a857d3eb59fd6f2d12627102eddc437d7c1cd1ebccc350ec29', 3948208], [84498, 'cdad0f807848c050177b412675a4c21f77b05f10ccadf5b6901a23aac8c38f3f', 3842004], [84550, 'ad604e92c8ffc936b36a431e03a6067b21ce5582da10c0b950c79943ad7de16d', 3759700], [84558, '82034b7b4e7ede1e6e4f7c41fb1a1ede7fcb79ef80473c194ce4b69a1c30dfb7', 3923545], [84592, '753a3fab9e67025b08dd0442723c21dabc54d2bb712dbdfb7cffe239357f06cf', 4035187], [84643, '2502b2722bd067c998e3a0da8514823ab1a96a5be4261e20558959363eab5480', 3761175]]
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
