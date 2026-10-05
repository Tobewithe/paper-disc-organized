ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='970247d5a0ded4e88924b3771f76a3f4f464c241e9721293b51050ab7679ea53'
ITEMS=[[32603, 'a85b995117ec99bf986a4b2c8caae88727e502cff415fac773ca03792a2d7e16', 3962730], [32609, '825c927375905be2867e3eb8e3ce0e31af1d2da28fd928fa93315ecf9890cd66', 3704274], [32627, '73c52bbea19b6394251627d3401c3f8cce6e877026ba11ff9a314947cf8d064b', 4215337], [32655, '4b43f90772b3abf7988dea92937c49b197e0ee3e2895f64f3979854b5cd897e1', 3682098], [32675, 'f33a319805e5dcdef53d27912a898184f4cf772cb4ec9ef9207f23dd803ddd9a', 3867810], [32695, 'fca8e73488c568ead839dee00dd3cb169f676d93ec2934281ccdbc505ac6ac10', 3841348], [32704, '5414d88f6387ebe913b626079fd08f42a577b6859a7c35e471d652f7c352e96c', 3675493], [32727, 'a78eca6db9fb9685336650056ae9a837a2d8f9d72d08557120c608159cdef8b9', 3846767], [32766, '13c20a2eba130c57f7dbd19382307bda72ee5ab0ee7c23969078e03ba5f9c14e', 3616041], [32811, '93fce4a83442e1a09d4943023480679a55755f0b648574f45dfb887f3ec8c881', 3577868], [32906, '284a76a19803a52b87ce318f6b005b23e01001bd4bd302e2ae6b775da5ced23d', 4207119], [32918, '68a7f153253205c166133074808b2b95e67d797dfaadd03da1b0230ad1bca38e', 4019506], [32941, 'eb8e54684714701fe99c57bcc6fa8a7487de5882c2db924279715133820cfb3b', 4181157], [33057, '61fb3ab0c0097a35294712c3e2bdaf80ff4649d2b786e12eecf6ee9be4a7b26d', 4352958], [33172, 'f67c2dfc255aaf77b02d4113e728206849309b0e34892b02990bf486fba8357f', 4542831], [33221, '05946bf6f1b6c83beea0a544c6890f642259f3ea15f498194efe6f2c9ba37631', 4163537], [33368, 'fcd8482525e03b407d01db22a6b5122655dd77f802fd91002cb97ed7edfebf0e', 3590646], [33383, '587dde143c4ed4510b7a8ad38f4da40249bedaab8155e369468605f2661099d5', 3857642], [33413, '15af5ae167d334eb6e6be36bf9b6631ca2623a7edce895c68351276afefe5033', 3827928], [33422, '73502512047522fdbf5521b1ace12190d5c8bafceed263cd8c949cc9b0a2d104', 748660], [33435, 'ad73fff76eb35d5ffa1b01e8fcee85fe5ab93e08c6d5e641da933fb20cc50cd3', 3761554], [33476, 'df4255aec67eae360b252b1f87d88e1122666341bfbb41aed57f0f875ce9afda', 4001867], [33633, '1d68a874ffbaa56f27eb3c6c65a211dcc222e64967a9a98a0bda9e6ce03ef3f5', 4200624], [33638, '1cce196f88c412d384c4f4b5c57be6577a3ea3e1f2d2b8444dbd001cb90ae149', 3886232], [33642, 'e0993490a377b7b11d24253442e5f297cc299bdb9897aa10023012e89c6efc30', 4083433], [33643, 'e9572dde37613336c7a0af30df047883565b1a61425eeca38829f18925a59ed0', 3755298], [33659, 'e67c507ec2578bcfb2ba61a0b9a993c19f0a28a62c0831f943908cc580c66b81', 3506464], [33707, '1309ebfd12f56d618866fe43d9821f6996e9fa69274d1581995b8d67b7eb5319', 4062032], [33727, '8c77acd5c458cde7a532797deda26c7efacd0ba89381db34879476f3e2f81e7b', 3762026], [33828, '7b7d289a93dca6da6223c72ccbdd1582618881ef9b612ba5994bbee131d13aa1', 4079205], [33891, '29f85fe0e0a5a8d7ac3678d0fa641115916aaab1aa1c0d2c1bcda08e2639bbd1', 3671018], [33930, '79ff50fb9a9aad1ad2a79979f5e38916f2e34ddb5da1a1f99c4fac59c7ad7363', 4220077]]
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
