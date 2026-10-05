ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='8181a55c76e8b0b5e1a46844118104e3f67186434532a96afdf48523932dc63d'
ITEMS=[[75841, 'ad72747c941ecf34818f46a917ad2214759da53ccc41c1a1b71ce74975d65005', 4028594], [75882, 'f130549577fe1f437f028db62b4ac2a71974c3a473885eb5045291d8c7a1e6fa', 3719216], [75884, '9bfe7635b7b074ad7ff09fbea6b68e7d8b50abf561108e86ed1748ca0b541e46', 3787996], [76028, 'df5c53c3beadcfece2376f616c52813179cf048ac40e0577ecde8251eaf68e30', 3879859], [76103, '1390aa22e7f28895b6cc35574ae1519f49a754a3b7470ce6e06958929143a526', 3918273], [76211, 'e500d1056ffff2b397059daaf673f121846989301fe6c485ef8969bc4a830ad9', 3731523], [76245, '3c30a101a01c70cf46e3ee30c6f827ce82f4935549dff8a7ceea83fd0595f915', 3964872], [76430, 'e169305943ca0e247c972cf33448668e94fc6c9edd5e2ac0ec0960a32cd89703', 3667487], [76462, '08b855bd595b109e59e06f240bdcfcb270f250810fc32e32ec90aac15609e679', 3631608], [76476, '42ff341bb03ae12cbb8b836c0bd7d83fc4424fbc36c204d9d4a956b84d208d65', 3719103], [76484, '886f5a4b59011332e646191f2ba2ce5ac089c46606cd17994885b0cff69dff1d', 3557728], [76547, '808dcc6703985cf2d7d199e6691b30cc0975a0b6dd5d4dc802924c4b708b4892', 4255776], [76571, 'b46620ab896c2142959da045860f819e6631e44a6f91d7a5771b0012ad4f9352', 3984835], [76588, 'e56a434f12406702394099112e7ef448c728669913763d93e4b7f413b7cbf345', 3874170], [76590, '05e2a90c007e0c0de0b1e5f8b5522122ae38a7c26b3d2245c6f774c94c539596', 4360883], [76607, 'f0d9e9e0f0a0689545d1742b12c43b01583693f5af54f2af135c81876a8d2903', 3897677], [76625, 'a2c4b3e2e68b2b1e428e40786a8e80b629f251dd2ee33009f38b10c5499795a5', 4231800], [76632, '2fd86bd6832007b650aa7e02f7db637a8a947e23d53b70860c7d5c040f225b70', 3937780], [76740, '0dd54c07e0b808c67e36f62b9bb316b5f9e102876950d17b95a435ff1652e074', 3920013], [76776, 'e9141b232cd66e1f2c6ae0483f611d04f3b9c4d69dbf796366dc84cf9554ec41', 3717261], [76792, '5bdc70535ad4d6b412a6bb2db9e968ce48df13af434b6ba49d267a5d1b1866a1', 3461009], [76841, 'f8a0568f2cf8dd950f560e709b0953ae249482e05ee4a3438f48f99ee2b58d7e', 3848698], [76846, '42594cb456ca6e083586ea4aa72fa465ca5ebe15d639746e2e59924d12c5b014', 3988491], [76893, 'a540343e4c41b4ac686d690d9b2564c7766e833db55e18e97b6d10a65601a964', 3700205], [76985, '7fc8931f38e51d6ee82757472f5e9dbdbfed2170c67035ad7607338abf4f5382', 3880628], [77001, 'ec732f75f7a482fb75421f5c79762eb734bd5c2efdb406b725788b48703714cc', 4002615], [77067, '7332765abf171b59c75cecd4150527b780e36a723ebe2266da8f44f87c5ce973', 4093495], [77102, '5402ac0991e389d2682160e9096c7b301f44ab9aa65ea8a389e8be714e7c497e', 4207105], [77137, 'efb533cedeaf39cf5f4fd8f2da3bd876ecc0fe62458ae0d71df8e4b50f21d4c4', 3837575], [77181, 'b7afdf369c76c50d764615621c26cda83db7a05eb051c358185cfeba48d5a1bb', 3997531], [77298, 'a9914f1cde2aceb8b319144b6fd73c32ce4087dec32a2e0472309fb75be6abda', 4108668], [77346, 'ce8c171f3aeceef09932fec36e951df4a86936569b64baac4be1802af46f771c', 4150246]]
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
