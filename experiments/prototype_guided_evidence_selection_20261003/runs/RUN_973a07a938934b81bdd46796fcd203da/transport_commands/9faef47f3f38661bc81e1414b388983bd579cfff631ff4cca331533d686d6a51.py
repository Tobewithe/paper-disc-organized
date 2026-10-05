ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='36c3242b0ee357fb03bcd1b207b4ea62db6eade7c0125a622de3a7a1965fa289'
ITEMS=[[27805, '1dab830fd5783e51351768890106b2d6a06bd69bde24688ae5d6fff866c39fdb', 3803047], [27862, '9c2bda05e7d7f76c39ba53cf263eb6937afeab5bb0d77b2db2b95b56764be481', 3701685], [27871, 'bea9bb5abb667d5eeac0ea4be5ed535905b52ca51ca7bf3fcb93e5b31a5e4d74', 4011684], [27901, '7fcbb4e928477a6c591babe205621ddf3507ce74fd645b66bf527cfc49b9d1b9', 3878169], [27972, '1f77ea87e84328745ab8872104990b07ade6fd660aba1a5564f2784f839adef4', 3746648], [28012, 'b7b81dd446fcdbe1855041d938e9bbd725a788317d750e7f844e6dcad89911f4', 3892422], [28171, '3a4050bb66f92df78e88d0b139e140b2055e80412b353f4c270a8d05e8ec5354', 3594739], [28314, 'b65d04ed25159a20369a86dfb929a531894b52c68824c04d93a9f09bde8e1d3b', 3885343], [28336, 'acac92aba09ad141f5fd4a47bb14ad7906daeb58896d752d0638b6750bff1648', 3840345], [28389, 'ead95aea2f690a923e1dc0c27b4b4e6990954c1473726b2a06268a1f17cbdcb2', 3562333], [28449, '037fffe244016b750b6f6cedc20e8323c1620f3d76b1aad890bd5ddde333a6f6', 3867376], [28560, 'fd7ca538539d600868c3127195f2259fca80c304ef8443c853754f14475da6f4', 4118249], [28576, 'fe32ba398357c7edd4f730aedb1e70cc8f97d6fc06fc9a57e95425bc86f0f7a5', 4156971], [28649, 'ffa09dca9269852fbb0d4b8b7ba32cba3de2783243a281a2ae059a0848467353', 3891309], [28688, '0e9280c65449023cec465595fc5bae8c191473e46870e8675dd02ed1f63e6575', 3988466], [28698, 'e3a7b7599b7541c389ba32652afda6e13f0cb6087cd1a96a4af7a5e883cc475f', 3933697], [28714, 'c333e197fd7f3d6218841eacb4dd5fb6b8b51c4847c0d1237ecb966fb12d0352', 3842229], [28802, 'f4d9a6d1d548b505149074354f456d09f91d69f57421cd5128fc4695ab01b48e', 4191517], [28809, 'e5462cadb6d36d3a5c392ef1c5d9dddd1ad0f219e7e485162d0b2fb670465cc0', 4073856], [28824, '6373c016ef17c294be8956c587b178e833a27b339b06b5d1ce35f66ce4bb3208', 3832497], [28858, 'e8e4f582eafb2fa57ee2a427608c8b5e180bc2f061353c7f29d39bcb9ea8e0ac', 4453294], [28988, '13be8bc936098194dd3a8336aa83191fbb65ca96ca9e84fae7f676f0f8099431', 4213234], [28995, '9b73df6531333b552998fef664cf85cd57f5070f2e50627306ec546c188b5228', 4051583], [29005, '0907f9af4f67eaa656d4062b3d9f82414f1adebc5cd7848f82291ed35f617bd5', 3353718], [29019, 'd90490530012ac93ec8eb720545d4c12dbe698b20ad0d4ad22d35c656b1eb8b4', 3718250], [29023, '5eb85a2c7829107a36b74d8a0e491555592092a484a5522d8343bf689673ea20', 3644983], [29056, 'd8c16ade6c9ef290dfd5bb7270be81eb62fbfbb0fd18322bf5d256d4efaf4918', 667549], [29114, '860c5b10a5ba3617367daaae80d1fe12340765e7bef8232951bba7ffac72589b', 3745205], [29327, '1d9d8cebf0bb03b7797474df56dc9b49acf7b727523c5288c75b6c1da007a41c', 3868981], [29393, '95c780d114aa4552fe969c9ecc1d5ce253bf6b52c5da071d64cc5de990c097b3', 3929533], [29622, '5bf4dc3bcc386cbd5890d2fca58129fcf5252ad97055df4bb217455b90c5f99f', 3665786], [29643, '46127a8b3811296f14d24be7c63d658894628828c3fd9648806b40e3f2cfcf64', 3843260]]
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
