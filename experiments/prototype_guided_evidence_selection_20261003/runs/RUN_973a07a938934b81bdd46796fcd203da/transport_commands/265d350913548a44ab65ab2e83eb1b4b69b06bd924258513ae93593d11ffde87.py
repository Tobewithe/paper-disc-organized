ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='ff5b84f500a93cb61479d503bc5532e15fc92c754e3501eb6e7380c542d5f571'
ITEMS=[[77467, 'e6afe99672c4240a9717fd8492a8dcc9fbdda6725a312ac90ed139e68bd5aa86', 4224199], [77473, 'beebdbaa95912d4aca7716e89bc3f31cda14c300fbcf8fe84366b76612a8ce07', 4036758], [77548, '68675489f9bb1afeca739d35bbf194eeb350ea3ffdfb5076b0cba591ca7dd131', 3711554], [77563, '6453466ef1a630279b8797dd16d7180448bb54fa8c7191cc074b29dac98f71e5', 3569250], [77591, '73c6eb2e95ae369081738d672143e749cd2e9bb74e4ccfab7cd0e77bc48a37fe', 3893592], [77595, '31b77eb8f230310162f1e2f0ca9b329684b18964ef88f29c3c69482d71a0d627', 3724836], [77624, '4033550fcebb8d266a2f72c0088c2b2bc7621ee6e9893951d0d956b06bbf970a', 3790491], [77811, 'e8e0b18939c953599f559d9b21a6a14f0ccc61813cc08fc9d6bf3b490a95ad98', 3744738], [77847, 'd9cd17873b4b487bc938d16c5df726fdcb6a173b7dd057887d3b0702a7e77c5f', 3895188], [77901, '2bdba78cbe144c7fcc7e676df4ea098619698370a9c854cdec169001c2110a49', 3666427], [77924, '6ae6e12710dca2b4a16a9c04be9bfd5367086e2e193a999a3c6486c04d7bf572', 3809202], [78026, '2e8f605bc7c4ed51735a7b7b2d48383ff9caad512da1a5c132c5603b29b8085f', 3742523], [78056, '05a95392c0ec902860669612094de7a5efb0a3bbf0f7e4750300278355ba8d6e', 3701486], [78068, 'f05e58d20887e20b957405103c13c650e122d5126c153883253643930e43bdc0', 4039533], [78283, '4e04bba84b1cff9cff6d1970397b95f93c54a9f97c89c146348cc66ba06b98f9', 3845683], [78364, '6a8eadd3b1453581226e6f85c37b4514fbda80d1c2d756b6b66f23a269d2b07e', 3586990], [78400, '92ac85b3a942b0f3f38ce3c1837ca2322558875091a451fcaec64631cde040bb', 3810832], [78404, 'fa595d91415cc3381756d8fa2e30d4590b56a222e9f3c239fd87290032fadd03', 3921997], [78457, 'c9b3f0c8f56d7afa4a28b78fbe021b44980a2fa6bd015a2e980dfe504992f8d8', 3908493], [78478, '21b0b8f801b2bc5aa0578f9638099a3f312a418aaf993cb762f0095234cfa91e', 3953591], [78511, '9b1fca76f89ac385213c57fd1e3bd349840db584caf5607e52346fdb1dc7313c', 4762122], [78517, '263cab4107e6e5539845db7a49fd099d454a8c95f75a1ae9f9277862ecef70c1', 4099755], [78604, '400315535d055c0475d7bb3373ce1ddb67fe7caeb9f3f828d61ac4517e99bf97', 3780149], [78610, '760022f1a83bd704fea486fa2ef996f0000d0f23e1ed4c4a64a9c164f2cb747b', 3786505], [78729, '1a0787bcc51947c805d2a5f8b9ffea3573feacbbd3be4dd4ba6a6d15eb1c036c', 3383097], [78741, '5e799b0d5a7fc0bb8fa7fbee65d4a504b38c25109a8cc1c9b427c10cde6a9f52', 3845701], [78748, '056e4ddc6a135a47c7a0728c32b03b3b54f90ecd9107307d73adc34624dca164', 4526437], [78781, '1226e5a4c3cd8a1928fcff973ec12b27abef00380c0bde8ce7da7076f63ec148', 3653384], [78864, '0816991acef51a5a2362e99ca99d0876a829ed0f7e61e24f6eca302c27e4a085', 3581786], [78865, '0db5181f63027c7ad5e22656975f13345f028d4174ce0c54413d8d11ecec4c70', 3987158], [78915, '43d3be8996a5015619e4b2d9a842c6bc1c1cf15b1859e2645ed374a282ba2be1', 4219699], [78959, '221a8d757f440324be7033e3e661807331cdea4aade8ecaddedcedd5fa6f8942', 4019319]]
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
