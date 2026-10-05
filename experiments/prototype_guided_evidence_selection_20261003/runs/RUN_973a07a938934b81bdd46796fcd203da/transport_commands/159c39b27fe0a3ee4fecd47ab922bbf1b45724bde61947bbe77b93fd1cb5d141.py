ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='3623b36023d3c311ec6fdc91f649d3b21d5ad0e74c3a5a1bddc0d05d90c410f5'
ITEMS=[[73280, '682291be409e657e091a9df4bd7ef7a1af2a2e6438fa4fc97941d4884486a62a', 3912638], [73333, '388acf079516841bc52ccaa2d38721c12decc4369ba81c68e20df4ceac482864', 3594725], [73353, '5e84f57f309654497eac18540c2e04b214dba41dc002732f4c5c8b20c0c1f93a', 3832549], [73420, '82f33cf2349fe925387d0b25187a42e422ca8249465982804e0371c846c6fdcb', 3674877], [73429, 'da897946165b8715bc47ec5fe2ab725526f70cd794f24fa9be87041673aa3cea', 3853553], [73533, '56c5ca56f50bc394906eac8b9b71c676f0b0f8fabccdb0ca630ac225a0a1a072', 3878550], [73584, 'd4f8170fc37a886dfa3914c7af29f56bd9989a034c697569070be6c376476023', 3747135], [73680, 'dacb78654eb59849d8d2ecd007b0cc3b3c2b6e22afb47ae68bc6e16ce0125574', 3851611], [73702, '4877f172c10d406db02ac9f059d35a21a9358b0983b65d97d4fce53b85054771', 3982469], [73707, '7c088d061c82a0154c6626b74166539be7d5bf83e7cf6179118083c1ff5bd32d', 3674928], [73726, '4c3921e44886d8274d43dcd29c186680872cb5793485e86abf7e7612c343d295', 3882942], [73808, '93b0139b23f5f597433532e422bbce9af934e3c436df7daac9bdd8ee82063697', 3872595], [73826, 'b2454aee06886c761a7955950cbfa6b9a105757953701e47c2abc6bfee1dcedf', 4290184], [73843, '88cb63212e78dc194284a2918d92a97c2f181d94cb3ba3a9ba40e282bb7d35de', 3898361], [73857, '378b6feaebc0bfcabcfc1c41a15ad932c6cff1a9836f79faa51da7512af42c25', 3651641], [73916, '869b8d045f41c463d1afb9d460092961511327814472cbdf888e6692a4abc340', 3894941], [73996, '99bc9ce33706f3eb92e88899823f9fbdc6ed0d7e06efb94905d5978cf346efe2', 3703520], [74124, '920e60adbb4b5e22ca674aae8e4956fb445b54a4c9ef95cf376bae433195aafc', 3876745], [74138, '00861fa9e73fccab4722bae4fabb6db84d2c2bebe0f91c6e80ee0f01d392e3db', 3781990], [74181, '0b8f400310c3d452659bfe124cd9e16d11e3ed6d1bdefe23df46787b8486b8c2', 4158966], [74253, '2018da6bd1d79763a7014b3fd8753b82e3c235a2851f9e0f227bf8f82bd0b5fc', 3726664], [74256, '5c81bf2a1cd02429d2cf750b286abf907bc37dd6c9aeb2b99d689403ca9c9b6a', 4256668], [74268, '9a083c25cdb30622d7b8d95cdf8a31235313a733d2d14825870c930612b67a9e', 3843174], [74357, 'ae81b33a3054cc83e9829eec2f49739908a1118c35eb41dab4e54e1d6bb077ed', 3972445], [74411, 'a2ee90de816a24078c227ee521b1d18a3e75dfce16d2f9d3534fd855236ea5f4', 3834847], [74434, 'c8082d630290d2ca78a5afad844f32ad9c29bfb5ad62bcb4d9511196eb8243d1', 3700215], [74461, 'fd0f72480072e802c896167691ababed6cbe3336be305fdf734986653d0f9177', 3948867], [74538, 'd845687c36397f1d97fd42d5f526aa0eba1e48296ced9ffb983be5d89b8b907e', 3781351], [74590, '6580922a9ac9cc039514c9cecf46ad750d6c22f2637879ae727691186d66ca22', 3785619], [74629, '50f0debf69fe057f391b25b3b2fa30f5b0cf2a6c898c4db89f46830cc97317d7', 4229305], [74646, '0bb5c060004596592ad8634d32410dff97965fb935797ba8bd2e0996d8b001bb', 3980509], [74651, '1dfd8de8a06f6cefadddb386e16e30b02d4851790d083e5499475d7e313f5b3f', 3718848]]
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
