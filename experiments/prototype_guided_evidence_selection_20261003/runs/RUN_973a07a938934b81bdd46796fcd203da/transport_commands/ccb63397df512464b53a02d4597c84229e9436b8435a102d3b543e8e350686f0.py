ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='c9620efbc9519c516e82ffd46bf2ce10066597821cebbf94864f20e3da559c42'
ITEMS=[[65859, '5408745b42b5d90508d80d444595a73d9f58788babf77c79a0007ad207c54277', 4157607], [65898, '660ba0df21e70f46a7aee6ab886f6aa40f41058a5b010de129de5835aefbfbf7', 4066579], [65900, '9368dc40b62f48a5fcafb8afc2d4c44e5d92c7c35f6b4000886c9272c2887959', 4253814], [65948, '1ec241b6d18066a30a4d63161c97b1fdcd8a3c5cc4612ed8620f6c3e68cb4696', 5047452], [65965, '98ade3cabd2e405711cad4fc8482dfde614c367a0c2c5d0e65b6e45c5969ad34', 3820986], [65968, '7e7ddbdd4e38623a7d3409e2510798c25faa1db286be2602c749fa133aeb409d', 3854556], [66038, '6be22846c3081252782ebf2d31a9fa11f85de21d17e2de2e93304a093b2b5362', 4305064], [66087, 'abb1538f977d9e573e5bcc941d773ddd19a3d0d5ef67558296342f6d63915813', 3966540], [66118, 'c6501a772bb6516ad306edfdc9f4261a48192e80b6c160762614f67eab3ab162', 3775464], [66126, '886e90fa00a78807d244d1ba414d698a70161ca5f2774c2b3ecfced43a6a91b8', 3908147], [66369, '40e96f3af236dc6ef430a5ef805b46ba00fa5c37ae79a53006e5441cfb043314', 4043997], [66412, '7af05219ece934556303608f1caec7a3bc9dbff6c01bb931b77bce6968d0cb7a', 3894580], [66483, '1f27ce9087bcb321dc311ca8e20deb0676541388be4b1fbbbb09127bc36bc261', 3777221], [66487, '58fd4d4f734cc50259fd9f9c0c57f1f3f528598ef96f93cb4095449d17347d40', 4007405], [66510, '2304d24ec3b92310c34177340b1b20cf60d15becfc14ed59681f938322866a11', 3824042], [66514, '4c093fc1f3a986c7ca45299ce6122adacd64100aab4a952e5fa90164e317c7fa', 3785683], [66537, '2f4066f26cdc9da1c1f00788423783a9f1e36223f0875c469c5e91f005c76236', 4498191], [66539, '14de2342da93d7d506970b5adadf2be52cf381fa95ad42a00cbc4d70b84117de', 3553274], [66561, '32116e4f3be90fa93a7fe48c843098fdb92c84283bf29ec8a536c97820069fb0', 3764811], [66635, '482cff864fe3247a7e849bb046463e7c93fd8dcc9254d70b522832312a599d47', 3686851], [66647, '136a6aafd538621a725bf33fb3a479fe03fce2e55ccfe324c4055ba47ba2b565', 3750052], [66675, 'b1ca635ff084c4539f201590964e9a2ff9e7cf0810abb7e4a3a232e4ce727d7f', 4038684], [66676, '3e9f32e81c6b0a24463768b979dd5c91d77448c3a022de22084379cf697d1e61', 3891255], [66746, '822f2360fd1069ab0d819c2c5adbacd3f56cc0ee255aeaccd9a732556187e015', 4200083], [66771, '175ec8f6ee0fe8c0adc49c8aca38da45f5970e843d5763a1059efbb2b859bee7', 4082723], [66800, 'a62f3a578c1c5d74b64ea6e2c72a0f8c0d92adff02c8150f9cb1b0c9d16e55ff', 3758898], [66841, 'a9a860222dde4686730b543f783ececce7f86ef420ef8bc3e7bc0f41fbe3a535', 3937251], [66926, 'c4ab5133189fea41e238179d1f69e28221616edfb7dbd6e189fad24c1f101c7c', 4018360], [66944, '392f28f0fee76d39812626effef1099eecac3188bd1322a11bcfc6a8448e2eb5', 4416089], [66959, '72baa54d1a48ff55dda1ace9aa1b2d3d0ee1d0b860ffbca5fdd7bb7479051075', 4019128], [67015, '68d90ef37e1c89c18d440501b66ed33ae882b1f85fec8ca0986c6596e9589285', 3904715], [67078, '623b0b1835c3cd1176863e78c9c44c7f890e7102578b4046cd5782d5971a1f44', 3999500]]
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
