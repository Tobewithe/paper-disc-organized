ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='851067eb886a828edaf36a50e9415832a1cf197fa19b0c0e1928eb527d58bf97'
ITEMS=[[77, '3f72ce4ca19da33f90b1d8009f58e9fb4c136197efbf64e5dcdf489eaea0a0ff', 4020532], [127, '10b51c259562f1c93f10ef0d78cfa5cda82bb26816e9df530abae449755038c3', 4356184], [394, '88a4bd51052d5b32a3894dfccb5492f434d57caea758190ec86bed3161150fff', 4065654], [395, '94721e180b6fb2f05569ada36d782897cb3974a149b038918537007080caf039', 4361460], [404, 'cac339e3401ae60f8595498e0de01ea6b5be4ac54e8cb193e6a3a70211b21f30', 3727835], [472, 'df4ec17437688ba6dcd90f1da436233754d91c5aa3127c3d5a3706b3e19a20d8', 3286930], [510, '6ff9b45d403f90669c2eb5da72d2609a88106acfd86cec19c9531f6cd3aeb553', 3923881], [514, 'a3c18c68ee738a8b272e412f495816efa996a14f9664ad2d17ad33ec2db462e4', 3587925], [581, '0664d386a3c6a4f9e652ae2edb625ea636feb2357a8ced3c85617b440a73ab35', 3990925], [605, '3826e981bcba962c92aca04d924ca970e3572c1cb2805d59a2be9fe71b8fa760', 3941574], [620, 'ae7a181e4233c91f153667389d47ac8fa920e4216f96dc47c0b63c0333733df7', 3877424], [625, '162d2ad8f3a14c622f0ba86d8b87f1b46449009e6aef8d6f30503c80f14bd9ec', 3888253], [692, '8185670cbd5b55975bae3aeaef9a0892bea4c88452689376b9594786b4bd836a', 3796934], [693, '7db872664d1fcd79b11137072ee7de3e749ea1ad8d9b28449f36ef7499b38eeb', 4193441], [715, 'd9a9022aa507ccb0c36b50e050f7474c4553e9e09de3c169561fc167efa076d5', 5296634], [722, 'fbe3cc01457bca7ca2593dbd78284583c2d1e86c4651ed989c2e2b4d090b338d', 4615432], [723, 'fc497535c093a8c5c1e52b35c5b138f12eeab6d15778a37acd4b5fd66128525d', 3851685], [781, '94cb3efd34d1296909bdb4a954dea393cb3ca9aa5227b80be334424cb0c1e88c', 3769586], [908, 'bd43692d4680a516b38fd89fb4fc8575dcbb851923fe835d3b4dc432e7f0b117', 4109809], [955, 'bfc270cd67f09644bb1ba92a007ffaa4c95bc782b0df8b761893e5a728e0b0d5', 3927700], [1083, '86ee10cf8b11492ccc9c57dd092a21d9069bf77efe6b337c443446dd05319cba', 3796174], [1183, '08315b16aa7a733a869e4d4ef5d0728d2d32e3838d903491aa3ed8f5f83b385c', 3850009], [1213, '4af4ae504b16155012066249ede686d8ad02eff9c098ca013e04c35d5fe83186', 3738665], [1244, '376248d1ef39908bb7f93540cd5257709bd6ffd6214623bc9bd5d6bddf253255', 3943628], [1307, '94934e4b59972036aa5b80081812b584a7bc41125cdba32866a3f21fef513838', 3928488], [1330, 'a8eeb0cf5196db11476a541b37b8a5727561229710b07a63888e841c1789fa5a', 3864877], [1342, '2b3a91e5e5183298d8af9f91088d3c52afe4eb8d79c30e21e0f56ea282cff54c', 4066917], [1347, 'e1076332347c540ab45c11484fd5399971b5ad30273d89d851afac20446231bf', 3976819], [1355, 'eed7179a9bbc65df54b2a9163515cd0a939ad91ab029a871c52b33947c2a33ba', 4021050], [1374, '2825bd8125aad6b9d161b1e78704fff619f2afaa5395c286ec9f2b1bffa053e7', 3750718], [1375, 'ea0f0769d28ffac5678309a0a5536ab150dc1164ac7ad05acd13ae03ae7ec3aa', 3785039], [1424, '54504d62710d68a4ddae61a53cd0c559d52817e889a6066cd071c72073e59f4f', 4394993]]
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
