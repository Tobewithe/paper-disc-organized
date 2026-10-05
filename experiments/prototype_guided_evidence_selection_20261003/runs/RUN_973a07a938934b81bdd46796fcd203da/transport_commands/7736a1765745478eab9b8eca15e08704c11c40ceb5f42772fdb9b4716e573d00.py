ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='c5df9c6e4cf1db3fe5d033353e2bdd43a00fd1649e3cafa535a6779a32182474'
ITEMS=[[31017, 'ffe9e8eb4f535837d3b96a7ad8e8fbc83bf4a8897a7971e0e2f31688176e528f', 3806839], [31050, '749dcf5f624615655a54faedabd9cac95956af2d0a72275b15c87dcf8980a935', 3734545], [31157, '6d8a104a858a8f4d86c62284edfef7f729826f7c96a314d9bd9bf2982790c3aa', 3957151], [31161, '273d9eff8e4616d905e81434bf66368636846dcff056dca41e2cf4e136f683c5', 4047465], [31164, 'd6cf56c26fe9231f8313bdc6c32996a95053704c68b94679a55d8f61beffa103', 3900052], [31217, '6300252dc8bb5aaf83967733b317fea74bc5da011f7afe7dc21ca237d15e009c', 3737539], [31250, 'e915036641d5a9cf84b597a9ffd6e5e9b0db6a44e9ff9b065773f02a116a1ba0', 4761748], [31282, '1cf9a3b1cb248c74321106b72bd558d307005bfa6b6b2e39db1eef36ad75b035', 3832382], [31322, '2dc64cc66047725f3cd7ddfe1ad4f5821c76d3766b34c68ed15cf5713bba6c8d', 4607897], [31329, 'e3bc7108551ffa3a79bce012e4cb490259f19ff6f3905d51fa202bd6fd715822', 4108729], [31333, '34949099e51bfbcef3c1088ad25205409ffa8a514984cb87466dfe5454c69f46', 3775737], [31339, '1043d9fcd202062dd0d30277cfa7469324e2890a6e70c405f19b6af5cdcd4538', 3939665], [31411, '137a28bb71d1866e16206e3a5906b1e8280e093a1f114f535fadf916a5f79f7c', 3766759], [31446, '653e6640ed52ef14ee4a58427bddaebc6791f0b24f807ad0e39a7002388eb66e', 3612104], [31471, '17103229c00db1b8ae811a575df67626f7e07da976b2c40da54bb06b78aa5292', 4351153], [31504, 'e22a444233017bb4057a593685e7b4b4cb708d68df052abcdde1cf2fd737683e', 3757693], [31561, '57d9604c4499b9a11224c0b09244c3f38f393980a220e71265afa0007a508396', 4050585], [31562, 'f31e6ffafcd76dac0ce05e054657ddc0aaaf70efd6e743da502bf3a77cb8276d', 3746235], [31567, '68d4567a395efdf3542f8bb51bfd38697190769aee62aad71fae9cb1745daf99', 3665140], [31736, 'b87bdd04b776168c0f4b55e9613482d7d60a511b5af6f8a8dcb7cc5d0588aa1b', 4001240], [31745, 'a730674fdeb493dc5a79d5ad953de035dc4c9db9da6ba1dda407fe62680710a4', 3879633], [31796, '257dd6498d2126c610c3eec34b6e71b1a427c6cf35211d4a6de6e60a265d2002', 4118339], [31798, '04d234a034a13e3c5fac75f84800503005a31f58e1303676fc78e1f789da109d', 4389574], [31904, '9ef66c3ebdca661a115c506a7ff72438adf96217330ee0246f4640c684bf2f53', 4134568], [32071, 'e903bac053060dd648562ae1c0f8b733ed98b1f467dd1612747463ee6f3fb303', 3647445], [32115, 'd1aac690f4eb391ac9b8a5f11e4abe5a795b8baa83b23f6555beef73e9b9fa41', 3947696], [32149, '446473388fe595b84316f30ed402b5a31a07a147e98a0b71afeae34b18524e37', 3611930], [32244, 'b80d3e8b2378fe6f70771245d53a8756dac50b9afaec7a495b4378d95d889687', 3736036], [32331, '5c2579dbb9cf1e98d349b0af0a24fe08ec6aae7aee2900b0c22977aebb41b86f', 3763761], [32391, 'c3c5241c5d0d9e51e7bd1a789fa9048e93991a1f3e6fc88bd326b94f543f5edb', 3857723], [32458, '27916a2a876e557d42f88fdd37c1bcd55d65f37d2f6ef633aac68b282246e440', 4977807], [32464, '9deb9ef2add7d2d6c50bf821390e6fb7ef00863e248d04250a5ba8fae2023f52', 3642558]]
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
