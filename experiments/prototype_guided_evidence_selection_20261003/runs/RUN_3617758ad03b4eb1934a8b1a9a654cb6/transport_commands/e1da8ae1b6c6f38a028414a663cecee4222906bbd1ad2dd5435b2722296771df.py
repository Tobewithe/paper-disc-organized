ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='b9e4b105ed0a48fb21c8971dbe083d7c7712fd5cd30d0bb2fe1b49ceca8c8ce8'
ITEMS=[[19852, '3b8f2fedeaac51f732c0d9c31db6720b64703ccc8452bc17006f299df75ea761', 3637963], [19957, 'c3c5da2ec41f3a889b38978f4cd3fdb2bac303f47df222cb19034934d8254dca', 3954709], [19974, 'df0edaadf0d0ab4fac90a0e324810e2bddf48c8dc676f4ac81eb2b924263d506', 3764588], [20070, '785745b0330dc62c043df3b7a4213bbcbaec64ebdf9f3b4427b4360616ae7aa5', 3729009], [20150, 'e0f812016de8bda0318c8819253fdfeb3ea8c00f41a4ea626cdaec74d5c8d68b', 4059572], [20156, '9f06dd14c4fbff85eb333f59be3a321729af6eaa9b5a6b03224fcda908269c55', 4093088], [20171, 'a53b25461bbfc488752a8ec518cd26c4df3f04bf25078b72c13aa3f724a6c308', 3721001], [20178, '663179174ec6a3c24345e64dc0657bac4bca541ab0ca5419bfe437b4c7b74972', 3827241], [20251, 'fed0a8d2ff78e2b30b149047eaa053933bbfc70f3f2cb975b88f0838017643cc', 5025467], [20254, 'b157f4aad71bfed299b94954b3bb3bd628ca888eacb2c54573701e17436d63bf', 3682387], [20255, '71aba12da3c923d3963dfbbac33eed8de96ebd6e4b776e51c7ae85d5cdc9dd77', 3740995], [20261, '162fdd38836f5e2ac64b2df1517ea0dd178b47e349ad6f29d219a7453f8c5417', 3418526], [20391, '9ee9ff9580bbd16fc7c7b7f320824d118a719092061f41ab7b7343a8c752b032', 4271674], [20487, 'fd973859e6c112722f7ca431cfca4cb697062507036a9d4013abe00fafe52543', 3778065], [20512, '5431ebd8820ee41e8cbe85a2c6e0938ffbc9cc050fa475a0b1b9bf2d0acb15c6', 3721805], [20541, '3318c5a2751375f1b5535dbb86ff32449d50404ebec20e0db7cdcb5b9ad9c6bc', 3808440], [20563, '5939ec416a00ad90f88cb044d8c9b9212c57ab6cdda2acc6d60779ba75315b89', 3797211], [20774, '21738164ff8d6f5fe261d61c724cbb58c6f7f90b446e94baab0bed0d736df7ec', 3982308], [20839, '07cd37816f8cb8d72187b565c346fbfe77d1b537d3937a3b255dfcc781c9eb6b', 4271861], [20925, '7605b7677563295fb9b58de95b3a807b47c897b49b2106aee9da7ab48f05d824', 3673051], [21029, 'e1dbd7db6d732a9aa5ec66188e6ded7ad9c7adec063b65562d07cbf195cb4cc4', 3979139], [21088, '0e3bb6a58cdfdd2e9458e2aff0590d0dfda28b966ec297e2dec94f436936426f', 3934683], [21204, '56f88e337129c491c9afea01d760a0b30913ffa3c55aa9e8e06e53ad847c091a', 4248674], [21206, 'ed4e748e78811d183fafecca7326a3a9402a3cf36b1c332023aebe6e910cf621', 3924246], [21276, '7306d4582ec7163d148089a6147751d0585547665231d9aa5ffe7ccb140d7f29', 4189297], [21343, '0a0705bed42d73fb90aa830c515c40d6f314f8e00a8e1b9d26a30aa3ea66cf25', 4257111], [21361, 'acb68dc41bdba7f223845a4fb86dd433d033b00a64bde281e51749e9d54b4901', 3940095], [21365, '3272175d758d2c12a9365a8cb0ca1f06c77fe527a510a00784e46f4a79f82794', 3710783], [21497, 'a771b38b5d2fab63ea3955c50fe1ed97a4fa3aad43bd948314b2b96efc3bf7ed', 3939349], [21498, '653c5192738e2874230ed15f5eb79a458e9d05589cba69983f757a086928ecf3', 3845265], [21551, 'b89ff3d00c80a4f50cdce6449705c2706a8b3e7ae4057102d1c23e0de432d55a', 4207369], [21553, 'fe104b3b85af276e7bfb7f26ea68f63f57c2119bc9c93fd540a46e72a2956b2e', 3764206]]
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
