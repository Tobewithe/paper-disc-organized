ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='7afa001c7864cca8f1aa77dd7c8767b841563ee2835ee091cde3759adb72abfc'
ITEMS=[[1472, 'fd16871c4052bca127c14284835e7aee974e04abbdd84cce54ba4cba3feeb7ed', 616114], [1579, 'b345b25fe3ecfcfc0181a00f53ee263c1467b74148fa2460c3d8a055f889b3fc', 3290078], [1647, 'a05f46da2a5c80696f148ce7ad93c4e94b785bef413b0d8bdc2c37e40bdcd019', 3513659], [1655, '79864b068e037a4fe605dfcff9603b4416a8a445cdc059d61a8b31f489084b7c', 3476490], [1688, '5bcacc66b46e7dfaf724631686cc4db48f35cbedaad2ef875b377a40bd08bf5f', 4220167], [1722, '8e9afce6d22051018a6bb0652a62e7d6c9a7485362b45013de803c384fbad37d', 4076766], [1774, '82a47455f7fd91a600c67f5f0a835323c97784a834da21173089b2d58642e0bf', 3751480], [1792, 'd250d01f1a13c603f27b439ed4b7836752361735dfd679b8801a015deb3a15ff', 3779340], [1815, '8f4720d9b6a8fb7ae394e9b8c17e43802e3bba0241e75eb0f6cc8f93c6d2ce59', 4043321], [1875, '219ceb6aca6b01ee8a0a39d3537c7850855b5d69e7efac2cb952771c18fb3aee', 3993340], [1895, 'de738a25d177064fc49e7dbb112686f4e8f8e1f90cea70e3d45077d5fba101a9', 3647254], [1943, '18921ab2216c45e2d787f38648671b76f5ea8dce4bc9577efda132b230d95225', 3774520], [1999, '6741c8588a3fe0183491a12be880fd15e9ff0ce04daeff556c9fdafab8c6ccf6', 3695140], [2498, 'f82f2000e8d5221c199117bc520e3abc2f0944ab8fdca3fc597546f8fbc56e81', 4064722], [2606, '07576552ca692ea78cadf92511687c7c3173e3a91b0a283ad8b934459678a605', 3745642], [2624, 'ab4f09c5f7d92faaf0566091828e78d9d2f20c1a063fb80037ff7beb62258b27', 3843704], [2664, '5606571423238d2e404e3b9d6bcb7e3c13b794adc251d91f0553359b1009a8e3', 4664464], [2691, 'ffb86ff95bf56753f2f7e0c36efc782dea3b96fe4204323c687c49861db0971c', 5088192], [2867, '8e18b886ee959160f9db2fdf48967fd9f7f19bf5295b6f99b6d2b2c9fe486c36', 4124592], [3217, 'fd852f16d0416d38f0738bf2c1250831e13b3c34db4222b4cb2b97cfd488ee2f', 3896351], [3464, '973587034ca04b5aec747f85e7b5719474799ae1f305cfda021945d067610939', 4146688], [3771, 'b7526d4fd984b5d500851ee24bece2f35156b703258dc4dddbac269e1d03929a', 3713541], [3920, '24490b8052bcef0f349d064f5f2dc80d0c70c097d623bc092a626c8779e7eeb6', 3725028], [3967, '965ce93a38144ff201ca4b5b3012a0a599625c347d2360d098f06a4410938723', 4199198], [3995, '360dd93ffa1c121ffcd47d55d58615910f6f1c764639cadcbf72f218859b30e6', 3762432], [4139, '30221932f6cb73bd481624c96b292580d0d55f2af97815cd89030cec327c7a97', 3664340], [4159, 'eaf1efe21ce0dce232519f126fac49e5a31097e7e20026861a2104494c018e81', 3757897], [4187, '2a033d1f3d99fb5bf38b9f558598a05751bce6afd1fb266e42c6afe447dbc447', 3764498], [4208, 'e4d6aeef34ff761c0486ce3bdbbafd80b435442cfb27c450cf8a2ea56c6fe3c0', 3937475], [4245, 'e92ea2ecc7f6a2681461fbce6644d52fcd54eefd5105fec63dc7952302458281', 3750006], [4260, '1f229cde1da51770c969a45e124cd70209645be28154c22f2d80878fa52dc758', 3861910], [4309, 'ea04c3ac09d2affd06fd235dee08af4d24db2232fe98e66542b35413bd55b3ab', 3694753]]
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
