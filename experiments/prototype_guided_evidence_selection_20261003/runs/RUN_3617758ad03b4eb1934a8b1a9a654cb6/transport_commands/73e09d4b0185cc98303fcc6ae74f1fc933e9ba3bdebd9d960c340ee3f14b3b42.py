ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='9ba34123c172876a94ff842b46aef5e64b623a55d6197f61e0df0498a8f8ea8c'
ITEMS=[[12728, '1c99c19bba0e937a82e59d381bc4226f444f80385bb6236941d9c54f4158300c', 3685428], [12754, 'de9a01a80f86768ba5b4fe79afa0e52ff4263207c9d7147d65e0bb0ab22f3407', 4446705], [12756, 'c781593813aa184db4dcdbedc9ae50745e8a63d4553f8509473bbb5d046c64a6', 3834024], [12873, '45aa159124dfabd35cb923bde13436b76fc359759761a0f0aa0e2247bd654778', 3906226], [12933, 'd289c6a2ad198eb21c198a03cb433726e0c66165207adf7fd3f8c3b1b33b4c1e', 3932799], [13160, '19ace1e045405d19ffe497f78af55bf1a5f270722c98928a16168bbd3eb9973f', 3516374], [13168, 'e6387a93c5e98ddf40b45ae53788f8e46e7f75a551f03112fa45a5e1969245e1', 3921272], [13273, 'c6cbe6c70917d5c752d15f917a01228565ef95293488d6ea6a436639e65b995f', 3785908], [13356, '1bcfe95a39aca08626b09c14f04b05b17e310ee7b82e8d294516e9c4fb2549eb', 3696806], [13524, '921f6eca3bb1c0fb7d9617b566506d2383c3307b93dab2cf5eed8a4cf6a9fc9a', 3773777], [13578, 'bc7b16110d6ba2123c95a03509fa629e086216c5306a5d20887673a0c69a6278', 3750056], [13605, '4410e59e5269c09526f774bfbc634aaaa988ef72f1d024876e0382e2308bef69', 4043222], [13636, 'a32ecbae512efc2821477cfa3856b2e7f8ef0e6f63da9692756302a692f6ffd7', 3964663], [13637, 'b7b1b18ee481d6db9adee71cd4675dd48745b16501f01143ca9450503313f151', 3782201], [13802, 'db7e1f82d0c2594e6bf3c0f37c0928272b69ff1b017f2a78402b3b017334d7fd', 3513115], [13867, 'd4fe33ca27d6f8c6b48708581037ed0cbe2fe7f530e071f637b8357605b53610', 3894908], [13912, 'da1de0c0535e4b30240cb5c9883df6788651bc2dcce1db64f95e772947bf006d', 3909833], [13921, '961b3dd55ef2ca29bc22cc3bf5315ee1dd03bf4d4d792daf634a3dca5a8388de', 3791668], [13944, '570fb7c02229979c9962bab465be86f4a6aa0fb9a5059f29a0893d5f42f73979', 3926300], [13992, 'aebd07f70daa791166470722a82cd1e0cb20ac5bde3f115eb2bc695e18d3a093', 3773512], [14025, 'b56239128213ecc9c86ed63dd8e638acab09b0cac3f345db3bfff06b3404c3ea', 3527957], [14027, '2063d8558089220994e9dba7e817698882904a1a16e373f8ec7185acfec3e458', 3972201], [14034, 'ed8c38508471a1688056566adfcd28bd8cad0fcdda20410f7e4dab1898d81f3d', 3648694], [14049, '7c68c3e75ec03f7492d59b3226c508364c982c890e943372133e19f31dd36994', 3819628], [14073, '327fb025b69803232052853ccf16b2c15f3c439ba1999f210081f23fdeea59a9', 3675340], [14125, 'ed159f1cdc157102edb42479a26cf7b9fb332f4e3ecf8cd12b9eafbc18fb1d3d', 4013851], [14285, 'c0b95eadacdc98960a24d9b0b89485304e5006ce92782981c04d2a450874ec57', 3961562], [14320, 'a7d06caa3900dec08002d41c83bb98d2bf7d5dc3322f5df50522cdb6defa63e0', 4004074], [14377, '0befae6f40446cc04c5b0a5dca2c49d8262f16898cd079836b113ea4c9bb4fca', 4011108], [14392, 'ef30eaea34879aceeb51840e99da9d19127188a5a5ea64a00a1c0e61bbe2a632', 3943687], [14430, '94d52b2d4cd79aa02414d409799932b61efee4263f8c8ecfc68cf8e829c24f56', 3754453], [14432, '38b1d1aa49e84583e3bfa2b907f3028220e15747d60b371838fc44f6e252010a', 3841938]]
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
