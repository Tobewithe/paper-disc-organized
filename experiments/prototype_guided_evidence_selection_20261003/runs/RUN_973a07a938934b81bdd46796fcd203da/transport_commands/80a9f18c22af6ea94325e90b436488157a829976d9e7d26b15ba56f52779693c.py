ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='56d42241d5fd54343b29352dff1b29f0d3e87e908cc1ae2d1614b1efd9775300'
ITEMS=[[29656, '1a1955f5f32433ace1584b50c0367e13902d6a6e3f3ab279fc9c45555ac5831a', 3852405], [29675, 'cafb3707d93c64881d995f6d1fdcabd3a1de4889175ca7b941e4bb2d3b1c7f02', 3896263], [29687, '525d2b0d8e44430d3cfa7f2dbeac305f50cc1aedca8ae2e82cdf57b92696858f', 4050237], [29719, '69ee917acbf804e1313604e0ea15a4ff4dc96476374bd067965722aa8f44ed11', 3712575], [29755, 'e22619fac01f280c104e879b24c9c7f7bab4a3930fa55e50a6a4e6bc361d1f4f', 3812756], [29793, '29c4cdd1baf512d9751d1d44c56eebc36d6e09786ab4d55bfb755deae9286078', 3964085], [29794, '7dcd1fdd173af0902d7d692079b1a23b9a5800ff6ba7791960d774ccf66a68c6', 3761592], [29839, '8bdc8517b146fca75df8f99ab1d4ff21276f15e977307898ebeae5d2adbb9a01', 3920564], [29852, 'f1c32f2c888bf4edb252104f41ccdb098b5c9a5dc45bded775b4e4568d2d4fc3', 3869903], [29936, 'f578c440d1e692dda4c765dc14a526638adfa788f089a9eddd7ef90bb6df720d', 3892463], [30000, '06ed3d3a7291a69cb4ee75904bc088276be7ef324a5b7432a78c60be6c11408f', 4136630], [30066, '03672fcef6cc75ea7ce5415df53a46a33afee19ab49c725c8936a043e3871c89', 3731879], [30139, 'ef8764bd90b50c2477dfde64728872fb488a2af56cded1cde47a1aa5d5e6d4ed', 4447266], [30160, 'fd064d824b9bf61fbdb299ddf7d63484e6e09c5db6828ad04b1bbe89cd928339', 3711708], [30281, 'db22b183a09203f259910f92fd876e41e4ac63410106ee0f5b85803789a92eff', 3791912], [30299, '4add07341d26f328f2252797fc5a5e49b3448d71543e5865d83fb63fc7e5e6e6', 4002633], [30355, '95f330206a29a54c2293a6ea2bf0ed4ffc30a73cf8b82d61d3bae43920b15545', 3727444], [30357, '0e57b50c70011d258ead9c16ecd48be26d3e5bdca2e386245e479a89253c69e5', 3687832], [30409, '94f0872411ae50ce72c96bc330d05a9f700098953db09d94eac212ae49024ee6', 3888482], [30423, '22a39b62a4192a1a72e089c193c5d2ad20fabcd7edc49253afed35e97588b846', 3755301], [30494, 'b0df369db3ebb5a8a65b25a7d3d93f0f051b479c46df977a13010de1d9fa0885', 3925112], [30565, '146cee04f271d4d7ac450cfc7de2b612dfc5f88a24c76c37a843f92d6586eef5', 4134868], [30583, '03add74a5a0dc2b0303ac864f50612821466f4b9058432646743e004b50209bc', 3696207], [30699, '1a36f212cc3be373f3d03cf001b27ff2f783de9540bf09528c4fa3c145fa7244', 4393393], [30712, '82395621a2477f4581b56f0592dcc0bb7da7e07a5e25b4750d360f8332c23bf4', 3936206], [30719, 'bc2a2db0a9e628835c1a0d2c7109811f950a7a90eaa02d931c0ef98dc970aa0a', 4091044], [30796, '137edfe135182d681ffbd2cea62a2b93d649e3b7069a9e2810c24af607cc10cf', 3921396], [30852, '73de53431d2693e78e8e922bc858e273c185e1cfc91f2517b95f68ee4d0153a8', 3807408], [30875, 'f44f7b95dc632b1d88a3acd103399b9fd7746f2d4e7cdbb9f16cdf983c97fe75', 3778029], [30915, 'd4959a44dd5d98685b88f2ee8283a6d447c2fa4748b91b37b4b14a6f5c7a2aed', 3519054], [30958, 'f16a2f15d0093f58501372bfff0d452fe39ba4306ce89b042b42c4397686b1c9', 3565205], [30983, '6d8222794b5511b91e4ef18a5417d96bafcae0c6bf2ee0099ea3717e3dc926f2', 3813605]]
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
