ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='a39f47930e1c1fad19aeb1aa34b3a4525e459be856838a66b47edbfa1b3cd2ed'
ITEMS=[[67099, '1a0ae3a7056509b6a2d2dab348d0264a69818a88911585d98dda8810c6dc483f', 4279925], [67191, '2c665e60c40f4aec7b356e4380294bc911c5d7b17cf8c5d5706b930dc103a6ce', 3743926], [67213, 'b6fb3175a8a4347579ac76d390fc243595175311c9612f5ad78863a54fbe5315', 4231016], [67248, 'b33d3a189da41122ed658c94ba242b59a5fe7df606644d6676090bdfcb6483d8', 3883544], [67252, '19b7887ef4ef5b4408ebd435b91e18f4e1a771bd73b3d32437bc413fd95a5ccd', 3834525], [67256, '2c7429bce9f64ee7e104dd15a611e3a09ecb7b2514432b07572b981da78294df', 4190871], [67262, '7c998473e6d517071bbe8ece0da35b947e46547beef0e4a4602f9d1445f45eaa', 3820452], [67310, '81085fd7a15239af387f2d99452d3607ff307ee3a1070f0cfd0be82b7d0bd2aa', 3999413], [67315, '5bfdbf5241b4cb2b2699ccc78316426e2954a783c7634fa62e0703377b6ee353', 3725490], [67431, '9d9858b363cc1720e9d8a1fd5c90a053e7674f056677c2abd104962dd52c1b29', 3901739], [67532, '539c4a9ef2f5a7229614abf475bdfee9226192c63a1324c477d55a7859e22610', 3910821], [67560, 'ab9851c051f70df153aadc9b0e88a3af3a1168eb2868c40d02c84514be197bb4', 4644473], [67616, '5108224168196f7d654baee72ce5f1e1b6c290f8e249d95eb422b180a27753bf', 5029297], [67680, '493c982e599036b7b214ea4f40236dc09abcf373bade932f861d893dcad439d4', 3746913], [67715, 'ba3e0af8f38d7f02d9b890df50208c7df21a49591428bd8b0ee6a78c6201a0b5', 3779292], [67749, 'aa6a20b01a112d3724d878627c7a2df66147e6e7de83df0c11c4ec956b98751f', 4424575], [67788, '57844fc9eedc93d3811e59bb2c1c79053a6a4fed6a28e81c2a9f74764adbf4fc', 3882672], [67881, '394762f839d6a3ecec8ffdca1f655d23c3ab4ef8107648ad45b527649ef557ce', 4191955], [67904, '745dbe5681cdb82993d5bd2e36ba32d1f2132e179e538574dd607b20f26e9b36', 3741315], [67917, '9b1a6d58b6f0c806d8f54d12e82725134f6c8460acda4ab968dd99bc864771f5', 3794746], [68032, '1aa6228397cdf96d0ad90d4a6028695889b70cd5cf8328d89ca9cc6b7c8ae7e3', 4182151], [68036, 'c83cf7fdfc297a665155ce6c9704dd78b22bd8904dfbbcd6013ecbbec5660a42', 3753446], [68078, '41895a2264c292a90050c425a7c9c60f5d8257571153982e2a47f6daad862766', 3710891], [68094, '93a39f5f88869a2a397085d0c44c4ef2febebcd659be67e1845b63ada5e815e7', 3753234], [68117, '9bd2301b34f470cdbff12bb3c7890c5e80c50b194ea20bf1b2b4e1c2472b0230', 3598189], [68130, '3557ca992c6e4e6f233e84b898ed6552bf2610a71f507fbbc7daafe812e53260', 4735566], [68147, 'd1ac02d8089fbcc61c1c901d9fcd3bb1aa63899b868bc9726dc380081c34e7a2', 3822436], [68159, '613819ef37e4858cf31bd368f9023303d1f36e6b99791e54e2e6734c2dc8978f', 4333699], [68167, 'd077d978851a1cf3f330e1ab633fbabe749a6a40ca2279494e4727a5c1bcadc7', 3600891], [68176, '525b55573a1aeed0c29915fa366f641d46ab4b0720638ff69008e87188dc9512', 4227012], [68206, '7c4303dcc707683d3440d0cffb6358a7a8e27f77bc55b151e6f6ab527b289fc0', 3570800], [68219, '56ed568774177c9ef31614a32cb10ae14d682041eb8da6cc71b2324dca1ff022', 3853803]]
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
