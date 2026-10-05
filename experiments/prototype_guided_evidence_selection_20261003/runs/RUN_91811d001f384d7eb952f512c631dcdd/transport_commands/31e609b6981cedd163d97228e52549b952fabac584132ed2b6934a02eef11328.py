ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_91811d001f384d7eb952f512c631dcdd'
KEY='9de5667cf34f63a8b97e5dfc9755e3eeb0656ec978a8962754f392a931e1ff66'
ITEMS=[[9895, '1fa2821dd395644efb28a3db7889e494c4be49e1f9584f1d60d84158c81fb2f1', 3877702], [9908, '6b5859724c3aa38a18b3ae621803b765d960513f96bef0db76bed8c014c64a31', 3745733], [9929, '4011ebab30fe5e5bce7cfc255f8b36eb24854a3e610d66acb6b5842ec940526d', 4793666], [10092, '380700a8486c92b1b647b87d32f5bc241960967f4a4589485bdededd7920b33d', 3866087], [10466, '3c8e101957c962332853718ab73bc0718f8dc6a64b08ae87290bf163185811c3', 3808933], [10711, '6b77866f4cf89d056eb7e9574c41d748853f507a29c74439bd19bf05992d3d59', 4249646], [10877, 'fa1e694448678120bade543a180ed25d091844966d26303a5893d62839e94465', 3983600], [10995, 'f5809997648225febc904fbf052ec72e09549bef7d6cdbdef44c5fc57f0fcbbe', 3569520], [11197, '1ddaa074624acdd426655399c2fd6473a0dbe6ccc1eeb89ebe4b8e521318f608', 4214934], [11227, 'ceffe676737b10491aab075f707c2b4d6103ec900fb94930d1a087de64071b6d', 3612068], [11349, 'b0a6097505cbbade6744ec9e577976545cf149e80b47ea988cb5a6a96ed18865', 3973604], [11613, '835c44af86e4a5704ea84b7d3748a7b6cdda118cb99798f500c5eacf64d1139b', 4348442], [11760, 'caa3c96965dd436dd0b1c3d4d33c94a41674e04fc3d4c4fd3ae99028a8730ada', 3865659], [12490, 'cee44eb38bb7d349a571ffb78844c39feb64adfd0ae3deb418238bc6bf8517cc', 4309356], [12639, '1fda86b162c5e8ca518ac7ebf1a24d216cab194cf52ea09bfd24bef90d07be14', 4354213], [12670, 'a2da96520e3b65fe0cee6253a12cb053578f60cf848e0c1aff806f4bbceb7b77', 4678899], [12748, '72e770b2c16e2e3cd39da32b40089bfce8b7d02f47220284628c4e1d4288fb56', 3921185], [13177, '0328977dadb84cd9b8e7ebddc11b9ef6ed2af1762deb96b6dedade30e04c0458', 3837884], [13284, '95cb2dd8ef46c0333deb42f4791b482ee4044eda247f0dab0f52cf20d6fb82a5', 3774491], [13348, '0fd7f7e213223edfa1ff5925154f983bda0face95805cd439e7b532bc99592df', 4031101], [13546, '1627cdcd5515d4d0e8acb33b99843fde0950c1677dd1e619c7bfb505bbd0b611', 3968004], [13659, '88b4604ea26204fdef021997b1f16e5677541b04a9e04297ae27fb73ca044373', 4415739], [13715, '00045ed3638aa267f0ac11b11fda4a9e41c06dadd4a5a69b1f4a01dad313904d', 3751535], [13774, '0b3fee147f1b35fa62b7e997df5aef85b9fb6290dc544920c80cfa73b7c5ae80', 3706362], [14007, '5cbf341d21c9f8aebded7be9768d6a5ebaee62dc098b1e3eba1bcef299afc7c8', 3643164], [14038, '443ade1a5f0b16f6f2f639ea2fc9b1d520097999509b857dca8de07d1af80ade', 4523258], [14159, 'c81b275fc19344d8c6ddb2afb68ae96b0c01c98fec2e947f71039cb1894f70b3', 4264147], [14226, 'de4ec9b324a5b7c53809eed7c8cfac34c88dc00e754841d04fd991fcfa1bc1d2', 3989604], [14230, '1e8755155d25625265489cbc2b6fdcc35a8d7976ebddd6b232a8465a259c890d', 4456401], [14795, '8fa534e0b777e2a7057adb1395ab2324803eed6fbbd39ed6890ac55755726aff', 3778760], [14855, 'c40d8cd990c261e598e731cf4571d65fb13c934342b2fea23cf71b4dc215dbf3', 3631831], [15016, 'c1423c4b9b522ad859079eaead7c087375626df6a82c8858c009a398efd1bafe', 3399484]]
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
