ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='36b6a11362389862c4e8890549c8b0d3de2cd17577bff30348a4b677f6d232bb'
ITEMS=[[93402, '5e0ad79f0c95e865d3936d061ff3aae6cad837ccdb629a29a092310c066fdb99', 3642231], [93434, '1975ed8ddb33983c3fc5c48ae1e8b4aabde6462937bb2885f2500d4998caa99d', 3754634], [93437, '765d176fd2d39070ac9eefb30d6cc8d22c301b092e82fd5ac4471a36e2199973', 4032049], [93451, '5c1fe256d224fd3ac14338aac1bf3524c40754ad716d14518b2bf56b0efecb3b', 3805857], [93493, '24bd2a800bc55275797fbd2c4b0eb23b65af0ee107838b7ae11bba8b4271c5ee', 3684025], [93506, 'bfffd5e25c93cdb7e0dc84702c382d495657b239768f6392cf714db04e498894', 3895628], [93587, 'a0bfe83d6355f6c24adb9e2b48ed8a34615ced3fd7cd791a73c511a0cb27e894', 3993192], [93590, '2c5a67b607624a2d784f64ee0063a42531a5d17c00e72ec982c7dae54310fe85', 3781560], [93597, '8ae3d7822449a237888aa806591e0eb6199735b2d2ca1c796f7ad6bb6300f618', 3782602], [93621, '67262b44e68f0f7688fe9f0d2e91ac21ed558104af7190ddf554a2e5acdf817f', 4082455], [93644, 'c6e6e9c89f6260dce4ccc910c6fab9f537023e6181156fba529bdfead25790dd', 4281279], [93673, '397459521b19b32f1c568fb0358290fdca562094c69b22fea76d9dc24da712d4', 3612069], [93717, '41fa32ed4fc2377b24c3e1c8cf8ea9c95d2c0f8ebfa0904011df815ddb970018', 4450478], [93732, 'c76d520d48cc217195281d1995f811e8b353fad6d31a29328b6f762eea994a66', 4112455], [93736, '535bcfadd758fb04646c914dc8e4c0f797759ff18b451962c3a7b779291d624f', 4071191], [93740, '11393b15dca5f86f28c790e9743bd6a193f8d513de14c137dc081a8372b2f060', 3983801], [93795, '34dc2584e92d49f0e2370cbd01e32b6dbe8d6d5a86fc7e42bedd3ba7abd7d3b3', 3730997], [93923, 'c376b5849314588f2d98e9939c3d08542a67f1d72c7064dc18d2deb8ecc7fb60', 3831424], [93946, '8dfbfd6f6c8ce6251c2600e1dc4c0d99b4953d9a0ff68efc0a3505eb33490bb2', 4198467], [93985, '4958c841bcab5bd2a3a05f1f24c6f7558c095627cd5b7dd3afc53eddd4674d9f', 3840509], [94012, '06b09befe01d1ee73ecbb2998e548a9b47c2e1cf4fa258d55f3d99fb3e110a06', 3581513], [94049, 'eb049ef55f1cd90c7fa1e8bcfd43824fc07f3aa84e69499c87f036508204a671', 3815658], [94157, 'f585999869057514c77dd5cd0589ab4dc0a0f259d55c3d47be11699c9a16ee9e', 4000920], [94300, 'b9fa667057b5a7b50e13e94346eb2e12527ad80f6e2a76bc881b4e77b644bb22', 3964035], [94350, '66503d5669fee7460c870677f1b5014a8f8e0081f1c2ee4ab8fcec07fa2f840e', 3904068], [94392, '0ea17ca283118801f4e53eb0d11408471c590a753c09fd095495b70b36568e3e', 3909926], [94550, '1fe0230914a42c406962af15d95be61f99daee6dfaf236d00356dacc29d1598c', 3632148], [94577, '8dd921e3402be6655bff3800df8103e3e01277912147cc8501ac25d432877faa', 3818279], [94619, '4d5ae3a1ac5498e04792471f5e7978fe4bd22182e80080f394a723f5da813da6', 3872596], [94678, 'b322615c20890e971f5a3dc8654c200ed601d8fd19f3d66ee366afbf672cba6c', 3840334], [94687, '84a7da7dd34332f5b98cd95db51126f2d76f067666ee7895e89e55796d8fa22d', 4487049], [94751, 'ddfd1ad21e6ba9b4abdd7cd386689807b1f7ccaaf20b6ed18410307cf14deace', 4187825]]
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
