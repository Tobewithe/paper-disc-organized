ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
KEY='528a21aca4f23e772dc0853164591410fea5abb45fbd326575a586c223b624cb'
ITEMS=[[6160, '91d2f38db75382547211806a7155c20ef6490b834fdf4c895374d1213359a8bf', 3654785], [6226, '0b89c245815eefe4d2957c9ef88c86533d3120266c0af1a6d02a0aab0b1b41ef', 3796531], [6407, '6de7a182642cfc08b3479d22bc4725dcc54ccb807d87a5d990b91b2dc3633244', 3932911], [6426, '3b0408a5a08941baa9e9f65f0955145b113a03ec56d97f6072c753b0c8a35691', 778287], [6593, '903096353f80b24c0d18ffe8693a8175ed2d4bd615ce8394a9406f1e674d73e9', 3753457], [6632, 'e20842df36c192b8a42415d16c11f5306c636c91e9518dbf37cfddb6b814856d', 3750420], [6664, '8a17d04ba1b582990040e94c65c30bcf43eb08b83343f7de2b7b8dc2fb480dfe', 4048745], [6675, '03a2eba479d0abe4aab33271058789aec880c7dc2adba9434812952372cd2162', 3764706], [6709, '36ac4a1b809af9cd373e8ac50f0ca53219ab5a7027387f666299e6c3ef737fbb', 4652946], [6847, '16b9013e7912424487d96ad37018531a68232675ec0f02312a65adccba1ab265', 3574376], [6914, 'f7c75819cd59750a10b0f3da449a68157e77b8e2a831bb4dde957daaf7e04c6b', 3906758], [6936, '8e32324790354d4a554b7e5b53045d1855e2a00c53ccab579bc3c28c6d30dd08', 3827307], [6943, '0d85aea55f33077ff3e107df16a5f8c62312774a99a9df46f2fbe61a03512b99', 4223026], [7228, 'ea00fd79a30f28ac29d63357b78560f8eddb3b32292fd9e16d4b219718cea3e7', 5024279], [7367, '4d423e995b0299d7a2502cd8e774e8af5089f645d9fac1a717ee534abeb1b05c', 3946569], [7503, '49eddb442eaa931904854e8c49306571ce66631547dcc15c7de97147d92260ce', 3862403], [7570, 'bc33f73dd50d1b6479ca0b448917fac91192c9b570697695796c66572d29bcdb', 4022532], [7592, 'de6f20dc133420a504663bf0aa6565e7ebe077724ef0e865de37a570e30dfd68', 474952], [7603, 'dfffc1fca893615615908de1cc67b5f4cbea5b387fd5c3f112ce8e98ee98bddf', 3922596], [7616, '37877e21f28c73f83bca47494a9f4573da6b360850950229a260993cd499a371', 3797472], [7623, 'c8f374f657b1434761da0bd2fd6864029af12c3b2fd33a4fcf96e22e3435f59e', 4085806], [7729, '6bfcc36e60a84c0322c63f960220cd8a77825336ede4bda512b6bc19019ec671', 3854480], [7735, '3923d6fd7f0408b8063aeadf6425c86be92043409889194f8c44c921597f7ae7', 4248128], [7785, 'b548faee454c5c01742a6dea848ee1ef26b94ef2c4e527e05f4e71cd9dc1bb78', 3602213], [7819, '284e341e7c03c34b9f7ec866424bebabc2155d7711d2af530508fa722774dcfd', 4330033], [7823, 'c73b7a6172817ecf1263bde22f70d486ccbe5832a5a961e1d27e32dfe720c663', 668339], [7867, '61709c69a4169bb6281ef846f0559dc3bc150791c2e7c52f5fc736ed2019339e', 3702438], [7961, '29d83a0efb6cb7102ac52f92d4503382bb4bd24da74262215c6f2614ff00b42b', 3887944], [7996, '7b2f8de40a783f5b9074920408254eac137b6a4e6450a20dd608e7e19ca596d5', 3793679], [8064, 'b8954af377dc1ff7ba901b9e312d95430d4049a37b6ff5e009b3ad7716f92c6a', 4060005], [8095, '48330506ae2803d5ecf0c7ce71bb4810ad561ebaffb5cd9a3a706ad6a6f1617a', 3939373], [8138, 'f2a2b8803c44062037732ccfae515774e265061c089d2a457d948a607a038763', 3769007]]
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
