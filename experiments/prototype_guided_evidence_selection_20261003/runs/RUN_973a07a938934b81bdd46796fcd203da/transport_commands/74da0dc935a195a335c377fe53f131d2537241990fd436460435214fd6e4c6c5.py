ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='255138a6df6f9a5863419ee15b08f399f5469895c74c7514fddb6bed8693499b'
ITEMS=[[87593, '8363d9c7ceffd5f2cd3e98b08eb4a364ff7332ff59d7a77a235e15571dee3862', 3822954], [87662, '70ee0b40600173d7f1a6aaf5482e202fad1ee08109bc60e2907b3633f40c51eb', 4196330], [87672, '1d879209df02a76658b03f7be77c00d62833a66098c50fb3acf3f2b30be33af6', 3757851], [87720, '2d422c311ed2639387e04ff0df1b038b211d59a903238a9472fd967c170b1585', 3729971], [87792, '027ee040dadd354a254124546bdc66579ce969010fd105e825ca9ff38fc01a1e', 3981043], [87912, 'c1bffcf64e560828a3811686dc791c847f84b7c822842880161c048fb45e03c2', 3824974], [87956, '8df585f600ff8a2136d3eac0b2f87ccd0dd8262e0d1c7040763550b93a0ea3b7', 3563413], [88045, 'f58675485c91a6ee62fb7c1f46a0c487f897449c7c99c0def5839bd1ca829c30', 3717029], [88092, '066474e3cbdd5ebbe9936dbee69ac8d0cb06334728a53be9dbac9689dff1433f', 3953682], [88185, 'bc8954a2d66979a1ef654bdbaaa54511812170c8ff14961d04614772ca8c0237', 4534411], [88208, 'e99dda740e1c638b69ae2569d5edeba1736cba35cfcc904b3fa254bbfe648129', 3637087], [88265, '749053a10b2215d57d9c582d1543c3e94897eeb670a96a4b45b6167dc19c29dc', 3795091], [88269, 'f5b9f3294849fc895fdd96ed2d00ada8a4808b4a319541c4e63bee48f974293e', 3874743], [88270, '8604cdd7e3550a8dff81d7e117bf09d69efefe91c4daf21321a92a91f8076c61', 3645443], [88287, '9c27af3240044820c918ea4caabfd53c36fa0beadbabbedec7adb56e5e721dd8', 4098431], [88325, '18aa0740f0a2edc85e4618ec1ff9aae434c6a60113f2497ea1b5b6f6653ae2e7', 4332442], [88345, '4243b928abff49a88cb550bf8bfc6d7acfe333d07f227c3872fc280a193760df', 3666129], [88414, 'b5839b618ab766fc6c4285431dd69493a79d3294260c9536e35c8d2b30fe935a', 3755672], [88445, '493592c426dc6556a791ef7afd9f31c400a4e9b1920bf67671fed0a0965c4a6d', 3764174], [88606, '04867a14ea6ac52140c1d9648c4e82226aa3918ecac3ef8dfdcf3f7c9350a679', 3603781], [88695, 'd7567ace79c88796a6a53ff0910f661ccccab49c335719a4467b47cf1e45d4fe', 3636991], [88735, '4f386ad34b927ab854535a6f297ff163f47b7e6fa99aec1e7f26a42b6b5825a5', 3769603], [88757, '9aaf62866529051882ca10acb1a86e914a6063db1a59fb73b984cce4efc28464', 3850253], [88815, 'b158e19079e2062f02005aa234d8032732f019e879d5e2ba88571b77b4b146ee', 3722991], [88845, '2206f619792dbf51061936f3705f8deac2beac242cdfc94b5503bb28d8c3d593', 3847770], [88967, '0bee305e151a9690e49ab9cb682d9e007731cd38a24a76e531698f49174e5e3c', 3628896], [88970, 'b3d52fe4f70f12fab4dd0ef25f95332a9b1a80f493414219b47f5c6ab88ce8f0', 3690352], [89045, 'f8f833d8cab871161a850e40e7eb9099aabaccc7465f238a30e9a6822227c9eb', 3918119], [89062, '76067ed3a93ba81f9c6bba95d12800b2585e18da76300e7d0a87e80fed804ac1', 3826735], [89072, '21bae29cf177364809c60a7afc3d70c7b83ec8214c53ce3f706d8a1cb098cbb4', 3716834], [89078, '9dcbd8bf26b11e48c73b2f538bef2e8de0d7c0a48fd79f9008cb72b69cdf8bc0', 3862352], [89119, '615afb525cf59d08aaede3a63dd2a364e5c8a0978da024e6a8812e30a09bc6ad', 3696200]]
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
