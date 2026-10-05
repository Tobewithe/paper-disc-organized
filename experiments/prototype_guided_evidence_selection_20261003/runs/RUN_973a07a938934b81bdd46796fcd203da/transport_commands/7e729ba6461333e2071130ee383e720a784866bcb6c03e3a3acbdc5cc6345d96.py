ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='38deced3538b863ebcabd992884c80ef74e0ec1622947fa60f50533008c83dc5'
ITEMS=[[85981, '8948a23c1006414ca3d1b7cb81e00900e4212a9aa6faf4609a0f557cbd04e074', 3857935], [86121, 'ad0feff7e3106aff6f26d3c7398da79b214e1efe82ef12276b0404b53d9d52a5', 4052850], [86136, '832eb86cb93d4e534a37071a488e7f2a61db8e6b146e06ca3af5ed5d62bae77e', 3784461], [86183, 'a3e5c069f4df9f2714b06ae3407299ff743a459bb15ba1e33ec81c7ac38a51a0', 3869046], [86217, 'e6e0cd00661bdb86029e4ccfefc9ba8d8501e3e147bde8cf585a61fa3a92d661', 3841756], [86282, '7afd92775e5a92b102a398f0c073eb4166ff8001bda1df363386982048545181', 3996537], [86357, '60adaca505a67485069eb120353b7dfdb82e9bd123e04e5e7e3f25c14c53a6a9', 3433823], [86429, '362a41328391eeb8ad7ea85479083453fece03af1f6228b8e5eb7deb123c00b9', 4353847], [86514, '1bb999cb8583ebd2cc501f50dfa492d71dc98aeda3e8059eab651736209b3a76', 3844752], [86524, '8072f8200f0b410b57d3561be4a2ef3c181c4855cd858cbea48ecd677f84e102', 3996093], [86552, '987ab5d27fafb5d47637499c4228823ec769e595a57babb8b36a43160127a9b0', 4100114], [86555, '9459d860a46400fea38621830dfe2610452b7b6e9f08a48145579ec15e9deb98', 4282948], [86582, 'a1f9ae493dab2bfd1860900888e86e4a000383c55c4243153ce4b819dfd019c8', 3930528], [86630, 'f8616630cc322d4efd988363e95b6bac03661d2420993a5d283091532d017606', 3772044], [86677, '5333d0baf6449bfc4632005d8fe79e8fef5abf1b56958c6191b1ffc5fe83302b', 3802348], [86755, 'a92abda9eb864a269d89d8fdf0a8da38d22d82a708c501791aca832bedaf6d76', 3850083], [86836, 'e4212e7c039a727eabf6cb0257046e36b09522420e90dcb0b3a05d85477351c3', 508363], [86932, '306cd772a120664b4916be8724daf1438b3616b771fe26f05620da9a3beb9e05', 4012620], [86956, '4ea0c7badfa3854d1881536e9acd4af777089d99837d312b86cc097ed0add2ff', 3898485], [87027, '331ac46c69c0217c0a2888a372a092a3349bc7f68c694c7ffb05009cf7cdab4b', 4114251], [87070, '0d9c39514eef9cd91483e4512e40fb0321162dd198b54f374bd7cf9d3539e295', 3766527], [87078, 'f00e00a2728d7805b8e841ac1f90279919d6e843ce85c087d3b518a41c4e2714', 3696310], [87105, 'bcc0f265048a1967c25a6d68beda8d28d4cc3c94d7f4445d15e3b4f689e6f0c1', 3829564], [87285, '67624d3722c12a9d601ec9d056dd3e4bd02d494761b09cd2156428abeb7156d2', 3788889], [87308, '76bc05c1b050826901000713e8179d2847e6853638f713d7c0e386e62b0bec67', 3758134], [87348, '648cbb561cfb9e3ced553602dbaaf07d875d064c6cb4a39b8790a8b04c60e9da', 3613726], [87382, '5bfb655a91c40ee9c8202abdcf54c113ae736358f5232e2141abe287fba1bd2f', 3881736], [87470, 'f4262f27af1ac2993c45371234581c4057fd072d55f2871221a1fc59e0998bb3', 4417728], [87476, '865aceabac6219a4a7944883fd4a12a69965e6bb2dc7929443530f91ae91afa4', 4120484], [87522, '8f8a454bc02cc7554c7d344f6e33721e145fa7865c0a5642b4450db6f7081083', 3907947], [87555, '136d63ee89d3026e57c932dd5b591305339fe4b3cd7e154622a76ccc603aefc7', 3741632], [87569, 'b7f9d2fda8649c0e1ac1447d576e1044ac6fb70c46e46ff596b77ff841a5e959', 3999079]]
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
