ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='4ceb2847f6d28453571c25f1cbebf7eff636d414442a325e26dd57201e1ed63a'
ITEMS=[[92014, '2e8290cb0f4354cd8fbf2fd8ca7383d008f90ed7eda3b7cc2cf901c849649650', 3924859], [92032, 'bf94d58de215075f61f804935de613fa43ca918c93c5f53aab099001630523ce', 3817073], [92041, '91a82a2b86ce358e00746dd58177b879a8b8e637223e2be51c1e1dbf3695f416', 3793011], [92050, '3b2dc23df6dc07711c3f4fc01c875b453ffaa8ba278c41303615713f6f0c4b32', 3779339], [92198, 'bf46003dc45f0feb7d431d29c5da9511fcb454e6b14dcedaf5fd09b1f9248724', 3959587], [92221, 'b72a2fbb1bd1292c6d6a39ca41f61fda0b21ed1a898368ab3bca77f266b869aa', 4218166], [92230, 'eb32a5b435ad47e9724d6856d7c37b056c6d6377edb7420d52af3bd0a7c1fd1c', 4463903], [92248, 'bc9c48ddcab5038b3825829b31084aa0765fd81dc5b66dcb2816136eab42712a', 3762594], [92301, '877075d4b6746f1a5004c8e53377d45f8cdae39399a7f5374a039737e5d9f141', 3806641], [92336, 'c7b39515fe54294be2cbecb147535026ce2be2679106d06e13f87962d0d71789', 3999727], [92346, 'b583f11fdb29582bd8004f907e9253202029f859e0d0528b171d2588835e0ecd', 3985699], [92416, '1025ee3662fcf1034e0b5cc6b3969aba4917938ea4fe69a6a0c8a2512ef638b3', 3991681], [92420, '2f2ea2e96c6e442a1afa1314faa3ffa4310f7fcd02141cefd5c7f63d32c1d5ca', 4587160], [92488, '523501d3236b6a26a7c984c5c4a56a1c566237a2e72b08c02d90dc032bb154fe', 3826591], [92529, '5a93990a8a8f0aa33244f87ea6c6c199ecd28fb96190360b2f16e4e956df8ec5', 3883756], [92558, 'd4e07f1219133829369e4b44c2fd3b708f3275a8e61b082f1f34a3de123510e3', 4001040], [92601, '57779c834dfd5579294087a1592b3e922a0e18808447ac6150163cd781138637', 3745819], [92602, 'a0cefab04c8657b134d439243b4e67633f5dd360ceea8701b8846b2d072f02a8', 3958210], [92604, '03031b4e31a1043049dbd3079afff96890fa6b761544eb890efe43394054cadd', 920278], [92614, 'bc914ec1e19eea049dd8d4f87ec922d8630e5aef43bba1ca50145729765affd3', 3630322], [92634, '78c46f5bebcb7a70ff8fbdf7268a1988d83a43feb719596bb90b721b585f58ab', 3659432], [92686, 'f981af077d5f31ac64609da068ba70b8947fbd2f0729aea72ef1cd16efeb75a2', 3812762], [92715, '606fff168e58982e555523fa0e59febc62cde865ed26102434ef719e688730ab', 3705180], [92799, '64f3af1c7c440a309737a2b97ddaa5e7aebb3dc8326794d134e646f2b973d564', 3703771], [92839, '36e04afce09c0323a95c6568331238a1d27a5bd18667f999e1d26258ec663eba', 3855474], [92914, 'b79f555f5fe909c02c825195ebd4b2f0725ab0287d66e8937f6df59f48fcf069', 3834602], [93176, 'eb5899accd97e7c559a07f5922bac6cf3fad104c08f003efc4a19d8918c0440c', 4075487], [93246, 'fbd79a49c98eba83db15327b77b11251df041b7e33a5dcb24cbbe9560981631a', 3989109], [93258, 'dcf302279d5295b6f772ba49a9a4ce9af297115f6a811901e8c0df1e497ce34a', 4226145], [93267, 'cd723ae65a3ca132d43acbba56baa475345e000bd8ece8bfb3301c718dca2fc0', 3913608], [93287, '4f0b16441dadd9cc5f3eb5554bdac77034dfa130d9ba1fbe4f3f057f32a6282b', 3828761], [93353, '1c90d253bcc8266dc0d808f08e61c0a444ab062d7fd62b7b1828246f5b2367c6', 3807602]]
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
