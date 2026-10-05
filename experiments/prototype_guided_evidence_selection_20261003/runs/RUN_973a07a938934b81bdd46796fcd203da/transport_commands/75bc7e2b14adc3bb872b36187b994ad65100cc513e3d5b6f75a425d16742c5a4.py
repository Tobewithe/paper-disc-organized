ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='99d09b15fabb777c58d5075ba3791247edd5ff852455ef31f187b447273acd12'
ITEMS=[[74739, 'dec3017222d2690109d5906723fa49c480d9351296dcd099ef436d427921454f', 4057400], [74832, '3c5e7ab184d1f95d4b58f2bfa2019fb2bcf49c9bb89e9e5cd4bc8492b0407dce', 3684196], [74838, '54a4d880dfc10eae06785c15ea22294a1a6fd0f92af12b184e2f8032b92fab58', 3753969], [74840, 'e58fa65d02b7ec29f7d82d5bd4db46a9bb632bc54c55f20e3cf41bfd32cd1bae', 4995847], [74860, 'ea2044693adbaad3391a2e6e3a606bff4a707ed675bcb1a04806a4e7620da6b2', 3575230], [74885, '30bfa21b3ef2f3a4b049f5e9b6bd267d07d55391f3119218b21b95c2eeadc78e', 3614728], [75051, '36d8af66aeb7d71746c3b0bf87f74702977e014df17446e9d7869c1a1f870406', 3683349], [75052, 'c5e6735c494a9a8dfc2dc96e7419143948031b8ad6ae5d54dd80aadbf9ed1f49', 3994436], [75065, '9c5e3d0cb9d13c29745e44c646bf2d82495b5504c44fe8374aafdf18012b0cf3', 3962760], [75101, '256fbacdbfe4f4adfc5fa81456b523900e9cd403ceca4e8d7848d1b9d7ee7ab0', 3636073], [75133, '47809cd99537d225151220237ab107af590b55841f58267a761bdfa19e2fd35d', 4153463], [75144, '9e381413c9e11ceef075731ef36500315bc27ba6527a29163bff538803102340', 4375151], [75170, '54c71300a284daa8ab15ad272162355daadef5a1db86d4f663c39c00def583b8', 4122871], [75199, '115ee60cc6af66de77f232a9f0dd38d1192c1b69116edb74c42ed4c8cf9f530e', 3786560], [75227, 'd5b2cef51ba1accb8a46600e4f4e49e8207fb18fdad4392cfd0968678a578737', 3705417], [75270, '6dbae5f47c2df4bfb5d1a998348f9230dc11385a3135c934ba55e93df2df6b9a', 4483213], [75296, '53890c2cbc0b3412ba1c3ea455dad04d132b7c3e77ec9fc211832c0f361037da', 4387657], [75311, '1eaba70fd7f3b8293eecaaf3b1e93937c5391b98f2db834182dcfa11d462f7ed', 4200900], [75331, '64779501f0f6e457716e363099622a5916625b54518c698489c4afab89b49802', 3931573], [75361, '2dcae236969164d4d715806da6ca7ad22b14c07dc69a345c3dbf79487841f0c5', 3928551], [75393, '12480b02b8a9c8edfb9ff7c7cb1347de8e318cb296a7d1dd37b3a5ebff47855b', 3755195], [75451, 'e4db9c37f725362e3f53edb7f0b76064d6cc8b9a0d4b22a17f78018490e698e2', 3864598], [75456, '1abbcd910a0bebbbdacc55a324cb1f50fd0a99889c7dedda60424bea757ce502', 3602837], [75509, 'd107d399dc16552e276eba1bb6ac139fcc7e7f10137e4b0698e9b5ae931eb02a', 3895501], [75565, 'af0155a0e8f75a70519ad12acaa53d2d8bb9cb215fcc36b04ae1a7b22a0f4f99', 3824523], [75612, '5af7eb64318de75d2d7af8a06b289e64dcf1a84c68c3e48fd9088dd566ee9f45', 4400036], [75668, '2c9251948769d361421216dd0a9ea9c03edd4a709b4fbc32f73a4925689f9e8a', 4694128], [75691, '8312d139f4037e88f8b35d0785738b5fc56e50f442253fa53761f607a65ed741', 3621425], [75726, 'e8ae83d50bf2f70b71a9ed2f0e30476695dc8adde2d66f25ef5ebb97187f39e8', 4038880], [75748, '189b4893834e0e98a476d90196294ab4bc84298f2f96eb1f6299223a1b2e3b33', 3627631], [75786, '9ee847387aab594df3b10612c5f2facc94aca3eb37551b984189a1906caeba52', 3776489], [75829, '681ed3a78164eb78c6d3ce685af1b9f09f237bd812a21aa66d68ba1d873fb212', 3711786]]
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
