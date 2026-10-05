ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
KEY='aab9d41b44356de6229800e9835a1a7a72e2aa150746b03df5b4860898ac9df0'
ITEMS=[[89132, 'ed8d704eedbbc2fde5386ca0e3536d940d838bdb8ec65a37741f6d6036875716', 3903464], [89225, 'ff4035e13168bad3690c1ef20df26fd3752b23020d6e1097f21598aa9de7084b', 4165519], [89296, '4556615156e5c40400ac351ab232b1a6ba80ca61a9558e59300341fd609d0cf8', 3965427], [89462, '88f7e70bc0b8aa070a21e55e33e2c6172cec400bb695b6901bac8f48af2a0907', 3802036], [89629, '65fc6bfd5fecf5a6cd9b5a5163d15872c5c4da1ad1a442085259ef7697e3a05f', 4331040], [89648, '41b05034c8e768b0ce29ccf396d7cf6622477aeea194ec092c3a3899f8da7434', 4965996], [89670, '386ae7305335403fd38f5e4ae7c9518070405c74d03cf6141ac8481169fdd0c1', 3657604], [89773, '8b4dab009d22c512cee2cb4259609c223a4906df2c317876a62ed240ecf5648c', 3841539], [89790, '3ee238e267d404e5704e1416bb01313e3b58ce83f3f6e3d6cc17c8b36f10f067', 4373420], [89792, 'b7b57029118f7296603aa06c51fcdfa22db61caeaa661f3c6d34a2c94bdbfb3f', 4205782], [89804, '61d20a682dcb80613f599a202e8e9adc0a8f79bd5e615a9753b09717df2f4f92', 3786187], [89861, '397a5cbffc2bd670f8a9c6bf9d3e87e5d2af5a8ffd57cd8b7673dedc276e9335', 3813791], [89870, '4cabf012dfd0cd77b6f6cc3a5e01f8f6074459a56e359460166552a2efb674b1', 3747258], [89884, '429751ddb9dcf592b20d1f6ca65d521caf08c630db2eb6ac463830ada4492e45', 3733557], [89894, '4b1cc10cdcbf83018f1a6abc9a6fe0db24a49533bee16ff502bf9f4e861a1245', 3760393], [89949, '68d72e02f4caa8602b46ff64c63cfb2f3df3d53c1f002da723f20db005882d74', 3969669], [90067, 'ab307a5cbc2e3c498c35ce28d6e2fbc6214dd2b632a9c1bc7a202578ea880d05', 3983899], [90138, '5e92f1784805639cbfa65ed046c8c34d5ecd69d1b7d71c604375053533fced27', 3857185], [90155, '715c01e8edb63255778f58be01d73a97492c142771949c4cc7433e6c7b87974a', 3703810], [90194, 'b8deffeacb03a90ffabae5a866e929593e7281c4665120810d85b315796aebbe', 3740840], [90208, '77f5dad8baf29571418b3cdb7c634ce2b58b07a7df4cc2429226745c8915f533', 3866510], [90237, 'a1bfcc6630ae528539ec41412f312c41f8f96444a391484219f074599283e6b6', 4472462], [90274, 'fb7ecd811fe40b481c992842ea51dcd435f2fdd6f21fc152d0de9ef743b96e1d', 3548584], [90278, '3126dca2f4e19ef7009efe0bee05a5451be43fa3abc6c2cf1d73045adb4e29e8', 4286285], [90311, '65a04dea00cc4e0b8985eff111407bb3870d7404672c6fc7c56593cc44fb7d12', 3813335], [90387, '9cfe644ca4ddc54b0966310f0c7f661e92ee50965b7671a8ffe4b25f5463806e', 3726392], [90523, 'e4ff4d45de41f353aa46f0f76371013ec1d49fd97aeee5fe21525469c601bb2a', 3681328], [90675, 'e49f46b7cb3cfb6fa442ca61e54a6132b1810077361f378f1dae9bbd6ed2513b', 3884037], [90754, '8400c3fedbfc4cbe6a515dbae5415b1fd85bafa37fdf42b4fe5496a8b91593dc', 3954878], [90818, '9ee417b19ec6ad53a3683957d9068552f1c00ef9f917d3af5e975ab4b7b181ab', 3897529], [90820, '678bb021a252d180749dd555aec886f06ee797c07829a0263db8b484227f3705', 3571049], [90888, '1902cd53a72360d31f96e3f5d87af35017cd42e51405534074d6dbebe1a1fe51', 3814356]]
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
