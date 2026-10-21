"""Verify plot marks and source hashes, without running models or evaluation."""
from pathlib import Path
import hashlib,json,re,xml.etree.ElementTree as ET
base=Path(__file__).resolve().parents[1];root=base.parents[3]
data=json.loads((base/'three_module_figures_20261015/chart_data.json').read_text(encoding='utf-8'))
for src in data['source_hashes']:
    assert hashlib.sha256((root/src['path']).read_bytes()).hexdigest()==src['sha256'],src['path']
ns={'s':'http://www.w3.org/2000/svg'}
def group(file,key):return ET.parse(base/'svg_output'/file).find(f".//s:g[@id='{key}-chartArea']",ns)
receipts=[]
def near(a,b):assert abs(float(a)-b)<.001,(a,b)
for file,key,source,origin,scale,field in [
('09_失败定位.svg','failure-split','baseline',310,540/8000,'n'),
('10_修正空间.svg','oracle-area','oracle',230,580/100,'percent'),
('13_面积分层.svg','area-damage','damage_by_native_area',250,750/16,'percent')]:
    bars=group(file,key).findall('s:rect',ns);assert len(bars)==len(data[source])
    for b,d in zip(bars,data[source]):near(b.attrib['x'],origin);near(b.attrib['width'],d[field]*scale)
    receipts.append({'object':key,'file':file,'result':'passed'})
g=group('12_自动修正探索.svg','pilot-effects')
for c,d in zip(g.findall('s:circle',ns),data['pilot']):near(c.attrib['cx'],420+(d['mean_pp']+4)*740/6)
for p,d in zip(g.findall('.//s:path',ns),data['pilot']):
    a=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',p.attrib['d'])]
    near(a[0],420+(d['low_pp']+4)*740/6);near(a[2],420+(d['high_pp']+4)*740/6)
receipts.append({'object':'pilot-effects','file':'12_自动修正探索.svg','result':'passed'})
g=group('14_选择性修正结果.svg','selective-effects');order=[data['selective_effects'][i] for i in [0,2,1,3]]
for c,d in zip(g.findall('s:circle',ns),order):near(c.attrib['cx'],396+d['mean_pp']*172)
by_y={float(l.attrib['y1']):l for l in g.findall('.//s:line',ns) if l.attrib['y1']==l.attrib['y2'] and l.attrib['y1']!='510'}
for y,d in zip([275,335,421,481],order):
    l=by_y[y];near(l.attrib['x1'],396+d['low_pp']*172);near(l.attrib['x2'],396+d['high_pp']*172)
receipts.append({'object':'selective-effects','file':'14_选择性修正结果.svg','result':'passed'})
g=group('14_选择性修正结果.svg','selective-damage')
for i,(b,d) in enumerate(zip(g.findall('s:rect',ns),data['selective_damage'])):
    near(b.attrib['height'],d['percent']*45);near(b.attrib['y'],510-d['percent']*45);near(b.attrib['x'],911.25+i*162.5)
receipts.append({'object':'selective-damage','file':'14_选择性修正结果.svg','result':'passed'})
assert len(receipts)==6
svgs=sorted((base/'svg_output').glob('*.svg'));assert len(svgs)==22
for i in [16,17,18,19]:assert '拟开展' in svgs[i-1].read_text(encoding='utf-8')
notes=(base/'notes/total.md').read_text(encoding='utf-8');assert len(re.findall(r'^# \d\d_',notes,re.M))==22
result={'source_files_verified':len(data['source_hashes']),'source_hashes':'all_match','geometry_max_rounding_px':.001,'chart_objects':receipts,'slides':22,'notes':22,'video_status':'proposed_only'}
(base/'validation/three_module_evidence_20261015.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
