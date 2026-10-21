"""Check data, identities, unchanged charts, and the approved presentation scope."""
from pathlib import Path
import hashlib,json,re,xml.etree.ElementTree as ET

base=Path(__file__).resolve().parents[1]
root=base.parents[3]
saved=base/'backup/before_gt_candidate_taxonomy_20261015'
svgs=sorted((base/'svg_output').glob('*.svg'))
assert len(svgs)==27
notes=(base/'notes/total.md').read_text(encoding='utf-8')
assert re.findall(r'^# (\d\d_.+)$',notes,re.M)==[p.stem for p in svgs]
all_text=notes+'\n'+'\n'.join(p.read_text(encoding='utf-8') for p in svgs)
for removed in ['C / I / L / S','C/I/L/S','O/M/X','MISS','candidate-relations','id="footer"']:
    assert removed not in all_text,removed
for n in [21,22,23,24]: assert '拟开展' in svgs[n-1].read_text(encoding='utf-8')

ns={'s':'http://www.w3.org/2000/svg'}
old={int(p.name[:2]):p for p in (saved/'svg_output').glob('*.svg')}
chart_rows=[]
for previous,n,key in [(7,9,'failure-split'),(13,15,'oracle-area'),(15,17,'pilot-effects'),(16,18,'area-damage'),(17,19,'selective-effects'),(17,19,'selective-damage')]:
    def group(path): return ET.tostring(ET.parse(path).find(f".//s:g[@id='{key}']",ns))
    assert group(old[previous])==group(svgs[n-1]),key
    chart_rows.append({'page':n,'object':key,'geometry':'identical to previously verified source chart'})
data=json.loads((base/'gt_candidate_figures_20261015/chart_data.json').read_text(encoding='utf-8'))
for source in data['sources']:
    assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256']
for n,name,key in [(7,'raw','raw-gt-states'),(8,'argmax','argmax-gt-states')]:
    assert sum(r['count'] for r in data[name])==36335
    tree=ET.parse(svgs[n-1]); area=tree.find(f".//s:g[@id='{key}-chartArea']",ns)
    bars=area.findall('s:rect',ns);assert len(bars)==5
    txt=' '.join(tree.getroot().itertext())
    for i,(bar,row) in enumerate(zip(bars,data[name])):
        assert abs(float(bar.attrib['x'])-675)<1e-6
        assert abs(float(bar.attrib['width'])-370*row['count']/25000)<1e-6
        assert abs(float(bar.attrib['y'])-(268+70*i))<1e-6
        assert abs(float(bar.attrib['height'])-24)<1e-6
        assert f"{row['count']:,}" in txt
        assert f"{row['percent']:.2f}%" in txt
    chart_rows.append({'page':n,'object':key,'scale':'0..25000 from ticks','geometry':'calculator and independent source-value assertions agree','category_slot':70,'bar_height':24})
previous_data=json.loads((base/'three_module_figures_20261015/chart_data.json').read_text(encoding='utf-8'))
for source in previous_data['source_hashes']:
    assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
retained=[]
roster=json.loads((base/'sources/gt_candidate_roster.json').read_text(encoding='utf-8'))
for n,previous in enumerate(roster['order'],1):
    if previous and n!=9:
        expected=re.sub(r'<g id="footer"[^>]*>.*?</g>\s*','',old[previous].read_text(encoding='utf-8'),flags=re.S)
        assert expected==svgs[n-1].read_text(encoding='utf-8'),n
        retained.append(n)
images=json.loads((base/'sources/gt_candidate_image_manifest.json').read_text(encoding='utf-8'))
for im in images:
    assert hashlib.sha256((base/im['file']).read_bytes()).hexdigest()==im['sha256']
receipt={'slides':27,'notes':27,'all_footers_removed':True,'rejected_relation_classes_absent':True,'source_hashes_verified':len(data['sources'])+len(previous_data['source_hashes']),'charts':chart_rows,'retained_pages_unchanged_except_footer':retained,'concept_images':images,'scientific_runs_started':0}
(base/'validation/gt_candidate_evidence_20261015.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'PASS: 27 slides and notes; 8 charts verified; {len(retained)} pages unchanged except removed footers; 3 concept assets; rejected taxonomy absent; no scientific run.')
