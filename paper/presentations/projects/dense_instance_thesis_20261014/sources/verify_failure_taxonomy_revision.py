"""Validate presentation evidence and retained artwork; no model execution."""
from pathlib import Path
import hashlib,json,re,xml.etree.ElementTree as ET

base=Path(__file__).resolve().parents[1]
root=base.parents[3]
saved=base/'backup/before_failure_taxonomy_20261015'
data=json.loads((base/'three_module_figures_20261015/chart_data.json').read_text(encoding='utf-8'))
for src in data['source_hashes']:
    assert hashlib.sha256((root/src['path']).read_bytes()).hexdigest()==src['sha256'],src['path']

svgs=sorted((base/'svg_output').glob('*.svg'))
assert len(svgs)==25
notes=(base/'notes/total.md').read_text(encoding='utf-8')
assert re.findall(r'^# (\d\d_.+)$',notes,re.M)==[p.stem for p in svgs]
for i,p in enumerate(svgs,1):
    assert f'{i:02} / 25' in p.read_text(encoding='utf-8')
for i in [19,20,21,22]:
    assert '拟开展' in svgs[i-1].read_text(encoding='utf-8')

order=[1,2,3,4,5,None,9,None,None,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
old_files={int(p.name[:2]):p for p in (saved/'svg_output').glob('*.svg')}
unchanged=[]
for n,old in enumerate(order,1):
    if old and n!=7:
        expected=re.sub(r'\d\d / 22',f'{n:02} / 25',old_files[old].read_text(encoding='utf-8'))
        assert expected==svgs[n-1].read_text(encoding='utf-8'),svgs[n-1].name
        unchanged.append(n)

ns={'s':'http://www.w3.org/2000/svg'}
charts=[]
for old,new,key in [(9,7,'failure-split'),(10,13,'oracle-area'),(12,15,'pilot-effects'),(13,16,'area-damage'),(14,17,'selective-effects'),(14,17,'selective-damage')]:
    def scope(path):
        return ET.parse(path).find(f".//s:g[@id='{key}']",ns)
    assert ET.tostring(scope(old_files[old]))==ET.tostring(scope(svgs[new-1])),key
    charts.append({'key':key,'slide':new,'result':'identical_to_verified_previous_chart'})

sources=[
'experiments/failure_dimension_analysis_20260913/REPORT.md',
'project/research_notes/refinements/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md',
'project/research_notes/refinements/coco-structure/MASK_ERROR_AP_RESULTS_20260913.md',
'project/research_notes/refinements/coco-structure/COCO_RELATION_TAXONOMY_RESULTS_20260913.md',
'project/research_notes/refinements/coco-evaluation/COCO_CROWDING_SUPERVISION_SPEC_20260912.md',
'project/科研项目通用规范/指标最小化与TAL分类.md',
'experiments/coco_raw_argmax_conditioned_20261005/REPORT.md',
]
manifest=[{'path':s,'sha256':hashlib.sha256((root/s).read_bytes()).hexdigest()} for s in sources]
(base/'sources/failure_taxonomy_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result={'slides':25,'notes':25,'retained_artwork_unchanged_except_footer':unchanged,'experimental_source_hashes_verified':len(data['source_hashes']),'chart_objects':charts,'taxonomy_definition_sources':manifest,'new_numeric_experimental_results':False,'video_status':'proposed_only','scientific_runs_started':0}
(base/'validation/failure_taxonomy_evidence_20261015.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'PASS: 25 slides and notes; {len(unchanged)} retained pages unchanged except footers; 6 chart objects unchanged; {len(data["source_hashes"])} experimental source hashes verified; 7 taxonomy sources recorded.')
