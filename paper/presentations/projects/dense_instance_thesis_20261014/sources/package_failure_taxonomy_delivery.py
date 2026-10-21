"""Package the visually checked 25-page revision, retaining earlier deliveries."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile,xml.etree.ElementTree as ET

base=Path(__file__).resolve().parents[1]
dest=base.parents[1]
src=base/'exports/dense_instance_thesis_20261014_184520.pptx'
stem='毕业论文汇报_密集检测分割与身份跟踪恢复_20261015_失败分类补充版'
deck=dest/(stem+'.pptx')
notes=dest/'毕业论文汇报_逐页讲稿_20261015_失败分类补充版.md'
pack=dest/'毕业论文汇报_实验图与分类依据_20261015_失败分类补充版.zip'
assert src.exists()
assert not deck.exists(),'Preserve an existing delivery; inspect before replacing.'
reviewed=[6,7,8,9]
render_dir=base/'qa'/src.stem
assert all((render_dir/f'slide-{i:02}.png').exists() for i in range(1,26))
with zipfile.ZipFile(src) as z:
    parts=sorted([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)],key=lambda n:int(re.search(r'(\d+)\.xml',n).group(1)))
    note_parts=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
    assert len(parts)==len(note_parts)==25
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    texts=[];shapes=pictures=0
    for i,p in enumerate(parts,1):
        t=ET.fromstring(z.read(p))
        body='\n'.join(e.text or '' for e in t.findall('.//a:t',ns))
        texts.append(body)
        shapes+=len(t.findall('.//p:sp',ns));pictures+=len(t.findall('.//p:pic',ns))
        assert f'{i:02} / 25' in body
        if i in (19,20,21,22): assert '拟开展' in body
    assert '框支持率' in texts[5] and '6,229' in texts[6]
    assert '邻居歧义误分率' in texts[7] and 'E(4)' in texts[8]
    assert '167,872' in texts[11] and '5.18%' in texts[16] and '0.20%' in texts[16]
    assert pictures==0
(base/'validation/失败分类补充版_投影文字核对.txt').write_text('\n\n'.join(f'--- Slide {i} ---\n{t}' for i,t in enumerate(texts,1)),encoding='utf-8')
shutil.copy2(src,deck)
shutil.copy2(base/'notes/total.md',notes)
with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED) as z:
    old=base/'experiment_figures_20261015';new=base/'three_module_figures_20261015'
    for name in ['01_失败实例划分','02_有限修正空间','03_自动修正探索','04_面积与损伤']:
        for ext in ['png','svg','pdf']: z.write(old/(name+'.'+ext),'实验数据图/'+name+'.'+ext)
    for ext in ['png','svg','pdf']: z.write(new/('06_选择性修正开发结果.'+ext),'实验数据图/05_选择性修正开发结果.'+ext)
    for p in list(old.glob('*.csv'))+list(new.glob('*.csv')):z.write(p,'数据/'+p.name)
    z.write(new/'chart_data.json','数据/chart_data_with_source_hashes.json')
    z.write(base/'sources/failure_taxonomy_source_manifest.json','数据/failure_taxonomy_source_manifest.json')
    for name in ['11_基础模型.svg','12_修正器结构.svg','19_身份保持模型.svg','21_遮挡恢复模型.svg']:
        z.write(base/'svg_final'/name,'模型结构/'+name)
    for name in ['06_失败状态分类.svg','08_像素错误分类.svg','09_关系与拥挤分层.svg']:
        z.write(base/'svg_final'/name,'分类指标/'+name)
    for name in ['失败分类补充依据_20261015.md','三模块修订依据_20261015.md','研究方法实验依据_20261015.md']:
        z.write(base/'sources'/name,'依据/'+name)
    z.writestr('README.md','# 失败分类补充版\n\n本版为25页。分类定义和用途见《失败分类补充依据_20261015.md》；前版页码以本版PPT为准。图像E0—E5实验沿用真实Run，模块二、三仍为设想。新增分类页未虚构O2O细分类数量。\n\n五组实验图来自原数据，模型结构与分类页SVG可编辑；所有结果保留原分母和开发范围。历史raw、O2M和当前O2O结果不混用。\n')
receipt={'deck':str(deck),'sha256':hashlib.sha256(deck.read_bytes()).hexdigest(),'notes':str(notes),'figure_package':str(pack),'slides':25,'notes_pages':25,'native_shapes':shapes,'raster_pictures':pictures,'rendered_with':'@oai/artifact-tool','visually_reviewed_changed_slides':reviewed,'postflight':'passed','svg_errors':0,'svg_warnings':0,'data_charts_retained':6,'scientific_runs_started':0}
(base/'validation/failure_taxonomy_delivery_20261015.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
