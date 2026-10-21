"""Package the verified deck, speaker notes, editable diagrams and source figures."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile,xml.etree.ElementTree as ET
base=Path(__file__).resolve().parents[1];dest=base.parents[1]
src=base/'exports/dense_instance_thesis_20261014_182335.pptx'
deck=dest/'毕业论文汇报_密集检测分割与身份跟踪恢复_20261015_三模块版.pptx'
notes=dest/'毕业论文汇报_逐页讲稿_20261015_三模块版.md'
pack=dest/'毕业论文汇报_实验图与模型结构_20261015_三模块版.zip'
if deck.exists():
    prior=json.loads((base/'validation/three_module_delivery_20261015.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(deck.read_bytes()).hexdigest()==prior['sha256'],'Deck was externally edited; preserve it.'
shutil.copy2(src,deck);shutil.copy2(base/'notes/total.md',notes)
with zipfile.ZipFile(deck) as z:
    slides=sorted([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)],key=lambda n:int(re.search(r'(\d+)\.xml',n).group(1)))
    note_parts=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
    assert len(slides)==len(note_parts)==22
    a={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    texts=[];shape_count=0;picture_count=0
    for i,f in enumerate(slides,1):
        t=ET.fromstring(z.read(f));body='\n'.join(e.text or '' for e in t.findall('.//a:t',a));texts.append(f'--- Slide {i} ---\n'+body)
        shape_count+=len(t.findall('.//p:sp',a));picture_count+=len(t.findall('.//p:pic',a))
        if i in (16,17,18,19):assert '拟开展' in body
    assert '167,872' in texts[7] and '5.18%' in texts[13] and '0.20%' in texts[13]
    assert picture_count==0
(base/'validation/三模块版_投影文字核对.txt').write_text('\n\n'.join(texts),encoding='utf-8')
with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED) as z:
    old=base/'experiment_figures_20261015';new=base/'three_module_figures_20261015'
    for stem in ['01_失败实例划分','02_有限修正空间','03_自动修正探索','04_面积与损伤']:
        for ext in ['png','svg','pdf']:z.write(old/(stem+'.'+ext),'实验数据图/'+stem+'.'+ext)
    for ext in ['png','svg','pdf']:z.write(new/('06_选择性修正开发结果.'+ext),'实验数据图/05_选择性修正开发结果.'+ext)
    for p in old.glob('*.csv'):z.write(p,'数据/'+p.name)
    for p in new.glob('*.csv'):z.write(p,'数据/'+p.name)
    z.write(new/'chart_data.json','数据/chart_data_with_source_hashes.json')
    for p in ['07_基础模型.svg','08_修正器结构.svg','16_身份保持模型.svg','18_遮挡恢复模型.svg']:
        z.write(base/'svg_final'/p,'模型结构/'+p)
    for p in ['三模块修订依据_20261015.md','研究方法实验依据_20261015.md']:
        z.write(base/'sources'/p,'依据/'+p)
    z.writestr('README.md','# 三模块版图表与依据\n\n五组数据图均来自已完成 Run；图表使用标准绘图工具输出 PNG、SVG、PDF。模型结构 SVG 可编辑，前两张为已使用图像模型，后两张为拟研究的视频框架。\n\n当前状态以《三模块修订依据_20261015.md》为准；旧 E0—E4 依据保留其历史时点，面积策略未执行的旧表述已由 E5 完成结果更新。所有结果保留原评价分母和开发范围；没有视频成绩或全量 AP 增益。\n')
receipt={'deck':str(deck),'sha256':hashlib.sha256(deck.read_bytes()).hexdigest(),'notes':str(notes),'figure_package':str(pack),'slides':22,'notes_pages':22,'native_shapes':shape_count,'raster_pictures':picture_count,'rendered_with':'@oai/artifact-tool','visually_reviewed_changed_slides':[1,4,5,6,7,8,11,14,15,16,17,18,19,20,22],'postflight':'passed','scientific_runs_started':0}
(base/'validation/three_module_delivery_20261015.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
