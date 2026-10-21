"""Mechanical revision of the approved deck roster; no slide artwork generation."""
from pathlib import Path
import re, shutil, json, hashlib, csv

base = Path(__file__).resolve().parents[1]
root = base.parents[3]
saved = base / 'backup/before_gt_candidate_taxonomy_20261015'
assert not saved.exists(), 'One-time roster revision; preserve the original snapshot.'
saved.mkdir()
for name in ['svg_output', 'svg_final', 'notes']:
    shutil.copytree(base/name, saved/name)
for name in ['design_spec.md', 'spec_lock.md']:
    shutil.copy2(base/name, saved/name)
old = {int(p.name[:2]): p for p in (saved/'svg_output').glob('*.svg')}
assert len(old) == 25
order = [1,2,3,4,5,None,None,None,7,None,None,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
names = {6:'GT候选判定',7:'原始候选GT分类',8:'类别正确GT分类',10:'像素错误分类',11:'拥挤条件分层'}
for n, previous in enumerate(order,1):
    if previous: names[n] = old[previous].stem.split('_',1)[1]
mapping = {previous:n for n,previous in enumerate(order,1) if previous}

spec = (saved/'design_spec.md').read_text(encoding='utf-8')
prefix, outline = spec.split('## IX. Content Outline\n',1)
outline, tail = outline.split('## X. Speaker Notes Requirements',1)
blocks = {int(m.group(1)):m.group(0).strip() for m in re.finditer(r'### Slide (\d+) - .*?(?=\n### Slide |\Z)',outline,re.S)}
prefix = prefix.replace('| Page Count | 25 |','| Page Count | 27 |')
prefix = prefix.replace('18—22 分钟','20—24 分钟').replace('| AI Image Acquisition Path | not applicable |','| AI Image Acquisition Path | built-in image_gen |')
prefix = prefix.replace('x=64..1216, y=175..640；页脚 y=665..704','x=64..1216, y=170..660；不设页底小字和页码')
prefix = re.sub(r'P(\d\d)',lambda m:f'P{mapping.get(int(m.group(1)),int(m.group(1))):02}',prefix)
prefix = re.sub(r'- \*\*Revision authority\*\*:.*','- **Revision authority**: 用户撤下不成熟的实例关系字母分类，加入全raw与argmax-correct两套GT五分类；保留真实实验和视频设想，概念公式采用示意图，所有页底小字转入讲稿。',prefix)
prefix = prefix.replace('| P09 | failure-split', '| P07 | raw-gt-states | horizontal_bar_chart | 20260916全raw汇总；36,335普通GT |\n| P08 | argmax-gt-states | horizontal_bar_chart | 20261005重分类；36,335普通GT |\n| P09 | failure-split')
images = '''## VIII. Image Resource List

AI示意图用于解释概念；真实实验数据图使用可编辑形状，不由生成模型绘制数值。

| Filename | Dimensions | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gt_candidate_identity_v1.png | 2304x1024 | P06解释IoU与同一候选联合达标 | Infographic | 完整横向概念图，原生标题在图外 | no-crop | ai | Pending | sources/gt_candidate_image_prompts.json |
| pixel_error_regions_v3.png | 2304x1024 | P10解释自身遗漏及互斥误报区域 | Infographic | 五个区域示意与统一分母 | no-crop | ai | Pending | sources/gt_candidate_image_prompts.json |
| crowding_definitions_v3.png | 2304x1024 | P11区分框交叠与边界接近度 | Infographic | 两栏几何概念与公式 | no-crop | ai | Pending | sources/gt_candidate_image_prompts.json |

'''
prefix = prefix.split('## VIII. Image Resource List')[0] + images

def block(n,title,content,composition,chart=None):
    result=f'### Slide {n:02} - {names[n]}\n- **Title**: {title}\n- **Audience move**: {content}\n- **Core message**: {content}\n- **Relationships**: none\n- **Topology decision**: no\n- **Composition**: {composition}\n- **Content**: {content}\n'
    if chart:
        result += f'- **Visualization**: {chart}=chart/horizontal_bar_chart; 675,245,1045,595; 0..25000\n- **Native-ready**: {chart}=no\n- **Data class**: observed\n'
    return result
new = {
6:block(6,'GT 分类：判断模型是否产生了合格候选','以真实标注实例为单位检查候选池。IoU=交集面积/并集面积，Box75和Mask75分别判断。好框与好掩码存在但不在同一raw，不能计为联合达标。','生成的概念示意图占主体；无页脚'),
7:block(7,'原始候选能力：每个 GT 划入一个状态','全部raw，不以类别或分数过滤；COCO 5000图36335普通GT。联合23805、仅框6688、仅掩码487、分别达标531、均不足4824。五类互斥穷尽，逐GT存在性不是最终输出召回。','左侧五类定义，右侧同零基线计数条形图，下方两句解释研究用途','raw-gt-states'),
8:block(8,'类别正确后，进一步区分 GT 失败位置','每个GT只看argmax类别正确的raw，仍以全部36335GT为分母。无语义44、框失败6335、掩码失败6459、不同候选484、同raw双达标23013。类别条件不附带conf或TopK。','五行类别、简明判据与计数条形图；明确6459是候选诊断对象','argmax-gt-states'),
10:block(10,'像素错误：区分自身遗漏与多分区域','固定同一预测；自身遗漏、同类独占、异类独占、邻居歧义、背景误报统一除以有效自身GT面积。四类误报互斥，自身遗漏另列；不能以收缩全部预测冒充个体区分改善。','生成的五区域科学示意与面积比公式；无页脚'),
11:block(11,'拥挤条件：分别记录框交叠与边界接近','ICI是同类GT框累计交叠面积/目标框面积，可大于1；E4是4模型输入等效像素范围内接近同类邻居的有效目标边界比例。分层记录与失败分类分开。','生成的双栏几何示意与公式；不展示实例关系字母分类'),
}
new_blocks=[]
for n,previous in enumerate(order,1):
    if previous:
        new_blocks.append(re.sub(r'^### Slide \d+',f'### Slide {n:02}',blocks[previous]))
    else: new_blocks.append(new[n])
tail=tail.replace('P05—P22','P05—P24').replace('18—22分钟','20—24分钟')
tail=tail.replace('sources/失败分类补充依据_20261015.md、','sources/GT候选分类修订依据_20261015.md、')
(base/'design_spec.md').write_text(prefix+'## IX. Content Outline\n\n'+'\n\n'.join(new_blocks)+'\n\n## X. Speaker Notes Requirements'+tail,encoding='utf-8')
lock=(saved/'spec_lock.md').read_text(encoding='utf-8')
rhythms={int(n):v for n,v in re.findall(r'- P(\d+): (\w+)',lock)}
rhythm='## page_rhythm\n'+''.join(f'- P{n:02}: {rhythms[p] if p else "dense"}\n' for n,p in enumerate(order,1))+'\n'
lock=re.sub(r'## page_rhythm.*?(?=## pptx_structure)',rhythm,lock,flags=re.S)
lock += '\n## images\n- p06: images/gt_candidate_identity_v1.png | source=ai | crop=no-crop\n- p10: images/pixel_error_regions_v3.png | source=ai | crop=no-crop\n- p11: images/crowding_definitions_v3.png | source=ai | crop=no-crop\n'
(base/'spec_lock.md').write_text(lock,encoding='utf-8')

# Only verified local leaf files, with the exact source preserved in the snapshot.
for folder in ['svg_output','svg_final','notes']:
    directory=(base/folder).resolve()
    assert directory.parent==base.resolve()
    for path in directory.iterdir():
        if path.is_file() and path.suffix in {'.svg','.md'}:
            assert path.resolve().parent==directory
            path.unlink()
for n,previous in enumerate(order,1):
    if previous:
        svg=old[previous].read_text(encoding='utf-8')
        svg=re.sub(r'<g id="footer"\b.*?</g>\s*','',svg,flags=re.S)
        (base/'svg_output'/f'{n:02}_{names[n]}.svg').write_text(svg,encoding='utf-8')
notes=(saved/'notes/total.md').read_text(encoding='utf-8')
old_notes={int(m.group(1)):m.group(0).strip().removesuffix('---').strip() for m in re.finditer(r'^# (\d+)_.+?(?=^# \d+_|\Z)',notes,re.S|re.M)}
note_blocks=[]
for n,previous in enumerate(order,1):
    if previous:
        note=re.sub(r'^# \d+_',f'# {n:02}_',old_notes[previous])
        # Preserve removed source/qualification text in notes, not on the projected slide.
        footer=re.search(r'<g id="footer"\b.*?</g>',old[previous].read_text(encoding='utf-8'),re.S)
        if footer:
            lines=re.findall(r'<text[^>]*>(.*?)</text>',footer.group(0),re.S)
            source=' '.join(re.sub(r'<[^>]+>','',s) for s in lines if not re.search(r'\d+ / 25',s))
            if source: note+='\n\n'+source
        note_blocks.append(note)
    else: note_blocks.append(f'# {n:02}_{names[n]}')
(base/'notes/total.md').write_text('\n\n---\n\n'.join(note_blocks)+'\n',encoding='utf-8')

raw_path='experiments/coco_raw_capability_5000_20260916/runs/RUN_a7c840bb695d45b4b66bc346715ca1aa/SUMMARY.json'
arg_path='experiments/coco_raw_argmax_conditioned_20261005/runs/RUN_83e9f6f1f0de4d969868fd9a04321786/SUMMARY.json'
raw=json.loads((root/raw_path).read_text(encoding='utf-8-sig'))
arg=json.loads((root/arg_path).read_text(encoding='utf-8-sig'))
raw_keys=['joint_good','box_good_mask_unavailable','mask_good_box_unavailable','separate_good_no_joint','neither_good']
arg_keys=['Semantic fail','Box fail','Mask fail','Box/Mask misaligned','Success']
raw_labels=['同一候选双达标','仅有达标框','仅有达标掩码','分别达标但不在同一候选','框与掩码均无达标候选']
arg_labels=['无类别正确候选','类别正确但无达标框','类别与框达标但无达标掩码','分别达标但不在同一候选','同一候选类别正确且双达标']
data={'denominator':36335,'images':5000,'threshold_box':0.75,'threshold_mask':0.75,
      'raw':[{'key':k,'label':label,'count':raw['by_geometry'][k]['gt']} for k,label in zip(raw_keys,raw_labels)],
      'argmax':[{'key':k,'label':label,'count':arg['counts'][k]} for k,label in zip(arg_keys,arg_labels)],
      'sources':[{'path':p,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()} for p in [raw_path,arg_path]],
      'caution':'Raw existence and argmax-correct raw existence are not final-output task recall; neither is AP.'}
for group in ['raw','argmax']:
    assert sum(r['count'] for r in data[group])==36335
    for r in data[group]: r['percent']=100*r['count']/36335
out=base/'gt_candidate_figures_20261015';out.mkdir(exist_ok=True)
(out/'chart_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for group in ['raw','argmax']:
    with (out/(group+'_gt_classes.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['key','label','count','percent']);writer.writeheader();writer.writerows(data[group])
(base/'sources/gt_candidate_roster.json').write_text(json.dumps({'order':order,'names':names,'backup':str(saved)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('27-slide roster prepared; prior deck backed up; footers removed from retained slides; raw and argmax data extracted from completed Run summaries.')
