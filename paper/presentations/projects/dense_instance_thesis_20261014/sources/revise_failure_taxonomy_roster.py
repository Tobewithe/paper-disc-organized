"""Mechanical roster revision only; new slide artwork is authored separately."""
from pathlib import Path
import re

base = Path(__file__).resolve().parents[1]
saved = base / 'backup/before_failure_taxonomy_20261015'
assert saved.is_dir()
# Retained source page numbers in their new positions; None means a new page.
order = [1,2,3,4,5,None,9,None,None,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
old_files = {int(p.name[:2]):p for p in (saved/'svg_output').glob('*.svg')}
new_names = {6:'失败状态分类',8:'像素错误分类',9:'关系与拥挤分层'}
for n,old in enumerate(order,1):
    if old is not None:
        new_names[n] = old_files[old].stem.split('_',1)[1]
assert len(old_files)==22 and len(new_names)==25

spec = (saved/'design_spec.md').read_text(encoding='utf-8')
prefix,outline = spec.split('## IX. Content Outline\n',1)
outline,end = outline.split('## X. Speaker Notes Requirements',1)
old_blocks = {int(m.group(1)):m.group(0).strip() for m in re.finditer(r'### Slide (\d+) - .*?(?=\n### Slide |\Z)',outline,re.S)}
mapping = {old:n for n,old in enumerate(order,1) if old is not None}
prefix = prefix.replace('| Page Count | 22 |','| Page Count | 25 |').replace('16—20 分钟','18—22 分钟')
prefix = re.sub(r'P(\d\d)',lambda m:f'P{mapping[int(m.group(1))]:02}',prefix)
prefix = re.sub(r'- \*\*Revision authority\*\*:.*','- **Revision authority**: 用户要求模块一补充已设计的失败分类指标，并说明分类的必要性与实验用途。新增三页定义，前移现有原生失败数据图；三模块定位与其它章节延续已确认版本。',prefix)
def block(n,title,content,composition,key=None):
    s=f'### Slide {n:02} - {new_names[n]}\n- **Title**: {title}\n- **Audience move**: {content}\n- **Core message**: {content}\n- **Relationships**: none\n- **Topology decision**: no\n- **Composition**: {composition}\n- **Content**: {content}\n'
    if key:
        s+=f'- **Visualization**: {key}=table/comparison_matrix; 自定义语义表\n- **Native-ready**: {key}=no\n'
    return s
new_blocks = {
6:block(6,'先分清框与轮廓的成败，才能选对改进对象','历史同类Box50固定配对下的五个互斥状态：无配对、框差轮廓差、框差轮廓好、框好轮廓差、双达标；每行解释研究用途。框支持率作独立代理量，0.95为历史诊断切点。仅复用定义，不混入当前O2O计数。','五状态×判定×用途三列表；支持率解释置于下方','failure-states'),
8:block(8,'轮廓错在哪里：把遗漏与误报分别量化','固定同一预测；自身遗漏、同类独占、异类独占、邻居歧义和背景五类连续面积比，统一除以有效自身GT面积。自身优先、crowd排除；自身覆盖率作为遗漏的互补量。说明减少邻居混入不能以漏掉自身为代价。','五类像素误差×定义×用途三列表','pixel-errors'),
9:block(9,'把拥挤条件单独记录，才能解释密集场景的收益','ICI记录累计同类GT框交叠，E(4)记录4个模型输入等效像素内的同类边界接近比例；C/I/L/S、O/M/X/MISS记录关系拓扑。按类别与大小分层检查恢复和损伤，复杂关系不等于失败，不将关联写成根因。','左侧几何指标定义，右侧关系分类；底部说明分层评价用途'),
}
blocks=[]
for n,old in enumerate(order,1):
    if old is None: blocks.append(new_blocks[n])
    else:
        b=re.sub(r'^### Slide \d+',f'### Slide {n:02}',old_blocks[old])
        if old==9:
            b=b.replace('横向数据条形图','保留原数据图；下方解释当前目标集合与匹配竞争的区别')
            b=b.replace('分割失败13378：无合格框7148、T6229、竞争1；T占46.56%。','当前O2O分割失败13378：无合格框7148、T6229、竞争1；T占46.56%。用于选择轮廓修正对象，匹配竞争另列。')
        blocks.append(b)
end=end.replace('P05—P19','P05—P22').replace('16—20分钟','18—22分钟').replace('sources/三模块修订依据_20261015.md；','sources/失败分类补充依据_20261015.md、sources/三模块修订依据_20261015.md；')
(base/'design_spec.md').write_text(prefix+'## IX. Content Outline\n\n'+'\n\n'.join(blocks)+'\n\n## X. Speaker Notes Requirements'+end,encoding='utf-8')
lock=(saved/'spec_lock.md').read_text(encoding='utf-8')
old_rhythm={int(n):v for n,v in re.findall(r'- P(\d+): (\w+)',lock)}
rhythm='## page_rhythm\n'+''.join(f'- P{n:02}: {old_rhythm[old] if old else "dense"}\n' for n,old in enumerate(order,1))+'\n'
(base/'spec_lock.md').write_text(re.sub(r'## page_rhythm.*?(?=## pptx_structure)',rhythm,lock,flags=re.S),encoding='utf-8')

# Verified project-local leaf files only; all originals are in the backup.
for folder in ['svg_output','svg_final','notes']:
    target=(base/folder).resolve()
    assert target.parent==base.resolve()
    for f in target.iterdir():
        if f.is_file() and f.suffix in {'.svg','.md'}:
            assert f.resolve().parent==target
            f.unlink()
for n,old in enumerate(order,1):
    if old:
        text=old_files[old].read_text(encoding='utf-8')
        text=re.sub(r'\d\d / 22',f'{n:02} / 25',text)
        (base/'svg_output'/f'{n:02}_{new_names[n]}.svg').write_text(text,encoding='utf-8')

old_notes=(saved/'notes/total.md').read_text(encoding='utf-8')
note_blocks={int(m.group(1)):m.group(0).strip().removesuffix('---').strip() for m in re.finditer(r'^# (\d+)_.+?(?=^# \d+_|\Z)',old_notes,re.S|re.M)}
notes=[]
for n,old in enumerate(order,1):
    notes.append(re.sub(r'^# \d+_',f'# {n:02}_',note_blocks[old]) if old else f'# {n:02}_{new_names[n]}')
(base/'notes/total.md').write_text('\n\n---\n\n'.join(notes)+'\n',encoding='utf-8')
print('25-page roster saved; 22 retained pages moved; new pages 06/08/09 ready for hand authoring.')
