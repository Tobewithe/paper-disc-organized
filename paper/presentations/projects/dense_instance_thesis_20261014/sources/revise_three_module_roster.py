"""Update the approved roster and mechanically move retained pages; no slide drawing."""
from pathlib import Path
import re

base = Path(__file__).resolve().parents[1]
backup = base / 'backup/before_three_modules_20261015'
assert backup.is_dir()
pages = [
('研究定位','面向密集场景的检测分割与持续跟踪方法及系统研究','研究背景、研究方法、系统设计、研究安排。','none','文字封面','anchor'),
('应用需求','连续个体监测的应用需求','数量、位置、轮廓和连续记录。','none','开放文本行','breathing'),
('技术进展','现有技术已能识别和跟踪目标，但目前在密集场景下仍难保持稳定','保留已有能力与复杂场景不足的转折。','none','两栏能力与应用限制','breathing'),
('主要困难','密集场景仍然存在的主要困难','认不全、分不清、遮挡后记录接不上。','none','场景与应用后果对照','breathing'),
('方法总览','三个模块衔接单帧识别与连续身份记录','一：检测与实例分割；二：连续身份保持；三：遮挡恢复。后二者未开展。','order: 单帧输出支持连续关联，丢失后进入身份恢复。','三条横行，标清状态和衔接','anchor'),
('评价指标','评价既看轮廓改善，也看原成功实例是否受损','先定义目标T，再定义图宏ΔIoU、任务修复率、成功损伤率。IoU与AP不是自创新指标。','none','四条指标解释与统一阈值','dense'),
('基础模型','基础模型与轮廓生成过程','冻结YOLO26m-seg；特征提取与融合输出框/类别/分数、32维c0和32通道P。原型与系数组合经原生解码得到轮廓。','link: 图像→特征→检测输出；P与c0共同生成轮廓。','原生模型支路图，冻结与新增修正分开','dense'),
('修正器结构','轻量系数修正器的实际结构','33×8×8空间输入→两层Conv64+GELU→均值64；F64+c32+box4+score1+class16；融合181→128→32维Δc。167872参数。','link: 空间与实例条件共同进入融合层，输出残差后修正c0。','双输入支路与融合结构图','dense'),
('失败定位','6,229 个失败实例仍有合格检测框','分割失败13378：无合格框7148、T6229、竞争1；T占46.56%。','none','横向数据条形图','dense'),
('修正空间','固定检测后，仍存在轮廓修正空间','GT辅助有限系数优化：122/256；GT面积分组38/148、44/64、40/44。','none','数据图与总比例','dense'),
('探索实验设计','探索实验逐项检验修正条件','空间置乱、正常空间BCE、同ROI soft-IoU、固定面积选择；前三者同2000图3轮1500步，最后零新增训练。','contrast: 各实验保持同一基线，逐项区分空间、loss和选择条件。','四行实验设计对照','dense'),
('自动修正探索','无条件自动修正尚未兼顾收益与保护','三种训练配置目标Δ−1.930、−0.996、+0.680pp及原95%CI。soft整体−0.262pp，损伤53/1024。','none','点区间图','dense'),
('面积分层','小掩码的修正损伤更集中','原生预测面积S/M/L损伤37/270、14/424、2/330；小掩码占26.4%却占损伤69.8%。','none','损伤率数据条形图','dense'),
('选择性修正结果','选择性修正在开发筛查中兼顾收益与保护','原生面积≥96²使用已有修正。目标+1.002pp，总体+0.089pp，损伤2/1024。原三项数值通过；复用开发面板非独立确认。','none','左侧效应区间图、右侧同面板损伤对比图','dense'),
('模块一结论','模块一的重要结论与验证边界','固定表示有机会；无条件修正有损伤；面积选择通过开发筛查。下一步同配置全5000图AP和全原成功总体损伤。','none','三项结论与单个下一步','anchor'),
('身份保持模型','模块二：连续视频中的身份关联与稳定保持','拟采用ByteTrack基线，参照BoT-SORT结合运动和外观。轮廓只在显式进入外观提取后参与身份判断。','link: 当前观测与历史轨迹经运动/外观关联产生持续ID。','拟研究框架图及待验证问题','dense'),
('身份保持实验','模块二的实验设想','同视频同检测，比较ByteTrack、运动外观、原生轮廓外观、修正轮廓外观；IDF1、ID切换、断裂和端到端速度。','contrast: 固定观测隔离关联收益，再比较原生/修正轮廓。','对照设计与统一评价','dense'),
('遮挡恢复模型','模块三：遮挡后的身份恢复与轨迹续接','沿用同一跟踪框架；lost记忆→候选筛选→连续观测确认→旧ID续接；不足则保持未确认或终止。','order: 跟踪中→丢失记忆→重现待确认→恢复旧ID。','带分支的拟研究状态流程','dense'),
('遮挡恢复实验','模块三的实验设想','默认保留→只延长→同预算加记忆→加多帧确认；所有合格重现事件共同分母，报告正确/误接/未恢复与延迟超时。','contrast: 等待长度、记忆、确认分开消融。','消融对照与事件级指标','dense'),
('系统功能','通用系统的使用流程','选图像或视频、查看位置轮廓编号、回看遮挡恢复记录、导出。拟建系统。','order: 导入任务→查看分析→保存记录。','使用者流程，技术细节不外溢','breathing'),
('应用案例','猪只群体监测的应用案例','猪只仅作应用验证；其它类别需分别验证。','none','应用问题与预期记录','breathing'),
('研究安排','现有基础与后续研究安排','图像已有实验；视频身份与遮挡恢复尚未开展；后续全量图像验证、视频标注与方法实验、系统集成。','order: 完善图像验证→视频两模块→系统论文。','基础、任务与交付','anchor'),
]
spec = (backup/'design_spec.md').read_text(encoding='utf-8')
prefix=spec.split('## VII. Visualization Reference List')[0]
prefix=prefix.replace('| Page Count | 15 |','| Page Count | 22 |').replace('正文约 12—15 分钟','正文约 16—20 分钟')
prefix=prefix.replace('方法按评价、探索、风险分析组织真实证据','方法按检测分割、连续身份保持、遮挡恢复组织；模块一展开真实证据')
prefix=prefix.replace('技术细节限于 P05—P12','技术细节限于 P05—P19')
prefix=re.sub(r'- \*\*Revision authority\*\*:.*', '- **Revision authority**: 用户确认三个任务模块，并要求详述模块一的模型、实验数据图和重要结论；模块二、三只写设想模型与实验。继续现有项目修订，不重启规划。',prefix)
charts={9:('failure-split','horizontal_bar_chart','310,230,850,500; 0..8000'),10:('oracle-area','horizontal_bar_chart','230,226,810,466; 0..100'),12:('pilot-effects','scatter_chart','420,224,1160,464; -4..2'),13:('area-damage','horizontal_bar_chart','250,220,1000,490; 0..16')}
middle='## VII. Visualization Reference List\n\n实验图采用既有数据与可编辑形状，标准绘图另存 PNG/SVG/PDF。\n\n| Page | Object | Family | Source |\n| --- | --- | --- | --- |\n'
for n,(key,kind,_) in charts.items(): middle+=f'| P{n:02} | {key} | {kind} | E0—E4，保留原分母和区间 |\n'
middle+='| P14 | selective-effects | scatter_chart | E3/E5，同目标与总体配对 |\n| P14 | selective-damage | column_chart | E3/E5，同1024原成功面板 |\n\n## VIII. Image Resource List\n\n无装饰图片。模型结构图采用可编辑节点、连线和文字；实验图来自真实 Run。\n\n## IX. Content Outline\n\n'
for n,(name,title,content,relations,composition,rhythm) in enumerate(pages,1):
    middle+=f'### Slide {n:02} - {name}\n- **Title**: {title}\n- **Audience move**: {content}\n- **Core message**: {content}\n- **Relationships**: {relations}\n- **Topology decision**: {"no" if relations=="none" else "yes"}\n- **Composition**: {composition}\n- **Content**: {content}\n'
    if n in charts:
        key,kind,area=charts[n];middle+=f'- **Visualization**: {key}=chart/{kind}; {area}\n- **Native-ready**: {key}=no\n- **Data class**: observed\n'
    if n==14:middle+='- **Visualization**: selective-effects=chart/scatter_chart; 310,240,740,510; -0.5..2.0. selective-damage=chart/column_chart; 865,240,1190,510; 0..6.\n- **Native-ready**: selective-effects=no; selective-damage=no\n- **Data class**: observed\n'
    middle+='\n'
end='## X. Speaker Notes Requirements\n\n- **Generation**: enabled\n- **Content**: 每页自然讲稿，技术、方法图和实验指标限于P05—P19。图像模型与实验按真实Run描述；两个视频模块和系统标明拟开展。\n- **Total duration**: 约16—20分钟，模块一为主要篇幅。\n- **Evidence**: sources/三模块修订依据_20261015.md；E0—E5和原文献来源。\n'
(base/'design_spec.md').write_text(prefix+middle+end,encoding='utf-8')
lock=(backup/'spec_lock.md').read_text(encoding='utf-8')
rhythm='## page_rhythm\n'+''.join(f'- P{n:02}: {p[-1]}\n' for n,p in enumerate(pages,1))+'\n'
lock=re.sub(r'## page_rhythm.*?(?=## pptx_structure)',rhythm,lock,flags=re.S)
lock=lock.replace('方法章节展开诊断指标、真实实验和结论','方法章节按检测分割、身份保持、遮挡恢复组织；第一模块展开真实模型与数据，后两模块仅计划')
(base/'spec_lock.md').write_text(lock,encoding='utf-8')

# Only rename/retain existing page artwork. New page artwork is authored by hand.
for folder in ['svg_output','svg_final','notes']:
    dest=(base/folder).resolve()
    assert dest.parent==base.resolve()
    for f in dest.iterdir():
        if f.is_file() and f.suffix in ['.svg','.md']: f.unlink()
retained={1:'01_研究定位',2:'02_应用需求',3:'03_技术进展',4:'04_主要困难',9:'07_失败定位',10:'08_修正空间',12:'09_自动修正探索',13:'10_面积分层',20:'13_系统功能',21:'14_应用案例',22:'15_研究安排'}
for n,old in retained.items():
    svg=(backup/'svg_output'/f'{old}.svg').read_text(encoding='utf-8')
    svg=re.sub(r'\d\d / 15',f'{n:02} / 22',svg)
    if n in (9,10,12,13): svg=re.sub(r'02 研究方法 · 模块[一二三]','02 研究方法 · 模块一',svg)
    (base/'svg_output'/f'{n:02}_{pages[n-1][0]}.svg').write_text(svg,encoding='utf-8')
print('Approved 22-page roster; retained artwork renamed; new pages await hand authoring.')
