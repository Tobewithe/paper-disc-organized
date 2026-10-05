"""Render the completed locked pilot into a numerical, qualified narrative."""
import csv,json,hashlib
from pathlib import Path

root=Path(__file__).parent
out=root/'diagnostics/ownership_ranking_eval_20260912'
dest=root.parents[1]/'refine-logs/coco-structure/OWNERSHIP_RANKING_RESULTS_20260912.md'
read=lambda name:list(csv.DictReader((out/name).open(encoding='utf-8-sig')))
task={r['arm']:r for r in read('task_summary.csv')}
family={r['family']:r for r in read('family_task_summary.csv')}
analysis=json.loads((out/'PAIRED_ANALYSIS.json').read_text())
aggregate=read('aggregate_summary.csv')
names={'bce_dice':'BCE＋Dice','coverage_only':'＋覆盖','rank_only':'＋排序','rank_coverage':'＋排序＋覆盖','pair_bce_coverage':'＋同像素 pair-BCE＋覆盖','ccl_coverage':'＋同类 CCL＋覆盖'}
text=['# S013：重叠候选像素归属监督的受控验证','',
'日期：2026-09-12。状态：18 组训练、500 图评价和 2,000 次图像簇配对重采样完成；预先冻结的判断为 **NO_GO**。停止这个版本，不扩大 5,000 图、不重启原端到端队列。','',
'## 判断与收益来源','',
'当前全局系数残差头上的排序损失没有超过同容量普通监督形成可靠增益。覆盖项提高自身覆盖的同时增加同类邻居及背景误分。该结果限制这一个实现与预算，不否定所有实例归属方法，也不能证明网络缺少局部竞争是根因。','',
'普通 BCE＋Dice 相对原始冻结模型有所提高，必须与候选方法的额外贡献分开。不能仅以候选相对原模型的 AP 提升，声称排序机制有效或密集专属改善。','',
'## 数据与完整预算','',
'从全部 118,287 个 COCO train2017 原始图像 ID 按冻结哈希选择 1,200 图，不按 ICI 筛图。实际 1,190 图有可训练目标：8,322 个匹配且有有效支持的普通 GT；811 个未匹配、11 个无有效支持单列，其中 10 个保留目标的自身正像素支持为空，主 BCE 仍保留，覆盖项跳过。多多边形通过 pycocotools annToMask 合并为同一 annotation ID。','',
'每轮 3,316 个有向独占区域配对、1,743 个几何 CCL 边，6 个同容量损失臂 × 3 个实际训练种子 × 15 轮。各组 2,235 次 Adam 更新，同种子初始化和图像顺序逐字节配对，270 个逐轮 checkpoint 全部保留；固定最后一轮评价。原型、候选、框、分数及分类不训练。','',
'500 张哈希选定 COCO val 图包含 3,616 个普通 GT，其中高 ICI 607 个（183 图）、低 ICI 3,009 个。先进行全体任务匹配，再按实例 ICI 分层；空间诊断另有固定框匹配的 3,209 个目标，其中高组 548 个（177 图）。相邻同类 GT 对 856 个，高组 638 对（172 图）。高组为同类 bbox 交叠面积和／自身 bbox 面积 > 0.5，经边界容差稳定；可大于 1，并非 GT 掩码重叠率。','',
'候选来自官方运行时 one-to-many＋NMS 路径，640 输入、conf=0.001、max_det=300；原来被过滤的空掩码/抑制候选不能恢复。推理完全无 GT、ICI 和配对输入；统一官方掩码解码。冻结候选的各组共同 Box AP=52.553。R75 指官方类别内 maxDets=100 的 IoU=0.75 GT 恢复率，未固定 Precision。','',
'## 所有损失臂','',
'AP 和恢复率均乘 100。± 为三个训练种子的 AP 样本标准差，不是置信区间。','',
'| 方法 | Mask AP | AP50 | 高 ICI R75 | 低 ICI R75 | 高组相邻对同时恢复 |','|---|---:|---:|---:|---:|---:|']
initial=task['initial']
cols=['mask_ap','mask_ap50','r75_high','r75_low','pair75_high']
text.append('| 原始冻结模型 | '+' | '.join(f'{float(initial[m])*100:.3f}' for m in cols)+' |')
for f,row in family.items():
    values=[f"{float(row['mask_ap'])*100:.3f} ± {float(row['mask_ap_sd'])*100:.3f}"]+[f'{float(row[m])*100:.3f}' for m in cols[1:]]
    text.append('| '+names[f]+' | '+' | '.join(values)+' |')
text+=['',f"普通 BCE＋Dice 相对原始模型：AP +{(float(family['bce_dice']['mask_ap'])-float(initial['mask_ap']))*100:.3f} 点；高 ICI R75 +{(float(family['bce_dice']['r75_high'])-float(initial['r75_high']))*100:.3f}、低 ICI R75 +{(float(family['bce_dice']['r75_low'])-float(initial['r75_low']))*100:.3f} 个百分点。排序单项相对 BCE＋Dice 的 AP 仅 +{(float(family['rank_only']['mask_ap'])-float(family['bce_dice']['mask_ap']))*100:.3f} 点，高 ICI R75 均值不变。不能从这些点估计宣称密集专属收益。",'',
'## 主比较：排序＋覆盖减 BCE＋Dice','',
'差值单位为百分点。像素指标是每目标比值的均值；邻居/背景错误分母为有效自身 GT 面积，不是全图像素或预测面积。','',
'| 指标 | 差值 | 95% 配对区间 |','|---|---:|---:|']
labels={('task','hit75','high'):'高 ICI R75',('task','hit75','low'):'低 ICI R75',('pair','hit75','high'):'高组相邻对恢复',('spatial','coverage','high'):'自身覆盖',('spatial','same_neighbor','high'):'同类邻居误分',('spatial','background','high'):'背景误分',('spatial','mask_iou','high'):'固定归属 IoU'}
for (kind,metric,group),name in labels.items():
    r=next(r for r in analysis['contrasts'] if r['treatment']=='rank_coverage' and r['control']=='bce_dice' and (r['kind'],r['metric'],r['group'])==(kind,metric,group))
    text.append(f"| {name} | {r['mean_pp']:+.3f} | [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] |")
text+=['',f"Mask AP 差为 {(float(family['rank_coverage']['mask_ap'])-float(family['bce_dice']['mask_ap']))*100:+.3f} 点，未计算 AP 置信区间。",'',
'相对覆盖单项，排序＋覆盖的高组空间 IoU 仅 +0.005 点 [−0.024, +0.034]；邻居误分 +0.079 点 [+0.003, +0.178]。相对使用完全相同额外像素的 pair-BCE＋覆盖，排序＋覆盖的高组 R75 为 +0.055 点 [−0.517, +0.666]，IoU 为 −0.045 点 [−0.094, −0.004]。没有建立排序形式的额外优势。','',
'## 覆盖与错误为何要一起看','',
'高组固定归属像素指标如下（×100）：','',
'| 方法 | 自身覆盖 | 同类邻居误分 | 背景误分 | IoU |','|---|---:|---:|---:|---:|']
for arm in ['initial',*family]:
    r=next(r for r in aggregate if r['kind']=='spatial' and r['group']=='high' and r['arm']==arm)
    text.append('| '+('原始模型' if arm=='initial' else names[arm])+' | '+' | '.join(f'{float(r[m])*100:.3f}' for m in ['coverage','same_neighbor','background','mask_iou'])+' |')
text+=['',
'普通监督减少了背景，但覆盖也下降。覆盖约束将自身覆盖拉回时，邻居和背景错误同时回升；加入排序未把这两种变化有效分离。结果与该软覆盖约束缺乏空间选择性的解释一致，但本轮没有直接证明是哪一层表征或哪个梯度交互造成，因此不能把解释写成确定机制。','',
'从当前损失本身可确定一个局限：对任意共同偏移 a，softplus(1−(z_i+a)+(z_j+a)) = softplus(1−z_i+z_j)。排序只约束相对差，不能单独保证正确实例为正、邻居为负；例如 z_i=3、z_j=2 满足正间隔，但两者在零阈值下仍都预测前景。BCE 另行约束绝对前景/背景，覆盖项则保护自身正响应；共享全局系数会同时影响多个位置。因此“排序更好”在数学上也不等于“二值掩码分离更好”。这解释了为什么必须设置同像素 pair-BCE 和覆盖/邻居联合指标，但未证明共同偏移就是本轮失败的实际原因。','',
'类别×COCO尺度的 77 个共同分层覆盖 2,518 个 GT；排序＋覆盖相对 BCE＋Dice 的标准化高/低 R75 点估计为 +0.704/−0.350。该组成控制没有区间，仅作描述，不能推翻预定主门槛或据此重新选择子组。','',
'## 统计、复验和限制','',
'2,000 次配对图像簇重采样，先对三个头种子的逐 GT/对结果求均值；区间条件于当前训练子集和三个种子，未涵盖完整训练随机性、未作多重比较校正。val 已被项目探索，不是独立确认集。AP 表是这 500 图的官方 COCOeval，不能称为全 5,000 图或 COCO 全量成绩。','',
'缓存 v1 曾因输入采样重放差 0.000213146 超过 2e-4 阈值而停止，失败目录保留且未训练。v2 使用官方双线性插值的输入网格直接索引，最大误差 7.6293945e-6，未放宽阈值。主缓存耗时 64.41 秒，18 组训练 356.89 秒，解码与官方评估 454.26 秒；不含实现、归档和审计。','',
'本地训练与评价归档已按远端 SHA256 核对；重新检查 270 个 checkpoint、逐轮样本数、配对初始化/顺序，19 个预测文件及原始 GT 恢复均值，验证通过。此类确定性检查不替代语义审计。独立审计已完成，结论 WARN、未发现 fatal、same-family/provisional，见 OWNERSHIP_RANKING_AUDIT_20260912.md/.json；审计者未从原始 COCO JSON 独立重算官方 AP，不能称为完整独立复现。','',
'## 对下一步和论文的约束','',
'本轮停止当前“共享全局系数残差＋排序＋软覆盖”实现，不继续调权重、筛类别或用外扩 AP 补救。此前诊断中的邻居内部侵入和 GT 辅助读出机会保留，但它们不足以宣布可部署方法已经成立。','',
'下一项方法若提出，必须先说明它能选择性修正邻居像素而保护自身前景的具体能力，以及相对当前普通监督为何有增量；不能仅重新命名损失。当前证据可用于问题定位和排除简单解释，尚不足以包装成已成功的方法论文。','',
'## 原始文件','',
'- 协议：experiments/coco_clean_20260911/ownership_ranking_protocol_20260912.json。',
'- 训练、逐轮权重与历史：diagnostics/ownership_ranking_train_20260912/。',
'- 19 组完整预测、task_summary.csv、spatial.csv、gt_recovery.csv、pair_recovery.csv、PAIRED_ANALYSIS.json：diagnostics/ownership_ranking_eval_20260912/。',
'- 本地确定性复验：diagnostics/ownership_ranking_train_LOCAL_VERIFY_20260912.json 和 ownership_ranking_eval_LOCAL_VERIFY_20260912.json。',
'- 同类 CCL 是六组中的一个适配控制，作用于冻结头，不是原完整端到端 CCL 复现。','']
dest.write_text('\n'.join(text),encoding='utf-8')
print(dest)

