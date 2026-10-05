"""Recompute and persist current exploratory evidence without selecting positives."""
import csv,json,hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S');now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    groups=[('coefficient_learning_eval_v2_20260911',['bce','diagonal','gram']),('conditional_adapter_eval_20260911',['plain','diagonal','conditional']),('pixel_ownership_eval_20260911',['own','neighbor']),('region_supervision_eval_20260911',['bce','neighbor','background']),('spatial_head_eval_20260911',['own','spatial']),('local_coeff_eval_20260911',['global','local']),('local_dice_eval_20260911',['global','local'])]
    entries=[]
    for folder,modes in groups:
        file=ROOT/'diagnostics'/folder/'summary.csv'
        if not file.exists():continue
        rows=read(file)
        for mode in modes:
            selected=[r for r in rows if r['arm'].startswith(mode+'_s')];assert len(selected)==3
            entry=dict(experiment=folder,method=mode,seeds=3)
            for key in ['mask_ap','r75_high','r75_low','pair_recovery75']:
                values=np.array([float(r[key]) for r in selected])*100;entry[key]=float(values.mean());entry[key+'_sd']=float(values.std(ddof=1))
            entries.append(entry)
    failure=read(ROOT/'diagnostics/failure_decomposition_all_20260911/summary.csv')
    lines=[f'# COCO 机制与可学习方法试验记录\n\n更新：{now}（北京时间）。\n',
      '本记录替代旧报告中“后续训练尚未开始”的状态。用户后来授权找到候选后自行实验；原六组正式 YOLO 队列仍暂停，以下为单独的冻结网络局部学习试验。所有试验使用远端原始 COCO，未使用 Faro 或另一模型的 80 图口头记录。\n',
      '## 当前决定\n\n目前没有一个新增候选同时取得可靠的密集实例收益和相对同预算监督基线的优势，不能宣称新主线已成立。固定原型中的改进空间存在，尚需证明可由不读取 GT 的预测模型学到。继续定位具体错误，不恢复原 CCL 正式训练。\n',
      '## 全量失败分解\n\n官方预训练模型覆盖 val2017 全部 5,000 图、36,335 个普通 annotation ID。空间计数排除 crowd 后 149 个实例无有效像素，剩余 36,186 个；这些排除项保留在逐实例表。没有用 ICI 筛图。逐实例按同类 ICI 分组。\n',
      '| 实例组 | 有效 GT | 未框匹配 | 框匹配且 Mask IoU<.75 | 仅消除同类邻居错误可越过 .75 | 仅消除背景错误可越过 .75 | 预测框不足以覆盖 75% GT 的失败 |\n|---|---:|---:|---:|---:|---:|---:|']
    for r in failure:lines.append('| '+' | '.join(str(r[k]) for k in ['group','gt','unmatched','matched_fail75','remove_same_neighbor_iou_rescued75','remove_background_iou_rescued75','crop_blocked75'])+' |')
    lines.extend(['\n这是固定 COCOeval bbox≥.5 归属下的原始像素诊断，不是官方 Mask AP/Recall。消除错误使用 GT，是机会估计，不能作为可部署方法收益；不同修正机会会重叠，不能相加。高组与低组的类别、尺度构成尚未标准化，不能解释为拥挤的独立因果效应。\n',
      '## 相同验证集上的可学习结果\n\n训练用预先哈希选择的 1,200 张 train2017 图片；验证用 300 张密集富集开发图，包含 3,635 普通 GT、788 高 ICI GT、1,174 对同类相邻 GT。这个验证集多次用于探索，不是未接触测试集。下表为三种子均值，指标乘 100；AP 为该子集官方 COCOeval，非 COCO 全量 AP。\n',
      '| 实验 | 方法 | Mask AP | 高拥挤 R75 | 低拥挤 R75 | 相邻对同时恢复@.75 |\n|---|---|---:|---:|---:|---:|',
      '| 固定官方模型 | 原始输出 | 40.4521 | 53.0457 | 59.2905 | 45.0596 |'])
    for e in entries:lines.append(f"| {e['experiment']} | {e['method']} | {e['mask_ap']:.4f} | {e['r75_high']:.4f} | {e['r75_low']:.4f} | {e['pair_recovery75']:.4f} |")
    lines.extend(['\n- 第一组只学习原系数头的最后一层：BCE、对角梯度变换、Gram 梯度变换，同 15 轮。Gram 的 AP 小幅高于 BCE，但高拥挤 R75 和成对恢复更低。\n- 第二组比较普通、对角和完整条件残差变换；完整变换未优于普通监督的密集指标。\n- 第三组为像素残差 MLP：自身特征与额外预测邻居特征对照，同参数量、10 轮。邻居特征没有稳定超过自身对照；不能用二者相对未适配模型的 AP 上升证明邻居机制有效。\n- 区域监督组（若表中已出现）：同 15 轮，邻居额外负权重与背景额外负权重逐实例总量对齐、均值归一；纯 BCE 作为监督预算对照。\n',
      '这些不是完整端到端 YOLO 训练。检测、类别、分数、原型、初始候选集合固定；原本为空的候选不能新增。三种子仅反映此冻结试验的随机初始化/样本顺序，不代表三次完整网络训练。每轮 checkpoint 均保留。官方 Mask R75 由 COCOeval 的分割匹配得到，与上面的固定框归属诊断不可直接比较。\n',
      '## 方向探针及无效结果\n\n固定原型与同等 logit RMS 预算下，GT BCE 的 Gram 方向在与首批 300 图不重叠的 300 图上，相对普通 BCE 方向使高组 IoU +1.743 pp [1.472,2.049]、同类邻居错误 −0.714 pp [−0.941,−0.520]、覆盖 +1.039 pp [0.729,1.408]。这是使用 GT 的小步诊断，不是训练效果、AP 或因果根因证明。其正结果没有自动转化为上述学习收益。\n',
      '早期 box-core 坐标错误、重复归一化方向、零梯度蒸馏权重对照归档为 FAILED_INITIAL_DIAGNOSTIC_CANDIDATES，不用于方法选择。详情见 CANDIDATE_AUDIT_20260911.md。旧结果保留，不将无效试验说成纯粹的负面科学结果。原始误写日志曾覆盖的事实不能用后续记录追补成完整运行日志。\n',
      '## 接受主线的条件\n\n候选须在相同监督预算下超过普通 BCE/等容量控制；同时改善高拥挤实例的分割恢复与空间归属，报告覆盖和背景代价。冻结方案后再扩大到未参与调参的评估数据并做图像簇配对区间、类别×尺度分层。全量 val2017 已用于描述性机制定位，不能称为完全未接触的最终测试集。未达到这些条件，不把“有改进空间”写成“提出的方法已解决问题”。\n'])
    report='\n'.join(lines);dest=WORK/'refine-logs/coco-evaluation';(dest/f'MECHANISM_LEARNING_STATUS_{stamp}.md').write_text(report,encoding='utf-8');(dest/'MECHANISM_LEARNING_STATUS.md').write_text(report,encoding='utf-8')
    results={'date':now,'three_seed_means':entries,'source_sha256':{str(p.relative_to(WORK)):sha(p) for folder,_ in groups for p in (ROOT/'diagnostics'/folder).glob('*.csv')}}
    (dest/f'MECHANISM_LEARNING_SUMMARY_{stamp}.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    with (WORK/'MANIFEST.md').open('a',encoding='utf-8') as f:f.write(f'\n| {now} | COCO mechanism continuation | refine-logs/coco-evaluation/MECHANISM_LEARNING_STATUS_{stamp}.md | report | Full-val per-instance failure decomposition, all completed 3-seed learned pilots, failed-candidate disposition and current decision |\n')
    print(str(dest/'MECHANISM_LEARNING_STATUS.md'))

if __name__=='__main__':main()

