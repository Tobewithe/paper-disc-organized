"""Summarize audited remote COCO diagnostics; writes Chinese, versioned report."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE/'diagnostics/mechanism_official_300_20260911'
PROJECT=HERE.parents[1]

def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def fmt(r,scale=100):
    return f"{float(r['mean'])*scale:+.3f} [{float(r['ci_low'])*scale:+.3f}, {float(r['ci_high'])*scale:+.3f}]"

def find(rows,**kwargs):
    matches=[r for r in rows if all(r[k]==str(v) for k,v in kwargs.items())]
    assert len(matches)==1,(kwargs,len(matches));return matches[0]

def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S');date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary=csvread(RUN/'summary.csv');capacity=csvread(RUN/'capacity_check/summary.csv')
    bounded=csvread(RUN/'audit/bounded_summary.csv');represent=csvread(RUN/'audit/representation_summary.csv')
    target=csvread(RUN/'audit/bounded_direction.csv');raw=csvread(RUN/'interventions.csv')
    audit=json.loads((RUN/'audit/AUDIT.json').read_text());assert audit['status']=='PASS'
    draws=np.load(RUN/'bootstrap_draws.npz');image_ids=draws['image_ids'];idx=draws['draws'];position={int(i):k for k,i in enumerate(image_ids)}
    interactions=[]
    for metric in ['mask_iou','same_neighbor']:
        sides=[]
        for high in [True,False]:
            sums=np.zeros(300);counts=np.zeros(300)
            for r in target:
                if (float(r['target_ici'])>.5)==high:
                    k=position[int(r['image_id'])];sums[k]+=float(r['delta_'+metric]);counts[k]+=1
            sides.append((sums.sum()/counts.sum(),sums[idx].sum(1)/counts[idx].sum(1)))
        boot=sides[0][1]-sides[1][1]
        interactions.append(dict(metric=metric,contrast='high_minus_low_bounded_ridge_change',mean=float(sides[0][0]-sides[1][0]),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    (RUN/'audit/exploratory_interactions.json').write_text(json.dumps(interactions,indent=2),encoding='utf-8')
    names={'coverage':'目标覆盖 ↑','same_neighbor':'同类邻居错误 ↓','neighbor':'全部邻居错误 ↓','background':'背景错误 ↓','mask_iou':'逐实例 Mask IoU ↑'}
    lines=[f'# COCO 固定解码器机制定位报告\n\n生成：{date}（北京时间）。正式训练按用户要求暂停；本报告全部实验没有更新网络权重。',
    '## 当前可以采取的决定\n\n保留“密集实例的掩码归属错误”这一研究问题，但暂不把“系数余弦相似是失败根因”作为论文前提。官方预训练模型中，推开系数确实降低了原始掩码间重叠，却没有显示出同类 GT 泄漏改善；在同一原型和框上，朝 GT 求解的系数方向移动能够同时改善覆盖、泄漏和 IoU。这支持进一步研究**实例系数如何选择、如何使更新方向针对错误的空间归属**。目前没有证明某种新损失已有效，也没有判定经过训练的 CCL 必然无效。',
    '## 数据和比较范围\n\n- 数据直接来自远端 COCO val2017 原始 JSON 与图片，GT 由 pycocotools 按 annotation ID 解码，多段 polygon 合并为一个实例。没有使用旧的逐 polygon 标签。\n- 模型：官方预训练 YOLO26m-seg，FP32 推理，640 输入，conf=0.001，max_det=300。网络、原型、框、类别、分数和 GT 归属固定。\n- 在全部 val2017 GT 上按哈希预选 300 张图，每图一对同类 bbox IoU>0.05 的普通实例。按该对的最大实例 ICI 分层抽取 200 张高组图、100 张低组图。\n- 262 对双方均通过官方 COCOeval bbox IoU≥0.5 归属匹配；38 对未双匹配，保留记录且不记成干预改善。有效目标 524 个，其中逐实例 ICI>0.5 的 242 个来自 177 张图。图像抽样分层与逐实例评估分层不同。\n- ICI 为同图同类其他普通实例与自身 bbox 的交叠面积之和/自身 bbox 面积，不是掩码面积，也不是每图最大值。拥挤组按自身 ICI 划分。\n- 2,000 次图像簇配对 bootstrap；区间为点态 95% CI，条件于当前富集样本和已匹配目标。不是 COCO 全量 AP，不代表未检出的实例或全体密集图像。\n- 300 图主干预约 56.4 秒；闭式求解约 8.8 秒；独立复核约 14.3 秒，均在远端执行。',
    '## 评价量\n\n去掉 crowd 区域后，令自身 GT 面积为 A；正确覆盖像素为 T，落在其他普通 GT 且不在自身 GT 内的像素为 N，落在全部普通 GT 之外的像素为 B。报告 Coverage=T/A，Neighbor=N/A，Background=B/A，以及 IoU=T/(A+N+B)。Same-neighbor 只统计同类其他 GT 的独占错误区域。自身与邻居 GT 重叠的像素优先计作自身正确覆盖，避免重复计错。\n\n因此同类邻居错误下降 1 个百分点，指错误像素/自身 GT 面积下降 0.01，不能写成“少了 1% 的误检实例”。原型网格上的 raw pair IoU 只比较两张预测掩码，包含网络补边区，不是 GT 泄漏指标。',
    '## 主比较：高拥挤目标、正常预测框裁切\n\n差值单位均为百分点，括号内为配对 95% CI。推开与有指导的小步调整均保持每个系数向量范数，每个向量转动 5°。',
    '| 指标 | 单纯推开系数，5° | 朝 GT 闭式解移动，5° | GT 闭式解完全替换 |\n|---|---:|---:|---:|']
    for m,name in names.items():
        p=find(summary,domain='cropped',group='high',degrees='5.0',contrast='push_minus_zero',metric=m)
        b=find(bounded,group='high',contrast='bounded_ridge_minus_zero',metric=m)
        f=find(capacity,group='high',metric=m)
        lines.append(f'| {name} | {fmt(p)} | {fmt(b)} | {fmt(f)} |')
    lines.append('朝 GT 解移动和完全替换是主干预结束后补充的探索性 oracle，使用了 GT，不能作为可部署方法或 AP 提升。完全替换的系数转角中位数约 77.56°，不能与小角度推开视为相同预算；小步诊断的 524 个目标均实际转动 5°。相同系数转角也不保证相同 logit 扰动幅度。')
    lines.extend(['## “推开”究竟改变了什么\n\n以下是全部 262 对的 5°推开减去零干预。前两项与正激活乘积用原单位，raw pair IoU 用百分点。',
    '| 量 | 差值及区间 |\n|---|---:|'])
    for metric,label,scale in [('coefficient_cosine','系数余弦',1),('logit_cosine','logit 余弦',1),('raw_pair_iou','raw pair IoU（百分点）',100),('positive_product','正激活乘积（原单位）',1)]:
        lines.append(f'| {label} | {fmt(find(represent,mode="push",metric=metric),scale)} |')
    lines.append('这些量沿预期方向变化，但 GT 同类邻居错误未相应减少，背景错误反而增加。这把旧故事中的薄弱环节定位到“预测间相似度降低 → GT 实例归属改善”这一步；不能用 raw overlap 下降替代对这一环的验证。')
    lines.extend(['## 有指导方向与随机、低拥挤对照\n\n随机对照包含固定随机切向方向的正、负两个 5°扰动，取两臂指标平均。高拥挤目标：',
    '| 指标 | 朝 GT 解移动减随机均值 |\n|---|---:|'])
    for m in names:lines.append(f'| {names[m]} | {fmt(find(bounded,group="high",contrast="bounded_ridge_minus_random_mean",metric=m))} |')
    lines.extend(['\n同样的有指导 5°调整在低拥挤组也有效，因此不能据此声称问题只存在于密集场景。',
    '| 指标 | 高拥挤变化 | 低拥挤变化 | 高减低变化（探索性） |\n|---|---:|---:|---:|'])
    for r in interactions:
        m=r['metric'];lines.append(f'| {names[m]} | {fmt(find(bounded,group="high",contrast="bounded_ridge_minus_zero",metric=m))} | {fmt(find(bounded,group="low",contrast="bounded_ridge_minus_zero",metric=m))} | {fmt(r)} |')
    lines.append('上述高低组差异尚未校正类别、尺度和 bbox 质量；它描述当前样本，不能解释为拥挤的独立因果效应。')
    lines.extend(['## 数学上如何理解\n\n设原型矩阵 P∈R^(K×HW)、实例系数 c_i∈R^K，则 z_i=Pᵀc_i，z_iᵀz_j=c_iᵀ(PPᵀ)c_j。系数空间的普通余弦并不是输出空间的相似度；像素 x 上的变化是 Δz_i(x)=p(x)ᵀΔc_i，同一个系数更新会影响目标、邻居与背景。降低 c_iᵀc_j 并不指定哪些像素应该变正、哪些应该变负。\n\n对任意可逆对角 D，令 c′=Dc、P′=D⁻¹P，可保持 logits 完全不变而改变普通系数余弦。本次对每对实例施加两个预先固定的幂次缩放，524 次变换的 raw 像素差为 0。这说明普通余弦不是参数化无关的掩码质量证据；不否认在一个固定训练参数化中施加正则可能有用。\n\n补充 oracle 只在固定预测框支持区域内，求解区域平衡的岭回归：min_c mean_pos(cᵀp−1)² + mean_neg(cᵀp+1)² + λ‖c‖²，λ=10⁻⁴ trace(G)/32。正区域为自身 GT，负区域为其他 GT 与背景，crowd 排除。使用闭式线性求解，无网络优化器、无权重更新。评价仍保留框外 GT 作为覆盖分母，并通过原来的预测框解码。可改善说明当前固定 P 存在更好的可实现输出；这不是 IoU 最优解或性能上界。',
    '## 未成功的方向也保留\n\n主实验还用全图的目标正项、同类邻居负项和背景负项构造 softplus 梯度，然后在球面移动 5°。这一步没有改善高拥挤 IoU（−2.418 个百分点），还减少覆盖。该 surrogate 同时受到区域权重、全图与框内域差异以及固定步长影响。它不能证明所有空间损失无效，也不能被补充闭式解的正结果删除。2°、5°、10°，push/pull/random±/spatial_oracle 的完整结果都在 CSV 中。',
    '## 完成的实现复核\n\n- 模型、原始 GT、ICI 元数据、实验源码哈希一致；无使用另一模型的 80 图口头记录。\n- 对预选 600 个 GT 实例从原始 bbox 独立重算 ICI；多段 polygon 仍归属于同一 annotation ID。\n- 300 图唯一；262 有效对与 38 未双匹配对完整；16,768 行唯一，所有既定变体齐全。\n- 主实验零干预对官方 process_mask 的像素差为 0；另取哈希选定的 16 图，用官方解码器及独立 NumPy 像素计数复核所有变体，指标最大差为 0。32 次 pycocotools RLE IoU 对照通过。\n- 524 个闭式解结果全部独立重算，基线和 oracle 的 GT 像素指标完全一致；30 项主 bootstrap 汇总独立复算通过。\n- 范数保持与转角检查通过；在双精度下核验 Gram 度量余弦等于 logit 余弦。\n- 这些是同一代理编写的两套实现对照，不是外部独立审稿。',
    '## 下一阶段怎么做，何时恢复训练\n\n1. 先在与这 300 图不重叠的 COCO 图像上冻结并复核“固定 P 下可改善的空间归属”观察。保留低/高拥挤分层，报告 bbox 未匹配率；预设类别×尺度×框质量的标准化或匹配分析，防止把组间构成差异写成拥挤效应。\n2. 对不使用 GT 答案的候选方法，只研究一个明确目标：在自身覆盖不下降的条件下减少同类邻居区域的错误正响应。候选可以是 GT 邻居区域重加权的标准 mask loss，或基于 p(x) 的局部空间约束；必须与普通空间重加权、原 CCL、相同优化预算对照。尚未选择实现，也没有开始训练。\n3. 只有确定可学习的目标与预先固定的验证标准之后，再修复训练异常并重新冻结三随机种子、相同预算的正式训练协议。推理阶段仍不得使用 GT；最终用官方 COCO AP、逐实例高低拥挤 R75/成对恢复和 GT 空间错误共同判断，不能以 oracle 或单一余弦替代任务收益。',
    f'## 原始结果位置\n\n- 本地：`{RUN.as_posix()}`。已同步逐实例 CSV、逐对表示量、抽样索引、协议、完成记录和复核报告。\n- 远端：`/root/autodl-tmp/coco_clean_20260911/diagnostics/mechanism_official_300_20260911/`；原型与系数 NPZ、逐图 JSON 继续保存在远端。\n- 执行源码：`frozen_mechanism_probe.py`、`frozen_capacity_check.py`、`audit_frozen_mechanism.py`，位于本地及远端实验根目录。\n- 当前正式训练队列历史状态为 FAILED；用户要求的后续训练状态为 PAUSED，未自动重启。'])
    text='\n\n'.join(lines)+'\n'
    # Keep table rows contiguous for correct Markdown rendering.
    text=text.replace('|\n\n|','|\n|')
    out=PROJECT/'refine-logs/coco-evaluation';version=out/f'MECHANISM_LOCALIZATION_{stamp}.md'
    version.write_text(text,encoding='utf-8');(out/'MECHANISM_LOCALIZATION.md').write_text(text,encoding='utf-8')
    (RUN/'MECHANISM_REPORT.md').write_text(text,encoding='utf-8')
    hashes={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*') if p.is_file() and p.name!='LOCAL_ARTIFACT_SHA256.json'}
    (RUN/'LOCAL_ARTIFACT_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(version);print(json.dumps(interactions,indent=2))

if __name__=='__main__':main()
