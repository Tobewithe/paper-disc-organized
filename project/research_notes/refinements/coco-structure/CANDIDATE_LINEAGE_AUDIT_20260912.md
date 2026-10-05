# S014 实验完整性审计

日期：2026-09-12。审计者：fresh GPT-5.6-Sol ultra，任务 /root/audit_candidate_lineage。review_independence: same-family；acceptance_status: provisional。

## 审计意见：WARN

新审计代理直接读取实验脚本和结果，已返回阶段审计意见：主 S014 代码/结果未发现致命完整性缺陷。以下保存该代理已返回的判断，额外校验整改由执行者记录，不将其冒充独立重算。可报告受限的GT辅助诊断结果，不能报告为无GT方法收益。

### A. GT来源：PASS

真实COCO实例标注，任务评价保留crowd，由COCOeval处理忽略。固定bbox50一对一归属来源是COCOeval，mapping唯一性有断言。证据：frozen_mechanism_probe.py:93；candidate_lineage_probe.py:112；relative_ownership_experiment.py:57。

### B. 分数归一化：PASS

没有将成绩除以模型自身输出最大值；AP/R75使用官方定义。空间指标除以自身有效GT面积，并保留crowd与自身GT优先规则。替代候选不修改类别/分数，GT最优选择明确标注oracle。证据：candidate_lineage_probe.py:189；relative_ownership_experiment.py:95。

### C. 结果存在与重放：PASS（限定文件与执行链）

300/300图NMS保留顺序精确重放；raw→final候选身份与最终掩码重放通过。主汇总和1,200份RLE预测的hash由审计者检查，无不符。官方原始Mask75逐GT命中重放通过。证据：candidate_lineage_probe.py:112；summarize_candidate_lineage.py:43；ANALYSIS_COMPLETE.json。

审计者指出原COMPLETE只覆盖汇总CSV，ANALYSIS_COMPLETE补充预测hash但不覆盖逐图NPZ/JSON。执行者已补远端FINAL_ARTIFACT_SHA256SUMS.txt，1,819项含逐图中间产物、预测及补充结果，本地完整校验通过。该后补校验属于执行者确定性校验，不改写审计独立性。

### D. 评价路径：PASS

实际首个NMS压制者在重放循环内记录；一对一最大匹配算法有效。官方重复型FP计算中，COCOeval.ious行对应分数排序预测、列对应原_gts顺序，普通GT列选择和dtIgnore/dtMatches对齐正确。证据：candidate_lineage_probe.py:38；candidate_lineage_probe.py:70；summarize_candidate_lineage.py:48。

### E. 范围：WARN

重复探索的密集富集300图；GT选择替代，172重复来源槽位允许并记录；每个GT的独立最优不等于全局最优分配。score/NMS阶段候选预算不同，不把最大匹配解释为实际召回/AP。bootstrap没有多重比较校正，也不体现训练种子不确定性。密集专属性、可学习性、新颖性及独立泛化未确立。

审计者指出tracker仍RUNNING，执行者已更新COMPLETE，并保持正式训练停止。补充0.02框质量容差在主结果后选择，标注事后描述控制，不能追溯称预注册。

### F. 评价类型

real_gt_oracle_diagnostic：真实COCO标注用于候选选择及评价。四组任务数值是GT辅助干预结果，不能称可部署方法。

## 未独立复现部分

审计者本地没有原始COCO JSON，未从原标注重新计算全部官方AP，也未重新运行模型推理。校验了代码、输出、凭证、hash和协议限制。原注释文件内容、实际GPU运行不能只由输出hash保证；不能称完整独立复现。

## 论文影响

支持：固定原框和原型时，已有候选系数存在改进空间；本实验合格候选丢失主要分类为同GT候选压制；候选质量/保留关系值得继续验证。

不支持：CCL或正交化因果链、候选筛选解释多数密集失败、无GT新方法收益、全COCO/独立确认/三种子稳定性。
