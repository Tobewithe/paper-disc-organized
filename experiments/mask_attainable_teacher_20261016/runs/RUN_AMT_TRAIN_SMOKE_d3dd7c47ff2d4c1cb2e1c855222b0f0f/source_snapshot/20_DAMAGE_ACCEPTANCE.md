# 任务损伤验收：固定有限总体的精确单侧上界

2026-10-13；这是当前待冻结后训练协议的统计补充。实现：[damage_acceptance.py](damage_acceptance.py)。基线成功分母及 Gate 仍以 [执行合同](../../experiments/mask_readout_mechanism_v2_20261012/EXECUTION_CONTRACT.md) 为准。本文确定可实施的估计方法，不是已取得的损伤结果。

## 估计什么

总体是 S0 全量账本中全部普通、基线 segm75 成功 GT，记为 S，大小 N。方法完整预测集合重新经相同 COCOeval 匹配后，GT 从成功变为失败记为损伤。目标是该**固定 COCO 开发总体**的 GT 加权损伤比例 K/N，不是图像宏平均，不是未来图片的风险保证。

图内 GT 可以任意相关。这里不把它们当独立同分布的损伤试验，而把固定 checkpoint 在固定输入/批次/解码配置下的全部损伤标签看成有限标签表，通过**GT 单位的简单随机无放回抽样**产生抽样不确定性。

## 抽样与冻结顺序

1. S0 固定全体 S、预测 JSON 和账本 SHA。S2 的 512 张效果开发图仍按主协议固定，用于 ΔIoU；损伤验收使用另外的抽样设计，不能直接把这 512 张图里的 GT 当简单随机样本。
2. 训练前固定损伤设计及 n=min(1024,N) 的预算，最多需要评估 1024 张不同图片；与效果图重合可复用。n=1024 是工程预算，不能在结果后增大。
3. 固定 **B 的最后一轮 checkpoint** 与推理合同后、查看该方法损伤输出前运行 select。默认用系统随机源生成一次 128-bit seed，再使用 Python random.Random(seed).sample 抽取 GT；保存版本、seed、清单、SHA 和时间。--seed 仅用于事先已登记的随机抽签或复现，不能搜索种子。抽样清单应在首次方法评价前归档。
4. 对抽中 GT 所在图片评估**全部正常预测和全部 GT**，维持基线 batch/rect 几何，不按抽中的 GT 裁剪候选或标签。损伤计数只数清单里的 GT；同图未抽中 GT 不增加抽样量。
5. B 是事前指定的主要验收对象，C 是机制对照。不能根据该损伤样本的结果在 B/C、epoch 或多次配置中挑赢家，再声称同一个95%界仍有原覆盖率。不能边看结果边补样；失败或精度不足按合同停。
6. S3 用同一 checkpoint 完成全部5000图，S 的全体 GT 都观察到，此时报告精确 K/N。有限总体抽样区间自然退化为该数值；这不是面向新场景的统计保证，独立确认仍属于 S4。

方法输出必须确定，推理顺序、batch 和解码若改变导致预测改变，就已经换了待评价有限总体标签表。缓存回放应保持每图原始 P/c0/input_hw；不能为了抽样而重建不同 rect 批次，再把几何变化忽略。

## 精确上界

对 N 个标签中 K 个损伤，无放回抽 n 个观察到 x 个损伤：

P(X=j | N,K,n) = C(K,j) C(N-K,n-j) / C(N,n)。

取 alpha=0.05，单侧上界为：

K_U = max { K : P(X≤x | N,K,n) ≥ alpha }，报告 U=K_U/N。

CDF 使用整数二项式系数及整数递推累计，二分反演；使用 ≥ 保守包含边界，无浮点尾概率下溢。不是直接调用离散分布的中央预测区间。零事件 x=0 时，非全量抽样通常仍有正上界。n=N 时 U=x/N。

这个选择根据[超几何分布的官方公式](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html)进行单侧检验反演；本仓库的区间及验收设计由该公式推导。测试枚举 N≤8 的所有抽样子集核对 CDF，并对 N≤20、全部 n/K 核对漏覆盖概率≤0.05。

- U≤0.01：damage_guardrail_passed，仅损伤护栏通过。
- U>0.01 且 x/n>0.01：damage_guardrail_failed。
- U>0.01 且 x/n≤0.01：blocked_insufficient_precision，不放行，也不伪装成科学失败。
- 缺行、样本变动、预测身份变动、混合 checkpoint、无完整同图 COCO 匹配：工程阻断，不能出损伤通过结论。

例如零事件少量样本也不能直接得出零风险；精确值由真实 N/n 决定，不使用固定“rule of three”近似替代验收。

## 工具与证据

从仓库根目录：

~~~text
python shared/evaluation/damage_acceptance.py select --baseline-ledger <FULL_BASELINE_PER_GT.jsonl> --checkpoint <B_LAST.pt> --inference-contract <INFERENCE_CONTRACT.json> --out <NEW_DAMAGE_PANEL.json>
python shared/evaluation/damage_acceptance.py score --panel <DAMAGE_PANEL.json> --baseline-ledger <FULL_BASELINE_PER_GT.jsonl> --method-ledger <SAMPLED_IMAGES_METHOD_PER_GT.jsonl> --baseline-predictions <FULL_BASELINE_NATIVE.json> --method-predictions <SAMPLED_IMAGES_METHOD_NATIVE.json> --checkpoint <B_LAST.pt> --inference-contract <INFERENCE_CONTRACT.json> --out <NEW_DAMAGE_REPORT.json>
~~~

score 核对样本可复现性、GT 总体与忽略政策、完整同图 GT、基线/方法预测行数与顺序及框/类别/score、所有产物 SHA。方法 JSON 必须按基线筛出相应图片后的原顺序排列。方法账本来自 build_native_ledger.evaluate_ledger 同一官方匹配实现；其命令行的 S0 full/smoke 预算不得偷改成方法阶段，可由 S2 runner 按冻结清单调用该函数。

checkpoint 与合同 SHA 是关联证据，离线 score 不会重新前向证明该文件确实产生了预测。真实 runner 仍须保存生成日志、checkpoint SHA、输入/缓存 SHA、无 GT 前向证据、完整输出重放核验；不能手填声明代替执行。

S3 全量评价后仍沿同一方法；在事前登记的全量分支以 --sample-size N 生成覆盖所有 S 的全量清单（B/checkpoint不变），score 全量方法账本。不得把这一事前计划的 S3 全量验收挪到 G2 精度不足后，作为临时补样放行手段。
