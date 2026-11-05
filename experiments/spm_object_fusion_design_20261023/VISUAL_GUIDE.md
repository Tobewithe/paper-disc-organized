# V6 技术与结果图集

全部图由真实训练/已存预测或选中权重重放产生，PNG/SVG成对保存。当前图分为原GPU/CPU Run中的冻结产物，展示版修订另列，不改原始数据。

| 图组 | 读图含义 |
|---|---|
| [全部训练与AP曲线](runs/RUN_SPM_V6_ACCEPTANCE_CPU_e77f5712304d/figures/00_training_and_outcome.png) | B/F loss下降，但每个固定检查点F均未超过B，两者都低于共同起点 |
| [官方成功转移](runs/RUN_SPM_V6_ACCEPTANCE_CPU_e77f5712304d/figures/01_success_transfer.png) | 2090共同成功、11损伤、5修复、757共同失败；不是同一预测的Box/Mask拼接 |
| [损伤案例GT10](runs/RUN_SPM_V6_ACCEPTANCE_CPU_e77f5712304d/figures/case_00_damaged_gt10.png) | 同检测IoU0.7627→0.7435，绿色TP、红色FP、蓝色FN |
| [修复案例GT532](runs/RUN_SPM_V6_ACCEPTANCE_CPU_e77f5712304d/figures/case_02_repaired_gt532.png) | 同检测IoU0.7381→0.7544；不代表全量净收益 |
| [系数与空间修正GT10](runs/RUN_SPM_V6_ACCEPTANCE_GPU_8f53fe7cfb2c/figures/trace_gt_10.png) | Δc^T P、(c+Δc)^TΔP与其和；参照是F自身原生Mask路径，不是F−B |
| [部位响应/背景/熵GT10](runs/RUN_SPM_V6_ACCEPTANCE_GPU_8f53fe7cfb2c/figures/attention_gt_10.png) | 末层实际注意力按8部位聚合；不把token注意力当已验证的身体对应 |

其余固定案例为GT37、582、18、27，与[CASE_SELECTION](runs/RUN_SPM_V6_ACCEPTANCE_CPU_e77f5712304d/CASE_SELECTION.json)一致；GPU目录各有trace_gt_ID、attention_gt_ID，CPU目录有6幅RGB/B/F/GT图。共20组PNG/SVG，6份原始NPZ图层。full160网格空白处为窗口外；全图统一色标可能压低局部变化，解释时应同时读原始数值与局部展示。


展示版补充：[局部响应GT10](runs/RUN_SPM_V6_PRESENTATION_22db0b1608ce/figures/gt_10_attention_crop.png)、[6例定量摘要](runs/RUN_SPM_V6_PRESENTATION_22db0b1608ce/figures/response_summary.png)、[源数据与SHA](runs/RUN_SPM_V6_PRESENTATION_22db0b1608ce/RESPONSE_SUMMARY.json)。局部图每幅显示绝对色标，不同面板范围不同，颜色对比不能直接当作强选择性。归一化熵为先平均4头权重后计算，不代表各头单独的熵。额外7组图来自既有NPZ的CPU重绘（Slurm248，9秒，0前向/0优化/0GPU），不覆盖原始Run图。
