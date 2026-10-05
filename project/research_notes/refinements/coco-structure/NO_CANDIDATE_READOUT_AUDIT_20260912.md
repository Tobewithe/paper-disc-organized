# S015审计状态

2026-09-12。实验和执行者确定性验证已经完成；fresh GPT-5.6-Sol ultra代理 /root/audit_no_candidate_readout 在交付时尚未返回语义审计意见。记录为ERROR / reviewer_verdict_not_returned_at_delivery，acceptance_status=pending，不能称审计通过。same-family描述代理关系，不代表已获得provisional PASS。

已提供原始与补充脚本、317目标状态、每目标读数、原始COCO RLE复核、收敛见证、范数见证、报告和完整归档，要求检查真实GT来源、归一化、结果存在性、实际执行路径、范围和评价类型。代理在较长审查后被要求按现有证据给出有限结论；未将无回复解释为同意。

执行者确定性检查已完成：310项主文件/分析校验；远端最终318项清单hash与本地相等；317个目标分母完整；原图像素IoU算术误差0；原始掩码官方重放、原型插值误差阈值检查及保存系数重新解码均通过；254个自由系数恢复原范数后二值掩码0像素差异。见LOCAL_VERIFICATION.json与SCALE_WITNESS.json。

这些检查不是独立语义审计，不证明原型表达上界、无GT方法有效、跨图泛化或三训练种子稳定性。报告主结论限定于已保存和可重放的同图GT辅助解。主结果220/254采用原始COCO RLE域；crowd排除有效域228/254单列，不混口径。
