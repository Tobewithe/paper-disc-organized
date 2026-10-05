# 本地读出实验有限预审

2026-09-12。审查代理audit_readout_input_preflight，same-family / provisional。范围是1200/300缓存、拟合、固定归属评价和全图COCO任务评价的快速代码阅读，不是独立重训、原始GT重建或完整独立复现。

审查返回：未发现使当前运行失效的致命代码错误。官方COCO转换/原生parser/Format的身份与顺序断言、训练仅读取fit及fit归一化、同初始化/参数/样本/预算对照、完整GT官方匹配分母均有对应实现。

非阻断缺口：task evaluator读取缓存和系数结果时记录COMPLETE回执SHA，但未重新遍历验证回执内每个文件hash；当前顺序执行已经完整结束，没有观察到输入损坏。下次重跑宜补充校验。另有空transfer或整臂空预测的边界未处理；当前300图和每臂非空预测没有触发。cache协议中CACHE_BUILDING是历史快照，最终状态应读COMPLETE.json。

synthetic代数/梯度检查不冒充真实实验；本次COCO缓存和评价使用real_gt。后续错误分解和GT像素编辑为主代理实施的描述性诊断，未包含在该独立快速预审范围内。用户要求快速推进，未增加完整审计循环。
