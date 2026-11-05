# 固定检测行条件下的背景动作成本 v4

正式优化0，第二轮端到端审查修正TRAINING_PROTOCOL_V1中“真背景按空Mask IoU计算成本”的不当代理，继承所有正owner实际640/独立native保护成本、unknown KEEP偏好、训练更新/数据/8epoch/超参数/18GPUh与完整评价Gate。

严格真背景由owner合同定义：正常box B内无任何GT前景。由于所有候选Mask只能在B内，任意动作A与任意GT的交均为0，其IoU=0。bbox/class/score/排序/检测行身份冻结，且两臂COCO loadRes保留bbox共同面积策略，故该行始终是同一计分FP；擦空Mask不能消除这条FP或提高AP。以空GT的empty/empty IoU=1奖励删除，会把无AP用途的擦除错误当成收益。

本版真背景C_KEEP=0、所有edit=.002，仅表达最小操作偏好；unknown仍KEEP0/edit1.002，已知正owner维持实际640成本和nativeMask75保护不变。候选A阶段的真背景task空Mask监督继续保留，以避免对象幻觉；只修冻结candidate后的操作目标，不改图/owner/主任务/部署或评价输出。COCO全图反例含高分背景FP和真TP，证明擦空后全部12stats/AP及FP身份不变，成本不再奖擦除。旧设计/Run保留，源更改后两轮审查重新计数。
