# 本地 Codex 当前入口：冻结预训练权重的 Prototype→Coefficient 后训练修复

> **执行权约束：** 首先核对 [当前科研总控](../../project/ACTIVE_PLAN.md) 中唯一激活 Gate 及本地真实任务状态，避免多个 Codex 会话并行启动同一个阶段。本文只在获得本次 Study/环境授权且上一 Gate 真实验收后允许继续。

**2026-10-13 用户明确主线修订：不以 TAL/历史预训练分配作为科研目标，也不把 10/12 的评价纠正与审计当作转向 TAL。** 本 Study 当前唯一执行规划是 [POSTTRAIN_RESCOPE_20261013.md](POSTTRAIN_RESCOPE_20261013.md)。原 `PROTOCOL.md` + `GATES_SPEC.json` 仍保留为**未冻结的旧草案**，其 E0 TAL 前置、5 臂×2seed×20k 启动路由现在**不生效、不得按它发起新训练**。不要覆盖任何实际历史 Run，也不要把新规划写成已经批准/执行的实验结果。

## 第一次进入（单一工作流，不追加旁支）

1. 在本地仓库检查 `git status`、当前分支及用户改动；读根目录 `AGENTS.md`、项目通用规范、`paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md`，再读上述新规划和相关历史证据。保持过去项目现有 Run、论文与旧草案不变。
2. **首先核对已经计划的** `experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md`：取得 v8.4.100 同 SHA 官方权重的原生正常最终输出、独立 bbox/segm COCOeval 对同一预测集合的逐 GT 任务账本，检查 `(image,output_detection_id,raw_id_if_available,class,score,Box,Mask)` 身份。官方公布 52.5/44.1 parity 与当前配置账本有效性独立判定；不可把本地自定义 51.980/43.518 无说明混为同一个来源。
3. **S0 合法后，按总控转入 G0 身份桥接与 G1 分析：** 在已经固定的正常最终输出中，取官方 segm75 任务失败且有正确类别 Box75 COCO 配对的 GT，再限定该**同一检测** MaskIoU<.75（集合 T）；在其同一 P、c0、预测框和 decoder 上按冻结 finite oracle 检查是否存在改 c 后可恢复的机会。复用旧求解器/缓存，仅重算涉及的最终输出检测，不重复扫描数千万 raw，不触碰 TAL。
4. G1 不通过预设机会门槛：在 `DECISION.json` 记录 `STOP_C_ONLY_OUTPUT_RESCUE`，不延长训练、不新增五臂、也不推断“P 无能力”。只有 G1 通过且资源/清单/版本/无 GT 输入/拟训练权重白名单 G0 工程检查通过，才准备 2000 train2017 初筛的一个 zero-init output-conditioned coefficient residual adapter；基础官方权重（包含原生 cv4 与 P）全部冻结。
5. 训练关联只在**训练标签构造**时使用 GT 类别 + BoxIoU≥0.50 一对一配对最终检测行，**不是 TAL**。模型推理、激活策略、候选选择不使用 GT。保留 baseline 成功样本保护，损伤和 Mask AP/真正恢复数一起按预注册门槛评价。
6. 小样本 G2 通过后，先用同一 pilot checkpoint 完成 G3 全 5000 图 AP/损伤验收；两者均通过后才按 `POSTTRAIN_RESCOPE_20261013.md` 的资源顺序用最多 20k train2017、多个 seed 和真实未见场景作必要确认；否则关闭当前后训练配置，不另试增量 teacher/gate/solver。不得以“之前 TAL E0 未完”阻止后训练 G1，也不得绕过 G1 提前启动大训练。

先读 [执行合同](EXECUTION_CONTRACT.md) 检查 P/c0/输出身份桥接和每阶段冻结项；按 [本地协同规范](../../project/科研项目通用规范/运行登记与本地协同.md) 接续真实 Run，不把旧 GATES_SPEC 当新 runner 配置。

## 工具入口与新配置

先按 [S0/S1 工具说明](../../shared/evaluation/POSTTRAIN_TOOLS.md) 运行合成检查；本地正式使用必须加 `--require-integration`，不能把缺依赖的 skip 当通过。[机制假说与预设配置](MECHANISM_AND_DEFAULTS.md) 固定 B/C 输入消融及数值/训练候选预算。已提供原生 P/c0/F trace、实际 Top-K raw 映射及 native decoder 重放代码，固定版本源码哈希并逐行验证 RLE；真实模型 smoke 尚未执行，不能宣称 S1 已跑通。先运行工具说明中的两个 --require-integration 检查，再以最多16图的独立 smoke Run 核真实 trace/replay。

C 已改为联合空间置换，不再用输入置零解释空间机制；GT/保持 loss 分别归一化。[损伤验收](../../shared/evaluation/DAMAGE_ACCEPTANCE.md) 固定为成功 GT 总体的简单随机无放回抽样精确上界，训练前登记预算、固定 checkpoint 后抽签，不能看损伤结果后补样。完整 adapter 训练 runner 仍需在获得相应阶段授权后接入这些组件。

## 冻结与交付

所有新科学运行必须在实际执行前保存明确版本、数据/预测清单 hash、候选身份、研究对象、允许修改参数、门槛及来源 Run；按工作台要求为新 Run 分配真实 ID 和 `run.json`。工作完成时交付：`G0_OUTPUT_IDENTITY`、`G1_FIXED_OUTPUT_ORACLE`、`G1_SUMMARY`、`DECISION`，必要时才有 G2/G3 checkpoint 和 COCO 结果。缺少数据/权重/结果就返回唯一阻塞，不伪造。用户历史已关闭路线、官方 COCOeval 匹配、GT 辅助 oracle 和此后训练方法的成绩不能相互替代。

**当前状态：规划重定向已写入，G0/G1 未在此会话执行，新 Adapter 训练未授权启动。**
