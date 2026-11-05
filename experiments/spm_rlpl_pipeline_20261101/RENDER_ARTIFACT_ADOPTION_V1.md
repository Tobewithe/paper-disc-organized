# 渲染产物与调度终态分离：仅994/998的明确来源采用

## 已有事实与修正判断
988/990/992/996为Slurm COMPLETED/0:0。994/998为TIMEOUT/0:0，实际分配1827/1824秒，**未满足单片≤30min资源要求，原状态保留**；不改原Run、原合同或调度记录。六片合计10674 GPU分配秒，未超过六片原合计3GPUh及生产总9GPUh预算。

回传比进度观察更完整：994/998各99图/33组、最终`SHA256.json`、根`RESULT.json`、应用completed账本均存在，stdout均正常到`Blender quit`。`run.json.ended_at`位于最终全文件哈希之前，不能把它当作进程结束时间。此前“需要CPU补齐缺失最终哈希”的预判据此修正：最终清单实际已经写完，不应另加一次重复的大文件哈希作业。

## 唯一允许的继续方式
在本地独立来源对账Run中，核两份真实回传SHA、Run/SHA清单、99个唯一group/view、33组三视图、全部RGB/EXR/保存场景的逐图即时摘要与最终清单一致，核每份record的合法/同渲染声明和实际完成stdout。通过才生成不可静默改写的`RENDER_ARTIFACT_ADOPTIONS_V1.json`，仅允许其记录的**同job、同Run、同run.json SHA和同产物清单SHA**被下游采用。

这不是把TIMEOUT判为成功：分别记录`application_status=completed`、`scheduler_status=TIMEOUT`、`resource_protocol_met=false`、`artifact_manifest_complete=true`。当前仅采用完整已生成产物，不代表像素/射线或数据release通过。下游decode按原SHA重读RGB/EXR，rays按原SHA重读保存场景，原release仍逐文件全量重哈希；任一不一致即停止，不换图/降分母/重渲染追结果。

本修订0新增GPU、0新增Slurm作业、0优化，不追加训练/样本/渲染或修改科学Gate。它落实执行状态与产物状态分离，并复用已有证据，避免重复诊断。此来源例外不适用于其它超时任务，尤其不能用于绕过正式优化步数或恢复规则。
