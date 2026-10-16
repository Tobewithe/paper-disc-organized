# CLD-v1 原400图留出评价执行登记

用户明确授权在已授权笔记本继续评价原400张留出图，固定已训练完成的REAL/PERM最终checkpoint，按原协议检查方向、修复收益与损伤，结束复核配置并按实验展示范式汇报。本登记仅启用已冻结的评价范围；不改变科学协议、模型、执行源码、数据、幅度、门控、统计或门槛。

## 输入及权限

- REAL：`RUN_CLD_FULL_REAL_bc58d9f5253847e588eb01ce2a309d7b/last.pt`，SHA `869d44ca30876cec7a243f774d3b82994d1a7bdcad85347cb646f88015003ff8`。
- PERM：`RUN_CLD_FULL_PERM_a131a5d6626845878b5b93b9b51a7bfa/last.pt`，SHA `8aa44343099384990324c7a1e9acae5beb982a3d2fe948c685216df68f467d20`。
- 两组均已完成1200步并独立核验；last-only身份已经在`TRAINED_CHECKPOINTS.json`冻结。本轮不训练、不换checkpoint、不更新初始化或重新求教师。
- 原`PROTOCOL.md` SHA `053aaefb1cbb0fd5f87dad2dabdf9e961660201ae81bae6c8e42e67d61d6d453`；原`EXECUTION_FREEZE.json` SHA `56e9f26f4ebf2f56df92de068044ccaafdae60278b6b91b9a204751058fa12a8`；其中既有evaluate_cld.py、cld_statistics.py、verify_evaluation.py、verify_cld.py保持原字节。
- 操作授权只开放完整评价和独立复核，关闭正式训练/smoke。旧训练授权按字节保存到`authorization/OPERATION_FORMAL_TRAINING_20261016.json`；历史正式控制冻结仍对应`formal_control_snapshot/`中的原授权，不能拿新授权重判旧Run。

## 实际评价与判定

原留出400图/42167行完整输出不变。581条/208图接受教师用于GT辅助等幅方向R/P/J/N对照；另按原协议读取拟合2510条/867图的方向表现作描述，不做梯度或模型选择。两模型生产forward预计2×(400+867)=2534次；独立核验再次执行2534次。基础模型forward、优化更新及新oracle均为0。

正常无GT候选M/S使用共同a=2.1651711144258443、原生面积≥9216门控和完整混合预测；保留原生几何、batch16/rect=True缓存、decoder及Box/Class/score/输出身份。原T490/203图、全部2246配对/394图、1808原成功GT全查不改。G为原soft-IoU+面积门控历史参照，训练过全部2000图；其输出差值不解释成同预算留出泛化比较。

K_dir：R−P、R−J、R−N均须图像宏均值≥.005且95%下界>0。K_use：M−N目标≥.005且下界>0，全部配对≥−.001，M−G目标区间下界>0，损伤≤18/1808及身份护栏通过。固定seed1729、2000次配对图像bootstrap。阳性/阴性/信息不足及后继动作完全按原协议第7节，不修改数值或挑选子组。

## 控制、预算与交付

启动前只读复核源码、权重、全部2000图/缓存/标注、旧参照文件、两个last及各Run原始产物和独立回执；重查资源/真实进程/锁。既有smoke已通过，不重新训练smoke。`scripts/prelaunch_evaluation.py`只读核身份，新增报告不作为模型Run。

`scripts/launch_evaluation.ps1`仅串行调用原冻结评价器及verify_cld.py。独立单次Windows任务`Codex-CLD-Evaluation-20261016`无周期触发、无重启策略；控制器状态、日志和控制快照单独保留，以实际运行时生成的Run ID登记。评价≤2小时，独立复核≤1小时；失败停止并保留错误，不自动重跑或换协议。控制器只持有本Study的EVALUATION_CONTROLLER.claim；实际评价器使用原.CLD.lock。结束停用本单次任务，不修改其它任务。

结束完整回传Run与控制产物、核SHA；笔记本独立重新forward、RLE/完整任务账本和统计核验，本机从回传逐行结果独立复算分母、图宏均值、区间、损伤及门槛。共享的官方decoder/COCO引擎明确标注，不能称第二套独立算法。写整体实验REPORT/配置复核、更新Study及ACTIVE_PLAN并检查路由。旧协议、训练结案及阴性保留；不自动启动S2/S3/S4，不提交推送。
