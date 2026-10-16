# CLD-v1 正式两臂训练执行登记

用户在工程结案与“唯一下一步是从原initial启动REAL/PERM各1200步正式匹配训练”后回复“继续”。本次据此在已授权笔记本YANGLEI执行两臂正式训练、训练结束后的配置/独立核验与产物回传。先前只允许工程smoke的操作授权按原字节归档，再登记本次授权；不覆盖科学协议、执行源码或既有Run。

## 本轮范围

- 同一个原initial.pt分别启动REAL和PERM，每臂1600拟合图×3轮=4800图访问、1200个AdamW更新；原拟合清单与三轮顺序、17参数/167872容量、所有optimizer flags、FP32及AMP关闭不变。
- a固定为预检Run给出的2.1651711144258443；方向MSE、.1全输出保护、实例置换均执行原冻结代码。本轮只授权训练及训练审计，`allow_full_evaluation=false`。
- 使用既有预检 `RUN_CLD_PREFLIGHT_a436d52ceed643638c6f32be616f9cfe` 和smoke `RUN_CLD_SMOKE_REAL_396e8c25780f4357b21bc76c7431db2a`、`RUN_CLD_SMOKE_PERM_0761445d8fd2437795f783d6b6acdba8`；只读复核，不重复运行smoke，不把smoke last作为正式初始化。
- REAL训练→独立训练审计→PERM训练→独立训练审计。每个正式训练最多2小时、独立核验最多1小时。工程失败或预算耗尽停止，原错误与已消耗预算保留，不自动重训。
- 两个正式last全部完成后保存明确SHA，才具备下一阶段留出模型评价的前提。本轮不读取留出标签开展模型评价、不计算新AP、不启动正式S2/S3/S4、不自动提交推送。

## 运行控制

`scripts/launch_formal.ps1`仅串行调用原冻结`train_cld.py`和`verify_cld.py`，不实现新的模型计算。控制脚本与本执行登记、实际操作授权另以`FORMAL_CONTROL_FREEZE.json`固定；原36项执行冻结与11项科学冻结不变。

为使训练不依赖SSH连接，采用一轮Windows计划任务 `Codex-CLD-Formal-20261016`，不设置周期触发、自动重启或下一实验。控制器保存`FORMAL_STATUS.json`、独立stdout/stderr日志和控制源码快照；每项真实训练的Run ID由冻结runner实际启动时创建。控制器仅持有本Study的`FORMAL_CONTROLLER.claim`，每个训练本身仍原子持有`.CLD.lock`。任务结束停用本任务；不改变其它任务。

启动前核真实进程/锁、全部2000缓存/原图、输入和权重SHA、两个smoke/预检产物、科学及执行冻结和资源。执行后的独立训练审计核4800图访问、1200更新及逐参数optimizer计数、初始化/参数白名单/环境、仅拟合目标与last合同；它不重新训练，也不声称用训练loss判定机制。

## 结果解释

本轮只回答固定预算训练是否完整且可核验。相对原生N、历史G和REAL/PERM之间的修复收益、损伤与机制Gate均要等冻结的留出评价，不用训练loss代替。训练完成后按整体实验链路汇报当前证据和唯一下一步，保留工程阶段和历史科研结论。
