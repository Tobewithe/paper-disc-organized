# PCDCR 第一版：Stage I 已完成并结案

**没有达到预注册放行条件，不进入 Stage II。** 本轮实际执行了8图完整链路检查、B/C/D各12轮训练、统一dev选模、正确/错配条件及单标量校准评价。未改变原模型、候选、框、标签绑定或解码。

评价对象为196张未参与本轮训练及选模的图片、1,346个固定官方one-to-one TAL候选；这些图片已被此前研究使用，因此不是新盲测。本轮也不是完整COCO AP实验。

| 必答问题 | 结论 |
|---|---|
| Q1：优于参数匹配静态头吗？ | 未建立优势。图片macro原图IoU差−0.0153个百分点，95% CI [−0.0685,+0.0506]；低于0.2个百分点实用门槛。静态头完成12轮后由dev选中了epoch0，因此所选输出等于原模型，并不代表静态函数类的最优水平。 |
| Q2：优于普通条件拼接吗？ | 差+0.0241个百分点，CI [−0.0227,+0.0775]，未建立可靠优势；两者都没有超过原模型。 |
| Q3：正确条件优于错配吗？ | 差−0.0036个百分点，CI [−0.0097,+0.0023]，未建立正确对应关系的优势。 |
| Q4：支持排序或空间组合改善吗？ | 不支持本轮改善主张。相对原模型AUC下降0.0100个百分点；相对dev选定的单bias，IoU差−0.0645个百分点且区间跨零。这不证明PCDCR与阈值完全等价。 |
| Q5：增加修复并控制损伤了吗？ | Mask75修复0个、损伤3个，达标数831→828，没有净恢复收益。 |
| Q6：进入Stage II吗？ | 否。按预注册停止当前版本，不追加rank、输入、门控或解冻实验。 |

上述阴性结论限定于固定166维条件、rank8、seed0、12轮训练及统一选模规则；不能据此证明所有原型条件化方法无效。错配替换整套条件，包含原型统计、原系数和框几何，也不能单独检验“prototype是否唯一缺失变量”。

完整统计、方法、数据范围和限制见[正式报告](runs/RUN_stageI_seed0_20261002/REPORT.md)。本入口是该冻结Run的结果摘要，数字以Run内机器产物为准。

- [RESULTS.json](runs/RUN_stageI_seed0_20261002/RESULTS.json)：全体及预定义分层、5000次图片配对bootstrap。
- [PER_IMAGE.json](runs/RUN_stageI_seed0_20261002/PER_IMAGE.json)、[PER_CANDIDATE.jsonl](runs/RUN_stageI_seed0_20261002/PER_CANDIDATE.jsonl)、[ABLATION_TABLE.csv](runs/RUN_stageI_seed0_20261002/ABLATION_TABLE.csv)：逐图、逐候选和对照表。
- [DECISION.json](runs/RUN_stageI_seed0_20261002/DECISION.json)：放行判定、条件行为及来源版本。
- [BASELINE_REPLAY_AUDIT.json](runs/RUN_stageI_seed0_20261002/BASELINE_REPLAY_AUDIT.json)：dev和val共2748个候选全部与历史正常原图基线一致，最大IoU差0。
- [run.json](runs/RUN_stageI_seed0_20261002/run.json)、[transfer.json](runs/RUN_stageI_seed0_20261002/transfer.json)：实际运行和回传记录；44个文件校验一致。

正式训练在笔记本28358lan的RTX 4060 Laptop GPU上执行，训练耗时262.391秒，评价105.734秒。训练使用已有冻结特征缓存，只更新新增的小型读出模块；这些耗时不能当作完整YOLO训练或端到端推理耗时。正式Run退出码0、产物齐全，科研判断为STOP_STAGE_I；两者分别记录。

最初准备启动中断的日志与未知退出状态保留在[runs/RUN_prepare_20261002](runs/RUN_prepare_20261002/)，随后准备重试成功。没有复用失败退出码来判断方法，也没有因运行中断改变实验参数。笔记本本轮训练已停止，临时单次后台任务已清理。
