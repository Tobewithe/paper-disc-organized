# S2配置复核：执行前后及两机独立核验全部通过

本次评价使用训练完成的唯一last checkpoint；相对原soft-IoU方法，训练科学变量仍仅为监督目标策略。原训练协议和冻结34份文件未变。S2数值规则沿用原协议第5节，新增执行细则只登记评价实现与预算。

| 范围 | 冻结与实效 | 当前核验 |
|---|---|---|
| checkpoint | AMT原2000图/3epoch/1500步last，b39df4f2… | 训练独立审计、文件SHA及实际加载state身份通过 |
| 基础模型/输出 | 原YOLO26m-seg权重、P/c0/F、box/class/score/输出顺序 | 基础模型forward=0；1311图/188708条zero-delta RLE全部精确一致 |
| 输入几何 | 原imgsz640/batch16/rectTrue缓存，每图保留原input_hw和ratio_pad | 不重新组合rect批次；adapter仍按原单图完整系数矩阵读取 |
| 原生参数 | end2endTrue、agnostic_nmsFalse、conf.001、iou.7、max_det300、mask_ratio4、overlapTrue、retinaFalse | 原profile、实际源码和原缓存绑定核验通过 |
| 解码/面积 | 原native两阶段decoder；G仅按原生原图RLE面积≥9216选择；门控外逐字恢复原生RLE | 原decoder/area_gate代码复用；烟雾及独立RLE核验通过 |
| 环境 | Ultralytics8.4.100、torch2.5.1/CUDA12.4、NumPy2.4.3、pycocotools2.0.11、faster-coco-eval1.8.0 | package版本、导入路径及源码SHA均与原冻结配置相同 |
| 数值模式 | FP32、matmul highest、CUDA TF32=False、cudnn TF32=True、benchmark=False；其余runtime_policy完全沿用 | 运行前后完整environment与原配置相等，数值模式无漂移 |
| 评价 | 同完整JSON含bbox，bbox/segm独立COCO匹配，maxDets1/10/100，useCats1，完整同图GT | 烟雾原生账本及全部正式bbox身份通过；笔记本独立重跑G全部COCO匹配一致 |
| 分母 | 原dev512/T655/Box75配对2925；22957成功GT中一次新SRSWOR1024；并集1311图 | 两机清单、固定seed重抽复核和合同绑定通过；不按门控缩分母 |
| 原对照 | 旧G开发效果＋旧S3全量混合预测，在新图片并集上重匹配 | 预测SHA固定；2925条旧G配对端点精确重放，原G目标均值完全一致 |
| 统计 | 图内均值再图等权，2000次配对bootstrap、default_rng1729、linear；精确超几何单侧95%损伤界 | 复用冻结原函数；两机复算区间、损伤、修复计数和裁决完全一致 |
| 成本 | 新增训练0，基础模型forward0，唯一新adapter推理及原生解码核验 | smoke/full各独立Run，原对照不重训、不重新前向 |

输入48项、源文件47份；见[EVALUATION_CONTRACT.json](EVALUATION_CONTRACT.json)、[S2_SOURCE_FREEZE.json](S2_SOURCE_FREEZE.json)、[S2_PREPARATION.json](S2_PREPARATION.json)、[本机启动绑定核验](S2_LOCAL_LAUNCH_VERIFICATION.json)。执行后配置与独立核验均通过；科学结果为S2_development_failed，配置通过不等于方法通过。

本机独立统计核验环境与笔记本推理环境分开记录：本机Python3.12.14/NumPy2.3.5仅解析回传证据和复算统计，不生成模型预测，也不作为推理配置替代。冻结统计脚本在两机给出完全相同的端点、区间、损伤和裁决；没有用更换环境生成新的预测或放宽容差。

官方发布52.5/44.1尚未复现与原生GT账本有效是独立状态。当前本地原生全5000图Mask AP为44.01417484774351；本次1311图AP只作辅助，不与5000图或官方发布值直接相减。

正式Run `RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357`；smoke与full交集8图/1470行四组输出完全相同。两Run各83个最终执行端文件，共166项回传大小/SHA核验一致，含独立匹配回执。独立任务退出码0，无Python进程、无锁、无自动触发。

证据：[Run配置审计](runs/RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357/CONFIG_AUDIT.json)、[独立匹配与统计](runs/RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357/INDEPENDENT_LAPTOP.json)、[本机统计](runs/RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357/INDEPENDENT_LOCAL.json)、[回传核验](S2_TRANSFER_VERIFICATION.json)、[结果与方向裁决](REPORT.md)。
