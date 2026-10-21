# 三臂联合适配结案配置复核

2026-10-19；复核范围为既有训练、评价与统计产物。新增训练步数、模型前向、新bootstrap样本均为0。结论：**核心配置、实际训练预算、产物身份和统计归约核验通过；科学主Gate仍失败。**

| 核验项 | 实际证据与结论 |
|---|---|
| 协议与配置身份 | PROTOCOL SHA `4d7c983e70f6d9c62266f768b2738cd4acf0d0e0249bd5ac746c5dd65a9f9f97`；CONFIG SHA `c885b1248b80b6d74e41ad3df93a5e0f608e6c3ee2bde1466127b96d465739ca`；10个真实Run冻结副本、源码快照和最终summary绑定一致。 |
| 冻结清单 | FROZEN_MANIFEST SHA `01726a24656636ab7e0f8c865cb3fda7473269755e035149a9759e406abf1948`；438个文件在服务器重新核验通过，本机独立SHA复核通过。未修改冻结源码/协议/参数。 |
| 实际环境 | Python3.12.3、torch2.5.1+cu124、torchvision0.20.1+cu124、Ultralytics8.4.100、numpy2.4.3、pycocotools2.0.11、faster-coco-eval1.8.0、OpenCV4.13.0.92、Pillow11.1.0、PyYAML6.0.2。8项关键Ultralytics源码pin一致。服务器RTX A6000；本机复核仅用Python标准库，不作为模型环境。 |
| 官方基础权重 | `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；其本体始终冻结。C/PC在各自私有掩码分支副本按白名单学习，不把PC误记为P固定。 |
| 数据与拆分 | train20k=原fit2000+固定种子抽取18000；val5000；无train/val图像ID交集。标注、25000张图的清单、1250训练源批/313验证源批及3轮微组顺序绑定。三臂TRAINING_NATIVE_IDENTITY与TRAINING_JOIN文件的独立远端SHA相同。 |
| 历史预处理恢复 | 验证5000图逐图历史输入/几何精确重放通过；原版本等宽高比排序造成的失败Run保留。未取消身份断言、改batch或替换图像。新服务器原生P/c跨设备逐位相等不是冻结要求，采用同服务器Ns/G作为比较锚点。 |
| 推理实际配置 | imgsz640，源batch16，rect=True；FP32，conf=.001，iou=.7，max_det300；O2O end2end=True、agnostic_nms=False；mask_ratio4、overlap_mask=True、retina_masks=False；无增强。原生mask面积9216门控不变，门外逐行复制Ns RLE。 |
| 数值开关 | highest matmul；matmul TF32=False，cuDNN TF32=True；benchmark=False、cuDNN deterministic=False、deterministic algorithms=False；CUBLAS `:4096:8`。三臂实际ENVIRONMENT与合同一致。 |
| 训练配方 | AdamW lr=1e−4、wd=1e−4、betas=(.9,.999)、eps=1e−8；foreach/fused/amsgrad/maximize/capturable/differentiable均False；无AMP/EMA/调度/裁剪/增强/早停。同ROI soft-IoU+0.1原生概率保护BCE，实例→图→4图均值。 |
| 预算与实际记录 | 各15000全局优化步、60000图记录、3轮；独立逐行检查每条训练记录的epoch/step/image_id与预定顺序完全一致，共180000条。未追加训练。 |
| 参数与梯度 | G167872、C854496、PC3158016；三臂全部白名单参数在正式训练出现非零梯度；PC的P/系数共32张量均有15000次有梯度更新。基础256个state张量前后核验通过，保持eval。 |
| checkpoint选择 | 三臂last第15000步是唯一正式checkpoint，训练summary/CONFIG_AUDIT/远端文件新SHA/评价summary四方一致；epoch1/2文件仅保留，未选用。smoke更新6步独立记录且弃用，不计入正式45000步。 |
| 评价与分母 | 五方法同5000图、539125输出、36335普通GT；同完整bbox+segmentation JSON及COCO默认阈值/类别/面积/maxDets语义；bbox/segm独立匹配。pycocotools与faster-coco-eval AP逐方法一致。 |
| 原生迁移 | Ns Box/Mask AP52.347950/44.013836，旧G44.152393；迁移误差在原.02 AP容差内。当前原生成功GT22958，历史22957的7条丢失/8条新增有单独清单；损伤采用当前完整分母。官方发表parity仍失败，与有效GT账本分开。 |
| 检测身份与损伤 | 服务器逐输出框/类/score/顺序全等断言通过；本机五份逐GT账本的GT身份、ignore状态、Box75状态及bbox匹配检测序号再次核对一致；逐GT精确重算原生损伤/修复以及相对旧G的变化。 |
| 区间统计 | 原2000×5000共享整图样本、seed20261018；五方法2000×5已保存AP数值与2000条进度记录逐项吻合。本机独立线性分位数实现重算全部报告区间，误差≤1e−10；没有重新执行COCO accumulation。五方法含重复图/crowd的字面RLE核验原回执通过。区间是点对点95%，非四比较同时95%。 |
| 成本与资源 | 三臂训练49m12s/46m15s/60m58s，均低于8h；评价及统计2h07m34s，低于12h。训练保留显存峰值G4.059/C4.121/PC5.371 GiB；运行及结束时44GiB上限检查通过。验证完整batch中位耗时比G1.3613/C1.2738/PC1.3131，逐批计时独立重算一致，PC≤4。精确评价显存峰值未单独落盘，不用训练值冒充。 |

两个容易误读的配置字段已沿真实执行路径核对：

1. `native_config/NATIVE_ARGS.json` 包含原框架`amp=True`、`epochs=100`、`optimizer=auto`、`lr0=.01`等通用默认字段。这里它是**原生val数据与预处理参数容器**，并未调用框架Trainer；实际后端显式`fp16=False`，训练为自定义FP32循环，实际AdamW与运行期数值开关见各Run的`ACTUAL_OPTIMIZER.json`/`ENVIRONMENT.json`，不能拿未生效默认值替代运行合同。
2. G有4个参数的实际有梯度更新计数是14996或14999，其余15000。全局优化器确实调用15000次；这是冻结实现保留的`grad=None`跳过行为，逐参数计数及政策已随Run落盘，不补造零梯度，不追加步数。C/PC各参数计数均15000。

本机独立核验产物：[LOCAL_INDEPENDENT_RECHECK.json](audits/closure_20261019/LOCAL_INDEPENDENT_RECHECK.json)；只读核验代码：[verify_existing_results.py](audits/closure_20261019/verify_existing_results.py)。第一次开发执行因读取源码pin JSON层级错误终止，修正为其`sha256_normalized_utf8`字段后完整通过；没有更改任何冻结输入或研究判据。

回传包SHA `c53ec099a6ced36fafc435f27ff11982a4bb0e6208ec74642de8e4ff6d4d2cac`，292个成员（其中1个是回传清单自身），291个内容文件逐个核验。远端10个Run共301文件新SHA与原Run内291个被登记产物一致；27个checkpoint/预测/大缓存等仍留服务器。原自动结案记录、原始Run、旧失败均保留。没有提交、推送、定时任务或新训练。

发现并修正的是**当前Study元数据滞后**：服务器自动结案已写45000步与结论，但`runs`列表仍保留G训练1625步的旧快照。本次以10个真实已完成run.json和Slurm退出码更新Study及总控；原服务器快照保留在审计目录，不改写Run证据。
