# 桌面独立 GPU 成本面板

2026-10-09，用户明确要求同时使用本地电脑，B指示先做实际GPU工程一致性检查。本协议继承 [原成本协议](COST_PANEL_PROTOCOL.md)（SHA256 `2211ae66a47560c81f658791e1a02680dbcccba5326c9288392c943be96277b0`）的输入面板、五臂、端点、必要路径隔离、warmup、重复/顺序与严格parity，以下仅覆盖执行环境和资源时序；不修改已执行的笔记本Run/来源。

实际环境为桌面 `C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe`，Torch2.9.1+cu128、RTX5060Ti，vendor `shared/vendor/ultralytics_8_4_100`，官方权重 `assets/models/coco_clean_20260911/yolo26m-seg.pt`（原SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`）。原GPU设备/软件与桌面分别登记，耗时/峰值不混合或平均。图片采用原面板相同JPEG字节；实际本地路径和通道/尺寸摘要留Run，若本地缺图片可仅回传所需面板原字节并保留来源链。

桌面短工程固定原清单前4张、五臂各warmup4张后实际GPU执行与完整冻结输出/输入tensor/身份/RLE parity检查；独立Run保留终态、PID、实际设备、source/协议锁和首个有效产物。已有原七臂与TriFlow参考必须真实可用，不以占位通过。Torch/硬件差异导致任何严格parity不通过时保留失败，不放宽阈值或冒称正式成本；可据具体差异修复兼容执行后新Run，不能改变科学方法。

当前桌面CPU Boundary有并发负载，允许先跑功能/一致性工程，但该工程时延不能充当正式部署成本。只有工程通过且原CPU重负载结束后，才执行独立正式前32张×3重复成本Run，记录测前/测中资源状态、桌面峰值范围。CUDA reserved peak是进程热态保留容量，不解释为单图新增或最低显存需求；CPU lifetime peak范围仍如实包含初始化/warmup/审计。

桌面与笔记本的独立工程/成本结果分别报告；保持原全部质量结果封存，不训练、不调方法阈值，不为占满GPU新增没有决策用途的实验。
