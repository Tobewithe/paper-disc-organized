# 猪对象学习 v0 训练包

**2026-10-23执行已完成：** 2304图、两臂各4000步和固定评价已结束，权重/18图已回传，独立复算通过。几何增量明确，但体形编码稳定性未过联合Gate；见[结案报告](REPORT.md)、[模型与推断](artifacts/README.md)和[图集](VISUAL_GUIDE.md)。以下接口与预算保留本轮启动合同。

本包把 V5 的对象定义实现为图像到身体模型：共同标准网格与特征场，实例体形与骨骼状态，以及结构化/自由解码两个严格同容量分支。完整实验合同见[PROTOCOL_V0.md](PROTOCOL_V0.md)，数值与资源见[CONFIG_V0.json](CONFIG_V0.json)，源码字节见[FREEZE_V0.json](FREEZE_V0.json)。当前执行状态只看[总控](../../project/ACTIVE_PLAN.md)和真实 Run。

## 已实现接口

| 文件 | 职责 |
|---|---|
| [export_asset.py](scripts/export_asset.py) | 从真实 Pig.blend 导出标准网格、骨骼权重和六姿态；与 Blender 求值逐点对齐 |
| [geometry.py](scripts/geometry.py) | 透视投影、蒙皮和三角形深度栅格；独立射线用于验证 |
| [prepare_data.py](scripts/prepare_data.py) | 32体形族的多视角/外观/遮挡成组数据、标准坐标及几何标签 |
| [object_model.py](scripts/object_model.py) | 同容量结构模型与自由模型，体形/姿态/特征场和稠密部位读出 |
| [train_objects.py](scripts/train_objects.py) | 两臂顺序训练、相同初始权重、相同采样顺序，逐项记录真实步数和曝光 |
| [evaluate_objects.py](scripts/evaluate_objects.py) | 留出体形族的完整对象/新视角评价、错误绑定和等幅噪声干预 |
| [check_package.py](scripts/check_package.py) | 小型三角形独立几何及模型前后向/权重重载，优化0步 |
| [check_actual_asset.py](scripts/check_actual_asset.py) | 实际猪网格的射线/几何/模型前后向核验，优化0步 |
| [visualize_objects.py](scripts/visualize_objects.py) | 资产先验图与学习后固定对象预测分开展示 |

## 首轮合同

2304图，24体形族训练、8族开发，全部来自同一猪资产。两臂各4000步，固定末步评价；单次最多1张A6000，数据30分钟、训练60分钟、评价30分钟。真实 train/dev/test 全部不读，不训练YOLO，不把本轮合成对象能力当成真实Mask AP收益。

v0为了先检验可运行的对象推断，使用已知相机、强合成几何监督、给定蒙皮和简化着色。姿态以求值骨骼仿射状态实现；受关节范围约束的状态、真实语义教师、多资产、真实体形适配和YOLO接口属于后续阶段，当前未完成。模型包会学到图像推断、体形/姿态状态、标准特征场与残差；骨架先验来自资产。

## 执行次序

CPU工程核验和资产导出通过后，保存ASSET_LOCK。正式GPU阶段必须有用户授权回执，绑定配置和协议SHA。阶段按“数据→两臂训练→完整评价→固定可视化”逐项提交一次性Slurm作业；根据上一项真实终态继续，禁止靠提交成功推断执行成功。

运行根为 `/srv/gpu-lab/users/a25023104/projects/spm-object-prototype-20261023`，解释器与入口在配置中登记。程序自身校验授权与源码，缺回执即拒绝GPU数据与训练；不能用修改布尔开关替代真实用户指令。当前AGENTS对远程训练的限制按协议处理，不触碰其它任务、共享环境或工作台索引。

## 能改变取舍的结果

在全部预注册指标、普通分支对照和干预共同支持时，只晋级真实训练域对象对齐的规划。若结构分支不超过普通分支，关闭该参数化的扩训。不能靠换形变族、重挑视角、增步数或调阈值把同一个冻结实验改成阳性。最终 +5 点真实分割目标继续保留。
