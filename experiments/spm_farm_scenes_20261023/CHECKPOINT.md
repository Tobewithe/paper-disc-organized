# SPM-YOLO现有成果保存与接续

2026-10-23，用户最新指令：“你先保存现有成果”。本检查点保存方案、源码、真实工程产物、失败修订及旧已训练原型。保存期间不追加正式生成或训练，不创建自动接续任务。

## 已有且可恢复的成果

1. **旧P1真实训练产物**：Slurm184/`RUN_SPM_PROTO_TRAIN_2e46d0049349`，投影2000步；256个128维fg/bg原型、投影权重及来源/重载核验。原始文件继续位于[旧Study](../spm_prototype_training_20261022/REPORT.md)，不是本次新训练。
2. **新场景生成器R2**：[render_farm.py](scripts/render_farm.py)、[运行入口](scripts/run_render.py)、[确定性场景计划](SCENES.json)、[配置](CONFIG.json)、[真实train参考](REAL_TRAIN_PROFILE.json)、[资产骨骼清点](ASSET_INSPECTION.json)及[冻结源清单](FREEZE_MANIFEST.json)。源资产为既有单个Quaternius Pig；骨骼生成腹卧/侧卧，不调用Death动画。
3. **工程样例已完成**：Slurm189/`RUN_SPM_FARM_SMOKE_c065813560a6`，exit0、51秒。8个独立几何场景、16张RGB、90头几何实例、180条可见实例标注；包含站立、走、进食、腹卧、左右侧卧，4种布局/地面和2类相机。标签640×384，32,768条相机射线核验0不一致、RLE往返通过、两外观ID逐像素相等。
4. **失败与修订**：185缺用户空间运行库路径；187不支持OpenEXR BW接口；188高数量规则摆放无法无交叠装下20头。186资产清点成功；R0/R1源分别在[render187](revisions/render187/FREEZE_MANIFEST.json)、[render188](revisions/render188/FREEZE_MANIFEST.json)，修订见[记录](ENGINEERING_REVISIONS.md)。失败没有记为完成或混入正式训练。

工程Run、原图、ID、RLE、相机与姿态元数据、日志、原命令和校验清单已从服务器回传；轻量文件进入Git，图像、张量和传输包保留在本地/服务器并另作完整备份。

## 尚未完成

- 正式128个几何场景×2外观的256图尚未生成；没有正式数据Run。
- 新版原型更新尚未开始，新增优化步数为0。旧P1权重保留作初始化。
- [TRAIN_PROTOCOL](TRAIN_PROTOCOL.md)是未执行设计；[prepare_training.py](scripts/prepare_training.py)为输入准备脚手架。新训练执行器`train_prototypes.py`、`train.sbatch`、实际TRAIN_CONFIG/INPUT_MANIFEST/FREEZE尚未完成，不能直接运行该脚手架。
- 尚未进行每实例Mask驱动的value/融合/YOLO头联合训练，也没有任何新增Mask AP结论。
- 工程图仍基于一个风格化低面数猪资产。8场景预览的归一化Box长边中位约.109，真实172张train的参考约.195；两者场景组成不同，不能作显著性或方法效用结论，但提示不能只采用拍完整猪栏的构图。

## 恢复时唯一下一动作

**完善相机取景分布：加入符合真实训练图猪体尺度的近景和局部群体构图，复核站卧及高数量样例，再执行已登记的正式生成预算。** 保留合理环境和姿态多样性，不为了凑数缩小猪体或删除个体。完成正式生成与输入冻结后，再实现并执行2000步原型更新；融合任务监督单独合同接续。

服务器根：`/srv/gpu-lab/users/a25023104/projects/spm-farm-scenes-20261023`。最新一次性作业189已结束；用户要求保存后未提交其它任务，无自动触发器。恢复时先读[总控](../../project/ACTIVE_PLAN.md)和Study，核本账号队列与真实Run，避免重复提交。
