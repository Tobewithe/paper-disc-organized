# 已训练真实迁移模型与完整身体候选库

实际来源：Slurm235，两臂各4000步。**该配置真实Mask AP低于原生模型，保留供科研复核；当前使用仍保留原生N。** 不是未训练脚手架，也不替代原对象v0的几何模型。完整来源与SHA见[MODEL_MANIFEST.json](MODEL_MANIFEST.json)。

| 文件 | 实际内容 |
|---|---|
| [structured_last.pt](structured_last.pt) | 46209参数的实例Mask logit修正CNN，4000实际步，正确身体布局输入 |
| [scrambled_control_last.pt](scrambled_control_last.pt) | 同容量/初始化/监督/预算的空间布局置乱控制，4000实际步 |
| [object_bank.pt](object_bank.pt) | 冻结v0从432个合成train观察预测的身体投影，8旋转×2镜像，共6912条；轮廓/标准坐标/已训练表面特征及随机投影/置乱定义 |

真实原生模型的SHA为`cdea4fb8e837dbbda092bbd5743abe28a8b01731817d9dc326133debd61fe40a`，服务器位置为`/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023/runs/RUN_SPM_SCALE_CONTINUED_8239d712ee7f/best.pt`。该文件未被本轮修改。

模型加载：以`scripts/model.py`中的`Residual(checkpoint['config'])`实例化，再严格加载`checkpoint['model']`。共同输入为14×32×32 RGB/原生logit/固定P2P3投影/ROI坐标，另加12×32×32的候选地图。候选须通过`transfer_io.retrieve`从原生预测选择；不得输入真实GT或猜相机。推理重放参考`scripts/run_transfer.py:evaluate`，必须保持冻结原生预测行身份及真实“上采样后裁框”的解码顺序。

二进制已回传本地且同时存个人服务器本Study的`artifacts/`，GitHub保留代码/报告/图/数值与SHA，大文件不入Git。原始2304图对象模型仍见[对象v0](../../spm_object_prototype_20261023/artifacts/README.md)，不能把本轮ROI残差权重当成完整3D对象生成器。
