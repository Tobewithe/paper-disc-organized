# 工程恢复 R1：JSON 几何字段比较

原 Slurm167/168 各在首批身份断言处停止，正式阶段未启动。原 Run、源码与 FREEZE 保留。

原 batch_identity 返回 ratio_pad tuple，而磁盘 JSON 解码为 list；Python 对二者直接比较为 False，不能据此判定模型漂移。R1 只将实际身份记录经标准 JSON 序列化/反序列化后比较，并在仍有差异时保存 actual/expected。所有哈希、原生预测、RLE、科学协议、对象、预算、阈值和运行环境不变。没有放宽数值精度。

重新运行各64帧smoke（shift仍9相位），通过才正式。之前各16帧前向作为失败工程预算单列，训练0步。旧配置不是科学阴性。
