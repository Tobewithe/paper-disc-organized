# 两目标模型独立增量成本

2026-10-09，在 [主协议](PROTOCOL.md)（SHA256 `cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057`）冻结配置下补隔离成本；不把共同20k提取或两个gate一起计算的耗时分摊成单臂部署成本。

固定同一原val5k清单前32图、baseline/target_I/target_H三臂、3次重复，各块顺序循环左移repeat_index位（3臂3块各位置一次）；每arm-repeat独立fresh进程、前4张warmup一次后测32张。工程固定前4图×三臂×一次重复只核合同，不作成本能力结论；正式在完整5k预测封存后执行，不与提取/拟合/评分重负载并发。

实际只在原笔记本28358lan/Torch2.5.1/vendor8.4.100/官方权重环境。可只读复用旧单臂RCMC必要路径，target_I/H分别加载其实际20k新fit的numeric模型，模型SHA/完整HGB参数及portable预测sign绑定；baseline不加载gate。每臂只算自身必需smooth/五features/gate及first64支持，不共算两gate再where、不读GT、不用旧1500-fit模型替代。

共同端点：计时外已读原始RGB→本臂预处理/官方forward/native及必要方法→全部正常原图binary masks CPU，头尾CUDA同步。RLE、完整身份/input/新5k参考逐byte审计、写盘、权重加载/初始化/warmup与child生命周期均在端点外单列；纯startup不能从整process_wall冒算。每次输出严格对应新完整5k本臂cache及原baseline，保留fallback/empty/范围外身份。

记录每图每repeat原始e2e，mean/median/population std/p10/p90及same-image/repeat相对baseline的有符号增量，负值不截断；未测CI未知。GPU每测reset peak、endpoint后审计前读取，resident/热reserved不是单图新增或最低要求；CPU为fresh-process累计peak，包含init/warmup/此前audit，不伪称纯部署峰值。保留实际资源状态及采样范围，不混两硬件或以原七臂联合时长排名。

工程与正式独立Run、固定源码/资产/参考/协议、完整回传manifest与transfer，失败保留并新Run修复执行口径；不据成本或工程质量改目标、阈值、模型容量或seed。独立消费重聚合保存记录/真实参考证据，不冒称重测墙钟。结果只限此32图/一次fit/设备/端点，不外推完整5k时长或训练seed不确定性。
