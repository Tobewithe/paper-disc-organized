# RSM-C1完成后的配置复核

2026-10-20。执行与证据完整性核验通过；严格科学数值Gate未通过。`CONFIG_AUDIT.status=passed`表示符合本次已冻结的“超差留证继续”执行合同，不表示`pressure_RMS_matched`或`solver_numerical_valid`通过，这两项均为false。

| 项目 | 复核结果 |
|---|---|
| 执行端 | 笔记本YANGLEI，原pytorch Python3.12.3；同一RTX4060Laptop。smoke5条后正式256条，均exit0。 |
| 原生版本/权重 | Ultralytics8.4.100、torch2.5.1/CUDA12.4及源码pins与原PREPARATION逐项相同。权重仍为原冻结SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。没有重新加载模型做新推理。 |
| 原生几何/解码 | 原batch16/rect缓存与640输入政策，不重组图片批次。原N×32 GEMM、首阈值与裁剪、原图scale/byte/RLE；selected与full decoder核对，独立完整重放13056次。 |
| 数值运行值 | FP32原生解码；matmul TF32=False、cuDNN TF32=True、matmul_precision=highest，均与原实际环境一致。FP64仅原几何求解，不替换FP32评价。 |
| 对象与输入 | 原256条/243图及159条C见证；原C/原生缓存/标注/图像和训练误差库不变。原505项输入执行前后SHA核对，拟合供体与验证图隔离。 |
| 科学源码 | 原V3的21项科学来源不变；与本次新协议/上下文/执行器共28项冻结，逐项快照及运行后SHA核对。 |
| 唯一执行差异 | 原固定点校正最多8次、误差合格提前结束及最终系数算法逐字符前缀核对。只把最终“幅度不合格即抛异常”改为记录false后继续评价；另记录实际FP32前后logit差。没有追加搜索、挑选历史最佳迭代或放宽容差。 |
| 求解/预算 | 原max_iter100、24轮、原约束及1e-8/2e-7标准，单条90秒上限，0.10*C_RMS压力与24方向。正式处理206.781秒、含独立复核总663.328秒；GPU峰值分配1044573184字节，低于原3GiB。 |
| 失败和分母 | 256条全部进入统计。218条接受、38回退，代理1条越限；原方向幅度3条配对失败，双读数8条配对失败，均保留。原5e-5/95%覆盖及全部效用门槛未修改。 |
| 无训练/方法参照 | 网络训练、基础模型前向、Adapter前向均0；有限系数求解256次。G_budget权重未取得，同面板指标null，未用旧G替代。旧G只作为预定历史辅助参照。 |
| 双端核验 | 笔记本13312项RLE/IoU、13056次full replay、原前113条系数/压力/掩码精确复现。本机纯stdlib重算统计/区间、RLE和失败标记；90项Run清单哈希通过。96项回传文件SHA一致。 |

原协议SHA `fc18e379f8679920ec8f472af2ad8ac955859ee449f7d99d32519d44d09dad12`、CONFIG SHA `d9773c5d0db940dbd9857bab6cc82ceb5ad57215f5d19c0bbc20c06639d14987`不变。本次协议SHA `f997fc16f636e0a1873758fba679295cd8effa626ee3ca7fbdc58501454c2570`。

证据：[原始配置审计](runs/RUN_RSMC_FULL_c130c3b5270744fabdb50df0b0326c95/CONFIG_AUDIT.json) · [执行差异](preflight/EXECUTION_SOURCE_DIFF.patch) · [完整报告](REPORT.md) · [本机核验](preflight/INDEPENDENT_LOCAL.json)。
