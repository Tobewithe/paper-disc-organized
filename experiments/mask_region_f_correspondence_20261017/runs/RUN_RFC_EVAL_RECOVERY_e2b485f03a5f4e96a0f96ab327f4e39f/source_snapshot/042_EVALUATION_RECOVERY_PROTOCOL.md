# RFC-v1 评价显存释放恢复：科学协议不变

原科学方案 PROTOCOL_DRAFT.md / DESIGN_SPEC.json 及 EVALUATION_PROTOCOL.md 全部沿用。本轮只处理 RUN_RFC_EVAL_FULL_82c717bc94c84c87959c84ae0051d6c4 的可验证工程中止，不改变模型、样本、阈值、decoder、统计、Gate或训练预算。

原Run耗时141.422秒，在下一张图片开始前触发 `CUDA allocated/reserved exceeds frozen 7 GiB`。81张图片的系数与完整行前缀已保存；run.json、ERROR、源码快照、未闭合JSON和原始产物清单均保留，不把它标成科学阴性。旧错误未分别记录allocated与reserved峰值，因此不编造旧峰值或断言唯一原因。源码中图像局部张量释放后没有回收CUDA分配器保留块；变尺寸图像连续运行可能造成保留块累积。

唯一计算执行差异：每图所有模型/对照/解码/指标记录结束、局部张量引用释放后，调用 `torch.cuda.empty_cache()` 回收未使用的分配器块。每图记录allocated/reserved及历史峰值。原峰值≤7GiB检查继续使用历史最大值，不能靠重置峰值或放宽阈值绕过限制。8GiB CPU、48GiB F/8GiB其它产物限制不变。

恢复前提与验收：

1. 验证原失败Run全部产物SHA；读取原已抽面板，同一1024成功GT、1294图、A/Q最终checkpoint、原N/G预测和已验证区域F。不得重新抽签、导出F、训练或改配置。
2. 在恢复运行中对原81张图片逐项对比保存的全部方法系数张量，要求 `torch.equal`；所有N/G/A/Q及适用控制组完整预测行/RLE精确相同。任一不符即工程阻断，不能提高数值容差。
3. 剩余科学评价时间上限为14400−141.422=14258.578秒；所有重复的81图计算也包含在剩余额度内。正式基础模型forward不新增，训练步数仍为0。
4. 只有完整评价、原Gate计算、配置审计、81图精确重放、资源限制和独立核验全部完成，才能给科学结论。若资源再次超限，保留错误；不得通过预算/阈值放宽追结果。

实现保存为独立 rfc_evaluation_recovery.py；旧 rfc_evaluation.py、EVALUATION_SOURCE.json 和旧Run不修改。新源码/协议由 EVALUATION_RECOVERY_SOURCE.json 封存，旧科学推理合同与损伤面板SHA继续逐项核对。最终仍按原协议评价A；Q/J/等幅输出不得事后晋升为部署方法。

恢复启动前的产物检查发现：原失败Run在预测流关闭前生成了partial manifest，进程退出又把缓冲尾部写入9份未闭合预测JSON。逐字节确认每个旧SHA对应的前缀完全未变，仅有末尾追加；其它全部产物SHA一致，最终前缀覆盖81图。原manifest不改，另存 `preflight/FAILED_EVALUATION_POSTEXIT_MANIFEST.json` 记录进程退出后实际SHA及前缀证明。第一次恢复因该严格检查在注册Run/计算前退出，其启动日志和入口源码保留。恢复版增加每图写入后的flush，保证异常退出时已完成图片落盘；这不改变任何预测。接受结果仍需原81图全部系数与完整预测精确一致。
