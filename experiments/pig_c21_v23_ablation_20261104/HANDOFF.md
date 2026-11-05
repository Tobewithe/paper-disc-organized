# C21／V23消融正式启动交接

用户要求正式启动后暂停会话。**正式推理已实际运行，评价已依赖排队；暂停的是助手等待，已提交作业继续执行。**

2026-11-04 03:50:56 UTC（北京时间11:50:56）核到：

- 工程1173：COMPLETED／0:0，2张预定图片、每臂54条输出；N／C21／V23原RLE精确重现，干预已执行，原权重恢复。实际17GPU秒／68CPU-core秒。
- 推理1174：RUNNING，`RUN_PIG_C21_V23_INFERENCE_20261104_56f1232f6402`，已记录8／432图，真实进程存在。10组输出包含N／C21／V23及七项组件消融，尚无完整AP结果。
- 评价1175：PENDING，依赖`afterok:1174`，纯CPU、未开始所以未创建Run；上游失败会取消依赖，不自动重试。

[协议](PROTOCOL_V1.md)固定权重功能消融、完整432dev／2863GT；不进行组件移除重训、不读test。新Study帽1.5GPUh／10CPU-coreh，串行峰值1GPU。55项执行来源冻结SHA为`1244cfede568fab05f219082ff07cd2401f260f2c9f0f12d150b9b06586225cf`，7项权重／标注／参考预测资产SHA已核。

## 服务器定位与接续

根目录：`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/experiments/pig_c21_v23_ablation_20261104`。

完整实现、冻结源、原始预测、逐例账本和日志均留服务器；本地仅保存协议、Idea、Study及本交接。服务器`START_HANDOFF_RECEIPT.json`保存上述真实状态、时间与进程核验，`ENGINEERING_SUBMISSION.json`／`FORMAL_SUBMISSION.json`保存提交命令。

接续时先核1174／1175的`sacct`、真实`run.json`及`INFERENCE_POINTER.json`／`STATISTICS_POINTER.json`，结合日志判断，不把队列为空当成功。完整评价成功后读取服务器`RESULT_SUMMARY.json`及统计Run的结果／SHA，回传精简结论；失败保留原Run，先定位，不重复提交已运行或完成的作业。

结果需反馈[具体Idea](../../project/ideas/IDEA_PIG_C21_V23_ABLATION_V1.md)、训练／表示／评价纲要、中心纲要与SPM主线。已有C21／V23训练及旧消融照原范围保留，功能干预不冒称训练贡献或独立泛化。
