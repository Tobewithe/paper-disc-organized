# 冻结 S2 checkpoint 的训练内诊断

2026-10-14。用户授权在笔记本完成本诊断，使用已有 checkpoint 和原2000图缓存，配置保持冻结，不追加训练。本协议是独立诊断，不改写 [S2冻结协议](../mask_readout_mechanism_v2_20261012/S2_EXECUTION_20261013.md) 或其阴性结论。

## 命题与输入

本 Run 量化最终 B 在其实际训练图片上的修复、总体变化和原成功损伤，区分“训练内尚未满足有效修复要求”和“训练内达标、开发集不达标”的位置差异；固定模型的训练目标变化作为解释证据。它不从一次结果断言唯一的优化、容量或泛化因果。

- B：`RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/last.pt`，SHA `201ab6a7372c167e772217904eac8f4ba9946c26e5862c2882e3c01bdcd58e5e`；不选择其他 epoch、臂或残差倍率。
- 缓存：`RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec/trace`，完整 manifest SHA `fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d`。使用原 SPLIT_CONTRACT 的全部 fit_ids=2000，不换图。训练GT子集 SHA `51477c03144651f01f941991661da85df0df17b6048a148feca0b7dad6bce26c`，核对原保存的 TRAINING_JOIN。
- 开发集：直接复用 `RUN_S2_EVAL_a24a564568c943eab362784abfbe2c54` 的 B 固定配对表、SUMMARY 和 CONFIG_AUDIT；本诊断不重抽损伤样本、不重跑开发集、不合并两个总体的 AP。
- 执行根目录 `D:\coco_wire\baseline_native_20261013`；解释器 `C:\Users\28358\anaconda3\envs\pytorch\python.exe`，原 `_site` 离线环境。依赖、原生源码和数值策略沿用 [baseline_profile](../../paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json)。

## 不变量和执行

基础网络不实例化；P/c0/F、预测框/类别/score/顺序、每图 input_hw/resize/pad 均来自原缓存。B 架构和 loss 函数直接使用训练时源码，核对原 Run 源码哈希；B 设 eval、参数 requires_grad=False，整个前向在 inference_mode 下，无 optimizer、backward、增强、新预处理或新 checkpoint。

每图使用与原训练/评价相同的完整系数矩阵、FP32 和两阶段原生 decoder；先逐输出验证 c0 重放 RLE 与保存基线精确相等，再导出 c0+Delta c 的全部输出。虽然缓存逐图读取，原生导出的 batch16/rect=True 几何保持不变，不重新分组推理。原 mask_ratio=4/overlap_mask=True 等框架参数保持原记录，新增监督仍使用原始 RLE→原图几何→proto 最近邻投影。

基线和 B 的完整 JSON（均保留 bbox）分别调用原 S0/S2 的 pycocotools ledger 实现，bbox/segm 匹配独立，maxDets=[1,10,100]、useCats=1、area=all、IoU=.50:.05:.95；严格核对原检测全局行号映射、全部 GT 覆盖、ignore 与 bbox75 匹配不变。AP 仅是产生账本时的训练总体附带量，不作为本诊断门槛。

## 量与预先判定

1. 按训练缓存的原生基线定义 T：普通GT、segm75任务失败、有bbox75配对、且该检测自身MaskIoU<.75。主量沿用每图等权的固定配对 Delta MaskIoU；总体护栏覆盖全部原生 bbox75 配对普通GT。含目标图内先均值、再图均值，无目标图不填零。配对图 bootstrap=2000、seed=1729、percentile/linear 95%区间；只描述本次训练总体的效果分布，不作为未见数据证据。
2. 损伤对本2000图的全部基线成功 S 做 census：B 完整 segm75 匹配从成功变失败的数目/|S|。全部 S 已观测，有限总体上界即精确比例；不把同检测IoU跌破阈值替代任务损伤，也不声称未见样本风险。
3. 训练内诊断门槛复用 S2 数值：T均值>=.005且95%下界>0；总体均值>=-.001；全体S精确损伤<=.01。三项全过定义 `fit_repair_supported`，任一不过定义 `fit_repair_requirements_not_met`。T少于200、统计不可计算或身份核验失败则 blocked，不缩小分母。此判定不放行S3/S4。
4. 以原Box50训练join和原loss归约，分别计算固定c0与B的逐图GT BCE、全部输出soft保持BCE及总目标 `GT+.1*preservation`。保持非空ROI分母独立、无监督图片GT项为零、图片仍计入2000图均值。总目标降低超过1e-6记为 `training_objective_lowered`；这是固定checkpoint差，不用在线epoch均值替代。
5. 保存同Box50监督配对的逐实例BCE变化与原图MaskIoU变化，及它与Box75评价配对是否为同检测的身份字段，供后继核验复用。两种配对不混合成一个指标。BCE变化容差1e-6、IoU方向容差1e-12，只作方向统计，不凭相关变化认定唯一根因。

唯一后继规则：训练内三项全过且原开发集失败，则下一步核验训练到开发集的效果落差；训练内任一不过，则下一步核验训练监督目标到原生掩码指标的对应关系。若后者同时总目标下降，结论明确为“目标降低与有效修复要求未同时成立”。这些是后继研究命题，本次不启动其训练或参数修改。只有发现输入、身份、解码或评价错误才撤销对应结果，保留失败Run并独立纠错核验。

## 预算、工程控制与结案

工程 smoke 为原 fit_ids 前8图，零训练，<=900秒；覆盖原生重放、相同GT join、完整B导出、双任务账本、精确损伤及参数前后哈希。正式Run为全部2000图，<=3600秒，新增产物<=8GiB；正式运行必须接受同代码/协议的smoke。前向前检查无同阶段进程/锁，原子阶段锁。任何不变量或资源门槛失败停止并保留已有产物，不运行中更改batch、精度、样本或门槛。

每个真Run独立登记原命令、解释器、源文件快照、协议/输入/权重/checkpoint/缓存散列和真实状态；结案前复查所有输入、模型state、参数无梯度、执行步数=0、源码/环境/原生配置、全部覆盖和统计分母。轻量证据回传后独立复算均值、bootstrap和损伤计数并校验SHA；运行状态与回传状态分别记录。
