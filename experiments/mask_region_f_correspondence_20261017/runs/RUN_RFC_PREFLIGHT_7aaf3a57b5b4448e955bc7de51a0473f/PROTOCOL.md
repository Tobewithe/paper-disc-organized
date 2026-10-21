# RFC-v1 原生重放、真实 smoke 与正式 A/Q 训练执行冻结

2026-10-17。用户明确授权：“补齐原生输入重放集成，在笔记本冻结环境做既定小规模 smoke，核对张量身份、zero-delta 掩码重放、梯度和资源条件。启动正式训练”。随后授权清理以前不用的文件。本文件启用 [原设计](PROTOCOL_DRAFT.md) 和 [DESIGN_SPEC](DESIGN_SPEC.json) 的原数值/结构/比较/停止规则，取代此前仅限设计和 CPU 工程的执行范围；不修改历史协议或已有结果。

## 科学命题与比较不变

检验同层 8×8 区域 F 与 P/z0 的空间对应能否改善系数修正。A 保留布局，Q 按原 SHA 排序置换整通道向量；原单点 F 保留，共享原投影和卷积，原 17 张量/167872 参数。原 2000 图、原 initial、原逐轮顺序、原 soft-IoU 与 0.1 原生保护项、AdamW 和微批1/累积4全部沿用；训练不按面积门控筛行。每臂 3 轮/1500 步，先 A 后 Q，最后 checkpoint 固定，单 seed、不选最优 epoch。

原生 N、已有 G、新 A/Q 的公平比较，原512开发图、T655/257图、全2925/508图，原成功损伤总体22957与 checkpoint 固定后一次抽样1024，全部保持设计。A−G≥.005且图像配对95%下界>0；A−N目标≥.005且下界>0、总体≥−.001、损伤上界≤1%、Box身份不变。A−Q/A−J及两项等功能幅度对照分别裁决机制，不能凭参数兼容或梯度非零宣布机制成立。完整阳性/阴性/精度不足决策出口见原设计第8节。本次只启动两臂训练，不启动正式S2评价、S3或S4，不预抽损伤样本。

## 执行设备、来源与冻结

- 唯一执行端为用户指定笔记本 `ssh 28358lan` / `YANGLEI`，工作根 `D:/coco_wire/baseline_native_20261013`；解释器 `C:/Users/28358/anaconda3/envs/pytorch/python.exe`。
- 原 `_site` Ultralytics8.4.100，Python3.12.3、torch2.5.1/CUDA12.4，全部版本/源码/数值开关按 G 的实际环境完整比对；不安装或升级依赖，不以本机CPU环境替代。原权重/数据/初始化/预测与缓存 SHA 均核验。
- 部署发现笔记本的《基线复现与配置一致性》仍为较早短版；当前完整规范在本Study的 `reference_snapshots` 固定，保留其原预登记SHA，不覆盖笔记本既有规范。`REFERENCE_LOCATION_OVERRIDES.json` 只允许这一份文档的显式位置映射，旧版SHA同时核查；所有模型、计算源码、数据和配置原位严格核验。
- `EXECUTION_SOURCE.json` 保存本次真实源码清单和 SHA、HEAD、dirty patch；每个 Run 开始时独立复制协议、源码、合同。`AUTHORIZATION.json` 记录本次范围。原设计文件和旧工程 Run 保留。
- 原训练缓存 `RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec`；原生验证缓存 `RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170`；初始化 `RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/initial.pt`；G 为 `RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/last.pt`。这些源产物只读。

## 原生输入重放集成

1. 从原 data YAML 建立原完整数据集及 rect 分组，再选择包含所需图片的原完整批次。使用原 dataset item/原 collate/preprocess，不把目标图重新凑 batch；保持 batch16、实际成员、顺序、位置、input_hw 和 ratio_pad。原标签 cache 缺失或无效时阻断，禁止向历史 dataset 写新 cache。
2. 复用原 `native_output_bridge.make_trace_validator` 的 Head Top-K 观测、原 native postprocess、冻结基模指纹和参数检查；通过独立子类只选择批次并输出新区域 F。不会重算科研 AP/GT 匹配；原 native serializer 逐图核对全部输出字段。
3. 每条上下文图均核对原图片SHA、预处理tensor SHA、P/c0/原点F/框/类别/score、raw位置、最终行身份与完整 zero-delta RLE。重放预测行索引按已核对的原 source prediction SHA/全局行映射绑定。张量精确相等，RLE精确相等；不放宽容差。
4. 仅为所需图写独立 `regions/fit|dev/<id>.pt` sidecar。fit 存全部输出的区域 F；dev 只存原生面积≥9216行，门控外训练无此筛选、评价保留原生整行。原缓存不覆盖，forward 输入不含 GT。

## 固定执行序列

1. **PREFLIGHT**：核真实环境、33项既有来源、原配置及资源。无YOLO前向/优化器更新。通过后生成执行冻结收据。
2. **INPUT_SMOKE**：原 `split.fit_ids[:8]` 和 `split.dev_ids[:8]`，重放它们所在的原完整批次；仅保存这16张所需图的 F。上下文图不加入训练/评价分母。
3. **SMOKE_A/Q**：各用上述8张fit图及原单轮顺序生成函数，最多2步；16张fit/dev图检查原初始化零Δc及原生RLE，使用已有非零G权重检查广播F精确退化、A/Q有效干预、同权重J=Q。原参数17张量须获得有效梯度，原输入无梯度；原GT配对须一致。smoke 权重只作为工程产物保留，绝不接入正式训练。
4. **INPUT_FULL**：两臂smoke全部通过、源码一致后，重放原2000训练图/218209输出，写完整训练F。当前不导出正式评价F或抽损伤面板。
5. **TRAIN_A → TRAIN_Q**：逐臂重新加载原initial，使用原训练循环，只将其loss绑定为原soft-IoU实现、模型forward接入已验证区域F。原逐轮顺序、优化器参数、GT join、更新次数及结束输入SHA均核对。每臂独立Run；有消费过正式预算的同臂不得自动重启。两臂固定last后保存 `TRAINING_PAIR_FREEZE.json`，科学状态为 `trained_not_evaluated`。

先前25项CPU组件测试和新增原批次选择测试是工程前置，不能替代上述笔记本验证。真实smoke通过后保存带来源SHA的 `REAL_INPUT_REPLAY.json` 和 `REAL_SMOKE_VERIFICATION.json`。

## 资源、清理与停止

- 原预算保持：smoke导出及两臂合计≤1800秒；正式F导出总预算≤10800秒（本轮仅训练部分，未来评价导出仍计入此总额）；训练每臂≤7200秒；CPU RSS≤8GiB，CUDA allocated/reserved各≤7GiB。启动前系统可用内存≥3GiB。
- 初始新产物空间保留64GiB；阶段推进后以“当前剩余空间+本Study已写产物≥64GiB”检查同一预留，不能每阶段重复要求已经消耗的空间再空闲。新区域F合计≤48GiB，其余新增产物≤8GiB。保持FP32，不压缩网格/改batch/缩图集来省空间。
- 用户已授权删除不用的旧文件。本次优先把6个旧传输归档包移存到工作站 `C:/Dpan/codexarchives/paper-disc-organized/laptop_20261017/`，逐项SHA完全一致后才删除笔记本原包。保留原Run/阴性记录、源图像、基础模型、当前P/c0/F缓存与checkpoint；清理前后空间和迁移路径另存收据。不是删除旧实验结论。
- 单次隐藏进程串行完成上述序列，使用阶段锁及序列锁；不创建定时任务或goal。任一阶段失败停止后续；不自动重试、续训、换seed、加epoch或扫阈值。可验证工程修复必须保留失败Run，另记修复原因及新源码；正式预算消费后不能从头重跑追阳性。
- 结束核查实际配置、checkpoint、日志、输入与代码SHA；科研效用在后续已冻结评价中裁决。当前唯一后继为完成已授权A/Q训练及配置结案，不自动扩量。
