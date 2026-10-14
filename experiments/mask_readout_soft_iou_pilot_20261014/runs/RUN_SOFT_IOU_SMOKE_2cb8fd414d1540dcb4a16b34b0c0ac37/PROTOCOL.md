# 同 ROI soft-IoU 单变量 B adapter 复验

2026-10-14；用户授权笔记本 `ssh 28358lan` 执行。先冻结本协议、输入 SHA、实际代码和每 Run 合同，再做 smoke、一次正式训练及一次 S2 评价。旧 BCE Run、配置和阴性结论保持原样。

## 要判定的命题

在原 B 的初始化、原 fit2000、3 轮及所有非研究变量固定时，仅把 GT BCE 换成同 ROI soft-IoU，能否使无 GT 推理的系数残差同时通过原 S2 修复、整体效果和损伤门槛。依据是 [BCE 分解报告](../mask_readout_bce_decomposition_20261014/REPORT.md)；不根据本次结果选择损失权重、epoch、seed 或样本。

## 唯一训练变量

继续调用原 `adapter_model.loss_grids` 和 `adapter_data.projected_targets`：全部最终输出共同计算 `z=(c0+Δc)P`；GT 仍是原图标签经原整数 resize/pad 和最近邻下采样的二值 proto 网格；ROI 仍是该最终预测框在 proto 网格的半开区间。

对每个原本可监督且 ROI 非空的实例，`p=sigmoid(z)`，`g` 为原 GT，`R` 为原 ROI：

```
I = sum_R(p*g)
U = sum_R(p + g - p*g)
L_GT(instance) = 1 - (I + 1e-6)/(U + 1e-6)
```

FP32 运算；不平方、不加 Dice/BCE 混合项、不扫平滑量。GT 在 ROI 内为空时仍使用这一公式并计入原实例分母；空 ROI 仍按原实现排除并计数，不改变 GT 关联。无匹配监督的图片 GT 项仍为可微零。

图片内 GT 实例分别求均值；保真项完整保留原实现：全部非空 ROI 最终输出上 `BCEWithLogits(z,sigmoid(z0).detach())`，先 ROI 像素均值再实例均值；`L_image=L_GT+0.1*L_preservation`。图片外有效 batch4 均值及尾组分母沿用原训练循环。正式执行通过仅替换原训练循环导入的 GT loss 函数复用循环；原文件不修改。训练前测试保真项数值及梯度与原实现相同、ROI 外 GT 梯度为零、空 GT/空 ROI 可用。

## 冻结输入与训练预算

- 研究线唯一原生配置：[baseline_profile.json](../../paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json)，SHA `ae8133e371d121c642e0a32070b3743f6f7657ded587d6ab0491456329198e45`。原生 P/c0/F、全部框/类别/score/输出顺序、每图实际 batch16/rect 几何和解码保持；不再次调用基础网络或改变缓存分组。
- 从 `RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/initial.pt` 加载相同参数，文件 SHA `ce2de59a316eef3cea6cfe6d1763221931fd10693377c8a91ce321843604bacd`。原始状态 SHA `ff3107ba29c0a330d1849c8373f3c7210c6c89d0382da5e1fcec367e376fd6a2`，167872 个新参数；全新 AdamW 状态；禁止从旧 last.pt 续训。
- 原缓存 `RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec`：同 2000 图、原标注及训练 join。逐轮直接复制原 `IMAGE_ORDERS.json`，3×2000 图访问、1500 次更新、微批1/累积4；训练后 join 与原文件 SHA 一致。
- 完全沿用原 B 合同：seed0，AdamW lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=.0001；amsgrad/foreach/maximize/capturable/differentiable/fused 全 false。无 scheduler、AMP、增强或 early stop；FP32；只选最后一轮。架构、所有数值运行开关和依赖环境与原合同逐项相等。
- 先独立 smoke：原 fit 前8图、原 dev 前8图零残差 RLE 重放；仅2次更新，检查梯度白名单与新损失；smoke 参数丢弃。正式训练重新加载原 initial.pt。预算 smoke≤900秒、正式训练≤7200秒、S2评价≤3600秒。新 Run 独立登记；失败保留，不自动重试已消耗的正式训练预算。
- 权重、缓存、GT、原训练合同/顺序/配对、原评价器/代码 SHA 记入 `INPUT_EXPECTATIONS.json` 和 Run 的 `EVAL_CONTRACT.json`；真实环境和导入源码记录在 Run。

预检已审阅的唯一历史工程差异：`audit_s2_completed.py` 在旧S2源码快照后，将已存在的审计JSON与新报告的比较改为双方 `json.dumps(...,sort_keys=True)`，使序列化后的list和Python tuple同值比较；逐行diff仅此一行。该文件不参与训练/预测/指标计算，当前文件与历史快照的SHA及差异固定在输入清单，其余旧训练/评价源文件必须逐字节一致。此差异先于本次smoke登记，不能用此豁免接受其它源码变化。

## 评价与原 S2 门槛

复用 [原 S2 执行协议](../mask_readout_mechanism_v2_20261012/S2_EXECUTION_20261013.md)、原生 S0 全5000图缓存和同一512张 dev（目标 T=655）。只评价本次 B 最终 checkpoint；不重训 C、不用旧 C 的结果声称新空间机制成立。评价直接调用原 `export_and_measure`、`image_effects`、`bootstrap`、`evaluate_ledger` 和损伤工具，固定完整输出矩阵及相同 decoder。

1. **目标效果**：T 是原任务 Mask75 失败且存在同条 Box75 检测、自身 MaskIoU<.75 的 GT；固定该检测计算原图 IoU。按图先取均值，再图宏平均 ΔIoU≥0.005，且配对图 bootstrap 95% 下界严格>0。2000次重采样、seed1729、linear percentile，单位0–1。
2. **整体护栏**：同512图内全部普通、原 bbox75 匹配 GT 的固定检测图宏平均 ΔIoU≥−0.001。
3. **原成功损伤**：遵循 [DAMAGE_ACCEPTANCE](../../shared/evaluation/DAMAGE_ACCEPTANCE.md)，保持 S0 的全部成功普通 GT 总体；新 checkpoint/推理合同固定后，首次方法评价前系统随机源只抽一次1024个 GT（简单随机无放回），完整评价其所在图片全部预测/GT。保持抽样设计和预算，不沿用已看过的旧 BCE 损伤标签选方法；发生工程重试只能沿用本次已抽清单。固定有限总体损伤比例的精确单侧95%上界≤1%；精度不足按原协议 blocked，不追加抽样。

按原 COCOeval 对同一完整 bbox+segmentation JSON 分别做 bbox/segm 匹配；框配对身份核验不等同 segm 匹配身份。评价图是 dev 与新损伤样本图片的并集；其 AP 仅为辅助开发数，不能称全5000图 AP。旧 BCE 与新 soft-IoU 仅在相同 dev 固定配对上描述性比较；选择和 Gate 仍相对原生 S0。

## 配置核验、停止与唯一后继

每 Run 完成后检查代码/输入/冻结权重前后 SHA、真实保真项/总损失、初始化、逐图顺序和配对、checkpoint 内 AdamW 配置及实际1500步、参数白名单、数值环境、完整评价覆盖、输出非 Mask 字段和 bbox 匹配身份；配置问题先工程阻断，不能裁决科学结果。回传 manifest/transfer、合同、日志、checkpoint、必要逐GT/效果表；独立复算 bootstrap、有限总体上界与 Gate。

三项 Gate 同时通过才支持“本配置满足 S2 初筛”，唯一下一步为同一 checkpoint 的 S3 全5000图 AP/损伤验收；本轮只完成 S2，不自动启动 S3。任一失败就拒绝本配置进入 S3/扩量；唯一下一步为同一 checkpoint 的同 ROI→原图配对诊断，分清新目标已学到的几何变化与原图损伤；不追加训练。仅当配置/实现证据无效时撤销科学裁决，先修复被定位的工程错误并登记独立复核，不用调参挽救结果。

结论范围仍是反复使用的 COCO val2017 开发数据。S0 原生账本有效、官方发表 AP 复现未通过的双状态保留。
