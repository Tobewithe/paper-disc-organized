# AMT-v1：已训练 checkpoint 的 S2 执行冻结

2026-10-16，用户在训练完成报告后明确指令“开始”。本文件落实 [原协议](PROTOCOL.md) 第5、6节，增加评价代码、资源与运行登记；原训练协议、34份冻结源文件和历史Run不改写。

**待检验主张：** 训练内可达目标能转化为无GT推理的额外修复。以新last的固定面积门控G为唯一主要候选，与原生和原soft-IoU门控结果公平比较。教师合格和训练loss下降不是该主张的证据；必须完成本次实际评价。

## 固定身份与执行顺序

- 新checkpoint：`RUN_AMT_TRAIN_e10144e6d3d94803824aef1c636d831f/last.pt`，SHA `b39df4f2720bf60f3ffe64f79ce426d2dc55ee28188c0068e6b1aa3061cf79b7`。只用这一份，不追加训练。
- 原生：`RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170`全部5000图缓存和最终输出；只读取本次图像并保持各图原batch16/rect几何，不重新跑基础模型。
- G：原生原图RLE面积≥9216应用新Δc，其余逐字保留原生RLE。U仅描述，不允许改选U；全部Box/Class/score/输出行和排序不变。
- 原G对照：开发配对直接复用`RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8`；本次新损伤图片从`RUN_S3_FULL_670fffd010664a648e096572fc7c476d/predictions_mixed.json`提取全部输出并重新做相同COCO匹配。旧checkpoint不重新训练或推理。
- 先在笔记本核原冻结训练、源代码、依赖/运行时、数据/权重/缓存与checkpoint审计，再冻结`EVALUATION_CONTRACT.json`。随后系统随机源一次抽取22957个原成功GT中的1024个，保存`S2_DAMAGE_PANEL.json`及时间/版本/哈希回执；**任何smoke或正式新方法预测前完成抽签**。已存在清单时拒绝新抽签。
- 实际评价集合为原dev512与新损伤图片的并集，其大小由唯一抽签决定，不能照搬旧1314图。每张图保留全部正常预测及全部GT。原T655、全部Box75配对2925不变。
- 烟雾使用该并集升序前8图，验证零残差RLE、行身份、面积选择、原生COCO任务/匹配及新U/G/原G完整匹配。独立核验通过才执行完整并集；正式和smoke共用同一合同与损伤清单，smoke结果不用于选择。

## 原数值规则原样沿用

1. G相对原生：T图内均值再图等权ΔIoU≥.005且2000次配对图像bootstrap的95%下界>0；全部Box75图等权均值≥−.001。
2. G原成功损伤：完整同图COCO segm匹配；仅计一次抽签的1024个GT。有限总体精确单侧95%上界≤1%；未通过且点估计≤1%记精度不足，不补样。
3. 新G减原G：同dev T配对的图等权增量2000次bootstrap区间下界>0。复用原`image_effects/bootstrap`，numpy default_rng1729和linear percentile不变。原G的+.010024491543988425仅作身份复核。
4. 并列报告U/G及原G的target/overall、同检测修复与COCO任务修复、同一新成功面板损伤。并集AP仅为辅助指标，不能当作5000图S3 AP。
5. 全部原S2门槛和增量门槛通过才推荐同checkpoint S3；本次不自动启动S3/S4。失败关闭AMT-v1，聚焦已有固定修复标签上现有输入能否区分应修/不应修及修复形态；不改教师、门控、损失或预算追阳性。

## 配置、审计与预算

沿用训练和原评价实际环境，含Ultralytics8.4.100、原packages/source hashes、FP32/CUDA/TF32策略、原native decoder、完整JSON含bbox、COCO bbox/segm独立匹配。推理只调用原adapter的`eval:0`，GT只在输出生成后的评分使用。每张图重放全部c0并要求RLE精确一致；缓存、checkpoint与源码前后SHA一致。

独立检查重新核对所有输出非mask字段/行顺序、G的原生面积选择、全部2925条配对RLE IoU、原G复用一致性、bootstrap和损伤计数；在笔记本对G重新运行完整COCO匹配。统计/身份轻量复核同时回传本机。原生smoke同GT账本和bbox/segm关联均核验。

资源：笔记本单一执行者；可用RAM≥4GiB、磁盘≥8GiB。smoke≤10分钟，smoke独立核验≤10分钟；正式评价≤60分钟，独立核验≤30分钟；进程峰值RAM≤6GiB，单Run产物≤4GiB。超预算或工程错误保留Run并停止，不自动重试/重新抽签。实际基础模型forward=0、训练更新=0。

真实执行各自生成RUN_ID、run.json、原命令/环境/源码快照、EVAL_CONTRACT、输入/产物哈希、SUMMARY/DECISION/CONFIG_AUDIT。独立Windows一次任务只串行执行smoke→独立核验→full→独立核验，任务断开SSH可继续。结案回传核SHA、复核配置并作通俗讲解；不自动提交/推送，不触及其它会话任务。
