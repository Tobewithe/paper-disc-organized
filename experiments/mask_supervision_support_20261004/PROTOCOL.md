# 掩码监督位置扩展：固定原生one2one系数分支

预注册：2026-10-04，Study `STUDY_cfa7c92dfdd7483d882d3e3e536dfe76`。仅本会话3080Ti服务器执行。桌面不执行模型，笔记本由另一线程负责。

## 问题、依据和竞争解释

检测分支需要控制重复输出，并不逻辑地要求掩码系数也只从每GT一个位置获得监督。本轮检验：保持原生网络、检测输出与原型固定，让原one2one系数分支接收同一GT的其他TAL初选位置监督，是否改善原one2one位置的跨图正常掩码？这是一项尚未证实的监督位置假设，不是已确认的失败根因。

已核对真实历史：fast-screen N已经同预算微调整个原生cv4；候选共识的SAME组已回放完整one2many系数头在相同raw位置的输出（没有稳定优于原头），不能重跑头移植。早期teacher的one2many少量正样本、P3额外框回归、7I–7V离线预测器、Gram预条件及原生共享末层证书，与本轮同一one2one头的掩码监督位置重分配不同。全项目检索未找到本轮实际Run。参考历史：[fast screen](../prototype_readout_fast_screen_20261003/REPORT.md)、[候选共识](../candidate_response_consensus_20261004/REPORT.md)、[监督评价域审计](../supervision_output_domain_audit_20261003/REPORT.md)。

文献已有丰富/稀疏监督并存的设计：[YOLOv10](https://arxiv.org/abs/2405.14458)、[One-to-Few CVPR 2023](https://arxiv.org/abs/2303.11567)。本轮不把多正样本当成新概念；只是用它提出针对当前模型的受控判别。8.4.100源码中one2one与one2many系数头独立，one2one使用topk=7、topk2=1；跨GT冲突处理早于topk2。源码以准备运行的SHA与配置快照为准。

竞争解释包括更丰富的实例特征支持、对邻近位置的正则作用、原赢家权重降低，以及更多像素计算带来的不同优化路径。阳性筛查不能单独区分全部解释；阴性只能降低本配置优先级，不能宣布密集监督普遍无效。

## 唯一变量和三组

- A：原官方模型，不训练。
- N：复用已完成fast-screen原生cv4稀疏监督末轮，`RUN_bd203c92371e45f986a3cdfe999ae143`。不重训。
- D：同一原生one2one_cv4，从同一个官方权重初始化；仅训练期掩码监督位置和其权重改变。不是JointCoefficientReadout历史D的证据模块；代码内部架构仍是`mode=N`、新参数0，产物中的D仅表示本轮实验组。

从原冻结one2one raw box/class输出及全部GT调用原TAL。复制原criterion，只令`topk2:1→7`，保持topk=7、alpha/beta/eps/stride、小框处理和跨GT消歧。不得把它称为官方one2many assignment。先完成全部GT竞争，再保留原one2one已覆盖GT；逐键验证原(raw_id,GT)是扩展集合子集。新位置允许来自另一尺度，报告原→新增层级计数，不宣称纯同尺度邻域平滑。

对GT g，原位置r及新增E：没有新增时 L_D(g)=L(r,g)；否则 L_D(g)=0.5L(r,g)+0.5 mean_{s∈E}L(s,g)。每GT总权重1；有效batch总体分母为原one2one候选数。不是“原监督保持不变再追加”：有新增的GT原位置权重从1降至0.5。禁止改为按扩展候选总数平均。

冻结assignment、backbone、neck、P、框/分类参数及所有BN运行buffers。重新运行整个原生系数分支，其卷积及BN affine参数可训练。无外挂、teacher、gate、阈值或新损失形式。仍采用官方全640 mask BCE、原GT框裁剪/归一化面积/seg gain。未改变检测监督、类别分数或最终候选集合。

## 复用与预检

同GT重复原位置并按上述权重归约，数学上等价原N。因此先在2张有效fit图验证：完整官方single_mask_loss vs缓存分块公式，在c0和固定随机增量处同值同系数梯度；重复原位置的完整原生参数梯度等于原稀疏目标。预设atol=1e-6、rtol=1e-5，不因失败放宽。输入/P/box/c0重放使用既有atol=rtol=3e-5。原图字节、overlap标签/annotation排序、永久候选身份必须一致。临时D一步更新证明只有指定分支更新。失败保留日志、修复明确执行错误后另Run；不能绕过检查开始训练。

N复用条件：初始化完整state hash、trainable名单/参数数、数据列表、每epoch图像顺序、优化器/学习率/BN策略/更新预算匹配。评价时A和N旧dev逐候选IoU/Mask75/Coverage/AUC/FPR要以1e-12容差复现。原始缓存只读，准备阶段使用私有标签/数据.cache目录。

## 锁定预算

原fit1024计划图与dev256计划图；不增加样本。N/D相同seed0、3epochs、micro2图、effective16图，AdamW lr1e-4、betas(.9,.999)、eps1e-8；多维参数weight decay1e-4，warmup1epoch，cosine eta ratio.1，clip10。不重置或改变对照的图像顺序。固定epoch3，无dev早停/选模。

smoke300秒，正式支持准备900秒，D训练1800秒，评价1800秒上限；达到资源上限记未完成，不把末次中间checkpoint替作预定结果。每次独立runner包裹，IDs见RUN_IDS.json。相同图像/参数/更新预算并不等于相同FLOPs：D多算额外mask像素，报告时间/显存/位置数量；记录D每步原梯度范数、实际更新幅度以及N/D最终位移，不假称梯度规模完全相等。

## 评价、判断和停止

统一完整P→固定预测框裁剪→零logit二值化→真实letterbox原图逆变换，COCO原实例mask评价。仅原固定one2one候选：fit前128计划图和全部dev256，不在新增训练位置上替换主评价。不做AP，也不称未查看的新盲测。

主比较dev图片macro正常IoU D−A及D−N。同时报告候选均值、Mask75净变化/repair/damage、coverage、连续logit AUC/FPR、原稀疏官方BCE；原成功/失败、框好mask差、P3/P4/P5预先分层。P3不等于目标小。图片配对bootstrap1000次，seed20261004，候选均值也整图重采样。

继续信号预设：全体D对A/N均达到0.2pp、区间下界>0且Mask75净增；或框好mask差组对A/N达到0.5pp、区间下界>0且全体不低于−0.1pp。其他指标如有明确实用收益需与全部损伤共同解释，不能挑偶然子组替换主检验。此为单seed探索门槛，不是独立方法确认。

没有信号就停止本配置，不扫topk、0.5权重或epochs，不增加网络。实际额外位置很少/未充分训练时保留限定，不宣布机制不存在。若有信号，先提出独立确认计划，不自动扩大数据或增加新模块。完整结果回传和报告后本轮结束，总Goal仍以实际独立收益验收。
