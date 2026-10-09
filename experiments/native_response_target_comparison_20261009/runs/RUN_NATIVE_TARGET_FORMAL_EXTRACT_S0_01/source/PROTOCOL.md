# 同容量 ΔIoU 与净阈值收益目标对照

2026-10-09，B授权一次有限目标替换实验：既有smooth动作是否受监督目标限制。H变换不是创新，也不是完整AP效用；不追加F局部动作、TriFlow补丁、分布头或逐编辑AP标签。实际使用 `ssh 28358lan` 原Torch2.5.1 pytorch与固定vendor8.4.100/官方权重（SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`），桌面另做SegRefiner，各环境各自串行。

## 一次共同提取

复用原TriFlow随机20k train清单，`SELECTED_IMAGE_IDS.json` SHA256 `ef3453e65b80a2d07997c882b0624a88f6d8161fcd32e2d629cdff3285ba3c63`；实际顺序、JPEG字节、标注（train JSON SHA256 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`）、vendor/解释器/输入输出及原始源码绑定到Run。不排除旧500 selection，不另造校准集，无模型选择。工程固定原20k前4图、原val前4图及整数阈值/unknown/new-empty合成合同，仅检验工程；硬门通过后自动完整20k提取/充分拟合两臂及固定5k，不依据小面板质量停选模型。

固定640/FP32/one-to-one/conf>.001/max_det300原生输出，动作仅no-op和smooth，τ(A)=.75/(1+(A/2304)^2)，面积A为原正常原图binary像素；保留原native解码/实际ratio_pad、完整GEMM及first64/protoROI资格，不补位。两臂共享同次提取的原五维response特征字节、相同可监督候选及顺序、sample权重；默认单位候选权重，两臂无Scaler/label标准化。新空trial真实回退baseline，未知GT不填零；资格外和全部正常空输出/身份均保留。

训练GT关联仅在baseline固定身份上按非crowd同类原BoxIoU最大且≥.5，GT原顺序稳定破tie，每prediction独立、允许重复GT；记录监督覆盖/未知/invalid。fit行仅为共同first64且protoROI supported并有有效关联的action-support候选；matched unsupported和范围外不加入0训练行。以相同原图GT mask像素账产生I_base和I_smooth；共同supported行的newly-empty trial等真实动作fallback两个target均0，不能按待拟合gate的no-op策略重写标签。

两标签为 y_I=I_smooth−I_base；y_H=(1/10)Σ_{p=10..19}[1(I_smooth≥p/20)−1(I_base≥p/20)]，向下跨越扣除。阈值定义精确为p/20；可用原整数TP/union交叉乘20*TP≥p*union实现，避免np.arange漂移，不另设epsilon或标签裁剪。

## 两个相同模型

两臂均重新拟合20k共同监督行；旧1500-fit RCMC只作为已知外部结果，不当同预算控制。HGB参数固定：loss=squared_error、learning_rate=.05、max_iter=100、max_leaf_nodes=7、min_samples_leaf=80、l2_regularization=1、early_stopping=False、random_state=20260915；完整get_params、实际sklearn/NumPy版本、threads与模型/特征/row/weight SHA留Run，其余默认值两臂相同。训练100迭代，不扫容量/seed/阈值，两臂固定predicted_gain>0才采用非空trial，否则baseline。

GT只用于训练提取及独立评分。推理进程GT-free，输出完整原图所有候选，类/框/score/顺序/empty不变；正式5k名单为原val_full SHA256 `b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db`，标注SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。全量baseline须与已封存native5k02逐input/候选/byte/RLE一致；失败保留、独立重试，不使用删空旧适配器。

## 主要判据与独立评分

主量为完整5k联合COCO Mask AP(H)−AP(I)，segm-only输入、原crowd/ignore/maxDets=[1,10,100]。同时报告相同baseline的连续IoU、原成功/失败分母修复/损伤、candidate/image macro coverage/purity/FP-G/FN-G、固定Boundary .02，以及真正隔离两模型必要路径的统一端点成本；未运行值未知，不以配对/IoU指标代替AP。

AP差95%区间用同一图像簇配对bootstrap1000次、np.default_rng(20261009)，从全部5000图有放回抽5000；两臂每次用同一draw。每次重复抽中图当独立副本，副本ID按抽样位置有序，保留候选/score tie稳定顺序及相应GT/crowd/ignore。可复用已实际COCOeval的逐图逐类阈值匹配/排序信息，但必须真正重累积precision/recall与101 recall-grid AP；禁止平均每图AP、去重抽样或用IoU区间替代。工程以含重复图、score ties/ignore/空输出的独立COCOeval副本对照验证该累积实现。percentile取2.5/97.5；记录原始draw/replicate/seed与有效分母。

同draw报告各臂baseline-success损伤率及H−I损伤差区间，baseline-success分母沿原final8 fixed association，不冒充唯一GT。只在AP差区间下界>0且H损伤点率不增加时，记值得后续独立确认的方向，不称安全保证；AP↑但损伤↑记权衡，区间跨0记未证实，上界≤0或两臂同策记当前配置无支持。每种结局结束，不补阈值/seed追显著。

本5k已用于主线研究，不称从未接触的最终盲测；单次拟合区间不含训练seed不确定性。阴性仅限此动作/五特征/容量/训练配置，不外推全部方法能力。工程、提取、拟合、推理、评分/独立核验分别新Run并保留真实失败和完整回传链，不手改run.json或旧接受结果。
