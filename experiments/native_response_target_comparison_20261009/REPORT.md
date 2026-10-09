# 同容量监督目标对照：20k / 5k 正式结果

当前配置未证明净阈值收益目标 H 比连续 IoU 增益目标 I 带来额外 Mask AP，且 H 增加原成功候选的损伤。两种校准器相对未经修边的冻结 baseline 均有约 0.438 个 AP 点的描述性提升；这个收益不能归因于 H 目标优于 I。

生产、成本、正式评分、1000 次图像簇 bootstrap、保存产物核验及 B 的四个固定 draw 逐臂独立 AP 重累积验收均已完成。B于2026-10-09接受完整固定20k I/H结果并结束当前H替换配置：不以H进入扩大确认，不补阈值/seed/动作或重新评分；本有限Study完成，整条研究线目标不据此标记完成，原桌面SegRefiner5k及v2后续链继续。

## 受控条件

共用原随机 20,000 张 COCO train 图的一次提取：2,186,397 条正常候选、912,598 个 first64、909,964 个 supported；235,427 条共同监督行来自 19,817 张有有效监督的图片，关联 128,767 个独立 GT，保留 106,660 次重复 GT 关联。未知标签保持 NaN、不参与拟合，invalid=0。

两臂使用相同五维特征字节、监督身份/顺序、单位候选权重、21 项完整参数、seed20260915、100 次迭代和最多7叶的 HGB；仅标签为 ΔIoU 或 10 条精确 IoU 阈值的净跨越收益。YOLO26m-seg 官方冻结权重与 native 解码保持；没有微调 YOLO。本轮两臂均使用原 fit_target.py，另存的缓存 v2 仅通过离线合同，没有用于本次模型或声明完整拟合加速倍数。

训练摘要的 fallback1197 含92个 baseline 原空，真正新空1105；监督 fallback546 含41个原空，真正新空505。原摘要不改名；真实回退标签为0。正式5k的215个 trial-empty fallback保持baseline。

## 正常任务与主判据

全部原 COCO val2017 5,000 图、原GT/crowd/ignore、segm-only输入、COCO maxDets=[1,10,100]。全部555,445正常输出身份/框/类/分数/顺序保留，first64 229,596、supported228,974、unsupported622；I/H采用动作180,324/181,331，不同决策6,309、不同binary输出6,249。shared first64 不等于两臂实际编辑集合相同。Box任务输入不变，本次正式评分表为 Mask AP，未重算Box AP。

AP采用0–100量纲；AP差写为AP点。

| 输出 | Mask AP | AP50 | AP75 | APsmall | APmedium | APlarge | Boundary AP (.02) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 43.685372 | 66.353710 | 47.294844 | 22.401825 | 47.670209 | 63.571734 | 29.571367 |
| target_I | 44.122977 | 66.485530 | 48.050173 | 23.036845 | 47.995499 | 63.710536 | 29.973818 |
| target_H | 44.123412 | 66.484708 | 48.057949 | 23.043117 | 47.982582 | 63.710753 | 29.974846 |

主 AP(H)−AP(I)=+0.00043518 AP点，1000次成对图像簇bootstrap的95% percentile区间 [-0.01449756, +0.01583064]，包含0。

预声明继续条件为 AP差区间下界>0 且 H 损伤点率不增加，本轮未满足。保存的机器decision=AP_damage_tradeoff只反映AP点值微高及损伤更高；不能表述为证实AP改善的权衡。当前动作/五特征/容量/训练配置下不支持以H替代I，不能外推全部方法能力上限。

Boundary H−I约+0.00102815点，Boundary区间未测，不能据此确认H优于I。

## 固定配对与像素账

沿原final8的86,600个 detection→GT关联，允许重复GT；baseline maskIoU≥.75成功36,266，失败50,334。此配对诊断独立于正常COCO重新匹配，不是唯一GT召回或COCO TP。

| 输出 | mean IoU | 损伤 / 36266 | 损伤率 | 修复 / 50334 | 修复率 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.66759717 | 0 | 0.00000% | 0 | 0.00000% |
| target_I | 0.66941773 | 739 | 2.03772% | 1257 | 2.49732% |
| target_H | 0.66940955 | 775 | 2.13699% | 1284 | 2.55096% |

H−I损伤率=+0.0992665个百分点，95%区间 [+0.0580342, +0.1431455]；H多损伤36个、多修复27个固定候选。

候选macro：coverage/FP-G/FN-G/IoU有效分母86,600；purity86,596，4个原空保持null。FP/G是按GT面积归一，不是FPR。

| 输出 | coverage | purity | FP/G | FN/G |
|---|---:|---:|---:|---:|
| baseline | 0.80947608 | 0.78839178 | 0.27553704 | 0.19052392 |
| target_I | 0.80011011 | 0.79889069 | 0.25680330 | 0.19988989 |
| target_H | 0.79966114 | 0.79926755 | 0.25590507 | 0.20033886 |

按图macro有效图片4,946、无关联54；不把缺失填0。

| 输出 | image coverage | image purity | image FP/G | image FN/G | image IoU |
|---|---:|---:|---:|---:|---:|
| baseline | 0.84729549 | 0.83427485 | 0.21331798 | 0.15270451 | 0.73001856 |
| target_I | 0.83949277 | 0.84310038 | 0.19800822 | 0.16050723 | 0.73158616 |
| target_H | 0.83914667 | 0.84340222 | 0.19732751 | 0.16085333 | 0.73158178 |

## 独立部署成本

笔记本RTX4060Laptop，同32固定图×3repeats×3臂，288计时/9 fresh child、平衡顺序；预加载RGB→该臂全部必要路径→全部原图binary CPU，初始化/依赖/warmup/参考审计/RLE/写盘/评分在端点之外。

| 输出 | 平均端点 ms/image | paired signed增量 ms |
|---|---:|---:|
| baseline | 63.40517 | 0 |
| target_I | 81.33645 | +17.93128 |
| target_H | 80.93439 | +17.52922 |

负增量记录I/H为8/9，保留有符号值；成本CI未测，约0.4ms均值差不证明H更快。CUDA allocated峰值baseline1,036,744,704 / I,H1,142,128,128字节，reserved1,981,808,640 / 2,220,883,968；CPU累计peak baseline1,378,091,008 / I1,501,011,968 / H1,500,217,344字节，含初始化/驻留/warm/audit，不是增量或最低RAM。v2 checker加强保存来源/冻结模型/回传/内存及childscope核验，没有重测。

## 来源、核验和边界

生产10Run+cost2Run的15,913原文件完整回传、原run.json保留；评分supervisor03、score02、bootstrap02、verify02的82原产物完整manifest回传。失败/stale与原错误路径记录保留；位置修复只引用真paired_triflow_vs_baseline，不重新GT配对。

portable abs0/sign0为生产者全235,427训练行的源码绑定存证；独立消费者未反序列化joblib重新预测。正式核验重建保存12指标数组、86,600固定配对、1000 RNG/固定簇统计与区间；AP仅复放4个draw并复用paired_aps。独立算法匹配证据来自此前fresh官方COCOeval重复图副本工程；B另行固定draw0/1/499/999×三臂独立图片副本展开/排序/累积已真实completed0/complete，最大误差2.7755575615628914e-16低于原1e-14；仅4个draw、复用已验累积核心，不是独立重算1000次AP或重新GT匹配。

区间条件于单次拟合，不包含训练seed不确定性；5k已用于主线研究，不称新的最终盲测。Boundary vendor精确六文件content-lock，上游commit未知。主判据、数据和参数不因结果补改，不增加阈值/seed或重新评分追显著。

## 固定引用

- PROTOCOL.md（SHA cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057）；COST_PROTOCOL.md。
- B审查范围见原EXECUTION_NOTES.md，含训练/回退计数、source消费与来源限制。
- RUN_NATIVE_TARGET_FORMAL_SCORING_CPU_S0_02/SUMMARY.json（SHA 160213a960bbe1a842e5a18962c2e54e3690e9fc56b32764ea30ec19f52617b8）。
- RUN_NATIVE_TARGET_FORMAL_AP_BOOTSTRAP_CPU_S0_02/SUMMARY.json（SHA 924f5b8af1bb782b9a3171244e5fda4853f8541427b9a5d9b06c8077be9ecf8c）。
- RUN_NATIVE_TARGET_FORMAL_SCORING_VERIFY_CPU_S0_02/VERIFICATION.json（SHA 28b0956677c065deec52e2a42985d0523a4dde5d6e8b2d7334e99e43db9e5272）。
- RUN_NATIVE_TARGET_COST_PANEL_VERIFY_S0_01/VERIFICATION.json（SHA f1eddab56f2606f9c8f47d21365f8ec4c3fb8451e7a740dc0d9f568309635a7a）。

- RUN_NATIVE_TARGET_INDEPENDENT_AP4_VERIFY_DESKTOP_CPU_S0_01/SUMMARY.json（SHA c164e7568c0998b887a5fa203b022ec34bfcf8897407c3f286910f96108070f9）；VERIFICATION.json（SHA ff9cb515f1e7046f35b0e315e0d8f030639ab6d051c8e9276cee30411a1fb4a2）。
