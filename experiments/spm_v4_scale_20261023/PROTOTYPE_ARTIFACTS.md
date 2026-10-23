# V4.1已训练原型与当前分割训练

原型已真实完成20000步：warm5000步、mixed10000步、real adaptation5000步。1272真实train、4096 cutout、1792合成训练几何提供7160份source episodes、791652个特征位置。最终形成fg16/bg16、共32个128维单位中心；全部槽位有支持，单槽4015–43579个训练token。前向坐标固定为同一已训练B*的P2/P3，原型投影512→256→128。

原型训练Run为`RUN_SPM_SCALE_PROTOTYPE_FINAL_959ef31d838a`，warm父Run为`RUN_SPM_SCALE_PROTOTYPE_WARM_d1c5f03299e3`。`runs/<final Run>/memory.pt`与`PROJECTOR_STATE.pt`已回传本地并逐项核SHA；大权重保留本地/服务器，GitHub保存源码、清单和图集。memory SHA为`6ad2e284824abf4a2bd7c7af97ea3f84e088167c9d09b551286cb83e2b076e96`。见[本地回执](PROTOTYPE_LOCAL_RECEIPT.json)、[原型训练摘要](runs/RUN_SPM_SCALE_PROTOTYPE_FINAL_959ef31d838a/SUMMARY.json)、[构库来源](runs/RUN_SPM_SCALE_PROTOTYPE_FINAL_959ef31d838a/BANK_SUMMARY.json)和[加载用法](USAGE.md)。

真实原型GPU两步检查已通过：新增系数梯度>0，第二步基底混合器梯度>0，原型坐标与冻结参数不变；见[实际GPU检查](runs/RUN_SPM_SCALE_ENGINEERING_TASK_GPU_de78cb87bd44/SUMMARY.json)。这证明训练链路接通，不等于Mask AP收益。

| 实际技术图 | 说明 |
|---|---|
| [方法链路](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/00_method_and_supervision.png) | 原型响应与原生空间基底组成64基底，独立RGB系数参与原生Mask解码 |
| [20000步学习曲线](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/01_actual_task_prototype_learning.png) | 三阶段实际loss与梯度；loss下降不是AP收益 |
| [前景原型样例](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/07_prototype_atlas_foreground.png) | 16个真实已学中心及最相似train来源；红十字为查询位置，未给中心虚构解剖语义 |
| [背景原型样例](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/07_prototype_atlas_background.png) | 实际背景、栏杆、地面与黑区域等来源 |
| [槽位与来源支持](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/08_prototype_support.png) | 全部32槽的真实/合成/cutout计数 |
| [嵌入PCA](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/09_embedding_PCA.png) | 同一学习嵌入按域和监督层显示，二维重叠不证明高维塌缩或跨域效用 |
| [128维中心余弦](runs/RUN_SPM_SCALE_PROTOTYPE_VISUALS_c612e80b2b44/figures/09b_prototype_center_cosine.png) | 32×32完整中心相似度，补充二维投影；描述性证据，不是效用Gate |

全量基线120轮/12720步已完成，最佳54.6798 Mask AP，末轮47.8063，选最佳权重作强参照。四臂分割训练正在并行，各8000步，正式科学判断等待完整选优与同F删除/错配评价。1000步中途F/B/G/R为56.9064/57.0871/56.9127/56.9418：F相对同预算B为−0.1807pp，当前不能称原型额外收益，+5目标未达成。

四臂实际B*、数据清单、抽样顺序和冻结上游SHA一致；F/R有效可训练参数相同，G多9280参数。中途证据见[四臂运行回执](FOUR_ARM_PROGRESS.json)。唯一当前动作仍以[总控](../../project/ACTIVE_PLAN.md)为准，完成既定训练与208评价；不因中途分数更换样本、阈值、预算或参照。
