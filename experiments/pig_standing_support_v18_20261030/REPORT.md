# 严格四脚支撑与真实双猪接触转移：固定资产修复通过

实际完成三个站立姿态的有限根刚体支撑修复、V14两份双猪几何/标签转移及6张RGB逐图审核。根pitch约−0.282度使原后脚.002566987体长间隙归零，骨架、原面与双猪形变均保持一致；固定资产范围通过。整体外观仍均匀、体型姿态有限，本次批准训练图0，完整数据release未完成。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| 问题 | V17实际后脚图和原几何共同显示，旧stand两后脚离floor0约.002566987体长。旧V8≤.01近地资格不等于严格四脚接触，需修复而不破坏已核形状、部位/owner或真实双猪接触。 |
| 机制假说 | 不适用论文机制归因；工程主张是两侧脚成对一致时，±1度范围内的全身根pitch+全局Z平移可使四脚接地。相反若最小点切换/左右差异造成无法同时满足，或与骨架不一致则拒绝，不逐点压地/扩关节自由度。 |
| 针对性方法 | 对同三站立求f(theta)=hind_minZ−front_minZ的固定60次二分，更新root矩阵和世界偏移，用原LBS独立重建，核每脚Z/rig/原边与面积。整体刚体变换同样作用于V14 clear/contact及reference，逐实例8脚检查，位移范数/锚点和source面身份保留，通过V17接口输出owner/source_face/parent/candidate。810固定6图，含修复前后脚部与修复后双猪全景；独立每图128射线，再全部实际看图。 |
| 整体结果 | 3/3根pitch−.282016984度，世界Z移约−.001151/−.001157体长，4脚最低Z≤2.17e−19；最大点位移.004674体长。rig重建误差≤1.11e−13，边/面积相对误差≤4.63e−13，内在形状不变。双猪8脚均接地，contact位移范数误差2.07e−17、非body锚点误差5.55e−17，原0交叉关系由同一已核刚体变换保留。810 COMPLETED、CPU76秒/0GPU；61原文件逐SHA，768射线全匹配无歧义，最大距离误差1.20e−6体长，修复前两图与V17逐像素差0。6图实际看完，后脚间隙消失，无明显倾倒/脚穿地/接触拉丝。相对旧近地资产新增严格支撑及联通标签资格，未比较学习方法AP、未新标定物理或证明真实泛化。 |
| 研究决策 | **准入修复后的固定3站立和刚体转移的clear/contact，优先统一资产、接触和表面控制的场景编译接口。** 现有构件都已有真实核验，继续孤立扫材质或新增模型不如把已核能力接入最终RGB/标签链直接。新POSE_CATALOG_V3操作白名单保留V11原左右侧卧、替换3个站立的后继入口，不回改旧V8/V14 Gate。唯一下一步按NEXT_COMPILER_DESIGN_V1构建有限能力编译器并接经过审核的脚端表面着色，再有限场景运行；尚未启动下一渲染，腹卧、多体型、外观/联合覆盖、完整T/S/P、回放及release缺口保留。 |

几何`RUN_V18_STRICT_SUPPORT_a98a7a715db4`；810/`RUN_V18_SUPPORT_RGB_1097486316e0`；独立`RUN_V18_SUPPORT_RENDER_AUDIT_b234dc802b17`。见[冻结协议](PROTOCOL_V1.md)、[实际6图审核](VISUAL_QA.json)、[回传SHA](RETURN_VERIFIED.json)、[新操作目录](POSE_CATALOG_V3.json)、[下一编译设计](NEXT_COMPILER_DESIGN_V1.md)。构造摘要0/1是实例下标，导出owner始终1/2，二者不混用。

失败保留：首几何`RUN_V18_STRICT_SUPPORT_73c9fb0669fb`漏算源根关节旋转中心，被rig重建断言拦住；按原门槛修正offset=R*(old_offset+pivot)+shift−pivot后新Run通过，未改变形状/求解角范围，见ENGINEERING_FIX_V1.json。本地`RUN_V18_SUPPORT_RENDER_AUDIT_aa247d88ae2e`在scp仍占用archive时被过早启动，输入未就绪，保留失败；等待同一传输句柄终态并逐SHA核验后才新建成功audit，远端没有重跑，见TRANSFER_TIMING_FAILURE.json。

本次只修源根刚体与支撑，不是新自然体型、腹卧或写实材质。点/边距离保持不等于源开放网格已获闭合体积/真实软组织资格；14退化UV面和细语义unknown限制仍在。没有真实既有标签/旧拟合/dev-test、模型训练、对抗、推送、定时器或工作台索引操作。

![修复后后脚](server_return/runs/RUN_V18_SUPPORT_RGB_1097486316e0/outputs/after_feet_rear.png)

![更新后的双猪接触](server_return/runs/RUN_V18_SUPPORT_RGB_1097486316e0/outputs/contact_overview.png)
