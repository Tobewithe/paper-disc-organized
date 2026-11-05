# 原面细表面控制与双猪接触标签接口：有限范围已接通

实际完成9个原面候选编译、4相机×皮肤/分组共8图逐张审核、512独立射线核验，以及V14真实clear/contact的全量面标签桥接。候选和原9父部位/owner均能保持身份，但不能当成鼻孔、眼球、完整内耳或全蹄的解剖GT。看图同时发现后脚离地.002566987体长，进入V18独立严格支撑修复；本次没有准入训练图。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| 问题 | V15材质没有解决细部位表达，V16证明内部脚关节Gaussian不能覆盖最低蹄表面。需要把有限语义控制落到可核查的原面区域，再送到实际多猪接触链，而非只用模型输出词或抽象点坐标。 |
| 机制假说 | 不适用论文机制归因；工程主张是rest中的source face ID可在保持拓扑的pose/contact后稳定连接parent/owner与可见投影。竞争结果是跨父级、错误实例身份、投影不符或候选区域解剖含义过强，分别拒绝或保留unknown。 |
| 针对性方法 | 源修复rest上冻结鼻端/眼周/耳朝前/四肢末端9组，存face/vertex/规则/面积/边界，原面序和9父级不改；固定4视角皮肤与发光图共8张，最终像素中心射线存source_face/owner/parent/candidate。独立NumPy每视角128射线，再按V2接口把原面组接到V14 clear/contact各2头的实际网格，对所有面owner/source tuple/父级多数票独立复核，错面序/重复owner负对照须拒绝。 |
| 整体结果 | 9组共5347面、无重叠/跨父级；可见像素每组93–7989，全部≥64。512独立射线原面全匹配、无共边歧义，最大距离误差6.29e−7体长，几何进入Blender误差2.98e−8。809 COMPLETED、CPU76秒/0GPU，54远端原文件SHA一致，8图全部实际看完。源鼻盘孔洞、耳内与分趾已有几何；鼻组有孤立背侧斑，眼组含眼睑，耳组含耳缘，脚组未含完整副趾，故只保留限定几何候选而非解剖叶GT。接口对clear/contact共89784面通过，4个身份负对照拒绝；比原粗父级新增稳定的候选通道，不是自然度/AP收益或已有方法优越性证明。 |
| 研究决策 | **保留源面ID→pose/contact→owner/parent/candidate→像素接口，优先修严格站立支撑。** 该选择由后脚实际图与原几何共同定位，问题可用有限根刚体变换检验，比继续材质扫描/加模型更直接。鼻/眼精确边界保持待审，不夸成完整fine GT；旧V8≤.01近地Gate保留。唯一后继V18已冻结并实际启动，若根变换不能使rig与四脚共同满足则拒绝而不逐点压地；腹卧、多体型、外观、T/S/P最终合同及release仍未完成。 |

编译`RUN_V17_FACE_CANDIDATES_28b590d2a490`；809/`RUN_V17_SURFACE_REVIEW_65ac7953c23f`；独立`RUN_V17_PROJECTION_AUDIT_eb21b77d9ebe`；V2接口`RUN_V17_CONTACT_SURFACE_BRIDGE_b5f7dffb54c2`。见[原协议](PROTOCOL_V1.md)、[V2接口合同](BRIDGE_PROTOCOL_V2.md)、[逐图审核](VISUAL_QA.json)、[控制目录](SURFACE_CONTROL_CATALOG_V1.json)、[支撑缺口](SUPPORT_REVIEW_V1.json)、[回传SHA](RETURN_VERIFIED.json)。

原面候选必须在rest中定义后按source face ID传递，不能在旋转/侧卧后的世界Z上重选“最低脚”，也不能用UV颜色反查做唯一语义。0候选表示未细分，owner0才是本单猪投影的非猪背景；当前只导出可见身份，不冒称完整T/S/P。14退化UV面保留，当前视角可见像素为0，不算修复。所有精确解剖边界、未见面及细分含义按目录保持unknown/candidate。

![原鼻盘与耳几何](server_return/runs/RUN_V17_SURFACE_REVIEW_65ac7953c23f/outputs/face_front_skin.png)

![限定候选区域](server_return/runs/RUN_V17_SURFACE_REVIEW_65ac7953c23f/outputs/face_front_groups.png)
