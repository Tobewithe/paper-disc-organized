# V4.1 +5点目标接续记录

当前goal已由用户明确要求后激活；目标是同432 real-dev、相对同8000步/同数据continued的Mask mAP50–95至少+5绝对百分点，并验证原型实际帮助。[goal补充验收](GOAL_ACCEPTANCE_ADDENDUM.md)同时要求相对保留的最强原生参照+5pp，防止续训对照退化带来虚高。未达到不标记完成；小幅正收益、AP75损伤及原型不必要等阴性如实保留。没有定时任务。

当前已完成：全1704原train图划分1272train/432dev、原始RGB/COCO标签来源核验；Slurm198真实标签缓存、200的4096 cutout；Slurm194的8几何×2外观渲染smoke（32768条射线0错）；Slurm202新原型链路CPU核验（零新增系数初始等价、直接梯度、加载、冻结坐标不变）。这些是工程完成，不能当新方法AP结果。

一次性作业依赖：195三分片生成2048几何/4096RGB；201等待195生成完整合成缓存；199运行全量B*120轮/12720步；204等待199完成后提取真实/cutout特征并训练原型warm5000步；205等待204和201，训练新增15000步并构32中心；206实际新memory的GPU smoke；207四臂并行各8000步；208重新评价、同F定向干预、实际技术/结果制图；209独立AP复算、CPU加载和打包。最多4张GPU并行，后续作业均已有依赖；**接续先核队列/指针，禁止重复提交。**

服务器根为`/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023`，个人账号`a25023104@192.168.71.10`。解释器沿用`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python`。Slurm日志在个人logs目录；仅操作本Study、保留全部失败与源码。全量原图、缓存、特征/恢复状态留服务器，轻量结果和权重按清单回传。

科学合同为[TASK_PROTOCOL_V3](TASK_PROTOCOL_V3.md)与[TASK_CONFIG_V3](TASK_CONFIG_V3.json)：P2/P3高分辨率实例任务监督，32原生Mask基底+32额外原型基底，新32系数从0开始；B/G/R/F相同来源/抽样/预算，F/R有效可训练参数匹配。旧V2数据工程冻结和全量基线合同保留，不事后改写。当前源码训练/评价/验收各有独立FREEZE；不要编辑正在运行或排队作业的冻结输入。

`.gitattributes`保留本Study原始字节（包含部分已冻结Windows CRLF）；不要在Git或编辑器中统一换行后重写冻结源码/配置，否则SHA变化。方法正式训练前的训练、评价/验收所有输入已在服务器逐文件核SHA并通过编译。

PROGRESS_RECEIPT是有观测时间的中途快照，不代表最新Run已全部完成。是否运行/成功看真实squeue/sacct、run.json和日志；没有指针不直接推出未启动。只有真实完成才生成相应stage指针。评价报告/图集尚未生成前，USAGE仅是加载接口说明。

当前唯一动作：推进既定完整实验并收取真实训练/消融结果；不自动增加seed、网格或重复训练。若有工程失败，保留Run和源快照，修复后另作业/另冻结；若效用未达目标，按实测证据选择一个有依据的改进方向，再登记新版本，不换验证集或弱化基线追5点。

可视化预检已实际执行：210因服务器根SCENES缺失在冻结预检失败，原Run/日志保留；补齐同一冻结配方，211/212 CPU生成真实预览。已查看并核SHA：最新Run `RUN_SPM_SCALE_VISUAL_PREFLIGHT_9711226b9b3d` 的实际方法链路图与8个固定布局/地面/密度/相机代表场景，见VISUAL_PREFLIGHT_ACCEPTANCE.json。CONFIG内旧scenes_sha字段与实际父/分片文件不符，真实FREEZE_MANIFEST逐文件SHA一致，2048条父配方与三分片并集逐条一致；差异登记在SCENE_PROVENANCE_RECONCILIATION.json，不改已冻结CONFIG或数据。
