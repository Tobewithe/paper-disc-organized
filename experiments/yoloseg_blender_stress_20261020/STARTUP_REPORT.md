# 首轮服务器启动与设计交付

2026-10-20 16:26:50（台北时间），Slurm **162 / YOLO-blender-smoke** 已实际启动，Run **RUN_YB_SMOKE_c6f0333ccc994518bdf0790b52265526**。1张RTX A6000、4 CPU、16GiB内存、2小时上限；`Requeue=0`。这不是仅预留空卡：正在渲染，首次进度核对44/384帧，后续会在同一作业内执行原生推理与描述性评价，结束释放GPU。尚无科研结果。

- 首轮：密度4/8目标、四目标相向交错、墙体前/后遮挡、栏杆前/后遮挡，共4组成对视频、8段、384帧。Cow现有资产，640×384、12fps、4秒/段。样片已回传并目视检查。
- 扩展设计：12类、马/牛/羊、三个布局种子、成对版本，拟216段/20736帧。该矩阵仅设计，未启动；需先查看工程核验的域识别与速度结果。
- 保留当前LCF Job154及其它会话Job160/161，未取消、改配或重启。旧训练的进度不属于本实验的新成绩。
- 从服务器冻结profile复制实际软件版本与数值策略：Ultralytics8.4.100、官方同SHA权重、FP32、原生验证batch16/rect/640及decoder，完整预测conf=.001；展示/操作点在保存预测上固定>.25。没有新增训练、跟踪器或时间平滑。
- 继承资产函数与原生重放工具采用独立源码副本，原文件只读；冻结19文件、20外部输入。外部资产/模型/标注/Blender哈希通过，19项本机回传哈希复核通过。
- 8项拓扑语义测试、类别无关一对一框关联、空预测/无关联及成对场景清单检查通过。真实标签射线与zero-delta原生重放将在作业内检查，不能把上述CPU测试当作这两项已经完成。
- `python project/check_active_plan.py` 通过。无定时任务、目标模式、自动重启或提交推送。

关键输出：`FREEZE.json`、`PROFILE.json`、`CONFIG.json`、`ENGINEERING_TESTS.json`、`LOCAL_FREEZE_VERIFICATION.json`、`EXECUTION_STATUS.json`及所属Run。

执行端根目录：`/srv/gpu-lab/users/a25023104/projects/yoloseg-blender-stress-20261020`。首轮Run内`dataset/PROGRESS.json`是渲染进度，`PROGRESS.json`是模型推理进度；以最终`run.json`和Slurm退出状态判断完成，不以进度文件消失/队列为空代替成功。`SUMMARY.json`、`NATIVE_REPLAY.json`、逐帧`TIMELINE.jsonl`和8段预览在完成后产生。

接续只检查这个Run，不重复提交。若失败保留`ERROR.txt`及所有片段；如果正常对照的类别正确Box50召回不足80%，应报告合成域有效性不足，不将漏检或猪等非COCO资产误判为原型结构缺陷。首轮只有原生N，相对旧G/LCF的收益未测。
