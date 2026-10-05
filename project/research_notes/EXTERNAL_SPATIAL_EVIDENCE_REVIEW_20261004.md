# 外部空间证据的去重、方案审查及执行去向

**执行状态更新：本候选已完成并停止当前配置。** 原始计划保留于下文，实际方案、五臂结果及边界见[正式报告](../../experiments/external_spatial_projection_20261004/REPORT.md)。正式Run `RUN_bb0160a72b9b4a93a43e5ffc7c4f3346` 已回传，不能按下文历史“尚未启动”重复登记。



当前[evidence_path_usage](../../experiments/evidence_path_usage_20261004/REPORT.md)已结案：额外通路已激活但效用小，不自动延长训练。下一步优先考虑外部冻结空间证据进入原YOLO prototype basis的零训练检查，先不训练新头。



## 已完成与真正缺口



- [local-view](../../experiments/local_view_coefficient_replay_20261003/REPORT.md)已经做过原RGB局部放大→同一YOLO再推理→原P回投，投影后macro IoU−3.6204pp；不重复。

- [7J-N](../../experiments/coefficient_neck_roi_20260926/RESULTS.md)及后续BGCR已经使用更上游neck ROI；不能把neck本身包装成新信息。

- 在实际产物检索中未找到SAM/SAM2/Mask2Former框提示结果→原P回投或教师蒸馏的已执行Run。文献note和官方转换器中可选SAM函数不算执行记录。未记录或未回传资产不能作绝对不存在保证。

- 未找到直接RGB块或P2/stride4浅层图进入新增系数读出的已执行证据。它需要新训练，短预算阴性仍难区分训练不足；暂次于零训练外部证据检查。



## 文献、实现与数学限制



[SAM原论文§7.4/D.4](https://arxiv.org/html/2304.02643v1)已经用检测器的预测框提示SAM做实例分割；引入SAM或框提示本身没有新颖性。框也可能对应对象、部件或邻近物，SAM标注习惯不保证与COCO一致。



[官方predictor](https://raw.githubusercontent.com/facebookresearch/segment-anything/main/segment_anything/predictor.py)支持每图一次embedding、批量框提示以及返回连续logit。候选设计用ViT-B、multimask_output=False、return_logits=True，不用GT选输出、不加点击或迭代反馈。输入可沿用同一640 RGB，SAM内部按官方预处理；明确内部尺度和外部编码器成本仍与YOLO不同。



这条路线新增的是独立预训练空间先验，不是仅用YOLO特征的等参数/等预算改进。即使有收益，也不能证明唯一根因、YOLO内生能力或可蒸馏性；还不能作为轻量部署方法贡献。它只检验一种无GT信号在原P里能否被兑现。



## 拟议单轮及关键反例



先核既有OGPS真实函数、固定λ/概率截断/8×8几何并复用，不重建求解器。固定现有dev256官方候选，原预测框；候选指标不冒充完整推理AP。主对照至少：原YOLO、SAM完整连续输出、同一SAM压成8×8直接解码、同一网格通过固定solver回投原P、原YOLO网格经同solver。不能省SAM完整输出和网格直解，否则压缩/投影改变会混在一起。对象不兼容的旧BASE-solve数字不得直接借用。



如果SAM完整输出有益且原P回投保留收益，支持外部无GT证据可在该P中兑现，不证明蒸馏已成立；完整输出有益但回投损失，需区分网格压缩与当前投影；完整输出本身无益，停止当前教师/提示配置，不自动加提示、扫λ或训练预测器。



预算拟定一次2图smoke与同dev256正式零训练，计算上限60分钟；下载与环境准备单独记账，不因超时选择已跑前缀作结论。具体复用函数、臂定义、检查点来源、停止阈值须在独立Study配置中锁定后才能启动。当前仅完成研究审查，尚未下载模型、登记或运行新Study。服务器专属本线程，笔记本不访问。



## 已核验复用接口（只读，未启动）



原函数：[ogps_solver.py](../../experiments/ownership_solver_replay_20261002/scripts/ogps_solver.py)，与local_view的副本SHA256均为`7d4c60d31add5be7cac8a1385c8e50b40753bb9eef04472135ab08d415ca3076`。`crop_pool_7o`按640预测框映射到特征分辨率，floor/ceil整数裁剪，再adaptive average pooling到8×8（不是ROIAlign）；`pooled_design_matrix`从完整P构造A[64,32]；`solve_ogps`截断概率[.01,.99]→logit，FP64求解：



`min_delta ||A(c0+delta)-t||²/(2*64) + .003*||delta||²/2`。



无额外jitter/自适应正则/系数截断，求解后转FP32正常解码。它不是BCE oracle，也不是fast-screen的`.1G`算子，禁止拿缓存16×16投影替代。`grid_to_canvas`/`add_direct`来自[已有适配](../../experiments/local_view_coefficient_replay_20261003/scripts/evaluate_local_view.py)。



当前fast-screen缓存完整P/c0/预测框/真实ratio_pad/输入与候选身份齐全，只需生成新SAM响应。新对照A逐身份复现1816候选；旧196图/1346候选OGPS结果和93.65%oracle恢复比例不能迁移。



**重要非交换性**：旧BASE_SOLVE用`sigmoid(A8*c0)`；如果SAM使用`pool(sigmoid(full_logit))`，本轮YOLO自投影也必须走`pool(sigmoid(full_logit))`。二者不同，不能借旧BASE数充当同预处理基线。固定λ=.003、网格8、不调参。SAM原尺寸输出与内部低分辨率logit要区分；拟用同一缓存640 RGB作为SAM输入，其返回原尺寸即640，再统一真实letterbox逆变换。单图embedding复用，box坐标保持640，不混原JPEG框或256低分辨率输出。具体代码实现前再核验缓存RGB通道约定。

