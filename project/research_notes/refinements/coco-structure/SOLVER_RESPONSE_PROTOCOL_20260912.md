# S041：正则求解响应的跨图可学习性对照

2026-09-12。执行前机器协议见run/protocol.json；本文在执行期间将同一协议整理为中文，不能称它早于启动。开始前已固定全部臂、参数和门槛。

待回答：S040同图GT辅助全局系数解的收益，能否由原有共享输入预测？先用相同头、优化预算和原预测框检验，不预设必须改结构或恢复CCL。

数据：S024缓存的1,200 fit图/7,811固定bbox50目标，300 transfer图/2,002普通GT，全部来自COCO train2017。transfer未参与这几个头的拟合，但已被项目多次探索。本轮仅筛查，正结果还需预先固定新的确认图；不得以“未参与训练”代替“从未看过的评测”。

三个种子分别继续S032 rawCOCO epoch15同一头（73→128→128→32有效输出）。保留Adam历史状态和随机顺序；每种子三臂额外15轮、每轮245步，共3,675更新/臂，135checkpoint全部保留，取最后一轮。使用本地conda pytorch、私有Ultralytics8.4.143；主干/特征/原型/框/分数/类别固定。

- direct：继续原始rawCOCO像素BCE+Dice。
- self_response：同一原始损失，加保存头响应的Bernoulli KL，权重1、温度1。
- solver_response：同一原始损失，加正则求解响应的同种KL。教师仅fit图GT生成，512位置同direct；不直接回归系数。

教师求解沿用S040 global32目标：meanBCE + 0.01 mean(Δlogit²) + 0.0001 ||w||²，原型基底按这512位置RMS标准化。FP64 Newton最多30步，梯度阈值1e−7。保留所有有限教师、记录未收敛和极端系数，不按teacher质量/transfer效果筛选；用sigmoid概率目标避免将系数范数当监督尺度。教师预计算耗时和文件单列，共享SGD预算相同不等于整个算法总算力相同。

评价：所有预测槽位完整解码原预测裁切，全部普通GT官方COCO匹配后按实例ICI统计。主比较是solver_response对direct以及self_response，不能只比15轮旧头。主指标正常Mask AP、高R75、nonhigh−high召回差、低/中/高分别统计与自身/邻居/背景。辅助按同一全局Precision≥0.9操作点报告分组Recall、类别×面积组成控制；全局操作点为评价曲线描述，不是训练选出的部署阈值。

三种子先按目标取均值，再对图像簇配对2,000重采样。点态区间未多重校正，不给AP或P90阈值伪造区间。模型、种子、轮数、损失权重不在结果出来后挑选。

筛查门槛：solver超过两个同预算对照的高R75，整体AP保持且高/非高差不恶化，才值得进入新图确认。否则停止当前配方，不调lambda/epoch追分，也不由此宣布全部蒸馏/条件信息研究无效。当前没有新颖性结论；GT辅助上限式诊断和教师训练目标也不构成实际方法得分。

运行输出：experiments/coco_clean_20260911/diagnostics/solver_response_20260912。
源码：train_solver_response.py；统计与P90：summarize_solver_response.py。
