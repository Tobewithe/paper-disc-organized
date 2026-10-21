# 已有外观跟踪方法补充对照

在新 PEA 确认结果读取前冻结。补充 Ultralytics 8.4.100 原实现 BoT-SORT 和 Deep OC-SORT，统一原检测、64维身份特征、conf .3/.1、寿命60、match .8、GMC none、proximity .5、appearance .9，均不调参/不新增训练。
BoT-SORT 保留其原 Kalman XYWH 和 EMA；Deep OC-SORT 保留其原骨架。用现有 frame_input 将同特征传入原 encoder auto。明确这是共同输入、无GMC的版本内对照，不冒充论文榜单的原生ReID/GMC配置。

完整运行 dev、历史五视频、horse_7、zebra_1/2、zebra_3/4。对已保存的 Deep OC-SORT dev/test 输出要求逐帧完全相同，验证适配无偏差。运行前2×16帧 smoke。新增参照不改原 Gate，但论文与报告必须同时报告相对这些既有方法的效用；没超过时不得说优于既有方法。

来源：[BoT-SORT](https://arxiv.org/abs/2206.14651)、[ByteTrack](https://arxiv.org/abs/2110.06864)。本项目的外观特征前端、保护约束与已知跟踪骨架分别归属，不能把已有组成当作首创。
