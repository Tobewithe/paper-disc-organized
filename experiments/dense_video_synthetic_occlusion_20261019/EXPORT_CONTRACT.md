# 标签格式导出合同

输入固定为已完成并独立核验的RUN_SCENE_PILOT_20261019T092948_31ae46d0，数据不重生成、不更改。

新增CPU导出：COCO可见分割、独立COCO完整投影分割、每序列MOT gt.txt、逐目标遮挡状态JSONL。类别名称为procedural_quadruped，不将几何素材标为真实动物。

- COCO images=144；可见目标V>0才写可见注释；完整投影A_in>0才写amodal注释。保留scene、track_id及asset_id，不预先声明train/test拆分。
- 分割使用COCO未压缩RLE，Fortran像素顺序；每条RLE解码后必须与原掩码逐像素相等，面积和框一致。
- MOT为1-based帧/左上角坐标；仅写V>0的可见框。visibility=V/A_full，包含遮挡和截断影响；外部遮挡率与截断率在JSONL分别保存。
- 全部288条目标状态保留在JSONL中，包括完全遮挡和出画。不得通过插值为不可见目标制造检测框。
- 预期252条可见分割/MOT观测、280条原视野完整投影、288条状态。若不一致则失败，不改分母。

GPU使用0，模型训练0步。本导出是已验证生成实验的交付步骤，不新增方法效用或机制结论。
