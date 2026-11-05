# 正常监督训练源隔离与只读QA V1

承接[本次完整流程的数据权限](AUTHORIZATION_AND_SCOPE.md)，不是合成生成许可。只允许原1272 real/train缓存GT用于正常分割监督及工程读取；**生成器不得读取本阶段产物/标注**。432 dev仅允许图片元数据用于拆分排重，不读其RGB或GT；test426不接触。

## 输入与防泄漏

可信入口是既有`spm-v4-scale-20261023/inputs/REAL_MANIFEST.json`。已审其生成源码`prepare_real_local.py`：来自COCO的images表，只含图片元数据/源SHA/分组/处理RGB路径和valid_rect，不含annotations。禁止读取内嵌RLE的DATA_INDEX/REAL_DATASET、全1704的REAL_SOURCE_COCO或DEV_COCO。解析后检查禁入字段，出现annotation/RLE/box等payload立即中止并记录，不能假装仍是纯元数据。

从该入口先核1704图片元数据的既有13/4组划分与1272/432计数、ID/组级不交叉；然后**只为train行**构造原`inputs/real_images/<id>.jpg`及`cache/real_<id>.npz`白名单。RGB逐SHA与旧manifest一致，既有GT缓存只读并新冻结SHA；不重新从混合train/dev原COCO计算或修改标注。

缓存语义沿用已运行基线：原640可见实例标注、归一化xywh框、160×160最近邻loss_mask、单类pig→训练class0、固定valid_rect。检查shape/dtype/有限性/框范围/ID对应、记录掩码为空和重叠等统计，不悄悄丢弃原实例或更改旧>16像素筛选规则。此处是已存在缓存的身份与读取QA，不重新认证人工标注的全部语义正确性。

只输出train-only路径/SHA/组ID manifest及汇总，真实像素和GT不复制进生成目录或主仓库。manifest可提供给后续模型工程/监督训练，不能用来调合成配方。读取具体训练图用于真实模型8图前反向须后续工程合同；本阶段RGB只做文件SHA与尺寸头核验，不用图像选择超参数。

## 执行与判定

个人服务器一次Slurm，0GPU、4CPU、8GiB、≤5min，0模型前向/优化。所有实际读入1272个GT缓存均如实计数，不再写“标注读取0”。源manifest、代码、逐文件身份、环境/日志和Run保存。未知源/缺失/哈希不一致/拆分污染立刻停止，不换图或缩小1272分母；最多2次已定位工程重试，不改数据资格。

通过仅开放这份train-only源用于正常模型工程合同准备，**不开放正式训练**；仍须新合成release、实际8图接口和冻结资源/训练合同。失败保留缓存/源身份，优先修读取/身份问题，不将其解释为RLPL方法无效。原读取边界事故见[勘误](READ_BOUNDARY_ERRATUM.json)，本合同不追溯授权或掩盖该次事故。
