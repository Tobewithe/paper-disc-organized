# 主张—证据对应

2026-09-23。数值均来自已完成原始结果；三种子、Boundary AP及固定先验/系数强控制的本地评分已完成。本文件是引用索引，不复制原始记录。

实验简称：M=`experiments/coco_spatial_calibration_mechanism_20260922`；T=`experiments/coco_template_source_20260921`；N=`experiments/coco_native_spatial_readout_20260922`。完整路径均相对于项目根目录。

|主张|证据入口|范围/协议|限制|
|---|---|---|---|
|相同响应仍有位置条件差异|M/runs/RUN_c1b150dc71c54c3e9abcfe07d6da9303/CALIBRATION.json|第二批500张确认图；GT框内硬标签，z∈[0,.5]|池化统计有实例组成混杂|
|同实例差异仍存在|M/runs/RUN_5df5c53a9c9c4727b551e94d20529ce8/RESULTS.json|1,713实例对，图像簇bootstrap；残差差6.832pp|第二批的后续分析；不是新独立确认集；logit区间不是严格完全相同logit|
|空间项超越统一bias|M/runs/RUN_89e14901533a4493a9617a16414e68e6/SUMMARY.json；M/runs/RUN_c1b150dc71c54c3e9abcfe07d6da9303/SUMMARY.json|固定同一候选；空间项移除/反向；独立训练scalar控制|固定候选IoU不是COCO AP，不覆盖检测漏失|
|有效方向大部分在原型空间内|M/runs/RUN_c1b150dc71c54c3e9abcfe07d6da9303/instances.jsonl；M/DECOMPOSITION_PROTOCOL.json|无GT投影；32原型、FP64、实际预测crop|仅投影aT，保留bias；不存在所有错误的表达上界证明|
|方法不依赖旧16格头产生模板|T/REPORT.md；T/runs/RUN_9de7e2ec9f5643d3874bbd4422262def/ANALYSIS.json|fit残差模板44.3654 AP；解析模板44.3327 AP|单种子，不宣称小数差异有显著性|
|原生读出本身可改善|M/runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66/RESULTS.json|6,240原参数微调；同14,204训练记录；5000val|只微调末层，不代表所有端到端微调方案|
|空间方法减少错误但有覆盖取舍|M/runs/RUN_89e14901533a4493a9617a16414e68e6/SUMMARY.json|覆盖90.334→88.818；纯度83.738→86.194|不能写“保护全部自身覆盖”|
|现有最终AP与修复损伤|T/runs/RUN_9de7e2ec9f5643d3874bbd4422262def/ANALYSIS.json；M原生评估|完整val2017；原框、分数、正式解码、COCOeval|本稿表格为seed0，不混入历史三种子平均|
|GT支持范围不是已确定根因|M/runs/RUN_72fc1ae7ebe14aa78fc25fd5bb305bee/RESULTS.json；M/runs/RUN_e4ca009989c245e59018b20432040e68/RESULTS.json|三组修正头训练支持消融|没有重放官方预训练；GT交集vs随机差仅−.0394AP|
|移除ROI卷积后空间作用仍保留|N/runs/RUN_1bb359da7b4942b7a7ab3f0be194ff49/RESULTS.json|同原生特征MLP空间44.2720、标量44.0431 AP|单种子；距原生末层alpha.5仅0.074 AP，不能夸大优势|
|三种子空间作用重复出现|N/runs/RUN_5c3523cfdb1f4328ab0901483037e5d8/RESULTS.json|空间44.2684±.0048、标量44.0257±.0152、末层微调44.1848±.0060 AP|SD不是AP测试图像置信区间；同一预训练权重与fit划分|
|边界有基线收益但未超越原生微调|N/runs/RUN_0425bd7208e149c38352d079248fcfee/RESULTS.json|作者Boundary AP实现；seed0空间29.9792，微调29.9911|不支持空间方法所有指标占优|
|原生头比ROI头更省时但并非免费|N/runs/RUN_ab6cb4908020457684da46187ecd1ad8/RESULTS.json|64图、3重复；中位16.953 vs19.662 ms；baseline11.067|0.001分数的评价运行点；不含RLE/读图，不等于所有部署速度|
|普通系数读出接近空间头AP|experiments/coco_spatial_readout_controls_20260922/REPORT.md及四个本地评分Run|完整5000图，普通系数44.2564、空间seed0 44.2720 AP|仅0.0155点差；未做AP等效性/显著性检验，新控制单种子|
|空间头相对系数头的R75差异集中在小目标|experiments/coco_spatial_readout_controls_20260922/runs/RUN_4e10bc2bf0094cbbaa9eb5400dd5fb69/PAIRED_MATCHED75.json|小目标+1.0286pp [0.6983,1.3892]；全部+0.4073pp|条件于seed0固定预测；探索性分组，未多重比较校正；不是AP差或唯一训练根因|

## 尚不可写入摘要的结论

- 全模型族/跨数据集稳定有效：未完成。
- 更换预训练权重或重新抽取fit数据后的稳定性：未完成；现有三种子仅复现追加训练。
- 精确的原训练历史根因：没有唯一定位。
- 官方原型完美、系数相似导致失败、CCL解决粘连：本论文没有支持这些命题。
- 实时、近乎免费、严格更优：统一计时和强控制尚不足。
- 一区录用：不属于实验可以保证的事实。

## 不得混用的量

AP采用0–100点；固定实例IoU/覆盖/纯度差采用百分点；Mask75修复/损伤为官方匹配GT集合变化。两者均不等于原始候选池的oracle可恢复比例。诊断GT选择只用于训练和解释，部署时不使用GT。
