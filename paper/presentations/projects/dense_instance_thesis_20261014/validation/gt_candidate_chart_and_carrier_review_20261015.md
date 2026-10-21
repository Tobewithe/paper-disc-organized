# GT 分类修订版核对

- P06：完整概念图解释 IoU、候选身份与联合达标；图片不承担实验数值。白底、原生标题、完整展示。
- P07/P08：五类按语义顺序排列，每类定义与真实数量同行；横轴从0起，两页共同上限25000。未使用连续漏斗或不同分母相减。
- P09：保留原生最终输出账本数据图，只明确标题中的最终输出范围；讲稿区分6688/6459/6229三个对象。
- P10：像素区域图解释固定预测的错误分量；四类误报与自身遗漏在讲稿中明确区分。公式分母为有效自身GT面积。
- P11：保留ICI/E4场景分层，关系字母分类不出现；公式是累计框交叠与有效边界像素占比。
- 其它21页仅撤下页底来源小字和页码，完整保留主体；模块二、三仍标记拟开展。
- 图像均由image_gen生成，仅用作概念图。3图分别使用一次，完整展示，无裁切；数据条形图保持可编辑。

verify-charts: 07_原始候选GT分类.svg | object=raw-gt-states | type=bar | mode=direct-calc | scale=0-25000 from ticks | calc=ran | svg=unchanged | slot=70,height=24

verify-charts: 08_类别正确GT分类.svg | object=argmax-gt-states | type=bar | mode=direct-calc | scale=0-25000 from ticks | calc=ran | svg=unchanged | slot=70,height=24

verify-charts: 09_失败定位.svg | object=failure-split | scale=0-8000 | source chart XML identical to prior verified delivery

verify-charts: 15_修正空间.svg | object=oracle-area | scale=0-100 | source chart XML identical to prior verified delivery

verify-charts: 17_自动修正探索.svg | object=pilot-effects | scale=-4..2 | source chart XML identical to prior verified delivery

verify-charts: 18_面积分层.svg | object=area-damage | scale=0-16 | source chart XML identical to prior verified delivery

verify-charts: 19_选择性修正结果.svg | object=selective-effects | scale=-0.5..2.0 | source chart XML identical to prior verified delivery

verify-charts: 19_选择性修正结果.svg | object=selective-damage | scale=0-6 | source chart XML identical to prior verified delivery

数据核验脚本：sources/verify_gt_candidate_revision.py。校验来源摘要哈希、五类计数总和、各条长度、分类比例、旧图对象一致性、关系分类撤下、27页讲稿对应和视频计划状态。
