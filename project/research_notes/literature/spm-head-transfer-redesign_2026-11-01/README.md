# RLPL复盘：归一化、保持与任务输出蒸馏的定向核查

对应[复盘/候选路线](../../../../experiments/spm_task_aligned_head_transfer_design_20261101/REVIEW_AND_ROUTE.md)。本轮6条arXiv定向查询；4篇PDF下载并读取指定方法段落，不是穷尽综述、逐页全文审读或代码复现。原RLPL的SwAV/DenseCL/PixPro近邻沿用[已有审查](../spm-prototype-candidate_2026-11-01/README.md)，未伪称本次重新全文阅读。

## 对设计真正有用的工作

| 一手来源 | 本次核查范围与核心做法 | 对本设计的支持 | 不可借用的结论/实质区别 |
|---|---|---|---|
| [Wu & Johnson, Rethinking “Batch” in BatchNorm](https://arxiv.org/abs/2105.07576) | 正文§1–2、§4.3–4.4与§5开头；BN跨样本计算，训练mini-batch与eval population统计不同；讨论FrozenBN | 合成不进任务target仍可影响真实输出/梯度；末点buffer交换与训练时混合是不同干预；冻结全部统计作为简化措施有依据 | FrozenBN并非普遍更优，论文也报告大normalization batch下可能不如普通BN；没有证明本项目AP退化就是BN造成 |
| [Li & Hoiem, Learning without Forgetting](https://arxiv.org/abs/1606.09282) | §2.2–3.1；保存旧模型在新训练图上的响应，与任务GT共同训练，保护旧函数 | 给K_N原生概率保持对照明确来源；输出保持不要求新增部署模块 | 原文主要是分类/增量任务，不保证本项目实例成功集≤1%损伤；本项目是同一任务固定最终检测的读出，不照搬类别温度/头部扩展 |
| [Kim et al., Offline-to-Online Knowledge Distillation for Video Instance Segmentation](https://arxiv.org/abs/2302.07516) | §3.1–3.3及架构图说明；离线跨帧教师，GT关联/过滤query，蒸馏对象embedding | 教师与学生的实例身份必须对齐；错误/无关教师知识会损伤学生；昂贵离线知识用于简单正常推理已有先例 | 原方法是VIS时序embedding/关联，不是本项目固定raw/ordinal的Mask概率；我们没有时序教师，也不把QFA或离线蒸馏当原创 |
| [Liu et al., Structured Knowledge Distillation for Dense Prediction](https://arxiv.org/abs/1903.04197) | §2像素KL、§2.1结构项、§2.2目标和§3.1实验设置开头 | 逐像素软概率蒸馏已有明确基线；直接核输出而非命名隐藏特征较适合当前正常Mask目标 | 论文还使用pairwise及对抗holistic项，本设计不采用；论文不证明仅像素KL足够，不能用其组合收益承诺本项目增益 |
| [Domain-Specific Batch Normalization](https://arxiv.org/abs/1906.03950) | 本次仅检索元数据/摘要 | 提醒“混合域分统计”是已知替代，不是新发明 | 未读取正文，不作为选择双BN的证据；本次不新增双分支归一化 |
| [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054) | 本次仅检索元数据/摘要 | 线性probe/保护预训练特征为已知方向，不能认领原创 | 摘要中的两层理论/分类结论不直接套到YOLO；固定末层的选择主要由本项目反复读出退化、已有算子与凸性推导支持 |

## 新颖性主判断

**“蒸馏＋冻结BN＋原生部署”本身不新；“只训练最后一层”也不新。** 当前选择是降低无效假设数量、争取有实测支撑的任务迁移，不是把RLPL换名包装成新机制。

本候选的可检验新增价值是：以同一实际检测身份和固定原生基底为约束，分开检查教师编辑的可达性、共享读出的可学习性、普通函数保持与更好教师的额外效用，并在完整真实任务上验证损伤和部署折衷。只有这些结果实际成立，才评估应用论文贡献；不保证高水平机制新颖性，也不将教师自身的原型知识归因转交给学生。

本次查询范围没有覆盖所有2D实例分割蒸馏/低秩掩码读出文献，不能声称不存在更近先例。若资格/筛选失败不继续扩综述凑贡献；若得到真实可重复增量，再围绕具体差异补直接方法基线和独立验证。

## 获取与审阅记录

- [queries.json](queries.json)：6项查询的原始结果，包括不相关检索命中，不把命中数量当相关文献数。
- [sources.json](sources.json)：4篇PDF及文本SHA、读取段落、临时缓存位置；缓存不入仓库、不保证永久保留。
- 首次`arxiv.org/pdf`请求SSL失败；改用`export.arxiv.org/pdf`获取。首次pdftotext默认编码不能按UTF-8读取，显式`-enc UTF-8`重提取后成功。没有绕过TLS验证，也没有因此声称首次下载/解析成功。
- 只审阅方法相关段落，未复现实验，未做新模型前向、训练或GPU任务。
