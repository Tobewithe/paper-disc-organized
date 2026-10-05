# S019审计记录

日期：2026-09-12。审计者：新上下文 GPT-5.6-Sol ultra `audit_grid_support`；review_independence=same-family，acceptance_status=provisional。

**最终状态：有限范围终审WARN。** 审计者已返回结果，未发现假GT、自归一化、虚构结果或四臂实质混组。没有重跑CUDA优化或COCO解码，因此仍是same-family/provisional。

终审补充：

- A PASS：审计者独立核对与protocol同SHA的本地COCO JSON，887个annotation均存在、唯一、图像归属一致、非crowd；其中155个多片polygon。
- C PASS：审计者独立核验主receipt490项和导出manifest501项，missing/hash_mismatch均0；复算10组均值/.75数量及C−A、D−C、D−A一致。4,435行=887×5，四个优化臂总计3,548项。
- D WARN：后补完整梯度/空支持witness闭合了初审缺项；保留浮点插值顺序10项1—4像素差、无75判断变化，以及3,548求解中1,352触120迭代上限。A与S018非逐比特一致，最大差约1.23/1.03IoU点。
- E WARN：反复探索val内预选失败目标的有限预算same-image oracle，不是共享头训练/全COCO总体/独立泛化/密集因果。
- F PASS：real_gt + same-image GT-assisted oracle diagnostic。

审计允许：主聚合用于固定预算oracle诊断，高C−A+17.465、D−C+5.227、D−A+22.692点；其余+19.226/+4.822/+24.048点。不得改为收敛上界、可学习方法收益、官方AP或密集专属因果结论。

以下保留初审和补查经过：

审计者已返回：

- A GT来源PASS：grid_support_factorial_probe.py:81,86读取COCO；crossimage_response_experiment.py:40—51先annToMask合并多片再letterbox；主脚本:110—111逐GT校验160标签与S017相同，:134原始COCO RLE作GT，无预测制造reference。
- B归一化PASS：:35—37使用grid面积×GT归一框面积，对应归档loss.py:586—588；:109的GT面积及参数尺度四组共享，无预测统计自归一化。
- C结果初审仅核验S018源887行/243图/高215其余672闭合；S019主结果当时尚在运行，因此代码初审没有声称结果存在性PASS。
- D四臂控制通过初审：:22及:143—146只改变网格/支持，:25—57同solver、原c0/area/scale/max_iter，:131—147最终同原预测框解码；640使用同P插值，不新增原型。初审WARN：主程序只比梯度cos，没有幅值比较；空支持分支跳过官方对照。
- E范围WARN：:72—80声明当前冻结、同图GT oracle、非AP/网络训练/容量上界。不能解释为密度因果或方法增益。
- F类型：real_gt_oracle_diagnostic。

作者根据初审补充并实际运行verify_grid_support_decode.py：全部3,548项保存系数重解码一致；完整初始系数梯度幅值最大误差1.7881393e-7，低于预设1e-6+1e-4 max|官方梯度|；空支持一项官方loss/grad=0。证据在DECODE_WITNESS.json和decode_roundoff_witness.csv。这是作者确定性核验，不能称初审者已重算。

全部结果已下载并通过作者本地核验：远端501文件manifest、4,435行含原预测/887唯一目标、各组IoU与.75数复算、执行源码hash一致。见LOCAL_VERIFICATION.json。审计者随后完成上述有限范围结果审查；不得据此写“全面独立复现”。
