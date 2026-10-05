# 位置相关的 prototype 通道动态读出

## 独立问题

此前 U/Q/P/L/B 已比较固定数量的空间证据选点，未超过原模型；7J、BGCR 与分区系数也没有证明“再读更大区域”能泛化。本轮不再选少量观测位置，而检验一个不同的读出假设：同一实例的 prototype 组合是否应随位置改变，且改变应由该位置的 prototype 响应参与决定。

固定原模型的预测框、候选身份、prototype 与检测分支，只训练原生 one-to-one 系数分支及一个零初始化的低参数读出。对候选 i、prototype 通道 k、位置 u：

\[
z_i(u)=\sum_k c_{ik}P_k(u),\qquad
z'_i(u)=z_i(u)+\rho_i\left(32\sum_k a_{ik}(u)c_{ik}P_k(u)-z_i(u)\right).
\]

\(a_{ik}(u)\) 是通道 softmax。两组使用同一个查询和框相对坐标门控网络；P 组额外把当前位置的 prototype 响应 \(P_k(u)\) 加入通道选择分数，G 组将该输入置零。这样 G 是同容量的位置门控对照，P 才检验 prototype-conditioned dynamic selection。\(\rho_i\) 零初始化，初始输出逐像素等于官方 \(z_i\)。推理不需要 GT，不增加候选或后处理。

## 执行边界

- 官方 YOLO26m-seg、Ultralytics 8.4.100、COCO one-to-one TAL 固定候选；服务器原图在线重放，桌面不运行模型。
- 1024 fit / 256 dev，seed 0，固定 3 epochs；N（普通原生系数微调）、G（位置门控不看 P）、P（位置+prototype 响应门控）三组同初始化、同预算、同 mask BCE、同解码。
- 只报告正常原图 IoU、框好掩码差组、Mask75 修复/损伤、coverage、AUC/FPR；图片为bootstrap单位。训练 loss 或通道条件数不能替代任务效用。
- 预注册筛选门槛：P 相对 G 的主失败组图片macro IoU至少 +0.5 个百分点且区间下界为正，同时全体 P−A 不低于 −0.1 个百分点；否则停止，不扫温度、rank、位置网格、训练轮数或新增 gate。
- 若 P 不超过 G，结论只否定本轮 prototype-response 动态门控，不否定所有位置相关读出；若 P 超过 G 且不伤害原成功组，才另行登记确认实验。
