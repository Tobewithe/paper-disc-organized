# S1 执行协议：同一最终检测的有限系数恢复

2026-10-13。用户已授权在笔记本按总控执行，先过 S1，再启动 S2。本文细化尚未运行的 S1，不修改旧 Run 门槛；与 POSTTRAIN_RESCOPE、EXECUTION_CONTRACT、MECHANISM_AND_DEFAULTS 共同冻结。

唯一命题：在冻结的真实 Mask75 失败、Box75 配对且该同一检测 MaskIoU<.75 的集合 T 中，量化给定数值预算找到的同检测系数修复比例。通过条件为至少200个预选不同 GT、找到 MaskIoU≥.75 的比例≥20%；结果是机会下界，不是部署恢复率或 P 表达上界。

- 来源：`RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170` 的原生账本、完整预测及同次 P/c0 缓存；采用原有 v8.4.100、权重 SHA、FP32 CUDA 原生解码与数值开关。继承原输入几何、批次和最终行身份，不重新前向。
- 正式面板：按既有 `posttrain_audit.make_panel` 的 SHA256(`image_id:annotation_id`) 升序取前256个 T；不看优化结果挑样，不补样。工程 smoke 使用紧接其后的4个 GT，独立 Run，不进入正式统计。
- 每个选中 GT 只改变其 Box75 匹配输出的系数；固定 P、原框、类别、score、检测顺序、decoder。按原图像的完整输出矩阵重放，避免改变矩阵形状导致数值偏差；同图其它输出保持原系数。
- 数值方案完整沿用 `finite_coefficient_solver.DEFAULTS`：CPU float64、zero-delta、mean BCE + .003×||delta||²/2、LBFGS lr=1、max_iter=120、max_eval=150、history=20、strong_wolfe、gradient acceptance=1e-6，单起点。CPU线程=1；送入原生 CUDA decoder 前只转换为基线 float32。
- GT 由原始 COCO `annToRLE` 生成，再按缓存真实逐轴 resize 比例和整数 padding，以 nearest 投影到原型网格。loss 仅在预测框的原型网格像素支撑内计算；最终评分始终使用原图完整 GT RLE。空 ROI、非有限数及身份错误保留并阻断，不能从分母删除。
- 控制：既有同配置解析非零最优点控制；真实缓存所有输出 zero-delta 精确 RLE 重放；单位基向量系数的原型通道直接解码控制；GT 投影/完整原图并集测试。工程 smoke 不承担科学门槛。
- 预算：smoke 最多600秒；正式 S1 最多3600秒；单进程，内存预检≥4GiB，新增产物≤1GiB。不重跑模型，不扫描超参。启动器使用独占阶段锁，前后核对代码与输入身份。
- 统计：分母固定256；数值失败保留。合法系数达到 .75 即计为已找到；未达到且存在预算未收敛样本为 `blocked_solver_budget`，不得宣称无表达能力。全部数值有效且未达到20%则停止该 c-only 配置；达到门槛才允许进入已授权的2k pilot。
- 交付：独立 `run.json`、源码/协议快照、输入与配置合同、目标面板、控制结果、逐GT结果及系数、SUMMARY、DECISION、产物清单。保存 eligible/attempted/completed/numerical_failure、面积层与距离 .75 的分组；不把单对 IoU 当完整 COCO 任务恢复。

成本理由：256个固定 GT 满足既有至少200的机会筛查要求，复用缓存把本轮计算限定在系数求解和原生解码；20%是当前路线的预设机会门槛，不因本次结果调整。
