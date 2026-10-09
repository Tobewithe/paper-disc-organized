# 单区域局部动作：有限机会未达投入门槛，当前配置结束

正式512图机会与独立动作核验均已完成；B按预登记门槛接受`stop_current_configuration`。完整数字、范围和成本见本文末A正式结果，B最终验收随后附录；本Study没有训练可部署方法。以下工程和中间审查保留各自当时的状态。

## 早期工程记录

正式本地 CPU 工程 Run [`RUN_LOCAL_EDIT_CORE_ENGINEERING_01`](runs/RUN_LOCAL_EDIT_CORE_ENGINEERING_01/run.json) 于 2026-10-09 00:38:26（Asia/Shanghai）完成，exit 0、产物完整。[十组检查](runs/RUN_LOCAL_EDIT_CORE_ENGINEERING_01/CHECKS.json) 全部通过：零动作和区外保持、数量与稳定顺序、增删方向、支持保护、空满/边缘/窄形状、距离边带、只读非连续输入，以及独立静态审查补出的桥接连通分量完整性和等面积最小列排序案例。

这是不读图像/GT、不执行模型的小网格 NumPy 检查。它证明列明的动作核心工程不变量，不证明真实 native 解码一致、局部动作可修复空间、RGB 可学性、部署损伤约束或论文创新。Study 保持进行中，真实训练面板及方法训练均未执行。

核心接口见 [`local_actions.py`](scripts/local_actions.py)：`generate_actions` 接收同网格 baseline、正负阈值 trial 和支持，返回零动作及稀疏单分量动作；`apply_action` 返回副本。适配器必须绑定原候选身份与 baseline，并按官方解码生成 trial/support；action_id 本身不是跨图候选键。零动作 native parity、负阈值 crop 外保护、逐动作解码内存和最终原图变化比例由 A 的真实适配阶段验证。

下一步依协议先固定 train 侧面板并检验 `best(G_eq+local)−best(G_eq)` 的有限 GT 诊断机会。真实数据运行前已纠正方向混淆：G_old 是旧五个全局收紧动作加零动作，G_eq 额外包含固定全局 tau=−.25 扩张；另报 `best(G_eq)−best(G_old)`，不把新扩张方向的机会归因于局部性。该纠正不改工程核心与其历史 Run，也不把负 tau 输入旧 multi_local 模型。机会不够即结束本配置，不扫描更多动作。机会存在也须通过信息／动作公平控制才能进入 20k 方法结论。A 的冻结强对照比较在独立 Study 继续。

正式来源摘要：核心 `f69206636730b18c4a5195177c51bd063874ab6522b6c30548a5a1dee772f3ef`；检查脚本 `655d958a8a47819908cfa3af7175a57b45db16b96be00b58dda2966563fe7a89`；CHECKS `d9a4a7e8bd95c5bcc5c13ea2c85c457a29f7e357a23a7c4aded31ee59b00fef8`。runner 已保存协议、源码和执行环境快照；Python 3.12.14、NumPy 2.3.5。

三个此前调试执行补记为独立 `RUN_LOCAL_EDIT_DEBUG_*_01`：默认 Python 因缺 NumPy 失败，另两个已安装环境的八组旧检查由 worker 报告通过。它们没有原始 stdout 文件、源码快照或准确执行时间，记录明确为事后 `agent_reported`，未知保持为空，不能代替正式 Run 的可核验证据。

## 2026-10-09 A：真实面板与原生工程

`RUN_LOCAL_EDIT_PANEL_PREPARATION_S0_01` 于 01:15:46（Asia/Shanghai）实际 exit0：原20k减旧门控校准交集362，合格池19,638；seed20261009均匀固定512，全部图像SHA/完整解码/原标注尺寸通过，无缺图或换样。PANEL SHA256 `ea51b6c8b1cb19de3d390d53b68595dbec9fae296410adbfa0d602405884e9d7`。该面板用于路线选择，不能再当独立风险校准或确认集。

`RUN_LOCAL_EDIT_NATIVE_ENGINEERING_S0_01` 于01:23:22实际 exit0，四图/268候选/5,204动作。与当次桌面官方native输入及原图掩码逐byte一致，候选身份、单区域、真实crop支持、区外保持、空动作及拒绝跨候选/改变baseline绑定检查通过。模型状态前后均为 `aa52514c89a60d349cc8f0e53217c3a4626f2058b639390003601b9bfdf1b3d5`，梯度为None、参数冻结、BN eval；无GT解析、无训练、机会统计未知。该Run摘要SHA256 `d74b5836a6ae32732f068a49ba010d96fcb143341bc2d3136fe81974ca587534`。

实测四图总22.439秒，GPU模型0.704秒、动作生成4.636秒、动作导出1.295秒、输入/绑定检查6.146秒、RLE编码1.641秒；包含关系的计时不全部相加。峰值CUDA allocated/reserved为848,378,880/1,136,656,384bytes，CPU peak约1.79GB。正式机会另含GT解码/像素账，不能把工程外推当实测或完成时间承诺。

独立 `RUN_LOCAL_EDIT_NATIVE_ENGINEERING_VERIFY_S0_02` 于01:40:43实际 exit0/passed：重核来源/512抽样及排除语义、已存RLE、稀疏分量连通/顺序、候选绑定及原图改动。核验不重跑模型、GT解码或匹配；原native逐byte相等来自实际工程的断言及对应零动作记录。首次核验 `_01` 因核验器把Windows两种路径分隔符误作不同路径失败，原Run和快照保留；仅核验器修为samefile+逐项SHA，F源锁未变。

`RUN_LOCAL_EDIT_OPPORTUNITY_S0_01` 于01:30:26真实启动，在同一锁定512图计算G_eq与单区域有限动作的GT诊断机会。该Run的完整结果、5000次图像簇区间和`.002`投入门槛以其封存SUMMARY及独立核验为准，中途数量不代表完成或方法收益。root/A已接手实现和常规修复；完整结果/实质阻塞/方向调整才送B判定，不以中间接线互报替代执行。


## 2026-10-09 B：正式机会结果的独立审查，动作级核验待完成

桌面正式机会 Run 于02:07:43实际 exit0，完整512图、57,147候选、9,449有效配对和3,610唯一GT；7张无配对图仍保留。B只读核对58项来源/快照/产物摘要，并从候选重聚合512图、独立重算全部九组固定bootstrap统计，差异均为0。主读数及范围直接引用[封存SUMMARY](runs/RUN_LOCAL_EDIT_OPPORTUNITY_S0_01/SUMMARY.json)（SHA256 `e20b9fa0c8e2cdccf60d4288bdf1f936530aedafa1d6dfbc3c3cb1cee8fc3b36`）：局部机会区间上界低于预登记`.002`，在完整动作级核验通过后应结束本有限配置，不追加动作、样本或RGB训练；这不否定全部局部修正方法。

首次正式独立核验`RUN_LOCAL_EDIT_OPPORTUNITY_VERIFY_S0_01`失败于smooth阈值检查。B全量定位发现：57,147条zero RLE实际面积均等于记录的baseline面积；57,147条smooth阈值全部精确匹配FP32倒数乘法公式，而核验器用直接FP32除法，其中51条差异超过`1e-7`，最大为`1.1920928955078125e-7`。首例`(image_id=10216,detection_index=58)`面积872，记录阈值`.656029462814331`、直接除法`.6560295820236206`。证据指向核验器算术实现差异，没有面积错接证据；A负责匹配计算语义、保留失败并新建核验Run，不能以放宽容差或修改生产结果替代解释。

B本次未重跑模型、GT匹配或GT像素评分，亦未逐一核对全部非零动作RLE、稀疏索引及真实argmax，因此尚不宣布动作级独立核验通过。上述汇总复算是只读审查，不登记成另一次模型执行；完整验证和最终路线判定随后仍在本Study原记录补齐。

## 2026-10-09 A：512 图机会完成，当前有限配置停止

[`RUN_LOCAL_EDIT_OPPORTUNITY_S0_01`](runs/RUN_LOCAL_EDIT_OPPORTUNITY_S0_01/run.json) 于02:07:43（Asia/Shanghai）实际 exit0、产物完整；[`RUN_LOCAL_EDIT_OPPORTUNITY_VERIFY_S0_02`](runs/RUN_LOCAL_EDIT_OPPORTUNITY_VERIFY_S0_02/VERIFICATION.json) 于02:20:33独立 exit0/passed。全部512图、57,147条正常native输出和1,122,315个有限动作完成；持续native零动作断言覆盖全部57,147候选，候选/源码/面板身份、原图RLE、稀疏动作、像素账、有限组最佳动作及5000次图像簇bootstrap重聚合一致。G_old含6动作，G_eq含7动作，加入最多16个局部单区域后每候选最多23动作，零动作只保留一次。

主分母为9,449条有效固定GT配对detection：原成功4,038、原失败5,411；7,161条局部零增量也保留。47,698条未配对输出的质量和修复标签保持未知（无同类普通GT16,976，最高同类BoxIoU不足.5为30,722），配对但GT不可标记为0。505/512图有有效配对，其余7图仍进入bootstrap，计数为0、质量未知；670个原图空动作保留。该口径是normal top-k/conf后固定输出诊断，不是完整raw、COCOeval一对一或AP。

| 预声明有限机会（IoU为0–1） | 有效配对候选均值 | 95%配对图像簇区间 |
|---|---:|---:|
| best(G_eq+L)−best(G_eq) | 0.0010857885 | [0.0009855769, 0.0011934040] |
| best(G_eq)−best(G_old) | 0.0097536632 | [0.0084198788, 0.0114277195] |

全部512图按seed20261009重采样5000次，主比率的有效抽样为5000。主区间上界也低于人为投入门槛`.002`，判定为`stop_current_configuration`：结束本配置，不加图、扩大边带/动作或训练RGB以追阳性。局部有小幅有限GT机会，但未达到投入条件；新增全局扩张方向的机会不能归因于局部性。该结论不预测AP，也不排除其它局部方法。

三组GT最佳动作的平均IoU分别为.69011927/.69987294/.70095873。原失败5,411中oracle修复为446/552/573条，局部相对G_eq额外修复21条（21/5,411=.00388098，95%区间[.00241139,.00553374]）。原成功上的oracle零损伤由零动作和GT取最好构造保证，不证明部署安全；实际局部删/补动作损伤计数分别为165/30,923和196/30,755个原成功**动作**，不能合并成唯一detection损伤数。

独立补表[`GT_GROUPS.jsonl`](runs/RUN_LOCAL_EDIT_OPPORTUNITY_VERIFY_S0_02/GT_GROUPS.jsonl)保留3,610个唯一配对GT及重复detection：局部正增量any/all为1,437/368个GT，局部额外修复any/all为21/1；它们不替代9,449条detection的主比率。逐动作原图改动及稀疏产物保留；删除/添加截断分量总数为1,026,972/1,039,400，截断像素为2,590,645/2,581,084。省略分量未重新生成，有限候选不是方法能力上界。

实测F总2,235.681秒（37.261分钟），模型forward16.911秒、冻结提取24.086秒、动作生成963.923秒、逐动作导出240.492秒、输入检查102.413秒、GT关联/像素账68.073秒、RLE编码312.405秒、产物写入93.605秒；包含项不全部相加。CUDA allocated/reserved峰值1,362,692,608/3,460,300,800bytes，CPU峰值4,768,006,144bytes。独立全量CPU核验另用417.824秒；没有重跑模型、GT解码或匹配，也没有执行GPU算术。GT对应正确性沿实际F的解码/固定关联，核验只独立核存储像素账与RLE等产物。

正式机会首次核验[`VERIFY_S0_01`](runs/RUN_LOCAL_EDIT_OPPORTUNITY_VERIFY_S0_01/FAILURE.json)因checker把CUDA的scalar reciprocal/mul序列写成NumPy直接除法失败，失败Run及原40c626源码快照保留。仅checker改为按锁定Torch2.9.1 CUDA/`Tensor.__rdiv__`顺序逐步FP32模拟，随后全部57,147条smooth阈值与存储值精确相等；没有扩大容差、改变F动作或读取收益后修改门槛。相关运算语义见[PyTorch2.9.1 CUDA除法源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)及本机`torch/_tensor.py`，后续核验为独立新Run。

来源：F SUMMARY SHA256 `e20b9fa0c8e2cdccf60d4288bdf1f936530aedafa1d6dfbc3c3cb1cee8fc3b36`；最终verifier源码 `98a70db5e8a6ad4601d807cb5a5c2572650a5cfa0bf4b87be7e512e838c63e1a`，VERIFICATION `233bb72a8e4d65c7734aef28de001fd60965d2f2eff2aadaa35118693bc5f333`。F推理源、PANEL及协议锁保持不变；完整核验前后1,078个文件SHA一致。当前仅结束这一预声明有限配置，创新、RGB可学性、独立风险校准、充分20k训练及正常任务收益仍未测，交B作下一次新方法方向判断。

## 2026-10-09 B最终验收与取舍

B接受`VERIFY_S0_02`覆盖范围内的独立核验，正式结束本有限配置。再次只读检查6个登记产物、源码/快照和前后锁记录，共1,090个唯一文件摘要一致；核验SUMMARY是VERIFICATION的准确投影，与B先前候选/图像及bootstrap独立复算一致。smooth修复采用限定环境的FP32倒数乘法，并改为逐值精确相等，未放宽容差或改变生产动作；两个历史失败核验仍保留。没有重新前向、解码GT、重做配对或生成被cap掉的分量，native/support等断言仍来自原执行回执，不能扩大独立验证声明。

投入决定为不追加本配置的样本、动作或RGB训练；局部小幅机会与新增全局扩张机会分别解释，GT最佳零损伤不作部署安全证据。接下来完成独立七臂和既定多角度读出，从真实收益代价中选择下一方法问题；[研究线原计划](../../paper/manuscripts/mask_boundary_calibration/MECHANISM_EXPLORATION_PLAN_20260920.md)已补齐旧内容修正和RGB细化强对照的边界，尚未成立新方法创新或启动新训练。Study结束不等于研究线或论文目标完成。
