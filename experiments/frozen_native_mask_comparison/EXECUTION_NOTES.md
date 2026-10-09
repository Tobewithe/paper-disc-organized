# 执行补记

`RUN_FROZEN_NATIVE_ENGINEERING_S0_01` 实际完成 64 图，runner exit0；原生/tau0/缓存及七臂身份检查通过，但 uint8 的 `any` 结果在决策 JSON 写成 0/1，未满足独立评分器要求的 bool 接口。警告不是自行崩溃的证据。

执行者为防止旧格式继续被消费，按实际 CIM commandline 与本 Study 路径确认后，主动终止旧 pipeline child13024 和已开始的 full child11092；两外层 runner 观察到 `failed/4294967295`（2026-10-09 00:55:50，Asia/Shanghai）。这解释停止意图和观察终态，不把退出码或 uint8 警告当作模型、数值或科研失败的根因。中断 Run 和部分产物保留；`PIPELINE_STATUS.json` 的 running 是被终止前最后一次写入，原 `run.json` 和该文件都未人工改写。进程选择/主动停止事实由实际执行子任务报告，runner 终态为已回传原记录。

修正仅将 `.any(...).bool()` 明确为布尔结果，未改动作、阈值、权重或图片。adapter 从 `16deaf262486a69a21c0e6263b1f4f8495e3e9cb8e00d0c5740efecdd0440d83` 变为 `0a11d7cee6f73ec680ce7895f9501a2f87be5d97a8904d3fe6b3a62a958fa0c2`；独立 `_02` 锁源重试，旧版本快照保留。

`RUN_FROZEN_NATIVE_ENGINEERING_S0_02` 实际 exit0，64 图/7,675 候选，五项 trial_nonempty 均为真实 bool，所有 native/tau0/参考缓存、七臂身份、empty/first64 检查通过。总耗时 100.6947 秒，其中 GPU 模型 2.7679 秒、同步 extract 4.5314 秒、参考 native decode 1.9490 秒、共享动作/特征/门控/RLE 77.1770 秒、落盘 4.4641 秒、assemble 1.5956 秒；计时存在包含关系，不能全部相加。峰值 CUDA allocated/reserved 为 1,712,268,288/3,617,587,200 bytes，CPU 为 1,503,084,544 bytes。

工程硬门通过后，`RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02` 于 2026-10-09 00:59:50.198 自动开始，实际进程 PID20648。其终态、正式指标及后续独立评分/核验以各 Run 原记录为准；中途数量不代表完整 5,000 图结果。工程外推约 2.19 小时只为成本估计，不是完成时间承诺。

首次最小回传包含六个 Run 的 135 个文件，逐文件清单已核验；transfer 标记 partial，逐图七臂 RLE 当时仍远端。最小回执核验不冒称本地重新计算完整预测。B 的独立静态/回执审查亦保留这个边界。

正式推理 `_02` 于 2026-10-09 02:47:02 实际 completed/exit0，5,000 图、555,445 候选及七臂身份、原生零动作、参考 RLE、空候选与 first64 检查通过。完整科学消费包 35,047 文件（5,366,495,041 bytes）已回传逐文件核验；ZIP SHA256 `1e9aa3efe42b467d6eca03bf221f2ed299ccbd587459169bb6e37867de4b9158`。旧 partial 回执及清单保留在对应 Run 的 `transfer_history/`；原始终态记录未改写。

桌面 `RUN_FROZEN_NATIVE_5K_SCORING_S0_01` 于 03:05:16 completed/exit0、产物完整。独立 `RUN_FROZEN_NATIVE_5K_VERIFY_S0_01` 于 03:07:56 failed/exit1：生产者对零候选图直接导出空 Counter，核验器访问不存在的 `candidate_count` 键。实际只有 image_id 267946、268996、374727、560371 四行如此，七臂原预测均为空；该异常不涉及已完成的 COCOeval 数值。

为保持已封存 Score 源码及产物原字节，新增消费者 `scripts/verify_comparison_readout.py`（SHA256 `582b5de6dc8fe0f53eecc7f3676e9be185b6b7085d1bfc84e0e8da4947c18279`），从 Score 归档加载原 scorer/helper。它仅在整数零候选、七臂均为严格空计数字典、身份不变量通过且当前原推理逐图 SHA 与 Score before/after 锁一致并确认 `detections=[]` 时，在内存补两个零键；缺臂、部分计数及非零缺键仍失败。失败 Run 与来源保留，新 `RUN_FROZEN_NATIVE_5K_VERIFY_S0_02` 独立重试；只读审查支持该最小接口修复，不替代实际指标核验。

2026-10-09 B审查桌面多角度读出工程：`RUN_FROZEN_MULTIVIEW_ENGINEERING_S0_01`实际exit0，28项来源/快照/回执摘要通过。合成检查覆盖像素公式、空预测purity为null、成对有效分母、重复GT、候选与图像宏平均区别，以及segmentation-only输入和普通/Boundary评价器；`ENGINEERING_COMPLETE.json`摘要为`6abfaeb14fb6cd4bc39d70282c56955c5675666f06cf0b9db2ca848e2f88cdac`。当前正式入口先要求推理与原scorer终态完整、固定final8配对和候选身份一致，再核当前baseline的12项普通AP/AR，方可继续七臂Boundary。B未见阻断正式读出的语义问题，接受工程范围通过；该工程未跑真实5000图端到端，正式身份、AP parity、质量及成本仍待实际Run，不以合成通过替代。审查源码为`multiview_readout.py` SHA256 `1076fc8da0de20b20d84b6aee2d871b17e69a39a7cebe41264e00f47f8242172`、`check_multiview_readout.py` SHA256 `fc3437a4084300ecea51bde7fb21a0810c088a3ec9547f2faac17f358a38f773`。

## 2026-10-09 B：完整评分的暂定审查，独立核验待恢复

SCORING_S0_01已完整结束，B实际流式核对133个关键文件（约2.40GB），包括7份整合预测、79份评分封存产物、原final8基线/固定配对源与标注，摘要均一致；35047项manifest位于Run内且存在，其中35000逐图条目与score源锁一致，未再逐份散列这些重复逐图文件。当前baseline整份预测SHA与final8相同，5000图清单一致，12项segm指标最大差1.11e-16。B未重算COCOeval、模型或GT关联，逐候选原生/首64语义依赖已封存执行记录。评分SUMMARY SHA256为`313794cb3944ba329848cc513b10bd1d625f692a4f783d5e0224acc9ef8f62aa`。

VERIFY_S0_01于03:07:56 failed/exit1，`candidate_count`缺键对应4张真实零候选图267946、268996、374727、560371的空Counter；当前未见总候选丢失证据，A负责保留失败、修复契约后新Run核验。暂不封存最终结论。科学解释采用与TriFlow相同的first64作用范围，但不假设实际修改集合相同：当前点估计呈AP、修复、损伤、平均IoU的权衡，没有全面优胜者；full增加修复和损伤，而相对first64的AP改变均不足.002点。配对分母是允许重复GT的86600条检测，不是唯一GT；七臂AP及配对区间未测。既定像素与Boundary结果到齐后，再判断较高AP的代价及下一内容证据问题，不以此给TriFlow或已结束F配置追加补丁。

## 2026-10-09 B：接受普通七臂评分，补充评价仍待完成

VERIFY_S0_02于03:16:34实际completed/exit0，B将普通七臂评分从暂定转为接受。本次只读核48个新产物及关键控制文件，原scorer与评分封存摘要保持不变；另读4张零候选图的28份实际逐图文件，均为`detections=[]`且摘要匹配，仅在消费者内存补零计数，未改写评分产物。VERIFICATION SHA256为`b3811dbbd8099a212add6fd44856887d49bbf3b167e56c40137c73a9bf005a3d`。独立核验重聚合5000图计数、七臂已存AP/AR数组、86600条固定配对及逐图/唯一GT标签，数值一致；其执行回执确认79份评分产物和40041份原输入再次散列通过，无不可用源文件。B不重复上一轮整批散列或COCOeval。

接受范围为本次普通Mask AP/AR、固定配对修复/损伤与来源完整性，允许按相同first64作用范围与既有TriFlow作有限比较。原生前向、COCO匹配和GT mask未重新执行；首64与baseline RLE语义仍引用封存scorer逐候选检查，不能把回执重聚合称为第二次推理。coverage/purity、相同有效分母差值及Boundary尚待正式multiview，AP/配对置信区间未测。当前支持的是预先列明指标间的点估计权衡，不是显著优胜或任何新方法机制成立。

## 2026-10-09 B：联合计时的成本解释边界

锁定七臂adapter对全部native rows共享计算五动作、两组特征及两gate，随后生成七臂；first64通过全量计算后的选择得到。正式SUMMARY的`all_action_decode_seconds`含共享动作、特征、gate、七臂RLE和内部检查，CUDA/CPU峰值也是联合进程峰值，不能平均分摊或当作独立first64延迟。TriFlow原final8的`extra_module_seconds`则含邻居构图、模块、传输和诊断整理，不含后续mask解码/RLE，口径不同。当前只能报告联合流程真实成本及其中局部阶段，不能给逐方法部署成本排名。

B据此将同口径独立成本小面板交A在其原笔记本环境执行，桌面继续multiview。它服务于同first64范围的收益、损伤与增量开销判断；先固定小面板和计时端点，独立冻结方法路径与baseline做parity及重复测量，保留单独Run和各方法重置峰值，不能修改旧科学结果或以测量替代新方法证据。执行协议与真实测量由A在本Study原记录补齐；本段不称测量已完成。

## 2026-10-09 B：接受完整多角度读出

READOUT_S0_01于03:54:44 completed/exit0，独立VERIFY_S0_01于03:59:25 completed/exit0；B接受本轮5000图、七臂的像素和Boundary读出。B复用已核验像素分母，独立重聚合8组已存评价数组的12项AP/AR，差值全部为零；另实查56个读出关键文件和22个新增/控制文件摘要，均通过。验证执行记录对35142份输入作前后散列，覆盖原源锁与封存产物，并逐候选重建固定关联、像素算术、空值、同分母差值及候选/图像宏平均；这不包含重新解码GT/预测RLE、重新计数真实像素、COCO匹配、构造边界或模型前向。读出SUMMARY SHA256为`3dffb22124ee1e6cade098c51081424fbfddc33eda3b6f629911fca880352c4f`，独立VERIFICATION SHA256为`bd1d3fa24045f8b8662ff80e80f9b6895a83bf0b9d45bd1776928647737d52ee`，事实与完整分母沿原Run保存。

当前结果支持评价维度间的真实权衡：RCMC_first64候选宏平均coverage下降1.0343个百分点、purity提高1.1140个百分点，同时固定定义Boundary AP提高0.41133点；不能从coverage下降推出Boundary AP下降，也不能把更高普通AP等同更小误伤或更大平均IoU。该臂purity的成对有效分母86596与baseline一致，图像宏平均覆盖4946个有效图并将其余54图记缺失；其他出现更多空mask的臂须使用各自成对有效分母，不能直接相减独立均值。当前全部差异为指定口径的点估计，无AP/Boundary置信区间，也没有本轮TriFlow Boundary值或因果机制结论。接受多角度评价不改变TriFlow和F既有取舍，不据此追加阈值扫描或声称RGB新信息已成立；独立成本与桌面环境验证仍按各自Run解释。
## 2026-10-09 B：接受笔记本独立成本面板

COST_ENGINEERING_S0_01与COST_PANEL_S0_01均completed/exit0。B实际核验两份回传manifest的81+767文件及ZIP摘要，并逐份核对工程20次、正式480次输出与冻结参考的完整身份/RLE；32图×3重复×5臂无缺漏，各臂独立PID、输入与资产绑定、必要路径隔离成立。逐次时延、配对有符号增量和峰值重聚合与SUMMARY一致；负增量保留。正式SUMMARY SHA256为`d280e83eafbfc25bb69fb2ab8aebf2c82d6951786a02f9b34ee4806efb757206`，COST_COMPLETE为`a58699ca0f23a40622aa3094b5e579f8ff21aa62be5c126624e381de6784543c`。实际为RTX4060 Laptop、Torch2.5.1/CUDA12.4、FP32、TF32关闭、4线程；使用脚本验证的vendor8.4.100，不用环境已安装包清单中的8.4.27替代实际导入来源。

共同端点从预读RGB到全部原图binary masks导出CPU；baseline均值59.03ms/图，global收缩、RCMC、multi_local、TriFlow各自增量均值13.56、18.33、289.04、678.51ms/图。该结果支持将RCMC视为当前较强的质量—成本对照；它仍有已测coverage及误伤代价，不是全面优胜或新方法验收。32图的重复观测不当作96张独立图，循环移位未完全均衡顺序，无置信区间；reserved是热态保留容量，CPU峰值含初始化/预热/审计，均不称最低部署显存。结果不外推full范围、桌面环境或生产最优实现，且不混入先前七臂联合计时。桌面跨环境失败及本机参考协议另行保留，不用笔记本通过替桌面通过。
## 2026-10-09 B：桌面独立参考与工程通过的范围

B已核对DESKTOP_DIFFERENCE_S0_01、DESKTOP_REFERENCE32_S0_01和DESKTOP_LOCAL_ENGINEERING_S0_01的完成回执、来源及新输出；旧desktop工程01/02仍保留failed。首图139重算发现296条记录、686项差异，仅为框坐标和置信度相关字段，输入FP32字节及300份RLE一致；最大原始坐标差0.00036621、原始置信度差2.0713e-6。这只说明首图，不能外推32图跨设备完全等价，也不把误差忽略后改称旧gate通过。

本机32图参考由原NativeMaskAdapter.decode_arms与原TriFlow evaluate_epoch_r3路径生成，未调用待测隔离executor；两个原路径共享实际提取结果后严格比较共同baseline，覆盖32条执行回执，第二份baseline未另存供离线重比。3813个native候选、8臂256份参考存在，身份/顺序一致，四个受限臂的8804条first64外记录与baseline相同。工程前4图×5臂20份输出逐份与本机参考完整identity/RLE复核通过，输入和所需资产/工作量隔离成立。REFERENCE32 SUMMARY SHA256为`c1b3bcfcc0a1a64694e6f81d4e90ae3fd5d778faef8070b3cfa2f0186d726482`，工程SUMMARY为`5f01979c9f1c88badcc5c7463455e30f4df992eeae25753886fe2d4dfe3b539d`。B未见阻断正式桌面成本的语义问题，允许沿已有授权继续；当前不称正式成本已测得。

32图参考没有空mask，4图工程没有unsupported ROI选中候选，相关一般性保障来自原代码而非本面板实测。参考Run未提供覆盖全部256输出的整体摘要清单，工程已绑定实际20份参考，正式结束依逐次参考摘要收齐确认封存范围。工程SUMMARY继承的“registered laptop environment”和旧final8参考措辞不准确，实际COST_INPUTS/子进程为Torch2.9.1+cu128、RTX5060Ti与本机参考；正式报告应更正表述，不改旧封存或因此重跑。桌面成本与笔记本质量/成本保持各自来源范围。

## 2026-10-09 B：接受桌面独立成本面板

DESKTOP_LOCAL_PANEL_S0_01于04:16:06–04:18:39执行完成，独立DESKTOP_LOCAL_PANEL_VERIFY_S0_01亦completed/exit0。B只读复核939份锁定文件共145677082字节，摘要全部一致；32图×3重复×5臂的480份输出精确覆盖预定集合，15个独立进程，五臂所需160份本机参考各使用三次，57195条检测的身份/RLE与参考一致。逐图同repeat baseline配对的有符号增量、均值、中位数、标准差及GPU峰值重聚合一致，global的一次负增量保留。正式SUMMARY SHA256为`30638e098fe9eac8da1bc994771c035a1c7afa51c8e278bac69dad1047db1349`，VERIFICATION为`c8cfc86a1666eb8982aca842a2dcf97b19a62775ad335a6df93e202eb04183b4`。此次没有重复原生前向；本机独立原路径参考的来源沿前一节与原Run保存。

接受范围是RTX5060Ti、Torch2.9.1+cu128、conda pytorch下固定32图当前实现的独立成本：预载RGB至全部原图binary masks导出CPU，baseline平均36.732ms/图，global收缩、RCMC、multi_local、TriFlow的配对增量均值分别4.743、11.219、159.529、210.375ms/图。加载、warmup、RLE、审计和写盘另计；CPU峰值是三个独立进程全生命周期最大值，GPU峰值包含常驻资产及热缓存。实际覆盖108条unsupported first64记录，未覆盖native零输出或空mask；三次循环换序未完全均衡位置，无CI，采样不能排除所有背景负载。原SUMMARY继承的laptop限制措辞由独立核验纠正解释，不改封存文件；本机成本不能与笔记本5k AP拼成已经验证的桌面质量—成本点。该接受不改变TriFlow/F取舍，后续桌面工作沿独立SegRefiner Study进行。
