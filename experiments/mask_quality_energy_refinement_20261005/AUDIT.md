# QCR Stage I 实现与协议审计

日期：2026-10-06（Asia/Shanghai）。Study：`STUDY_6f0c44af49e846f0a14f44a4b43d38e4`。

本审计只核对已回传源码、协议、配置和研究记录，不改训练代码、不启动模型、不重训。当前对象是已经训练的实现版本；补充评价可以确定该版本的效用和机制，不能追认它为完全符合先前协议的实现。

## 当前判定与证据等级

1. Stage I 原 YOLO 的参数冻结边界在源码上成立：训练只实例化新增 EvidenceEncoder 和 Direct32/QualityHead；完整 prototype 作为固定数据参与 `c -> Pc -> Q`，Q 推理能够对临时 coefficient 求导。没有发现本次源码解冻 backbone、neck、prototype、box/class、原生 coefficient 或 BN 的证据。
2. 已确认训练 seed、排序标签/损失、Failure 随机半径、实例权重、采样比例与部分输入描述存在偏差。部分偏差相对于研究笔记明确，相对于简写版执行协议则有歧义，下面分别注明。
3. FIT抽样10图中6个P为FP16且这6个均无input_uint8，另4个P为FP32；FIT全量精度/缺输入及入选训练占比仍未知，不把6/10外推为60%。完整DEV遍历已确认2,000图P存储均FP32，FINAL为5,000图流式FP32；W10数值核验通过只支持所测10图/102候选，不能据存储类型或缓存指纹宣称全量FP32数值等价。
4. 当前候选范围的完整DEV和FINAL评价、原图pixel v2、分层统计修订及W10均已完成并回传。FINAL严格目标组D−B为−1.5904pp、95%CI [−1.9249,−1.2650]，六项数值放行条件全部失败；完整结果见REPORT.md。DEV全R_arg严格组仍unknown，FINAL严格组限于固定TAL目标交集；不能把训练loss下降或当前结果当作原设计完整合规复现。
5. **停止当前实现于Stage I，no Stage II。** 不重训、不自动加模块、扫超参、延长epoch、改半径/步长、扩大数据或全模型微调；评价实现修复和统计映射修订没有改变已保存模型的训练事实。

本文件中的“源码已确认”指下列哈希的本地回传副本及对应Run快照；“未知”保持未知。终点本地核验 `assets/remote_receipts/VERIFICATION.json` 记录errors=[]：36个Run目录、433个选定文件、155个源码快照与回传archive哈希均吻合；主DEV/FINAL、原图pixel v2、W10及统计修订产物已本地可读。该核验采用文件/导出字段处理，没有本地加载或执行模型，也没有重新访问远端；它验证选定回传证据的完整性和链接，不等于整目录镜像、全量张量重放或科学协议放行。源码SHA与运行绑定使用各Run snapshots，不将当前脚本追认为所有历史Run的源码。

## 审计来源与 SHA256

所有下列路径相对于 `C:/Dpan/codexproject/paper-disc-organized/`，行号为本次审计副本的 1-based 行号。正文使用简称：`train`、`runtime`、`prepare`、`select`、`config`、`protocol`、`note`、`evaluation-note`。

| 简称 | 路径 | SHA256 |
|---|---|---|
| train | `experiments/mask_quality_energy_refinement_20261005/scripts/qcr_train_fixed.py` | `52836de5af4b5923ef6c6c527e1d3740319261a6efcbd7b6a94a323f6594ebb2` |
| runtime | `experiments/mask_quality_energy_refinement_20261005/scripts/online_runtime.py` | `56e0909c3dd3374c78593d982018739126cd52ecb243db7600ded93ae4ec2dbb` |
| prepare | `experiments/mask_quality_energy_refinement_20261005/scripts/prepare_qcr_cache.py` | `c0f32b020f506f897a45fa98fae053258b293042a0a6026ff41a30eb57888ac4` |
| select | `experiments/mask_quality_energy_refinement_20261005/scripts/select_candidates.py` | `56c032ff5eeabccfde423e709e065089dcf1809721a187efcd0416c917215464` |
| oracle | `experiments/mask_quality_energy_refinement_20261005/scripts/build_oracle_selected.py` | `ca12d5d425e4f6a7b03029b5a780a57b07235368b047d717760ff65f81f9d2fa` |
| config | `experiments/mask_quality_energy_refinement_20261005/RUN_CONFIG.remote.json` | `7c2e561cd6f1b8847d9e7cbe6be4369d49b7d3236b965264040e9484f36b8d62` |
| protocol | `experiments/mask_quality_energy_refinement_20261005/PROTOCOL.remote.md` | `d8cb1deba60349c15afa0cbb6d9b37158cd5b02cd2bfebe0466df984186a27a7` |
| 本地协议 | `experiments/mask_quality_energy_refinement_20261005/PROTOCOL.md` | `d8cb1deba60349c15afa0cbb6d9b37158cd5b02cd2bfebe0466df984186a27a7` |
| note | `project/research_notes/MASK_QUALITY_ENERGY_REFINEMENT_20261005.md` | `ead82b86243d4bb69e06fcf65deb25284d39a03346d506655152fcf8fad59c1a` |
| evaluation-note | `project/research_notes/EVALUATION_SYSTEMS_20261005.md` | `ddbe1c6b56ea7031a9d1d3a2ef3f39005b8fa476a14899518ad47b49b9506f4e` |
| FIT/DEV 划分 | `experiments/mask_quality_energy_refinement_20261005/IMAGE_SPLIT.json` | `a1a3ee0a0e39582dfc13a403faa91ddd63ed5ca5ab16017a0ff84af028765291` |
| FINAL 划分 | `experiments/mask_quality_energy_refinement_20261005/FINAL_SPLIT.json` | `00f8420d2aa16bf8dad421bca425c7d19c3e508f3ebe4836cf73d4380468eff8` |

已只读 `_maintenance/qcr_package.zip` 原执行包（ZIP SHA256 `60262d3d2b2548138ae9e4ca20518cf38ad29661db25211eed62a3d400edc40b`）：其 entry `experiments/mask_quality_energy_refinement_20261005/PROTOCOL.md` 长 8,062 bytes，SHA256 同为 `d8cb1deba60349c15afa0cbb6d9b37158cd5b02cd2bfebe0466df984186a27a7`，与当前远端原文、本地恢复版和 Stage I Run 协议快照一致。没有发现执行包至当前远端的协议文本修改；下面硬排序、margin、r、显式输入等细节的历史承诺来自研究笔记，对简写协议没有定义的部分保留歧义。

`CODE_MANIFEST.sha256` 是旧执行包清单（自身 SHA256 `cf0f254a7af048d57c14cbbcbad800dbf7d141bd006844b3044bb4005f1dc017`）。其中未列 `qcr_train_fixed.py`，其 `online_runtime.py` 与 `prepare_qcr_cache.py` 的旧哈希也不同于本次副本，因此不能用旧清单单独认证当前实际训练版本。旧版 Run 和故障 Run 应原样保留，不能替换其代码谱系。

对应实际运行快照和产物（下表路径相对本实验目录）：

| 用途 | 保存路径或实际绑定 | SHA256 |
|---|---|---|
| Direct32 训练代码 | `runs/RUN_stage1_direct_seed0_retry5/snapshots/c4c75f8c4df2f7e27838ea730bc822613f4512912c74bdb6278d31c718824294` | `c4c75f8c4df2f7e27838ea730bc822613f4512912c74bdb6278d31c718824294` |
| Q 训练代码 | `runs/RUN_stage1_quality_seed0/snapshots/9f39e53b0d9f718660d68a3f8a46a0faf89c1c314574dad0fad1f90d551b778b` | `9f39e53b0d9f718660d68a3f8a46a0faf89c1c314574dad0fad1f90d551b778b` |
| Direct HISTORY | `runs/RUN_stage1_direct_seed0_retry5/HISTORY.json` | `c7806d7c234996a8973b4e54bd4495b9376091fca9727024cbe8bd720d407b8b` |
| Q HISTORY | `runs/RUN_stage1_quality_seed0/HISTORY.json` | `79f225c25a1cdbe02fcadd437f25e1f30c44d762fb07993f1e3b59bd57d72590` |
| Direct final checkpoint | `runs/RUN_stage1_direct_seed0_retry5/final.pt` | `25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8` |
| Q final checkpoint | `runs/RUN_stage1_quality_seed0/final.pt` | `07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065` |
| Direct 简 DEV | `runs/RUN_stage1_direct_dev_seed0_retry1/SUMMARY.json` | `efb8319ae8615ce9b22c8a625260fecc797d980947a8eddb9efb3245c595cc1f` |
| Q 简 DEV | `runs/RUN_stage1_quality_dev_seed0_retry1/SUMMARY.json` | `9f39fbda739ae36e4464b4f64bd9dbb0c985670ab5d179a0b66307083b4272ee` |
| 冻结训练清单 | `assets/CANDIDATE_MANIFEST.json` | `a3729b36604c28c00eec82a1e4cba9d64eb638e1f470939e394b817a6d0e5c0f` |
| 清单计数 | `assets/CANDIDATE_MANIFEST_SUMMARY.json` | `31a4d5afe699483119c2a7008b59d11facb6b23bacc4bf2ff6bdaf69469ad8a1` |
| 有限 oracle 摘要 | `assets/ORACLE_failure.SUMMARY.json` | `1b8677d3d8b90cebb596a211f9e351a14dbc5fee1eca40e4b8b34ca95e110c53` |
| 缓存身份 | `cache/CACHE_IDENTITY.json` | `95e2c260ed127e7f079c93b1a584f30ecb3426121562be4e1b3d43d68f61c50f` |
| 缓存完成回执 | `cache/COMPLETE.json` | `37b7ab3eab9aaca6012fd6db1c89b14a6a3ff61760dce0880907349c2e58cca6` |
| 缓存存储抽样审计 | `runs/RUN_cache_storage_audit_20261006/STORAGE_AUDIT.json` | `590eb6000d3432a08aa41e7b5ab2dd7fec9a97b1a4af92a77ff82b8030feb329` |
| FINAL主统计修订 | `runs/RUN_stage1_final_metrics_repair_seed0/SUMMARY.json` | `3e36aec8320e95726840576f0b6575be38955e0a43d85c3be3d1a4814d039c6c` |
| DEV主统计修订 | `runs/RUN_stage1_dev_metrics_repair_seed0/SUMMARY.json` | `350a525da99f27bddd4da7da67c4e49d395c1e4b05b84e9d20e35764a9e0d364` |
| W10数值witness | `runs/RUN_stage1_dev_numeric_witness_seed0/NUMERIC_WITNESS.json` | `0c6b2d0ae387df4cfdef78af363d3311e98bd27d766f0b032cea3e32d3d05373` |
| 本地回传核验 | `assets/remote_receipts/VERIFICATION.json` | `568681c24a2d2475a646b06fb164c879bd03e36b6b9b562c1edaadf467f5f7fa` |

已比较训练快照和当前 `train`：Q快照到当前仅改原图decode的scale_masks维度和评价记录真实位移，未改训练目标；Direct快照到当前另修随机方向device/dtype，该段不进入Direct训练。正文为便于查阅引用当前 `train` 行号，训练实质偏差也出现在各自历史快照。`ORACLE_failure.pt` 的训练输入哈希由两臂 run.json 一致记录为 `91215967aec2b20cbfc9b6929505382a19938053b0506492769cd215e86f53cf`；未在本地反序列化模型或oracle。

两训练Run的内部run_id分别为 `RUN_4c129c2d79044dc8a3a79aa5df8efbf5`、`RUN_2867e3a6f86f40bca6751a88c3fe473f`，简评价Run分别为 `RUN_30a5b8cdd7d34b149b5d74420012339a`、`RUN_13053770912e485aa0ea6122a549aa32`。目录名和已有ID保留。初次transfer为2026-10-06T18:11:30.861143+08:00；最终 `assets/remote_receipts/TRANSFER_SUMMARY.json` 记录19:10:48.061131+08:00、36目录/433选定文件回传，archive SHA为 `44100362295642cad7f40c9c23583e32193381ed25d32b609b7418961a8f5f80`。final checkpoints、完整主评价导出记录、JSON/logs/snapshots已本地可读；完整FIT/DEV tensor cache与epoch checkpoints仍远端。传输、模型执行、产物和文件可用性分别记录，不声称完整远端目录镜像。

## 已确认偏差及影响

### 1. Run 名称 seed0 与实际训练 seed 不一致

`config:3`、`protocol:162,166` 和 `note:147` 锁定 seed 0。`train:14` 定义 `SEED=20261005`；`train:156` 使用该常量初始化 Python、NumPy、CPU/CUDA 随机数，`train:163` 用 `SEED+epoch` 排图片，`train:142` 也用该常量产生状态方向，未读取 `cfg['seed']`。

因此 `RUN_stage1_*_seed0` 是历史目录名，实际训练随机种子应记录为 **20261005**。不可改名重建 Run，也不可当作 seed 0 复现或独立增加的一个 seed。两训练Run的 `run.json.scope.seed=0` 也是名义记录，不是运行时种子的证据。配置 seed 0 可用于缓存原模型准备，不能据此证明新增头训练用 seed 0。两臂 Run 输入哈希相同，快照训练排序仍使用相同常量；偏差破坏的是预注册 seed 与复现标签，而非证明两臂用了不同数据顺序。

### 2. Q 排序标签来自 soft-IoU，正常原图 hard-IoU 未进入训练

`note:51-54` 明确规定 soft-IoU 回归、正常二值解码 Mask IoU 排序；`protocol:65` 仍要求每状态计算二者。实际 `train:143-147` 只计算 `soft_iou(...)`，将其同时用于 Huber 和 rank；`hard_iou` 定义于 `train:66-70`，本训练路径未调用，正常原图函数 `decode_original_iou` 也未用于训练状态标签。

实际 soft 标签在 640 分辨率、预测框支持内对官方 overlap GT 计算（`train:58-64,103`），评价则用原图独立 COCO annotation mask（`train:72-78`）。因此本模型学习的是框内 soft 质量地形，不能直接声称训练了正常原图 hard 质量地形。二者可能相关，但相关和梯度效用必须由 held-out 评价证明。补 hard 标签评价可以量化泛化差距，不能改变已有监督目标。

### 3. rank 是幅差乘积形式、margin 0.05，不是 sign ranking、margin 0.02

`note:65-68` 写明，对真实 `q_a>q_b` 使用 `max(0,0.02-(Q_a-Q_b))`。`protocol:72` 只简写“同实例状态排序”，未明确宣告替换 margin 或权重形式。实际 `train:150-152` 为：

```text
diff = soft_a - soft_b
valid = abs(diff) > 0.01
L_pair = relu(0.05 - diff * (Q_a-Q_b))
L_Q = Huber(delta=0.1) + 0.5 * mean_instance(L_pair)
```

这是 **按真实 soft 幅差缩放的乘积约束**，不只是换一个 margin。若 `diff>0`，零损失要求 `Q_a-Q_b >= 0.05/diff`：例如真实差 0.02 要求预测差至少 2.5，真实差 0.10 要求至少 0.5。激活时梯度又被 `diff` 缩放；小差状态被要求更大的分数间隔，却承受更小 rank 梯度。`|diff|<=.01` 的状态对全部被排除。它可能改变标量校准和局部梯度，而不能称为既定的 sign margin=.02 排序损失。

Huber delta=.1、rank 总权重=.5 与当前配置一致（`config:30-31`）；实质偏差在 rank 内部。补充 Spearman/pairwise/真实增量一致率可检测这种已训练地形的效果，不能补救训练损失本身的更改。

### 4. Failure 随机状态使用全局 rho，不是实例 oracle 位移半径

`note:49` 明确 `r=||d*||`，极小位移不放大。`protocol:62` 保留 `r u1,r u2`，正文未另定义 r。实际 `train:89-93` 的 Failure 随机状态为 `c0+rho*u1,u2`；`rho` 是所有 Failure oracle 位移中位数（`train:157`）。

这使小 oracle 位移实例的随机控制可能被放大，大位移实例的随机控制可能缩小；随机控制不再与该实例 oracle 方向幅度匹配。它改变质量头的训练状态分布和“是否只是修正幅度越大越好”的控制含义。Success 使用 8 个固定小扰动，最大随机幅度 `.25*rho`，不求 oracle，符合最新协议允许 Success 小扰动的方向。不能从早期“每候选均 oracle”段落推导出 Success 必须 oracle，因为最新 `protocol:153,163` 已明确豁免。

### 5. 输入缺显式类别/分数、尺度 embedding 和 c-c0；完整 F 也未直接输入

`note:26-30` 要求实例特征、归一化框/尺度 embedding、c 与 c-c0、8×8 Pc 响应、类别/目标分数。实际 EvidenceEncoder 输入仅 `h[64], box/640[4], c0[32], summary(Pc0)[68]`（`train:17-25,44-54,103`）；QualityHead 的状态输入为 `c[32], summary(Pc)[68]`（`train:34-37,47`）。summary 包含全 prototype 网格 8×8 平均池化及均值/标准差/max/min。

预测类别/分数、pyramid_level/尺度描述虽存于缓存（`prepare:294-306`），本头未读取；也没有显式 `c-c0`。c0 被编码在 e 中，因此可能被网络间接利用，但不能称为显式位移输入。归一化 box 已包含几何宽高信息，不能把“无尺度 embedding”写成“完全没有尺度信息”。实际使用候选 coefficient 分支隐藏向量 h，而不是完整 neck feature map F；相对 `protocol:52` 的 `e,F,B,c,Pc`，若 F 是完整特征输入，则实现简化了该承诺；若 F 只是 h 的简称，则文本有歧义，不能武断判定完全违约。

两臂共同省略这些输入，所以 B−D 仍是该共同输入下的比较；该结果不能判定包含原设计全部描述的 Q 模型一定无效。已有训练不能通过在评价时补字段修复，补字段会改变模型架构和训练任务，本轮不自动进行。

### 6. 每图内部实例等权，跨图是图片等权；不等于所有实例全局等权

`note:71` 要求训练 batch 内实例权重相等。实际 `train:137` 将一图中 Direct32 实例损失平均；Q 的 Huber 对该图全部 8-state 实例平均，rank 再按该图实例平均（`train:147-152`）；`train:166-170` 每图的 loss 除以 8 累积。对一个完整 8 图 update，其形式是：

```text
L_update = (1/8) * sum_image [(1/n_image) * sum_candidate L_candidate]
```

故同一图内实例等权、不同图等权，但一个含 1 个入选实例的图中，该实例权重可大于含 10 个实例图中每个实例的 10 倍。没有使用像素和/大框面积对质量头直接加权，但存在隐式的每图 `1/n_image` 权重。是否将 `note:71` 解读为仅一图 batch 内等权或整个累积 batch 全局等权，需要保留歧义；不能报告为所有 FIT 实例统一权重。

末尾不足 8 图时仍除以 8（`train:168,172-173`），没有按尾批实际图数重归一化。若每 epoch 12,635 图，则尾批为 3 图，仅有 3/8 的正常 update 归一化幅度。两臂在同数据顺序下共享该处理；评价统计可以按实例/图片两种口径揭示范围，不能改变已有优化权重。

### 7. Failure/Success 不是 1:1；Success 按身份排序截断

`protocol:151` 和 `note:147` 要求按实例近似 1:1。已回传清单计数为 Failure pool/selected 20,257、Success pool 91,392/selected 30,000，总入选50,257；Success/Failure=1.481，Failure 占 40.31%、Success 占 59.69%。`config:18-21` 为两池各自上限 30,000、各自最低 20,000，没有平衡到较小池；`select:39-41` 按 `(image_id,annotation_id,raw_id,pyramid_level)` 排序后各取前 max 条，而不是随机抽取等量。

因此训练保护 Success 的暴露比例高于 1:1 设想，且被截断的 Success 在身份排序中靠后，其图像分布不一定代表全部 Success。图片等权又使这两个全局计数不能直接换算为实际梯度比例。只满足各池最低数量不等于实现近似 1:1；“近似”的容差未事先定义，不能事后宣告 1.481:1 已通过平衡约束。两臂共享已冻结清单保留了本轮内部公平性；补原失败/成功分层只能描述效用，不能重新平衡已完成训练。

Failure/Success 的入池 Mask IoU 是 640 输入空间 `process_mask` 对 overlap GT（`select:28,33-37`），不是 `annToMask` 原图 IoU。阈值附近可与正常原图状态不同；清单应保留原身份和原分类，原图评价另报告交叉表，不能事后重标后称训练用的是新标签。

### 8. 3 epochs 是 selected-FIT 图片遍历，并非全部 20,000 FIT 图优化

划分文件实际包含 20,000 FIT、2,000 DEV，图片重叠字段为 0。已核对cache/COMPLETE与清单：缓存19,891有效FIT图、144,431固定one-to-one TAL candidates；入选训练只有12,635图、50,257 candidate。`train:106-114,159-166` 只遍历清单中有入选实例的图片，所以“3 个完整 FIT epoch”（`protocol:162`）实际是 **3 次 selected-FIT epoch**。缓存回执把109图同时列入no_positive/skipped_fit；prepare:318-320也可按无argmax-correct Box75跳过FIT图，因此不能仅据该字段把109图一概解读为原图无GT或无官方TAL正样本。

两臂成功Run HISTORY:3-33均已确认每epoch12,635图、50,257 candidate、ceil(12635/8)=1,580 optimizer steps；3epoch总4,740steps、150,771candidate exposures，候选数覆盖完整清单。Q每实例固定8状态，故402,056state labels/epoch，Direct32每实例只优化当前预测；相同image epochs/optimizer steps不等于相同状态前向数、FLOPs或墙钟预算。Direct训练完成时间13:40:49→14:08:24，Q为15:54:13→16:33:12（2026-10-06,+08:00，run.json）；不能因两臂训练loss都下降就推断效用匹配。

配置同时保留 `microbatch_images=2,effective_batch_images=16,batch_candidates=64`（`config:22-25`），训练循环均未读取；实际是每次一图、8 图累积（`config:41`）。优化器、lr/weight decay、clip、无 scheduler、固定 epoch3 则与最新锁定一致。不能将旧配置的 effective batch16 写成实际 batch。

### 9. 两臂同 encoder 架构，分别训练参数；参数规模不同

`train:39-43,158` 每臂独立实例化 QCR 及其 encoder。它们使用同 EvidenceEncoder 架构和同输入，但不是跨 Run 绑定、共用同一组已训练 encoder 权重。最新协议“共享 encoder”（`protocol:38,164`）若指架构则满足；若指字面权重共享，则未实现。分别端到端训练 encoder 属于当前实现，不能用“同 encoder”模糊该差别。

从Linear层尺寸静态计算（无需本地实例化模型），EvidenceEncoder=38,144参数；Direct32总=58,784；QualityHead总=84,097，D/B=1.431。已与两Run MODEL.json:2-5核对一致。它们属于相近量级，不是相等参数量。D−B同时包含头的结构、状态表示、参数量和不同学习目标，不能自动归因为梯度优化这一单因素的严格等参数因果效应。

## 完整 prototype、冻结范围与缓存数值边界

`prepare:274-307` 在原模型 eval/no_grad 下取官方 one-to-one TAL 候选，保留每图完整 P、c0、boxes、h0、overlap GT 和永久身份；`runtime:157-175` 检查 Ultralytics 8.4.100、官方权重 SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，原模型 FP32/eval/requires_grad=False，`runtime:184-191` 检查 BN 和 buffers 不变。

`train:56,124,136,143,180` 使用完整 32×160×160 P 做 einsum，而非 ROI/Gram 代替 P。质量头只读取 Pc 的 summary，但 Pc 本身从完整 P 生成。`train:178-182` 仅将临时 c 设 `requires_grad=True`，`summary` 和 `z_from` 未 detach Pc；能沿 c→Pc→Q 求梯度。缓存 P 在准备时 detach、在训练中无参数梯度是 **冻结 P 的合理实现**，不会切断对 c 的梯度；不得将它误判为需要解冻 prototype 才可工作。detach 质量标签（`train:144`）也不会阻断推理时 c 的链路。

但当前 `runtime:117-121` 允许 P 以 FP16 存储，再 `.float()`。转回 FP32只改变计算 dtype，不能恢复量化丢失的有效位。`runtime:112-116` 仅在 input_uint8 存在时验证原输入字节；`runtime:135-137` 标记 compact_cache 和存储 dtype。这里 `original_server_forward=True` 是程序赋值，不是重放证明。读取器验证压缩文件哈希、身份、full overlap mask，却未重新比对 `meta['proto_sha256']`。

准备脚本的声明是“完整 P/GT/uint8 lossless”（`prepare:4`）、“original FP32 eval batch1/noTF32”（`prepare:207-211`）。重试允许保留旧 fingerprint，但会写入当前 signature（`prepare:214-221`），只严格比 weights/split/annotations；它不能证明此前缓存压缩保持原张量数值。准备时的 LOSS_EQUIVALENCE 只审计 FIT/DEV 各首个候选图（`prepare:308-310`），验证的是官方 mask BCE 实现和梯度，不是压缩后全缓存、更不是 Q rank 目标。

已回传 `runs/RUN_prepare_qcr_seed0_retry3/compact.log:1-3` 记录 compaction计数8,172/15,047/21,919（SHA `cf1d4eb49a72d793c854d61f4bb0af325218f93a6c701b50c885078249b2582f`）；这是发生过缓存compaction的证据，但日志未逐文件列dtype或唯一身份，21,919也大于最终21,891个缓存，不据此换算全量FP16占比。`runs/RUN_repair_cache_metadata_seed0/REPAIR.json:4-9`确认对21,891文件重新hash、12,478个压缩文件哈希变更、无错误（SHA `b7a7cfce5fbd191c987e949f85cc8fad3bb676f49ce8203fa5e3af0d03aad948`）。修复快照 `snapshots/02be8ec7f1e7eb7952783d7edb0cd2fdd534a98bab250f6383db0bf38b12136e:50-59`仅将当前压缩文件digest覆盖元数据，并未比较P或恢复输入/原数值。修复在2026-10-06 01:56开始，早于两训练Run。

2026-10-06新增的 `runs/RUN_cache_storage_audit_20261006/STORAGE_AUDIT.json:2-3,5-244` 按固定FIT/DEV index中十进制image_id的SHA256最小值，各取10图，并非按结果选择。该审计没有实例化模型、没有修改缓存（:246-247）。已确认抽样结果：

| 固定抽样范围 | P存储FP16 / FP32 | input_uint8缺失 | h0 / c0存储 | 当前archive哈希与meta一致 |
|---|---:|---:|---|---:|
| FIT，10图 | 6 / 4 | 6；均为这6个FP16图 | 各10个均FP32 | 10 / 10 |
| DEV，10图 | 0 / 10 | 0 | 各10个均FP32 | 10 / 10 |

因此“FIT缓存存在FP16 P和缺输入”已由实际文件核实，是相对原FP32 lossless声明的已确认存储偏差。FIT抽样不是全量普查，不将6/10外推为全FIT的60%。20个archive与meta哈希吻合仅证明当前文件完整性。DEV全量P存储类型随后由完整评价逐图记录确认：主DEV修订SUMMARY.metadata.prototype_storage_images为 `{'torch.float32':2000}`；该结论来自2,000图遍历，非10图抽样外推。FINAL同字段为5,000图FP32。存储类型确认不等于全量P/h/c0/boxes或方法输出的数值重放证明。

以下事实仍未知：FIT全量FP16/缺input文件数、入选训练实例的精度占比、各图具体转换内容/发生时间及FIT量化对方法效用的独立影响。DEV全量P为FP32已知，不能再写其全量存储dtype未知；FIT全量和FIT/DEV全量数值等价仍不由该字段确定。两臂同缓存保留当前版本的内部比较，但不能宣称原FP32 lossless训练表示。重新hash不能恢复FP16量化丢失的原有效位；存储抽样与数值witness均不修复已完成训练。

固定DEV清单前10图的FP32重算与W10数值witness已完成，覆盖102候选（与SHA抽样storage audit的10图选择不同）。`RUN_stage1_dev_numeric_witness_seed0/NUMERIC_WITNESS.json`记录六身份字段配对完整，P/h/c0/B/target在atol=rtol=3e−5下全部allclose，A原图二值mask逐像素完全相同。`VERIFICATION.json.W10_local_evidence`又确认原缓存与 `RUN_stage1_dev_fp32_witness_seed0` 在102行的A/B/D1/D IoU、Mask75、B/D1/D位移范数、Q0/Q1/Q2及held-out Q值逐项exact。该结论只支持所测10图/102候选；不认证全FIT/DEV数值等价，不追认为原FP32 lossless训练，不补救FIT量化或监督偏差。FINAL官方FP32流式评价已完成，训练科学偏差维持锁定，不重训，不放行Stage II。

## A/B/D 评价范围及不能混用的 strict MaskFail

A 是同一冻结候选的 c0；B 为该 Direct32 checkpoint；D 为该 Q checkpoint 的两步更新。框、类分数、候选身份保持固定。`train:179-182` 的两步 rho/2、rho/4 和半径约束符合 `protocol:74-84`，原ZIP和两训练Run启动时已保存此协议SHA。早期 `note:81-88` 则是一步主结果、两步次要诊断；这是研究笔记至执行包的版本变化，当前证据支持两步已在Stage I运行前登记，不能判为DEV事后调参。一步仍是放行条件第2条所需的独立诊断，不能用两步结果替代。

评价对象是官方 **R_TAL one-to-one 正样本**，准备代码按官方 GT 分配保留每GT至多一个 owner（`prepare:276-281`）。它是固定训练监督候选机制诊断，不等于完整 raw 池 R_all/R_arg，也不等于无 GT的正常推理候选集合 R_out。GT没有进入 Q 的特征或推理步长，但使用 GT-TAL 选定评价对象，因此不能称为官方完整 COCO AP、正常最终输出部署效果或全 raw能力。

`train:202-203` 的 `strict_maskfail_candidates` 只筛 `BoxIoU>=.75 && iou_A<.75`，遗漏类别 argmax=GT；也没有计算该子组 macro/CI。`train:119` 取评价记录时未保留 predicted_class_id，`train:195` 输出又丢 branch/target_gt_idx/area/class；这些字段可从原缓存按完整永久身份恢复，评价必须补齐，并核对两臂A基线和候选集合完全一致。

补齐类别后DEV得到固定TAL候选中argmax-correct+Box75+原Mask<.75子组2,550行，但其完整R_arg判定保持unknown；空strict导出组不证明没有严格失败。`evaluation-note:84,310`的strict定义为有argmax-correct Box75、整个R_arg没有任何Mask75，单一TAL候选失败不能排除其他raw成功。FINAL流式评价已经补R_arg判定，其严格终点为这一完整raw条件与固定TAL class-correct Box75 baseline-fail的交集，5,304行/2,228图；不覆盖未分配GT或所有raw位置。旧val5k的6,459组还使用不同输入几何，不能直接替换或拼接。两套集合与分母在REPORT.md及主SUMMARY中明确保留。

训练筛选的640 overlap MaskFail、DEV/FINAL原图TAL MaskFail、全raw argmax strict MaskFail是三个不同口径，分别记录。size与P3/P4/P5必须来自固定GT面积与raw身份，不把pyramid level等同对象大小。

## 历史简 DEV 结果（已复核回传，主终点见后文）

两臂均为2,000图、13,690候选；下面增量以原图IoU的0–1数值乘100换成pp。原summary所谓strict数量2,670漏类别条件，不采纳为严格组数。

| 保存的独立评价 Run | 完成时间（+08:00） | 全体图片macro ΔIoU | candidate mean ΔIoU | repair / damage / net |
|---|---|---:|---:|---:|
| `RUN_stage1_direct_dev_seed0_retry1` | 2026-10-06 16:56:59 | −0.556601 pp | −1.022799 pp | 164 / 557 / −393 |
| `RUN_stage1_quality_dev_seed0_retry1` | 2026-10-06 17:04:04 | −2.722706 pp | −3.502959 pp | 267 / 1,407 / −1,140 |

同范围摘要差D−B=−2.166105pp（全体图片macro），尚非严格MaskFail配对差及其CI。当前明显风险是成功实例损伤；不能用较多repair掩盖更大的damage。训练Direct32 loss0.282294→0.277785、Q loss0.028951→0.026113（已核对HISTORY）是fit目标变化，不是原图收益，Q的不同loss也不能与Direct按数值大小比较。两MODEL与SUMMARY的rho均为12.0853271484375；oracle摘要double范数中位数为12.085327488712064，微小差别来自训练float32范数计算，没有DEV调半径的证据。

成功训练checkpoint原远端路径为 `/root/autodl-tmp/qcr_run/runs/RUN_stage1_direct_seed0_retry5/final.pt` 和 `/root/autodl-tmp/qcr_run/runs/RUN_stage1_quality_seed0/final.pt`，本地回传SHA已与runner记录相符。物理回传记录包含Direct32失败原Run、retry2/3/4及成功retry5；交接曾提训练retry1，但本次未找到 `RUN_stage1_direct_seed0_retry1` 目录，其执行/保留/回传状态unknown，不能断言该Run已保留，不造占位。已存在的故障Run不纳入科学效用比较。简DEV的scale_masks维度故障在其独立评价retry1修复（此评价目录实际存在）；执行修复不消除上述科研偏差。

## 已完成评价终点与指标版本

主FINAL与DEV均已执行完成、回传并通过本地文件/链接核验。主统计来源为 `RUN_stage1_final_metrics_repair_seed0/SUMMARY.json` 与 `RUN_stage1_dev_metrics_repair_seed0/SUMMARY.json`；原始逐行来源分别是 `RUN_stage1_final_complete_seed0_retry1` 和 `RUN_stage1_dev_original_metrics_seed0`。修订仅将不变的原生pyramid_level 0/1/2统计映射为P3/P4/P5，没有修改raw行、执行模型、训练预算或checkpoint；全体/strict主IoU与CI逐项exact保留，分组合计通过。旧简DEV及 `RUN_stage1_dev_complete_seed0` v1作为历史证据保留，不替代主终点。

FINAL处理5,000图、36,213固定one-to-one TAL配对行，IoU图片分母为4,952有候选图；48无候选图和122未配对普通GT不补零、不进入该IoU分母。严格交集为5,304行/2,228图，已有完整FINAL CI；DEV为2,000图/13,690行，其全R_arg严格条件仍unknown。全部主表、repair/damage、held-out诊断、分层和限制见[REPORT.md](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/REPORT.md)，不把TAL机制评价升级为全GT召回、COCO AP或部署后处理效果。

**v1的pixel AUC/FPR是640 letterbox辅助指标；原图primary AUC/FPR v2已完成全量FINAL/DEV评价。** 两版本Run与scope分别保留，不覆盖v1后追称其原图指标。主v2的AUC使用原图连续裁剪logits、FPR使用正常二值mask及相同原图预测框支持，未定义值保持NA。统计与源码谱系见两主SUMMARY/METRICS，指标修复未消除训练科学偏差。

首个FINAL `RUN_stage1_final_complete_seed0` 因eval images symlink落到系统盘导致ENOSPC失败，失败Run与部分产物保留；这是执行故障。其retry1仅将eval-only images放到数据盘，随后已完成5,000图，未重训、未改模型参数或固定rho/eta/步数/checkpoint。此前1,226/5,000截面属于历史进度，现以已完成终点和最终分母为准。

当前候选范围FINAL统计的 `validation.errors` 与 `validation.incomplete` 均为空；`protocol_gate.observed_numeric_status='failed'`，六个数值条件全部失败。`protocol_gate.status='incomplete'`/eligibility=false来自仍存在的scientific_scope_drift，**不是FINAL执行或统计缺项**。held-out随机扰动状态是训练后追加诊断，不能追认成预锁定oracle方向测试；原设计完整合规确认仍不成立。停止当前实现，不登记/启动Stage II，详细终点及决定由REPORT.md维护。

## 补充评价的作用与边界

在不改保存checkpoint、不改训练预算/输入、rho/eta/两步、不事后挑checkpoint的前提下，补充评价能完成：

- 原图IoU、Mask75 repair/damage/net；全体、原failure、原success、argmax/Box条件和scope明确的strict目标组；按图cluster bootstrap配对A/B/D及D−B，报告空目标图处理与分母。
- 正常原图coverage、pixel AUC、FPR及固定支持范围；small/medium/large、P3/P4/P5分层；新增推理时间；保持GT仅用于标签/评价。
- Q held-out pairwise accuracy、Spearman、ΔQ与真实ΔIoU的一致率，以及固定第一步真实IoU上升比例；提前固定状态构造/随机种子、明确soft和hard两种标签、tie处理与图平均口径，不用DEV/FINAL选状态/参数。
- 补齐image_id/annotation_id/branch/raw_id/pyramid_level/target_gt_idx和原类别/GT类别/area，核对两臂共同A与原候选谱系；若缓存没有final条目，使用独立官方FP32流式FINAL、逐图生成官方TAL/P/h/c0/boxes并解码，不能调用缺final INDEX的旧evaluate然后把空结果当成功。
- 记录FIT/DEV缓存实际精度/compaction统计和固定witness数值差异，并把评价表示精度写入每个Run。

这些补充不能补救已经完成训练的seed错标、hard排序监督缺失、rank形式变更、随机半径变更、缺输入、实例权重或Failure/Success比例偏差，也不能凭小样本witness证明全量缓存数值等价。若要测试协议原设计，应另行登记独立新研究版本与Run并获当前任务授权，本轮不自动修复重训；当前任务只完成已训练版本的评估和诚实归档。

## 阶段放行与报告结论

最新 `protocol:109-116` 要求同时满足：held-out pairwise≥65%；第一步真实MaskIoU上升比例>55%；strict目标组D−B≥+0.2pp且图片bootstrap95%CI下界>0；D−A目标组≥+0.5pp且全体CI下界≥−0.1pp。原图scope与strict定义必须随判定一起报告；缺项不得按通过处理，训练loss不得替代。

已完成FINAL的pairwise53.0261%、一步真实IoU上升17.7367%，strict D−B−1.5904pp且CI完全为负；六项数值放行条件全部失败，详细表见REPORT.md。当前实现明确STOP于Stage I，不登记/启动Stage II，不用超参、额外epoch、gate或上游解冻挽救。prototype/backbone/neck/box/class保持冻结。DEV strict unknown、FIT全量存储未知与W10样本范围保留；原设计完整科学合规确认未成立，但这些限制不能被当作继续挽救当前失败配置的放行理由。训练偏差及SHA关系不因追加评价、回传核验或像素/统计修复而改变。

最终报告须分别陈述：本次实际训练版本、与预注册/研究笔记的偏差、统计结果与CI、精度/候选scope限制、仍未知事项及停止或放行依据。阴性只能支持“此预算、输入、目标和范围下的当前实现未兑现”，不能因偏差和范围不完整而升级为所有质量地形方法不可能；反过来，即使补充出现某个阳性子组，也不能把已有偏差抹去或把子组收益当全体FINAL收益。
