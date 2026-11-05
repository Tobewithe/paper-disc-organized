# 首轮500真实参考图：已执行、复核并保留局限

已用部署的双Qwen模型实际完成500图向量、40图深度属性/语义向量、知识目录和技术图；Slurm741为18分05秒、742独立CPU复核通过，216文件逐SHA回传。该轮输出仍是待核验构建证据：全图聚类只有一个大簇和37个离群，9B在少数稀疏群养栏上把设施类型与猪只占用量混合，不能直接自动发布场景类型或训练集。

| 维度 | 本次完整链路 |
|---|---|
| 问题 | 13图部署不覆盖原train群养/个体/分娩设施和跨来源条件；资料必须先形成可审查证据，才能建设固定合成数据集。 |
| 机制假说 | 构建假说是全图表征、来源/相机中心和结构描述能提供场景候选；竞争解释为猪体/成像共性及聚类尺度混在全图距离中，且类别定义可被占用量影响。本轮不声称独立因果归因。 |
| 针对性方法 | 固定52原train+224Faro+224多视角，2B生成500×2048向量；HDBSCAN10/5；20代表/12边界/8离群共40正常RGB由9B提取同Schema属性，未知尺度null。固定模型/提示词/图像/参数，不据输出扫参；无其它视觉基础模型、0训练。 |
| 整体结果 | 500有限单位向量，40属性JSON/0解析失败、40语义向量。一个非噪声簇463图及37离群，38候选组/9字段组合候选，不等于38物理场景或9已验证类型。742核500图/40属性/40语义身份、250000余弦与93统计行，JSON/Parquet相同，余弦最大差6.728e−7。三图实际查看，四个宽群养栏被标individual_crate的例子已登记。无人工Macro-F1、无原生/方法Mask AP比较、0可量产场景；实际0.301389GPUh/峰值1。 |
| 研究决策 | 通过工程完整性验收，自动场景真值准入未通过。停止继续全图GPU扩量和扫聚类参数，优先核验以固定设施/几何定义的群养结构，避免按当前猪数判断栏型；第一步用已登记错误及真实参考图冻结结构定义、局部证据和首个群养模板的相机/资产依据。独立结构证据或物理核验通过后才生成训练场景；本轮不追加Reranker/27B或分割训练。 |

## 实物入口

服务器 `/srv/gpu-lab/users/a25023104/projects/pig-scene-synthesis`，本轮真实Run为 `runs/RUN_QWEN_CATALOG500_75e06a7b4476`。复现冻结批次：

```bash
cd /srv/gpu-lab/users/a25023104/projects/pig-scene-synthesis
sbatch _bootstrap/catalog500.sbatch
```

这是新增一次作业的命令，当前没有自动重跑。可直接查看已有 `knowledge/scene_catalog.json`、`statistics.parquet`/JSON、`generation_config.json`、`report/index.html`与`ANALYZED40.png`。图像/标注来源仍在`data/reference_v1`，500/40选择和每张角色在Run的`INPUT_MANIFEST`/`ANALYSIS_SELECTION`/`ANALYSIS_ROLES`，冻结数据/代码/模型/协议SHA在`EXECUTION_CONTRACT`及`_bootstrap/CATALOG500_ACTIVATION.json`。

本地交付 `delivery/RUN_QWEN_RETURN_9337599ac93c`；[SHA回传核验](RETURN_VERIFIED.json)、[资源账](RESOURCE_ACCOUNTING.json)、[图像QA](VISUAL_QA.json)、[失败与不确定性](FAILURE_AND_UNCERTAINTY_LEDGER.json)、[协议](PROTOCOL_V1.md)。完整代码与两模型部署/正常RGB/fresh重载证据另见[部署报告](../pig_scene_qwen_deployment_20261027/DEPLOYMENT_REPORT.md)。大权重在服务器，模型锁和哈希随交付保存。

## 具体复核发现

`real/1635`、`real/922`、`real/1634`、`real/1244`的可见设施是宽敞群养栏式空间，但模型的individual_crate解释使用了“中央隔断”“两头猪”等理由。该复核是助手看图指出的类别定义/占用量混杂，不冒称独立人工真值准确率，也不主张这些不同来源族一定是同一物理猪栏。原模型输出不修改，作为新结构核验的反例保留。

全图PCA/相似度仍呈来源/视角差异，不能因HDBSCAN选出大簇就断言Embedding完全没有结构信号；表示混合与簇选择尺度仍未区分。当前证据支持先完成结构定义/局部与几何核验，不支持盲增模型或把簇数改到期望值。

深度字段只分析40图，其余460图不复制多数标签成为逐图真值。模型报告的29群养/9个体/2分娩不是已核准的实际分布；同样，候选38份生成配置均保留待几何/资产核验，真实米制尺寸未知。没有把未知背景作负例，没有读取dev/test426；新合成release与实际训练/+5/完整交付goal仍未完成。
