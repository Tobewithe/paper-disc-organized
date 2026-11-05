# Qwen 猪场场景归纳系统：第一阶段部署验收

2026-10-27 已在个人账号 `a25023104@192.168.71.10` 的 `/srv/gpu-lab/users/a25023104/projects/pig-scene-synthesis` 部署并实际运行 **Qwen3-VL-Embedding-2B + Qwen3.5-9B**。两模型32个文件约23.60GB已逐SHA核验；1426张真实train参考图与原标签已接入。13张跨来源/相机RGB完成全部流程，fresh离线重载及独立统计通过。

| 维度 | 本次部署的整体链路 |
|---|---|
| 问题 | 用户要求用全Qwen视觉模型在服务器打通真实参考图分组、场景属性/证据和可追溯知识库，为固定合成数据集建设提供工具。原服务器没有该环境或这两个模型，官方HF直连不可达。 |
| 机制假说 | 不适用：本次是工程部署。没有对象知识新训练、人工标签准确率或分割增益实验；工具可用性由真实模型加载、正常RGB和fresh重载验证。 |
| 针对性方法 | 独立venv和模型锁；小文件取固定HF revision，大权重取官方Qwen ModelScope且与HF LFS SHA一致。CPU一次性Slurm安装下载，GPU逐模型加载；全图向量/HDBSCAN/9B属性/语义向量/三级目录/统计/生成配置提案/图册贯通。源身份/SHA、未知几何与异常输入均有检查。 |
| 整体结果 | 738完成13张全部RGB流程，13份属性JSON/0解析失败，2048维单位向量、3候选组/2候选类型；生成提案3份，已核验可量产场景0。739 fresh重载3 RGB向量误差0、cosine1，1份9B结构化输出精确相同；740独立核39身份字段、169余弦值、25统计行及JSON/Parquet一致。134文件回传逐SHA核验，三技术图已实际查看。累计GPU0.116667小时/峰值1卡；无额外模型训练。 |
| 研究决策 | 第一阶段工程部署通过。优先继续真实场景证据与几何/资产核验，依据是资料包含群养/分娩栏、跨相机/材质/光照，而当前自动簇较粗、物理尺寸仍未知。唯一下一步为另冻500图/有限代表分析协议，输出可审查场景证据和冲突；不把部署或候选目录当作合成release/+5效用通过。用户随后“继续”已授权这一接续。 |

## 使用

```bash
ssh -p 22 a25023104@192.168.71.10
cd /srv/gpu-lab/users/a25023104/projects/pig-scene-synthesis
sbatch slurm/analyze.sbatch data/reference_v1/ACCEPTANCE_MANIFEST.json 13 13
```

完整通过Run：`runs/RUN_QWEN_PIPELINE_1a3d8c2ff61b`。主要结果：

- `knowledge/scene_catalog.json`：图像、候选场景、候选类型及来源证据。
- `knowledge/statistics.parquet`/JSON：模型报告频率与分母。
- `knowledge/generation_config.json`：结构提案；未知尺寸/相机值为null，需几何与资产核验。
- `report/index.html`、`SCENE_ANALYSIS.png`、`COSINE_SIMILARITY.png`、`EMBEDDING_SOURCES.png`。
- `global/embeddings.npy`及逐行`INDEX.json`；`semantic/semantic_embeddings.npy`。

独立venv实际为Python3.11.6、PyTorch2.8.0+cu128、Transformers5.2.0；CPU工程Run保存关键库源码SHA与环境freeze。Qwen9B使用官方PyTorch参考路径，13图属性生成252.52秒；Embedding13图前向2.71秒，加载/进程/其它步骤另计。GPU峰值分配约19.17GB，无量化。以上是本小批测量，不是所有图像/上下文的性能保证。

## 证据与边界

732环境/模型68分37秒；733/734/736工程与元数据；735的13向量已生成但日志set序列化失败，原failed/18秒保留，736独立审计向量有效，738重算与735差0；737为该批真实向量的CPU聚类；738完整5分58秒、739 fresh44秒、740独立统计1秒。见[资源账](RESOURCE_ACCOUNTING.json)、[失败与修复](FAILURE_735_AND_REPAIR.md)、[回传核验](RETURN_VERIFIED.json)、[图像QA](VISUAL_QA.json)、[协议](PROTOCOL_V1.md)、[GPU冻结计划](GPU_STAGE_PLAN_V1.json)。

权重在服务器`models/`可离线重载；本地交付`delivery/RUN_QWEN_RETURN_9fbb219a979d`保留代码、配置、原始输出、日志、Slurm/Run与权重SHA清单，大权重不重复回传本机。两个官方模型revision：Embedding `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`；9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。

背景/设施特征接口需要带原图SHA的核验ROI，本次13图没有该类ROI，实际裁剪0，不以未知区域造背景真值。Reranker、27B和Qwen-Image尚未加入。Blender入口及待核验生成配置已提供，物理几何/资产和实际渲染E5未验收；E1–E4人工真值准确率与E6真实Mask AP未测。test426未读，整个模型训练/+5交付goal仍未完成，不推送GitHub、不创建定时自动化。
