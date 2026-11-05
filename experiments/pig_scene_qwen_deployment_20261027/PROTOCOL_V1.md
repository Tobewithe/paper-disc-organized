# Qwen 猪场场景归纳系统部署 V1

2026-10-27 用户提供完整方案并明确要求“在算力服务器上部署这个系统”。交付对象是可运行工程系统，不是新分割模型效用实验；不改旧Run/Gate，整个模型训练/+5目标仍保留未完成。

## 版本与职责

- 服务器：已登记 `a25023104@192.168.71.10`，个人新目录 `/srv/gpu-lab/users/a25023104/projects/pig-scene-synthesis`，先核无既有目录再创建。
- 首阶段仅部署 `Qwen/Qwen3-VL-Embedding-2B`（revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`）与 `Qwen/Qwen3.5-9B`（revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`）。官方公开 Apache-2.0 模型；原文件和逐权重SHA留存，GPU上真实加载与推理。
- 所有视觉基础模型均为 Qwen；不安装/调用 DINO、SAM 或其它品牌。Reranker、27B、Qwen-Image 不在本阶段下载/部署，按用户方案留作可选扩展。OpenCV/聚类/几何/Blender 是几何统计与渲染工具，不作为视觉基础模型。
- 首选独立 Python3.12 venv，PyTorch2.8.0/torchvision0.23.0 CUDA12.8（官方 Embedding 工程推荐 torch2.8，服务器595驱动），Transformers5.2.0 首先检验是否具备官方 Qwen3.5 和 Qwen3-VL 所需接口；若不兼容，只在新环境修复并记录确切版本，不改其它项目环境。
- Embedding 官方脚本只作已审查的源依赖；对图像处理失败必须报错，禁止其异常 fallback 的纯文本NULL向量混入图像聚类。9B输出受 Schema 验证，失败JSON保留，不写入有效知识库。

## 部署与数据流

提供来源/标注导入、全图/合法背景裁剪Embedding、余弦距离和HDBSCAN、代表/离群样本、9B可见属性提取、跨图汇总、source/camera关联的三级知识库、结构约束到generation_config、几何检查与Blender入口。设施裁剪必须有来源框/明确提示及复核状态；缺乏依据时不用自动伪Mask。姿态/框/未知标签不混造实例分割真值。

属性带 observed/inferred/unknown、原图证据和模型/revision/prompt。单图真实尺度、相机高度/角度未标定时为null；模型猜测不能静默成为米制Blender参数。场景簇不自动当成物理猪栏ID。生成配置缺几何量时明确 pending_geometry_review，保留可用的语义与证据。

本轮用已收集且原始train身份可追溯的参考资料验证；真实dev与保留test426不进入归纳。先用跨来源/相机小批真实图做正常端到端部署验收，支持用户后续扩大至500图及完整参考库；这不是E1–E4人工真值准确率或E6真实Mask AP研究验收。

## 有限资源与安全边界

- 安装/下载/权重哈希放CPU一次性Slurm作业：4CPU/16GiB，最多90分钟；项目新增磁盘上限100GiB。预检只做轻量登录I/O。服务器外网直连超时，允许使用当前用户已配置的本机HTTP代理，经**临时SSH仅绑定服务器127.0.0.1的反向端口**下载官方文件；完成后关闭隧道，不开放公网服务、不留守护进程。
- GPU验收：每次1张RTX A6000、8CPU/48GiB系统内存、最多45分钟，最多2次；另允许1次最多10分钟的fresh重载检查，总上限2GPUh、峰值1GPU。两模型依次加载，BF16、短上下文/有限图像像素和生成长度，不为下载占卡空转。
- 不在登录shell运行GPU任务；仅在有效SLURM_JOB_ID/CUDA_VISIBLE_DEVICES下运行GPU入口。记录每次JobID、命令、环境freeze、代码/数据/模型SHA、run.json和实际原始输出。只查看/提交/取消本任务本账号的作业，不操作他人作业。无定时器、自动轮询任务、GitHub推送。

## 部署验收

1. 两模型权重实际下载完整，safetensors与索引一致、官方LFS哈希（如可取得）及本地SHA逐项核对，CPU可导入所有入口，pip check无冲突。
2. 正常RGB实际图像Embedding有限、2048维、单位范数；图像身份/哈希和Embedding行一致，不静默丢图；源/相机分层样本和近重复关系可追溯。
3. 9B真实看图产生通过schema的可见场景属性，原文与解析均留存；未知尺度保持null，模型猜测不提升为几何真值。
4. 聚类—代表图—场景知识库—generation_config入口实际贯通，可重复调用；无人工标注准确率时不宣称场景识别已科学验证。
5. fresh进程离线重载与最小重复图推理、输出Schema和关键统计复核完成；给出项目路径、CLI/Slurm命令、输入格式、运行日志与后续使用说明。部署完不把合成release、分割效用或整个goal标为完成。
