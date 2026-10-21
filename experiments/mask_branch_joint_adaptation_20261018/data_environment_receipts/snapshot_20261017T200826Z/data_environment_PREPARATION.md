# 新服务器的数据与私有环境准备

本记录仅描述输入准备及可验证的传输恢复。科学训练范围、20k 选择、三臂方法、预算与验收由本 Study 的冻结协议定义；下载完整 COCO train2017 不扩大训练预算。本文不是完成凭据，以独立 `data_environment_READY.json` 和真实 Run 为准。

## 位置与版本

- 执行身份：用户授权的 `a25023104`，主机 `master-01`。
- 项目：`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized`。
- 私有环境：项目下 `.venv/bin/python`；未修改 `/opt/yolo`、系统 Python 或共享缓存。
- 该 venv 的解释器依赖个人 `.cache/python3123-source-runtime`；此路径是正在使用的运行时，不能按目录名当作可删临时缓存。
- 图像：`/srv/gpu-lab/users/a25023104/datasets/coco2017/train2017` 与 `val2017`，**没有中间的 `images/`**。
- 原始 GT：同数据根下 `annotations/instances_train2017.json` 与 `instances_val2017.json`，使用官方实例 RLE/多边形原始 JSON。
- 固定科学包：Python 3.12.3、torch 2.5.1+cu124、torchvision 0.20.1+cu124、Ultralytics 8.4.100、NumPy 2.4.3、pycocotools 2.0.11、faster-coco-eval 1.8.0、opencv-python 4.13.0.92、Pillow 11.1.0、PyYAML 6.0.2。全依赖以成功安装 Run 的 `requirements-resolved.txt` 为准。

## 真实执行与工程恢复

| Job | Run / 作用 | 真实性边界 |
|---|---|---|
| 97 | `RUN_DATA_ENV_8394c214fcc14e9cb8a4698e4b2598ee`：官方 train/val/annotations 与私有环境初始准备 | val/annotations 已成功；仅 train 单连接下载及极慢的 standalone Python 下载被后继工程接管，中断日志保留，原 Run 不改成成功。 |
| 98 | `RUN_DATA_ENV_RECOVERY_10b640b4f66e444ab46ae244a38487ec`：同官方 train ZIP 的 8 路 Range | 逐响应核 206、Content-Range、ETag；保留原约 323 MiB 前缀并在最终拼接核同前缀 SHA；完成后核 ZIP CRC、118287 文件与完整 ZIP SHA。因父 Run 的环境被另外接管，聚合状态可能失败；实际 train 组件须独立通过。 |
| 99 | `RUN_DATA_ENV_PYTHON_RECOVERY_9b71ff58e43f4eda93210e49d1c7d9af`：同版本 Python 源码编译 | 源码校验及私有编译、标准库导入通过。官方 PyTorch 索引元数据解析过慢，后继仅在相同官方 wheel SHA 验证后接管安装；旧中断记录保留。 |
| 100 | `RUN_DATA_ENV_WHEEL_RECOVERY_de188f6c52d645eda5490c27e4d2cc2c`：固定官方 wheel＋Tuna 依赖 | 两份官方 wheel SHA 已验证，保持所有科学版本；实际环境成功须见该 Run 的 `ENVIRONMENT.json`/`RUNTIME_READY.json`。接管分支完成所有数据组件后才写主 allready。 |
| **101** | 独立输入聚合核验；`afterany:99:100` | 单次、CPU-only；重新全模块导入和精确版本校验、三 ZIP SHA、GT SHA、图像集合/数量/非空、GT 注释数。成功写 `data_environment_READY.json`，后续实验只依赖 **`afterok:101`**。 |

上述均为一次性 Slurm 作业，没有建立重复定时任务或目标模式，也没有进行模型前向或训练。下载、解压、编译及依赖安装均无 GPU 申请。失败的传输 Run 不是科研阴性结果，也不能伪装成成功；独立组件及最终聚合必须分别保留。

## 固定来源与校验

原始 ZIP 来自 COCO 官方 `images.cocodataset.org` 的 `zips/train2017.zip`、`zips/val2017.zip`、`annotations/annotations_trainval2017.zip`。官方此域 HTTPS 连接失败，初始数据使用官方 HTTP 地址；Range 恢复仍为同一对象，ETag 和完整 CRC/SHA 连续核验。

- instances_train2017 SHA256：`610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`；118287 图、860001 注释、80 类。
- instances_val2017 SHA256：`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`；5000 图、36781 注释、80 类。
- Python 3.12.3 源码：华为镜像 `Python-3.12.3.tgz`，与 [Python 官方 3.12.3 发布页](https://www.python.org/downloads/release/python-3123/) MD5 `3c5498a34d5226c9b746b1199f0bf2d9` 一致；实测 SHA256 `a6b9459f45a6ebbbc1af44f5762623fa355a0c87208ed417628b379d762dddb0`。仅编译安装到个人 `.cache/python3123-source-runtime`，没有安装系统包。
- torch 官方 cp312 Linux cu124 wheel SHA256：`bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7`。
- torchvision 官方 cp312 Linux cu124 wheel SHA256：`d1053ec5054549e7dac2613b151bffe323f3c924939d296df4d7d34925aaf3ad`。
- wheel 哈希来自官方 `https://download.pytorch.org/whl/cu124/{torch,torchvision}/` 索引；实际官方 S3 文件逐字节匹配 SHA，未以别的 CUDA 构建替代。

## 运行依赖与凭据

提交清单、脚本 SHA 与最终依赖见 `data_environment_submission.json`。服务器上的 `data_environment_status.json` 是组件聚合状态，必须同时有 `execution_status=completed`、`all_required_inputs_ready=true`。独立 `data_environment_READY.json` 才是训练输入放行凭据；只看到目录存在、venv 创建、下载开始或 Job 已提交均不构成通过。

用于原生验证/工程 smoke 的环境先行凭据是实际成功 Run 的 `RUNTIME_READY.json`，它不表示 train2017 全部就绪。正式 20k 训练仍必须依赖 Job 101 和冻结训练输入检查。
