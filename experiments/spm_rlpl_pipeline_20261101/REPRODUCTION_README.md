# RLPL原生模型复现入口

**当前状态以`study.json`及真实Run为准。此说明不代表正式权重、评价或部署资格已经完成。** 本Study的三臂均固定最后2000实际更新，不提供“最佳dev轮”选取入口。

## 正常推理

交付后权重位于`final_delivery/weights/{N,B2,C2,D2}_NATIVE.pt`，逐文件SHA见`NATIVE_WEIGHTS_RETURN_VERIFIED.json`。只加载可信、摘要相符的本Study权重。CLI及依赖源码须与完成的部署资格Run的`source/`一致，而不是随意使用后续编辑的工作区。

```bash
# 使用该Study登记的Python/Torch/Ultralytics环境
python predict_native.py --weights D2_NATIVE.pt --image prepared_640_rgb.png --output predictions.json --device cuda --image-id 1
```

输入必须是**已经按共同协议处理的640×640 RGB uint8**；入口不会暗中resize/letterbox。它不是对任意原分辨率图片作通用预处理的包装。输出包含同原生Box/Class/score/raw_id/检测ordinal及原生Mask RLE，不加载训练期辅助、字典、检索、GT或后置修补。类别id=1为本单类任务。正常筛选为score>0.001、O2O Top-K300、无NMS；COCO评价maxDets100另由评价器执行，不混为部署检测数量。

`RUNTIME_ENVIRONMENT_V1.json`记录实际解释器、5个包版本及347个Ultralytics/pycocotools源文件身份；不是整个Torch/CUDA/操作系统全文件镜像。正常CLI的fresh子进程一致性和实际成本以部署资格Run为准，不能由源码存在推断已通过。

## 同服务器训练重现

正式运行来源应使用各自Run的`CODE_FREEZE.json`、`source/`、`EVAL_CONTRACT.json`和实际提交收据，不能用后来修订的工作区替代旧代码。需要：

1. 原checkpoint及SHA、1272 train原RGB/独立Mask缓存；保留真实GT重叠，不从owner图重造。
2. 完整200组合成发布的`MANIFEST.json`、`RELEASE.json`、2000步`SCHEDULE.json`和`COORDINATES.npz`及逐项摘要；同组160 train/40 QA。
3. `FORMAL_TRAINING_V1.json`、共同原生初值/辅助seed、实际参数白名单、LR/E2E/U时钟。源路径为当前登记服务器的真实资产路径；迁移设备或数据根需要另作身份/环境核验，不能静默修改冻结指针后冒称原Run。
4. 每臂从原checkpoint开始；不用954 scratch、不自动恢复。恢复状态`RECOVERY.pt`不是部署权重，也不是供dev选择的候选。事务计数未决时必须停止并另案对账。

复现入口是`prepare_workflow_sources.py`和`submit_workflow_once.py`，它们先核前置真实完成/回传状态并拒绝重复提交。**重复正式训练需要新Run和明确预算，不应通过删除收据绕过保护。** 所有GPU执行仍走一次性Slurm；无自动依赖链、定时检查或自动恢复。

## 评价和证据

三臂完成后才评价432 dev；test426封存。`evaluate_formal.py`重算N/B2/C2/D2并核旧V23预测身份；`verify_final_statistics.py`独立重算COCO和簇区间；`verify_deployment.py`核纯原生入口/实际成本；`make_final_figures.py`仅绘制既有已验收产物。

报告必须区分相对N的收益/损伤和相对不同预算V23的收益/损伤。D2−C2是本轮原型筛选主增量，不是原型内容已获因果证明；单seed/四来源/复用dev不构成独立确认。大合成资产、训练恢复文件留服务器，用SHA清单关联；元数据、预览和原生权重分别回传，不把其中一种说成全部数据已经交付。
