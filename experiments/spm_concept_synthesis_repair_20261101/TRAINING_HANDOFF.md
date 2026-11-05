# 数据交付与原生训练接入边界

## 已可使用的接口

发布Run：`RUN_CONCEPT_SYNTH_RELEASE_22ad2471b761`（Slurm1102）。本地数据在本Study的`private_data/auxiliary_dataset/`；远端原路径在[提交回执](SUBMISSION_release_v1.json)和[PORTABLE_DATASET.json](delivery/1102/runs/RUN_CONCEPT_SYNTH_RELEASE_22ad2471b761/PORTABLE_DATASET.json)。

在带NumPy/Pillow的环境中，进入便携目录：

```python
from data import resolve_manifest, load_auxiliary

rows = resolve_manifest("MANIFEST.json")
train = [r for r in rows if r["split"] == "train"]  # 72图 / 24组
qa = [r for r in rows if r["split"] == "qa"]        # 18图 / 6组
image, labels = load_auxiliary(train[0], verify_hash=True)
# image: uint8 [640,640,3]，原480高RGB上下填80，没有resize
# owner / part / foreground: [160,160]
# boundary: [3,160,160]，顺序普通环境、其它猪、设施
# owner_valid / part_valid / foreground_valid / boundary_valid：分别用于loss
# part_spatial_control：逐owner保持类计数，背景/255不变的预登记空间控制
```

实际服务器已加载全部90图并验证数组/字节、一个2图train NumPy batch和便携相对路径loader；本机只做标准库字节/元数据复核，没有为此安装NumPy/Pillow、没有本地模型执行。每张RGB/标签SHA在MANIFEST，183个包内数据/代码/说明文件SHA在SHA256.json（该清单本身另由archive SHA绑定）。

所有合成标签只供训练期辅助监督/QA，`normal_segmentation_target_allowed=false`。255必须ignore，不能将unknown/隐藏部分/画幅padding变成背景或负例；边界valid与part valid不同，不混用。A/B/C同组不得跨split，QA不进入训练；90图不是90独立场景。空间控制还带形状/计数信息，不是完美因果反事实。

## 已完成与未完成分开

| 项目 | 当前状态 |
|---|---|
| 固定合成语料生成、身份/表面映射、逐图射线和有限预览 | 通过；旧失败与一个旧数值超限保留 |
| 合成train/QA个体与旧source group隔离 | 通过，224/56个体、41/11旧组无交集 |
| NumPy辅助接口、保守有效域、控制标签类计数、便携包 | 通过；不是Torch训练接入资格 |
| 合法TRAIN-only原160独立Mask/boxes实际新接口资格 | 待执行；沿用[标签合同](../spm_pig_concept_training_20261101/LABEL_QUALIFICATION_V1.md)，不把160上采样冒称原640精细GT，不打开混合原标注 |
| 原生可训练参数精确白名单/数量、非BN共享梯度 | 待执行；候选全原生非BN适配不能沿用旧只训Mask分支的参数/资源结论 |
| FrozenBN、O2O detach、native criterion训练分支 | 待执行；全局eval或递归train均不可替代逐模块控制核验 |
| 辅助头删除、原生导出及正确正常推理 | 待执行；设计上零额外部署参数不等于已经通过 |
| λ/归约、优化器/时钟、匹配R/G/Q/C、配对seed与预算 | 待冻结；2k单seed只能初筛，旧RLPL预算不挪用 |
| 正常COCO末点评价、定位/原成功损伤/尺度保护 | 未执行；共享上游适配可改变框/分数/排序，不能沿用旧检测逐值不变承诺 |
| 真实部位迁移/Mask AP效用/N+5 | 未测，没有新权重或效用阳性 |

## 唯一下一步

在[原概念训练执行Study](../spm_pig_concept_training_20261101/study.json)另冻结一个有限的**原生参数—梯度—数据接口资格**合同，然后才执行对应工程smoke。优先回答“部位与几何监督是否进入预期原生非BN参数，同时真实任务、BN和O2O分支正确”。这比继续增加合成图或新增模块更直接地决定当前概念训练主张能否被合法检验。

通过才冻结正式效用实验与资源；只到辅助头、真实targets语义错误、BN状态漂移或任务分支错误则先修同一接口，不开正式训练。非零梯度、合成loss下降、QA部位可解码都不能替代匹配对照下的真实AP收益。该后续未启动，本文不是正式训练预算或新Gate。
