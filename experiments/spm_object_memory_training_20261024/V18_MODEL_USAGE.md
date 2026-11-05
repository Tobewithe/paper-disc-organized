# 原型记忆分割模型的重载与输出

V18已完成30轮、20130更新及完整验收，固定末C30和原生权重均已回传。**该融合模型效用阴性：Mask AP57.343500→47.697063（−9.646437点），关闭本配置加训；当前推荐仍为原生候选。** 本页供研究复现，科学结果与方向判断见[完整结案](V18_REPORT.md)，技术与结果图见[C图集](C_VISUAL_GUIDE.md)。

## 对象知识如何进入一次预测

冻结的原生YOLO先输出框、类别、置信度、掩码logit及P2/P3视觉特征。每条正常检测用自己的框截取视觉ROI，通过查询器检索持久对象、边界和遮挡内容；检索得到的空间内容由共享decoder重建。细化网络再结合当前图像细特征、该条检测自身的原生Mask概率和重建内容，输出Mask残差。

最终Mask为原生logit与残差在同640画布上相加后阈值化。基础YOLO、框、类别、分数、排序和raw/output行身份不变。原生概率是本检测的条件，预测不读取GT或参考图；训练中用到的参考图及标注不需要随推理图片提供。

| 保存内容 | 尺寸与作用 |
|---|---|
| 对象空间内容 | 128项，每项128×8×8，以同一实例框坐标表示跨样本内容；不是128张RGB猪图，也未证明128项都形成不同语义。 |
| 边界和遮挡内容 | 各64项，每项128×4×4，按局部patch匹配并组合；是否学会精细类型以实际能力记录为准。 |
| 查询器及检索投影 | 查询当前实例和邻域，生成全局支撑和局部权重，内容不能脱离这些投影独立解释。 |
| 共享重建器 | 将三类空间内容解码为重建特征、细化上下文和辅助结构输出。 |
| 实例细化头 | 输入512通道当前特征及1通道原生Mask概率，再与64通道重建细节融合；新增64个输入权重，初始为0。 |
| 归一化和PCA | 从train拟合的512→128映射内嵌在末权重`projection_state`，推理不用重新拟合或读取特征缓存。 |

## 文件和环境

实际交付两份权重：[native_candidate.pt](weights/native_candidate.pt)与[末轮MODEL.pt](runs/RUN_SPM_V17_FORMAL_C_NATIVE_CONDITIONED_8621cffce7a1/MODEL.pt)。原生SHA为`0e40602f1dbf0697fd49fa8e9f887c068970e3c5c39bd618f66e483c8712c8d6`；C权重20273562字节，SHA为`22ffe8320b9741cc2b033a8a6b135decf8b1514f93e57184f963112d28172df1`，实际元数据为`stage=C`、`epoch=30`和`model_variant=native_mask_conditioned_v1`。主包716文件与额外融合图111文件SHA验证，严格CPU重载、有限数值和PCA精确检查通过。只交付三个码本不足以重建完整推理。

复现环境沿冻结合同：PyTorch2.5.1+cu124、Ultralytics8.4.100、原生FP32、新模块显式BF16，matmul/cuDNN TF32均关闭。源文件随对应Run保存；在服务器使用已分配GPU的一次性Slurm作业内执行，沿[已冻结验收链](V18_ACCEPTANCE_CHAIN.md)。

## RGB重载接口

使用[candidate_native_conditioned_adapters.py](candidate_native_conditioned_adapters.py)中的`NativeConditionedRGBInference`。下面只展示接口；路径应指向最终已核SHA的两份权重和实际640×640 RGB画布。

```python
import numpy as np
from pathlib import Path
from PIL import Image
import sys

study_root = Path("experiments/spm_object_memory_training_20261024").resolve()
sys.path.insert(0, str(study_root))
from candidate_native_conditioned_adapters import NativeConditionedRGBInference

native_path = study_root / "weights/native_candidate.pt"
memory_path = study_root / "runs/RUN_SPM_V17_FORMAL_C_NATIVE_CONDITIONED_8621cffce7a1/MODEL.pt"
runner = NativeConditionedRGBInference(
    native_checkpoint=native_path,
    native_sha="0e40602f1dbf0697fd49fa8e9f887c068970e3c5c39bd618f66e483c8712c8d6",
    memory_checkpoint=memory_path,
    device="cuda",
)
rgb = np.asarray(Image.open(image_path).convert("RGB")).copy()
result = runner(rgb)
```

输入固定640×640×3、uint8、RGB。接口不隐式缩放任意相机原图；已有数据使用的画布变换与评价坐标必须一致。输出`detections`是原生检测行，`masks`是同顺序640²布尔掩码，`raw_ids`和`ordinals`保留原生输出身份。不能在重载后另行排序、换框或把不同检测的Box/Mask成功拼在一起。

Slurm345已从12张预固定真实RGB重新前向，与完整开发集缓存评价逐行身份及Mask RLE精确相同，原生/SPM/PCA状态不变。随后Blender诊断的8张真实参考又与原C30全量产物精确一致；这些是重载/实现核验，不是科学效用阳性。

## 成绩和适用范围

完整任务使用432旧dev、2863GT、12899正常输出；共同原生/融合Mask AP为57.343500/47.697063，Box AP均66.884503且非Mask检测身份精确不变。Mask AP75下降12.018728点、小目标下降7.338921点，修复35、损伤405/2126。预设至少62.358486和全部护栏未通过；旧开发集不作为独立外部泛化证明。

全量AP、同检测修复/损伤、48组图、权重SHA、独立48项AP差0与RGB重载均已验收；[本地权重检查](LOCAL_C_WEIGHT_READBACK.json)和[视觉QA](C_MODEL_VISUAL_QA.json)分开记录。末轮阴性保留，不以中途较高分或loss下降替代；用户要求结案后的[Blender零训练诊断](../spm_blender_domain_replay_20261024/REPORT.md)也已完成，后继训练未启动。
