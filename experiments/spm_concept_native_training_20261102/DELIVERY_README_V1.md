# V1原生权重与复现交付——阴性实验，不是成功模型

12个R/G/Q/C×3seed末点均未获采用；推荐保留原生N。当前正确部位路线的效用/独立机制及零训练挽救必要条件均未通过，Goal未达成。此包用于完整保留和复现，不能当作+5模型发布。

## 内容

- `weights/N_NATIVE.pt`及`weights/{R,G,Q,C}_{20261102,20261103,20261104}_NATIVE.pt`：共13原生权重，无辅助头/检索/后置Mask修补/优化器；SHA和来源在DELIVERY_MANIFEST.json。
- `source/`：正常推理CLI、原生加载器、正式训练和辅助/数据/时钟代码、冻结环境347源文件摘要及输入路径/身份、完整正式/评价/重放协议。训练数据、原标注与GT不打包；不复制混合train/dev原索引。
- `evidence/`：四臂完整评价、控制可比性、损伤/来源区间、重放结果、12末点索引及训练矩阵核验。大预测及RECOVERY/辅助权重留登记服务器，以SHA/源Run关联。

## 正常推理

使用冻结Python3.12.3、Torch2.5.1+cu124、NumPy2.4.3、Ultralytics8.4.100、Pillow11.1.0、pycocotools2.0.11环境。冻结的是指定版本及347个库源/二进制摘要，不是完整OS或跨平台容器。未声称在另一机器重跑训练；已经在登记服务器对12末点逐值fresh重载，随后13权重正常输出再次全432图逐值重放通过。

```bash
python source/predict.py --weights weights/N_NATIVE.pt \
  --image PREPROCESSED_640_RGB.jpg --output predictions.json --device cuda
```

必须显式提供640×640 RGB；不隐藏resize/letterbox。输出是正常O2O TopK300、conf>.001、原生c×P、原640裁剪RLE；无GT输入。自定义yaml/state_dict包不能冒称可以直接交给官方pickle模型CLI。

## 训练复现

`source/train.py`使用同N初值、INPUTS.json中的1272 TRAIN-only缓存及72合成train、FORMAL_V1.json固定seed/2000更新/λ/优化器。源码会检查协议、输入及实现SHA，要求Slurm和CODE_FREEZE/TASK合同；调用入口为原Study的`remote.py submit tasks/train_<arm>_<seed>.json`构造独立新Run，而不是覆盖旧目录。

本轮矩阵已全部执行，原任务key不可重复提交。复现实验应在另一个明确登记、获资源授权的新目录生成新任务key/来源快照，保持数据字节、采样算法和所有冻结数值不变；路径迁移须登记新的位置映射，不换数据或样本。scratch权重不作初始化，最后2000点不以dev择优。正式先全部12末点，再统一432dev；test426在本包中没有开启权限。

## 证据与结论

正常C平均56.494576，N57.343500，V23历史参照60.753278；C−Q+.088211且不稳定。保留N检测/无GT框关联的共享覆盖重放C57.384169、Q57.372034，差+.012135，不支持预定受限适配新版本。详细完整实验报告为FORMAL_REPORT_V1.md。不得把旧V23成果记为本次训练贡献。

校验DELIVERY_MANIFEST.json所有SHA后再使用。manifest自身由整个archive SHA绑定，清单不自包含其自身SHA；本地回传回执单列。此包不包含测试数据、未授权新实验或自动后继程序。
