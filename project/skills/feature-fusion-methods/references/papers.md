# 论文依据与阅读边界

本包核对原始论文并选择性阅读与方法有关的正文／理论段落，未逐篇全文审读、复现实验或开展计量式影响力排名。下面按“论文能支持什么”和“不能由此推出什么”区分来源结论与应用迁移。

方法正文中的YOLOSeg组合、主任务优先变体及部分代数说明是基于文献的综合或推导，文中已作标识；不能把它们作为原论文已测结果。

## BIFPN

**EfficientDet: Scalable and Efficient Object Detection**

Tan, Mingxing, Pang, Ruoming, Le, Quoc V.。CVPR 2020; read arXiv v7。

- 阅读版本：[1911.09070v7](https://arxiv.org/abs/1911.09070v7)。
- 核读范围：§3.2–3.3 cross-scale connections and weighted fusion。
- 方法依据：双向多尺度融合与归一化可学习权重，比较静态融合方式及效率。
- 结论边界：常用标量融合权重不等于每图或每实例动态可信度。
- PDF身份：`08deba0f1bdb13b8d8ea69fb10df4f5960f101413f5d5e314ddc28b3dd2b9c7d`。

## CBAM

**CBAM: Convolutional Block Attention Module**

Woo, Sanghyun, Park, Jongchan, Lee, Joon-Young, Kweon, In So。ECCV 2018。

- 阅读版本：[1807.06521v2](https://arxiv.org/abs/1807.06521v2)。
- 核读范围：§3 sequential channel and spatial attention。
- 方法依据：顺序执行通道与空间重标定，并使用平均和最大池化摘要。
- 结论边界：原分类／检测证据不能直接变成某个YOLOSeg位置的保证增益。
- PDF身份：`8fd75f994928659f6f89d50ba45589627459d588f09f27e0d7b18b9925984060`。

## CONDINST

**Conditional Convolutions for Instance Segmentation**

Tian, Zhi, Shen, Chunhua, Chen, Hao。ECCV 2020; read arXiv v4。

- 阅读版本：[2003.05664v4](https://arxiv.org/abs/2003.05664v4)。
- 核读范围：§2.1 instance-conditioned filters and §2.3 Dice objective。
- 方法依据：动态实例参数与相对位置使共享特征产生对象特定Mask，并采用相应目标。
- 结论边界：支持实例条件的重要性，不证明任何部位／门控组合都有增量。
- PDF身份：`6c3643a3002fc887938c00fbbbf2c89d2fb3f6595deec3b70b1a1f8a59434981`。

## DCN

**Deformable Convolutional Networks**

Dai, Jifeng, Qi, Haozhi, Xiong, Yuwen, Li, Yi, Zhang, Guodong, Hu, Han, Wei, Yichen。ICCV 2017。

- 阅读版本：[1703.06211v3](https://arxiv.org/abs/1703.06211v3)。
- 核读范围：§2.1 deformable convolution equations and sampling。
- 方法依据：以输入相关偏移改变固定卷积网格的取样支持。
- 结论边界：自由空间支持不等于已建立正确物理对应，也不解决标签语义冲突。
- PDF身份：`f87b0e43d5cd1b8d770188c524baee8144354c78179fdb790c378753c0939115`。

## FILM

**FiLM: Visual Reasoning with a General Conditioning Layer**

Perez, Ethan, Strub, Florian, de Vries, Harm, Dumoulin, Vincent, Courville, Aaron。AAAI 2018。

- 阅读版本：[1709.07871v2](https://arxiv.org/abs/1709.07871v2)。
- 核读范围：§2.1–2.2 conditioning and visual reasoning model。
- 方法依据：条件输入生成逐通道缩放和平移，对视觉表示进行特征级调制。
- 结论边界：支持条件化形式；未证明任意对象条件都可靠或能处理空间错配。
- PDF身份：`765acea6f2a78b1d76961392099cd0af49cb2e7d6f06d0b3d82f1dbe9cdcd363`。

## FPN

**Feature Pyramid Networks for Object Detection**

Lin, Tsung-Yi, Dollár, Piotr, Girshick, Ross, He, Kaiming, Hariharan, Bharath, Belongie, Serge。CVPR 2017。

- 阅读版本：[1612.03144v2](https://arxiv.org/abs/1612.03144v2)。
- 核读范围：§3 top-down and lateral construction。
- 方法依据：以自顶向下路径和侧向投影融合不同层级，给细尺度带来语义信息。
- 结论边界：不证明再加任意尺度路径或额外颈部必然改善当前YOLO实例分割。
- PDF身份：`3a33e7ad38c17037624defde57f380f49dee918c1c14abd5dbcad032bae91428`。

## NONLOCAL

**Non-local Neural Networks**

Wang, Xiaolong, Girshick, Ross, Gupta, Abhinav, He, Kaiming。CVPR 2018。

- 阅读版本：[1711.07971v3](https://arxiv.org/abs/1711.07971v3)。
- 核读范围：§3.1–3.3 non-local operator and residual block。
- 方法依据：按内容关系聚合远处位置，以残差形式加入，并讨论零输出投影保留初始行为。
- 结论边界：不保证相关性就是正确实例归属或训练后的不损伤。
- PDF身份：`415319e3506ca0d95851eafbce6ae46df76f86584375e09cbf118001a1968268`。

## POINTREND

**PointRend: Image Segmentation as Rendering**

Kirillov, Alexander, Wu, Yuxin, He, Kaiming, Girshick, Ross。CVPR 2020。

- 阅读版本：[1912.08193v2](https://arxiv.org/abs/1912.08193v2)。
- 核读范围：§3.1–3.2 sampling and fine/coarse instance representations。
- 方法依据：组合精细特征与粗对象预测，在不确定位置细读出，讨论区域特定信息。
- 结论边界：不表示原生YOLO每种边界错误都应增加同款点头，也不保证标签粗糙时的收益。
- PDF身份：`dc544aba7f76d1c395154d638c52e6196095eb9468700828b3b35d1fa250244c`。

## RESNET

**Deep Residual Learning for Image Recognition**

He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian。CVPR 2016。

- 阅读版本：[1512.03385v1](https://arxiv.org/abs/1512.03385v1)。
- 核读范围：§3.1–3.2 residual learning and projection shortcuts。
- 方法依据：以残差形式参数化网络，维数不同时可用投影支路。
- 结论边界：支持增量学习结构的动机；不保证任意新模块训练后保持已有任务输出。
- PDF身份：`1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a`。

## SE

**Squeeze-and-Excitation Networks**

Hu, Jie, Shen, Li, Albanie, Samuel, Sun, Gang, Wu, Enhua。CVPR 2018; read extended arXiv v4。

- 阅读版本：[1709.01507v4](https://arxiv.org/abs/1709.01507v4)。
- 核读范围：§3.1–3.3 squeeze/excitation/recalibration。
- 方法依据：全局汇聚生成通道描述，再学习非互斥通道权重重标定特征。
- 结论边界：图像级通道选择不天然区分同类实例，也不能恢复不存在的细信息。
- PDF身份：`d6e451c492c7c492fdcf1e40265bae9bf51fd9ecb45e624aaaeeebcdb38b2828`。
