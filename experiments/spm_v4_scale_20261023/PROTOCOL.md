# SPM-YOLO V4全量数据与长程训练合同

2026-10-23用户在快速首轮完成后明确要求“规模不上去怎么训练出来，继续”，授权扩大数据和训练；既有个人服务器、多卡和一次性Slurm授权沿用。本Study取代上一轮“先独立test、不追加训练”的当前执行顺序，保留快速首轮所有协议、Gate、权重和阴性干预。没有定时器或自动续训服务。

## 科学目标与判别

快速首轮F对continued/G有+1.3045/+1.3725pp开发Mask AP支持，但对原B*只有+.2306pp，AP75较B*下降，删除推理记忆不损失AP。因此当前主张H是：先充分训练猪群编码器，再用覆盖更多真实/合成场景的局部监督训练原型与更长程head适配，能否保留相对同预算基线的效用；不预设更大的库必然有用。竞争解释H_alt是训练信息/普通Adapter或优化轨迹造成收益，部署检索没有独立价值。

固定K=32、128D和原有fg/contact/对应/防塌缩损失，不因扩大数据自动增槽或换结构。M1依据是上一轮真实Run和未出现检索必要性的现象；M2预测是F在长程同预算训练后高于continued/G，而部署检索若真正有用，关闭/错配key-value/空间置乱应使同checkpoint成绩降低；M3三臂共享真实/合成数据、抽样、增强、起点、步数和选优规则，G共享原型预训练投影，披露128参数差异；M4保存同F checkpoint的alpha=0、错配value和空间打乱评价。没有定向干预增量就不称部署检索机制成立。

## 冻结规模与资源上限

- 原始PigLife训练部分全部1704图，不读取原始426 test图像。沿用快速首轮13/4个文件名family划分，得到1272 train/432 dev；旧54 dev全部继续留在dev。family不是已核实物理猪栏，原始test含相邻视频帧，不冒称跨场景独立测试。
- 真实RGB在本地按PIL bilinear共同640 letterbox，JPEG quality95/subsampling0；逐原图和处理图SHA。服务器由原始COCO polygon/RLE解码可见Mask，再按同rect nearest映射至640；所有模型共用同一缓存，不把新AP与旧PNG/54图AP直接作数值涨幅。
- Blender 2048个几何场景×2外观=4096 RGB，1792几何train/256 dev；姿态、体形、密度、布局、相机、光照、皮肤、地面随机化，沿用无交叠与可见ID标签。摆放最多6次确定性重启，全部失败则工程失败，不删除动物/静默缩预算。4096射线/几何、RLE往返、成对标签一致；ID误差比≤0.5%。单一Pig资产限制保留。
- 4096张train-only真实cutout合成；全部base实例可见Mask扣除覆盖，新增donor，不从dev取图。无合格donor的训练图仍用于原生训练，仅不作为抠图donor。
- B*：官方COCO yolo26m-seg.pt→单类pig，完整1272 real-train，全参数120轮；batch12，106步/轮，**12720优化步**。AdamW peak lr .0003、wd .0001，前300步warmup，cosine到峰值.02；640画布，FP32，clip10；不早停。
- 原型：在所选B*冻结坐标内训练**20000步、batch2048 token**，Adam lr .001；每source每C/B/I/BG最多24 token，域×stratum池均匀抽样，损失权重沿用1/.5/.2/.1/.01，fg/bg各16槽、20轮球面K-means。构库后query投影、归一化和bank冻结。
- continued/G/F：同B*起点、冻结上游/BN，原生Segment26 head及相应分支白名单训练。每臂**8000步、batch12=8real+4synthetic/cutout**；完全相同抽样/增强序列。head peak lr .0001、branch .001、alpha .01，wd .0001，warmup200，cosine到.02；alpha初始0、限幅±.5、残差RMS上限原特征.25，原生O2O detach保留。
- 使用最多4块A6000并行：3个独立渲染分片与1个基线；之后单卡原型、3组各单卡独立训练。单GPU训练作业最多8小时，渲染分片最多6小时，CPU预处理最多2小时；一次性Slurm依赖链，无常驻占卡、无定时器，不操作其它用户任务。失败保留并只修工程，不自动换研究配置/seed。

## 评价、选择与出口

同432张real-dev，原生O2O最终输出，conf .001、max_det300、process_mask upsample/crop/logit0、COCOeval maxDets[1,10,100]。segm loadRes前去bbox，指标百分比/差值pp。基线每1060步及最后、三臂每1000步及最后评价；最高dev Mask AP选优、平分取早，同时保留last。E2ELoss以总优化步数设epochs并逐步update，沿用显式进度日程。

主比较F−continued Mask AP，另报F−G、相对B*、AP75、Box AP、4 family损伤和延时。开发阳性定义主差>0，若F对G也>0则保留该配置做后续外部核验；不据单seed作统计确定性结论。F不优于continued/G则明确保留阴性，优先选择实际最佳基线/Adapter，不继续扩大库。若删除/破坏检索差<.1pp，停止把部署检索作为收益解释，保留有效训练权重；≥.1pp仍需G对照，不能单独证明机制。工程不足/非有限/冻结漂移只修工程，不改变dev、阈值或预算。末步前不因开发成绩低临时加步数。

## 交付

真实Run、源码/数据/权重SHA、Slurm日志、step/epoch/sample计数、best与last、训练原型和来源、真实预测及全量指标。延续重点技术图：训练与AP曲线、数据姿态/尺度、原型近邻/来源/PCA、真实检索与gate/残差、RGB/GT/基线/F比较、接触放大及改善/退化样例。最终按完整实验给出问题—机制—方法—结果—方向决策。尚未完成的阶段明确标为未启动/进行中。
