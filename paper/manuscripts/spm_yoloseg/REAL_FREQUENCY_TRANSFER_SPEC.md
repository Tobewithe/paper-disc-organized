# SPM-YOLO：真实猪场纹理／高频统计到合成图的迁移规范 V1

**2026-10-22 | DESIGN DRAFT | 无现成代码/真实实验**  
[总实施蓝图](IMPLEMENTATION_BLUEPRINT_V2.md) · [首轮对照](../../../experiments/spm_yoloseg_20261021/PILOT_AND_ABLATION_V2.md)

**基础实现优先级（2026-10-22）：** 当前先实现[基础链路](BASE_IMPLEMENTATION_SPEC.md)。本模块作为独立可关闭选项保留，默认关闭；真实高频处理、PBR贴图和相机处理分别开关，不要求高频先获得增益才允许构建原型。本轮只完善设计，以下算法仍未实现/执行。

## 1. 问题：真实「高频」不只有一个东西

视频像素中的高频至少来自 **(a) 猪皮毛孔/毛发/褶皱/污渍微纹理；(b) 真实猪、栏杆、地面等物体轮廓锐度；(c) 镜头 PSF/去马赛克/锐化、ISO 噪声、帧压缩和下采样伪影**。三种来源不是同一种变换：
- (a) 属于材质/物体局部，理想情况下在三维 UV/PBR 材质阶段产生，并随姿态、透视缩放；
- (b) 属于几何边界，优先由正确的三维猪体姿态、遮挡 Z-buffer 与相机成像产生，**绝不把真实猪的外轮廓直接贴到合成 Mask 附近**；
- (c) 属于整帧成像管线，应该在 Blender 渲染后统一模拟，不能只给合成猪加噪声而不给背景。

**完整候选方案**：Blender 负责正确几何/遮挡、Pig/ground 类别真实纹理做材质、统一真实条件镜头+噪声+编码；频域统计用于进一步校准和进行结构化增强。基础版先用可核实的程序性材质与基本成像；不能在没有比较结果时把完整候选称为最强方案。

## 2. 真实素材采集：train-only 的 patch/频谱/成像参数库

### 2.1 来源和对齐
- 首先**按 farm/pen/video/camera/time 完成 train/dev/test 分簇**，本模块只可读 train RGB；最终 test 即使无标签也不能进 style library。
- 真实帧保留原始 codec、bit-depth、色彩空间、原分辨率和目标网络 640 letterbox 的变换。频率单位使用**最终入网像素的 cycles/pixel**，防止 1080p 的细毛发统计直接用于 640p 的同一频率。
- 有可见实例 Mask 的真实帧：为每猪内侧生成 erode(M, r) 有效区域，首版候选 r=3–7 px（在最终 640 像素尺度），或按可见面积自适应；从内侧裁取固定尺寸（如 64、128 px）且有效像素覆盖率 >=95% 的 RGB patch。离背景/相邻猪边缘不够远或 GT 不可信则跳过。
- 无可信 Mask 的训练帧：可以采集**画面级相机噪声/模糊**统计；不能未经审核就当作 Pig 纯纹理 patch。
- 背景可先人工粗分类地面、墙、栏杆、食槽，缺少可靠背景语义标签时只建 general-bg 类并降低强度，禁止把土壤纹理注入猪皮。
- 一个视频上限、单猪上限和同质重复 patch 去重；每 patch 记录原帧 SHA、ROI、category、mask SHA、camera_id、lighting_bin、compression、network-pixel scale、license、抽取脚本 SHA。**必要时与真实照片使用许可一致**，不公开原图也能发布经过统计聚合的频谱。

### 2.2 多尺度测量
- 对每个 patch 在 linear RGB 或亮度 Y 与色度 CbCr 中记录均值/标准差、每通道协方差、梯度强度/方向、局部 contrast。
- 以 Gaussian/Laplacian pyramid 估计多尺度残差，建议 640 入口先用**纹理波长 2–4、4–16、16–64 像素**三类计能区间；频率频带由 1/波长转换，单位 cycles/pixel。它们是分析与首轮候选，不是预先验证的最佳频带。
- 对去均值并加 Hann window 的同尺寸 patch 作 FFT；聚合 per-category/per-camera/per-illumination 的**径向 log-amplitude 分位数、方向能谱**以及跨通道统计。样本少的条件桶按可记录的 shrinkage 退回 category/global 统计；不直接复制少量 patch 的相位。
- 从纹理少/平坦区域（需人工核验）估计相机随机噪声与亮度的条件关系，或从连拍/视频差分中先消除运动再估计。一个压缩 H.264 视频中的平坦图像可能存在编码网格，**测到的是相机+编码整体输出，不是纯 CMOS 读出噪声**。

### 2.3 输出数据
- bank/patch_manifest.jsonl：来源、SHA、标签可信度、pixel-scale、is_train_only=true；
- bank/category_psd.npz：不同条件和频带的中值、IQR、有效帧/视频/猪数；
- bank/camera_profile.json：PSF/噪声强度-亮度曲线/白平衡/压缩质量经验区间；
- bank/sample_gallery.html：随机样本的 Pig 纯度、模糊/分层和误采集检查；
- bank/qa.json：泄漏校验、重复性、采样数、频谱稳定性、无效 patch 率。

## 3. 合成图的五个有目的版本（同几何、同实例标签）

**V0 Raw/Photometric**：原始 Blender 物理渲染+标准弱光照/颜色随机化；对照母体。  
**V1 Low-FDA**：整体视图使用 FDA 类低频幅度替换/插值，目标仅从真实 train frame 选择，保留源相位；更改总体风格，不声称迁移高频。  
**V2 High-PASTA**：不看真实图片的结构化高频幅度扰动，按频率增大扰动；检验「随机频率多样性足够」竞争解释。  
**V3 Real-HF**：**真实语义条件统计 + 保持合成相位**的局部高频幅度匹配，Pig / floor / bg 单独处理；并施加真实训练相机 profile 的图像退化。  
**V4 Hybrid**：三维真实纹理材质+真实低频风格+V3 高频+真实相机退化，原则是每层可关闭，不能把几个变化混在一起后宣称高频单独有效。

实现时拆出 `V3a=real-HF only`、`V3b=camera only`、`V3c=both`；这些是独立开关标识，当前不启动三臂实验。V0/V2/V3a共用基础相机映射和非研究变量。相机几何warp一旦变化就产生新geometry_hash，不能继续标成同几何视图。原PASTA是“高频扰动相对更强”，不等于严格只改高频；本地裁频带/局部语义版标作PASTA-inspired，待与[作者实现](https://github.com/prithv1/PASTA)核对后才可称原方法复现。

**公平数据量**：每个几何 scene_id 在每臂只采**一张**训练图；不同 variant 虽可离线生成配对 QA 图，训练时每 epoch 按固定 seed 调度，不能 V3 训练了 3×样本却拿 V0 当无额外数据对照。权威 scene 及 visible Mask SHA 逐一一致，除非对镜头几何有同样作用于标签的合法几何变换。

## 4. Real-HF 的可编码算法：分两支，不能只套整图 FFT

### 4.1 支路 A：真实皮肤微纹理 → Blender PBR（最推荐）
1. 提取 train-only 高纯度猪体 patch；剔除外轮廓、背景残留、强投影阴影，不直接贴整块 RGB。
2. 分离照明/低频颜色趋势与局部 Albedo 残差；记录纹理物理可比尺度（体表大约的长度与当前像素投影关系，不准确定时给保守尺度范围）。
3. 制作 tileable / UV blend 的贴图，采样 Pig 专用污渍/斑点、粗糙度、法线细节，随机旋转/相位/亮度，按猪体部位变更细纹方向可作为后续拓展。不能把地板 texture 映射到猪体。
4. 随相机投影自然产生 perspective shrink、遮挡、阴影和 specular；高清渲染再用相机 PSF→下采样，这通常比在最终图像复制静态纹理更接近真实成像。
5. **必要消融**：全程序性猪皮纹理 vs train-real UV patch 材质；若真实皮纹只导致对某个猪场肤色过拟合，保留 30–50% 完全随机材质样本作为候选。

单张压缩RGB不能唯一分离albedo、法线、粗糙度与光照；这里得到的是近似外观残差。法线/粗糙度优先用程序性先验，任何由RGB估计的物理材质参数须标为估计值，不能冒称真实PBR真值。

### 4.2 支路 B：类别条件「真实高频幅度谱」适配（容易先实现）

局部 patch 记为 S，真实训练类别桶聚合的目标谱为 A_real,c,bin(f)，源谱 A_syn(f)，源相位 φ_syn(f)。为避免目标图片的解剖轮廓偷偷进入合成标签，**不拷贝 φ_real**，只改变高频振幅。

    window_j = Hann_window_for_patch_j
    F_j = FFT2(window_j * (S_j - mean(S_j)))
    delta_log_j = log(A_real,c,bin + eps) - log(abs(F_j) + eps)
    gain_j = exp(alpha * smooth_band_mask * clipped_delta_log_j)
    gain_j[DC] = 1                         # 偶对称正实增益
    D_j = iFFT2(F_j * (gain_j - 1)).real   # 只重建频谱增量
    delta_image = sum_j(scatter(window_j * D_j)) / max(sum_j(scatter(window_j**2)), eps)
    delta_image[uncovered_pixels] = 0
    S_new = S + valid_interior_mask * delta_image

其中 **A_real 是从许多 train patch 平滑聚合的谱包络，不是某张真实猪照片的 FFT**。f_min/f_max 在最终 640 像素单位定义，首版可用 1/16 到 1/2 cycles/pixel 作为高频候选，alpha ∈ {0,0.25,0.5,0.75} 的候选要经 dev 冻结。实际实现需要 patch Hann window、overlap-add 平滑、避免局部相位/振幅数值爆炸，并将倍率（建议初始 0.5–2 的试验夹紧）记录到 manifest。色彩统计尽可能与频谱能量分离，避免整体变色。

**数值约定：** gain在每对共轭频率上相同，或用合法rFFT实现；alpha=0时gain恒为1，D_j恒为0，输出直接等于原S，不能用窗口化patch重建替换原图而留下Hann衰减。eps只用于对数/分母保护，不凭空向零能量频点注入eps幅度；这也意味着纯增益不能恢复原本严格为零的频点。真实/合成谱使用相同窗口、归一化与patch尺寸。未覆盖区域保留原图，不除以近零窗口。记录重建虚部误差、增益夹紧、RGB越界/裁剪比例。

**频率和空间的边界：** 内部soft mask、overlap-add与RGB裁剪会产生额外频率成分，不能声称最终图像严格只有高频改变。局部均值/低频漂移及边界带变化按预定数值容差核验；保源相位也不保证可见边缘位置绝对不变。频谱统计与变换必须处于相同采样网格；若改变分辨率，重新换算频带和统计，不直接复用640尺度参数。

**图像合成**：对 Pig/背景各有效内部区域分别求 S_new，再用 soft-eroded region mask 混合；在实例轮廓 3–7px 保护带内不使用真实局部 donor 谱或 donor residual，轮廓只受物理成像统一模糊/锐化影响。边界保护带的宽度应随最终像素尺度和 PSF 调整，不能用遮罩羽化制造「人工毛边」。

**更轻的备选实现（能先做 QA）**：
对 Pig/背景内部 patch 计算高通 H_syn = S - GaussianBlur(S,σ)，从类别/频带对真实 H_real 的稳健方差估计 target_std，生成

    H_scaled = H_syn * clamp(target_std/(std(H_syn)+eps), lo, hi)
    S_new = GaussianBlur(S,σ) + alpha*H_scaled + (1-alpha)*H_syn

此法只匹配高频能量而不复刻真实纹理，预测收益可能弱于 UV 真纹理，但可作为干净的同条件**能量匹配对照**。sigma 和 alpha 需要 dev 冻结。

### 4.3 支路 C：全图成像链（和贴图及 FFT 独立）

在高分辨率线性空间得到理想 RGB 后按顺序：
1. 光照/曝光/白平衡/色彩映射；
2. PSF/景深/运动模糊（必要时模仿 rolling shutter）、镜头畸变（几何变换**同步作用**于 Mask/depth）；
3. 像素积分/下采样到目标相机输出后再 resize/letterbox 到网络尺度，记录每一次采样与标签几何关系；
4. 亮度依赖噪声（近似 Poisson-Gaussian，只做经验一致不虚构相机 RAW 参数）、相机锐化/去噪、压缩编码/重采样；
5. 用该镜头条件下真实 train pixel edges 和 flat-patch 频谱统计对比；如真实部署视频本身低清/强压缩，在 pipeline 末端重现其伪影。

**禁止**：RGB 先镜头畸变而 Mask 不变；像素上混出明显双边界后沿用旧标签；训练图未压缩、真实测试图强编码差异却全靠随机加白噪声近似。

## 5. 高低频分别做什么：将假说与已有方法拆开

FDA（CVPR 2020）强调替换**低频**幅度使 synthetic 更接近 real 风格；PASTA（ICCV 2023）强调更强的**高频幅度扰动**提升 syn-to-real 域泛化。这里的 V3 则检验不同主张：真实皮肤/环境/摄像机**条件统计**是否比单纯高频随机化额外改善真实实例分割。三个不是互相替代的写法。

一个额外的**泛化护栏**：即使 V3 能拟合单摄像头的 PSD，也可能记住该摄像头压缩块和猪场纹理。故每张 scene 可以随机选择**不同 train camera profiles**、部分概率切回 V0/V2，并在独立 camera/pen 测真效果；是否做 profile 条件采样由数据中 camera 数决定，不拿 test camera 建 profile。

## 6. QA 自动化／退出判据（首版建议，非已冻结数值）

1. **来源检验**：patch_manifest 中所有 frame_sha 都归 train；若引用未知/测试源立即失败。
2. **标注一致性**：V0..V4 每个 scene_id 的 visible mask SHA 相同；镜头几何变换时需共同 warp RGB/Mask 且另记 geometry_hash。
3. **结构守恒**：相同场景下各视图 Mask/bbox/接触对数相同；对非几何纹理处理，edge displacement 必须来自真实物理图像退化而非复制 donor 源轮廓。
4. **真实纹理纯度**：随机抽检不同 pig patch 每种光照，mask-valid>=95%，无相邻猪/栏杆；出现大面积污渍/毛发 ghost 记 bug。
5. **频率统计**：按类别和相机输出 PSD band energy、IQR 对照和高频注入系数分布；监控 2–4px 高频能量是否过大、是否有异常 ringing，避免强度只为降低 PSD 距离而失真。
6. **低频保持**：V3 明确只改高频时，局部均值/低频照明不能有未登记的大偏移。V1 是低频变化的独立对照。
7. **图像保真审计**：随机而非挑选的至少约 50 个 scene 的 raw/variant/mask overlay，并标注穿模、漂浮、贴图重复、边缘光晕、可见 ID 错误比例。
8. **最重要的外部效用**：即使 V3 的 PSD 离真实更近，若在严格同数据同训练预算的 real dev 上没有 Mask AP/困难子集收益或反有损伤，就不晋级论文高频贡献。

## 7. 最小可实现命令合同（规划中的脚本，尚未写）

输入三件：synth RGB + visible instance-id mask + 真实 train bank stats；输出一件：**同 mask 的合成图变体** + view_manifest。算法建议先写纯 Python/Numpy/OpenCV/torch 可测试函数，不与 Blender 绑定。

    collect_real_patches(real_manifest, split="train") -> patch_manifest, bank_stats
    calibrate_frequency(bank_stats, group=(region_type,camera,lighting)) -> envelope
    synthesize_variant(image, instance_map, env_map, envelope, mode, seed)
          -> transformed_image, labels_unchanged, transform_manifest
    audit_variant(scene_id, view_manifests) -> errors, PSD_report, overlay_gallery

需要的测试：
- test_no_test_source_in_bank、test_repeatability_given_seed、test_mask_sha_invariant；
- test_foreground_background_band_isolation；
- test_fft_identity_at_alpha_zero、test_complex_inverse_real_numerics；
- test_intended_band_gain_and_bounded_low_frequency_leakage、test_no_nan_blowup、test_color_range；
- test_split_video_heldout、test_PSFWarp_mask_sync；
- test_donor_texture_not_containing_other_pig_edges。

## 8. 首个建议判别与可推翻性

**先做 V0、V2、V3 三组**，几何、可见实例标签、真实训练帧、训练步数全部一样。若 V3 相比 V2 只在已有 camera_id 变好、换 camera 退化，说明高频条件适配可能过拟合：下一版优先做 camera-profile 混合/随机程度加大，而不是加深原型头。若三者性能相同且 PSD 已接近，停止将频率迁移列为方法主贡献，直接保持最省成本的 V0/V2 进入原型阶段。

### 相关研究
- FDA, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- PASTA, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Chattopadhyay_PASTA_Proportional_Amplitude_Spectrum_Training_Augmentation_for_Syn-to-Real_Domain_Generalization_ICCV_2023_paper.html
- Learning Texture Invariant Representation, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Learning_Texture_Invariant_Representation_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2020_paper.html
- BlenderProc lens distortion: https://dlr-rm.github.io/BlenderProc/examples/advanced/lens_distortion/README.html

本文件是实现规范与待验证研究假说，不表示本线已有频谱统计、贴图库或实测结果。
