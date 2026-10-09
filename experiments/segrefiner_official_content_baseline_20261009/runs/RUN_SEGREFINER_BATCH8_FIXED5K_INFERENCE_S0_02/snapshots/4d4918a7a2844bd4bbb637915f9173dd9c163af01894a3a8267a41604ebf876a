# 执行前固定补充

2026-10-09，B已确认按作者公开声明训练来源开展官方LR强对照：锁定LVIS v1 train的100,170图与本COCO val5k的ID/URL交集为0；checkpoint只有455个float tensors、没有训练历史元数据，实际完整训练来源仍未知，不能声称已证明无污染。具体来源沿 [TRAINING_PROVENANCE.md](TRAINING_PROVENANCE.md) 和真实审计Run。32工程忠实/成本可行后自动固定5k，不再重复询问来源确认。

随机规则在工程结果前固定：base_seed=20261009，每图种子为base_seed+image_id。保留作者六步5→0与前五步torch.rand，不挑seed；按原候选有效/tiny顺序及batch_max32执行。逐图设置并保存所需CPU/CUDA RNG状态及种子证据，隔离/恢复外部RNG以便32工程、5k、失败重试和独立计时在同一图重现。本轮只是一条预注册随机推理，实现复现不构成跨seed能力结论。

适配限定framework registry/BaseModule/init_cfg、head/loss构造及GT-free图片/预测元数据入口；原作者U-Net、diffusion、crop/resize/paste和阈值方法保留并绑定原源码SHA。严格加载全部453 denoise与2 Sobel state键，不用strict=False吞未知键；FP32/eval/no_grad，初始化与加载状态、实际依赖/设备留Run。无需覆盖现有Conda或编译核心不需要的旧MMCV CUDA扩展。

工程必须分别证明源码核心未改，以及同设备/同权重/同输入/同RNG下原作者核心调用与适配入口的张量/最终mask一致；不能仅用适配自己生成的摘要证明忠实。normalization/RGB次序、maskbbox+pad20、256bilinear与>=.5、原half-pixel粘贴、area512/empty/tiny规则和valid/tiny ordinal恢复均核对；可使用明确GT-free合成fixture覆盖空/混合边界，但不冒充真实32样本覆盖或能力结果。

真实推理读取冻结normal原图RLE与同字节RGB/standalone图元数据及native资格，入口/依赖不打开GT JSON。无候选、无支持候选及全tiny时在外层保留原输出/身份并记录实际跳过；范围外候选不补位、不改bbox/class/score。返回全部原图CPU masks后再独立RLE、审计/写盘/评分，披露不能统一的计时端点或加载/采样开销，不加门控挽救工程质量。
