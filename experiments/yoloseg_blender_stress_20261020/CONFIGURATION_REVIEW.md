# 首轮完成后的配置复核

Run `RUN_YB_SMOKE_c6f0333ccc994518bdf0790b52265526` / Slurm162。正常退出0；冻结源文件和外部输入在runner首尾核验，模型原生参数与实际runtime在本机回传后再核。所有列出的核验通过，范围如下。

- Blender4.2.1 LTS，384帧/8段，640×384；每帧128点，共49152次几何BVH与独立scene.ray_cast归属一致。原标签未降采样放大；可见⊆独立投影、表面三角形身份、相机投影检查在生成时逐帧执行。
- Ultralytics8.4.100，Torch2.5.1+cu124，torchvision0.20.1+cu124，numpy2.4.3；8个关键Ultralytics模块的源码SHA与冻结pins一致。软件版本以ENVIRONMENT实际值为准。
- 权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；FP32，imgsz640，batch16，rect=true，end2end=true，agnostic_nms=false，max_det300，conf=.001、iou=.7，原生验证decoder。全部native_args及runtime_policy与PROFILE逐键相同。
- **实际模型输入为416×672，24个批次均一致。** 这是被冻结的原生验证矩形批次预处理结果；640是imgsz配置值，不应把它误写为实际640×640张量。原图仍640×384。本轮没有为了方便改batch、填充或解码。
- 33967条完整预测保存；384帧zero-delta原图RLE精确一致。展示与诊断从保存预测固定取score>.25，没有再次推理或阈值搜索。P/c0/输出身份和每帧输入张量散列保留在NATIVE_IDENTITY.jsonl。
- 实际单卡A6000 UUID：`ab05fc6f-e5a0-21a3-769f-b01a8e62e605`。Slurm分配1卡/4CPU/16GiB，用时15:11；已结束释放，无自动重新排队或训练。
- 本机59项返回文件SHA与原始ARTIFACT_MANIFEST相同；包含完整预测、逐帧账本、摘要、配置、全部8段预览及40张固定时刻截图。完整清单849项，其余原始场景/图像/标签仍在服务器。CPU复算8段检测/错类/分裂/混入计数及IoU总和，384帧/输出ID/配置一致性通过。稀疏点时间翻转未独立本机重算，不把部分复算声称成所有字段完全独立复核。
- 三组控制的80%可识别性Gate失败、交叉控制通过，数值与失败结果保留。工程成功不等于合成域或机制Gate全部通过。真实视频迁移、旧G/LCF比较及COCO AP均未执行。
- stderr存在PNG重复eXIf元数据警告；任务退出0，完整图像/推理数量及回传SHA通过，未发现致命执行错误。

依据：同Run ENVIRONMENT.json、native/NATIVE_ARGS.json、NATIVE_IDENTITY.jsonl、NATIVE_REPLAY.json、dataset/RENDER_SUMMARY.json；本Study LOCAL_RESULT_VERIFICATION.json。
