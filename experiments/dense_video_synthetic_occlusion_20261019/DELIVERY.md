# 交付与当前状态

- plan_status：generation_validation_complete。
- execution_status：completed；115、117—121成功，116环境安装失败保留。无本Study活动Slurm作业。
- scientific_status：合成几何与标签有效性在当前六段工程样本上通过；模型训练0步，真实视频效用未测。
- artifact_status：6段1600×900视频、标签解释图、场景总览、数据曲线、离线HTML和标准标签包已完成核验。
- transfer_status：原始399文件及额外15个标准导出文件大小/SHA核验通过，结果与展示保留桌面和服务器副本。

原始交付包：77,176,941字节；SHA256 bcd8bff6217d6fce525cfd3929c102279d606671b771f24e34f41f3486ea6502。

标准格式交付包：76,734字节；SHA256 6e9cb3a8f8c47a9c557d9f31f1a0d4ebb87e6e073dde15aafdcaca14cad183b6。

核验：6,912个几何点逐点对照、532个官方COCO消费者掩码还原、144个展示视频帧解码、19项HTML依赖均通过。三张主要图已视觉检查；没有声称HTML已经浏览器截图验收。

桌面展示：paper/presentations/三维遮挡场景与自动标签_20261019/index.html。

完整原始数据副本：C:/Users/Public/occlusion-scene-20261019/delivery/verified。

标准导出副本：C:/Users/Public/occlusion-scene-20261019/delivery/exports_verified/exports/v1。

服务器项目：/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion；原始数据位于本Study pilot Run，标准格式位于exports/v1，展示位于showcase/三维遮挡场景与自动标签_20261019。

当前推荐的唯一下一步：冻结多资产、多动作、按资产家族及场景分离的遮挡—重现训练数据协议。保留Blender生成路线，扩大真实外观/姿态变化后再训练遮挡状态与关系模型；该训练尚未启动。
