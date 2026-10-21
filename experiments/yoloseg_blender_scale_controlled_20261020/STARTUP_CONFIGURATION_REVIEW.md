# YB2启动核验

- Slurm **164**，`YOLO-blender-v2`，个人账号a25023104；1×RTX A6000、4 CPU、16GiB、2小时、no-requeue。训练0步；未触及154/160/161等既有任务。
- 预冻结协议SHA `b20d158036b716eeac0aab76a9e8b5e796f769a6d33e07f5bb0adba9e89d0069`。
- FREEZE SHA `ae51089783babd047cc2444e53b8f6f77df024eaccf57cd0cad35ad48ef828ab`，19个执行/科学文件、20项外部输入；本地19项逐字节一致。
- smoke `RUN_YB2_SMOKE_4631088af30d47d4a42c320a879c6c21`：16帧、48实例观测、427条原生输出，全部zero-delta RLE通过；2,048次场景API射线交叉核验。最大居中投影偏差0.0000329426356984186px，焦点amodal精确一致。
- 45个正确匹配掩码IoU及1,202个表面点命中、48项投影、本机时序/配对复算通过；完整83项产物、22,039,603字节回传SHA通过，含原RGB/GT和Blender场景。
- 第一seed的分散smoke：中心4/4、全体12/12正确识别，真实画面已检查。此为工程抽查，不替代正式576帧的适用性裁决。
- 实际Ultralytics8.4.100、torch2.5.1+cu124，GPU UUID `ab05fc6f-e5a0-21a3-769f-b01a8e62e605`；所有111项NATIVE_ARGS与V1无差异。FP32、O2O、batch16、rect=True、实际输入416×672、原图384×640、padding(16,16)，profile所有显式参数一致。原profile的COCO预算字段改为本Study的16/576，不改变native_args/runtime。
- 正式 `RUN_YB2_FULL_9154f356ba5f435f8c73db254817abb3` 于2026-10-20 17:15:37 +0800开始。此文仅为启动快照；科学结果仍待完整评价与复核。

证据：[原样本机核验](LOCAL_SMOKE_VERIFICATION.json)、[回传核验](SMOKE_TRANSFER_VERIFICATION.json)、[smoke Run](runs/RUN_YB2_SMOKE_4631088af30d47d4a42c320a879c6c21/run.json)、[协议](PROTOCOL.md)。
