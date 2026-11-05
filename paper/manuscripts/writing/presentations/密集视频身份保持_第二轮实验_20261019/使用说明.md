# 第二轮成果使用说明

双击本文件夹中的 `index.html`，可离线查看结果、切换数据集、播放四段完整对照视频。图表、数据和视频均为本地文件，不依赖互联网。若应用不支持HTML预览，可直接打开MP4、PNG或REPORT.md。

推荐汇报顺序：先看斑马1/2的CGS-T独立确认（IDF1 43.41→50.63），再看同容量正确性对照51.87与Deep OC-SORT50.54，说明目前新增监督贡献尚未建立；最后展示horse_3和zebra_4反例，说明为何继续聚焦错误补接。

- `REPORT.md`：完整实验链路、全部成绩、原因分析与研究决定。
- `comparison.png`、`per_sequence.png`、`topology.png`：真实数据图。
- `architecture.svg`：可编辑矢量方法架构。
- `annotations/`：现场稀疏身份、GT间断原因及原图证据。143条可用标签为一次AI视觉复核，不是人工密集GT。
- `videos/`：四段1600×900、15fps的配对视频，指标仍由全部原始帧计算。
- `results.json`、`scores.csv`：全部分数和来源Run。

完整执行工作区在笔记本 `D:\thesis-video-v2-20261018`，共同原模型和数据仍引用 `D:\thesis-video-20261018`。请保留两个目录。

新视频分析后端已经用15帧样例真实跑通，页面/状态/视频接口通过进程内测试。后台网页启动和SSH转发被自动审批拦截，返回“blocked by policy”；浏览器另不允许file协议自动预览。未声称第二版实时网页已启动，未绕过限制。本地文件可直接查看。

若要在笔记本手动运行网页，在其终端执行以下命令，再在笔记本浏览器打开 `http://127.0.0.1:8872/`。此命令不会新增训练；上传的新视频最多分析前30秒，并保存到本机jobs目录。终端关闭后服务结束。

```powershell
Set-Location 'D:\thesis-video-v2-20261018\code'
& 'C:\Users\28358\anaconda3\envs\pytorch\python.exe' -m shared.dense_video_identity.review_server --root 'D:\thesis-video-v2-20261018' --port 8872
```

当前没有后台训练任务；当前候选的固定配置已结案，后继身份层恢复实验尚未启动。
