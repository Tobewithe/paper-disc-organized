# 关机接续记录

用户2026-10-19通知机房电脑将关机。本轮服务器训练、23段×8组评价和补充裁决均完成，无待训任务。

- 服务器：a25023104@192.168.71.10，项目 /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2。训练Slurm145，评价149，补充裁决150均成功。
- 本地完整回传：experiments/dense_video_occlusion_transfer_20261019/verified_delivery；1660文件SHA核验通过，RESULT_RECEIPT与TRANSFER_VERIFICATION在Study根目录。归档在C:/Users/Public/occlusion-training-20261019/result-delivery-v1.tar.gz。
- 模型三臂各2000步，87877参数；合成FULL F1=.904537。真实11段FULL IDF1=67.807643，C=67.544717，R=71.978721，新增+.262926，CI[0,.499987]；仅duck_3 +.647179，其余10段不变。SHUFFLE不劣，创新机制未成立，当前加权配置不扩训。
- duck_3正确补接GT74：967→1124→1182接回967，新增269 IDTP；BoT-SORT也已保持，不称完全遮挡恢复。CASE_ATTRIBUTION.json保存一对一GT核验。
- 物理补充层以adjudicate_pointer的SUMMARY为准：41/132、严格0/11。原149的54/132口径作废，主指标不变。
- REPORT.md、MODEL_AND_NOVELTY.md、CONFIGURATION_REVIEW.json已生成。
- 本地展示目录：paper/presentations/可见性学习与真实视频身份恢复_20261019。若最后渲染被关机中断，用conda pytorch Python依次运行本Study的build_videos.py、build_casefigure.py、build_showcase.py；这些仅做本地可视化，不重训或改结果。需要把showcase状态及研究线README最后更新为完成，再交付用户。
- 唯一后继研究建议：冻结跨遮挡身份正负对与拒接监督的新数据合同；没有启动新训练。

关机前最终确认：本地两个视频、关键帧对照图及展示页面均已生成，所有本地渲染进程已正常结束；服务器squeue为空。Study与研究线状态已更新，不需要为补做展示而重训。
