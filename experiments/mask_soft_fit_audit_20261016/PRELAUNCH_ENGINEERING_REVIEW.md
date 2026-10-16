# 启动前工程审查

第一次输入/源码冻结于2026-10-15T17:14:49.058488+00:00，尚未创建smoke或正式Run、未计算本checkpoint的训练集结果。启动前静态复查发现独立整数RLE核验器只接受JSON字符串，而pycocotools直接返回GT RLE时counts为bytes。补充ASCII解码，并增加同时覆盖bytes和字符串的工程用例。

首版FREEZE_MANIFEST、PREPARATION、INPUT_EXPECTATIONS、verify.py和test_statistics.py保存在笔记本`prelaunch_archive/`；重新运行相同预检后生成第二版冻结。科学协议、输入、模型、预测/评价实现、统计规则和预算未改变。该工程修订先于任何smoke或正式结果，旧冻结凭据完整保留；无训练或评价Run失败被覆盖。
