# LEA-v1导入隔离工程恢复：科学协议不变

原smoke RUN_LEA_SMOKE_6b56df7df3d64803bc21c2824fbced70 完成4图/4配对、706原生RLE精确重放、参照重放、全部支持域/对照构造和完整COCO损伤计算，最后调用独立核验器时报 TypeError: verify() got an unexpected keyword argument 'replay'。

原因是新脚本使用通用模块名 verify，而依赖路径中旧修复—保护Study也有verify.py；实际导入旧模块。原Run保持error/blocked_engineering，原错误、代码、源快照、统计和所有输出完整保留，不回写为成功。

恢复只隔离模块名：lea_run_v2.py使用lea_verify_v2.py；lea_common_v2.py只增加V2冻结文件及恢复Run/控制器身份。原科学计算lea_ops.py、lea_stats.py、lea_eval.py以及协议、CONFIG、输入/样本/区域阈值、随机种子、门槛与预算逐字节保持。lea_run_v2.py相对原run_audit.py仅有两个import替换；lea_verify_v2.py相对原verify.py仅替换common导入。

新注册SMOKE_RECOVERY Run复制原smoke科学产物并验证SHA，不重新生成科学结果；先用正确核验器独立复算，再独立重放原生区域定义。新恢复Run和旧失败Run分别保留。只有恢复核验通过才启动唯一正式Full Run。恢复不增加阈值、优化求解或训练预算，未以smoke科学读数修改设计。

新增导入测试确认实际模块文件路径、函数签名、科学代码字节不变及import-only差异。V2独立冻结早于恢复与正式执行。旧单次Windows任务保持停用，恢复使用独立一次性任务；不自动重试任何科学失败。
