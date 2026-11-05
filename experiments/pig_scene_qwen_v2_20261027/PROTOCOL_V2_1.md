# V2.1输出适配修订：保留744失败，视觉定义与验收不变

父协议为[PROTOCOL_V2.md](PROTOCOL_V2.md)。743CPU预检过；744在首两图因模型输出wire格式不符而按冻结规则停止，69秒/单卡/0更新，真实Run `RUN_QWEN_SYSTEM_V2_9d6a85b0263e`失败，原文保留。没有结构回归结果，不能将它记为分类阴性或通过。

具体已观察缺口：state写成known、漏animal_overlap、relations用relation+有序evidence_ids。现在新增可审计适配器：仅known→inferred（不提升到observed），缺重叠→unknown，显式二元有序关系键名转换。不得修改任何结构类别、坐标、证据ID、占用数字或视觉描述；其它不合法仍失败。保留raw_model_object、canonical model_claims与逐项wire_adaptations；分别报告raw严格schema通过数和规范化记录通过数。证据编译/降级/ROI规则全部不变。

**提示词与视觉推理配置完全不改。** 同一模型/权重、同一500/40身份、同一RGB/greedy/像素/token预算；4个结构反例和4窄栏控制、32/40ROI门槛及未知/冲突/生产阻断全部沿父协议。CPU先对744两原文离线重放，核视觉value/全部坐标逐项相同，再跑同一批40图；新推理不冒充旧缓存复用，也不把适配器当视觉识别增益。

资源仍使用**本Study累计<=0.85GPUh**：已花744的69秒；修订正式作业最多40min，fresh最多5min，累计申请上界(69+2400+300)/3600=0.769167GPUh，峰值1；最多3个GPU作业（含失败744）。增加一次CPU预检，独立CPU复核不变。与原V2差异为增加一次工程修复作业并缩短正式时间；无借旧Run预算或科学Gate更改。若结构/ROI未过，保留阴性，停止自动扩量；不继续改prompt或追加模型。

除wire适配/Run记录/独立验证读取两个对象和协议SHA外，其它计算模块不改。源按新内容哈希版本保存，旧v2_f33332cbe62a和744不覆盖。独立人工准确率仍null，整个goal仍active。
