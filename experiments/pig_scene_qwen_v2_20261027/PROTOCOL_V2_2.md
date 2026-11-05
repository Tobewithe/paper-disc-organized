# V2.2：在生成时约束Schema，停止逐格式补丁

父版本为[V2.1](PROTOCOL_V2_1.md)与[V2](PROTOCOL_V2.md)，所有原门槛保留。746出现多种关系别名和无效JSON，已完成12回答为4有效/8失败后提前停止，实际380秒；Run `RUN_QWEN_SYSTEM_V2_857d8bff7a4c`保留failed和原始文件。它否定“少量wire别名适配足以可靠衔接复杂结构记录”的工程主张，未裁决视觉分类/局部效用。

**本版主判断：必须在自回归生成时约束合法语法，而不是继续列输出别名。** 使用[LM Format Enforcer原始实现](https://github.com/noamgat/lm-format-enforcer)的字符解析器与Qwen tokenizer前缀树，经Transformers `prefix_allowed_tokens_fn`过滤每一步可选token。它不是视觉模型，不改变仅Qwen视觉模型约束。依赖固定lm-format-enforcer0.11.3/interegular0.3.3/pydantic1.10.24，装在个人项目独立_vendor目录，不改原venv或共享环境，128文件/8463780字节留SHA。

保留同一9B/2B、40图/500全图缓存、RGB、prompt、static schema/定义/证据编译/ROI与全部回归门槛。新增grammar约束会影响允许的token和条件生成路径，因此不能说视觉输出一定与旧版相同；它的收益只先验保证可解析的允许语言，**不保证语义、坐标、证据关联正确**。raw后仍做完整jsonschema和证据降级。Schema中数值/唯一性等若库未完全实施，以后验校验为准，不能用库保证代替真实检查。

限制连续空白<=8、按required字段顺序，帮助避免格式性游走；仍greedy/2048新token/524288像素，无训练。CPU用真实Qwen tokenizer逐字符/逐token重放一个合法全对象、验证EOS只在对象完成后可用、拒绝markdown；过后才GPU。tokenizer前缀树每次模型加载只建一次，各图独立parser。缓存新增grammar/依赖清单/torch/CUDA绑定，旧不受约束缓存不混用；fresh实际受约束forward。

**累计资源上界仍0.85GPUh：** 74469秒+746380秒+新主34min+fresh5min=(449+2040+300)/3600=0.774722GPUh，峰值1，最多4个GPU作业（含两个失败）。新主内部分析上限1900秒给编码/报告留空间。CPU预检<=5min，独立CPU<=5min。没有扩大原总GPU小时；没有借旧Run预算。若本版仍有格式失败或结构/局部门槛未过，停止新增GPU重试，保留结果并转具体证据复核。

比较和停止/阳性/信息不足沿V2；4个已知结构反例和4个窄栏对照不改，至少32/40局部覆盖不改。未知不作负例/460未分析不传播/生产0/独立准确率null/原goal仍active。
