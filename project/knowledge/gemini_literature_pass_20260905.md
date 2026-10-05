# Gemini 文献检索补充批次（2026-09-05）

## 结论

本批次确认 Gemini CLI 可认证，但不能完成 ARIS 文献检索：简单模型回显使用
`gemini-2.5-pro` 成功；进入文献搜索时默认尝试读取不存在且位于工作区外的
`scholar-megasearch` skill，或在受限无工具提示下超时。故本批次 Gemini 的**可核验命中数为 0**，不能把 0 当作文献不存在。

四组查询的完整命令、输出和失败状态见 `raw-search/gemini_pass_*.txt`。关系组和猪/动物组返回了模型记忆候选，但未联网核验；其中关系组把 arXiv:1906.05896 标成错误标题，说明不能直接引用。所有候选均保持 `unverified-model-only`，未写入主登记册。

## 调用审计

| 组 | 查询 | 调用 | 状态 | 可核验命中 |
|---|---|---:|---|---:|
| dense | dense crowded occluded instance segmentation touching overlapping objects | 2 | skill path error；90s timeout/abort | 0 |
| fragmentation | instance fragmentation low IoU partial masks duplicate proposals query/candidate competition | 1 | 20s timeout/abort | 0 |
| relation | global consistency instance segmentation relation reasoning occlusion ordering layering set selection | 1 | model-only output；未联网核验 | 0 |
| pig_animal | pig livestock animal instance segmentation crowded evaluation | 1 | model-only output；未联网核验 | 0 |

模型路由测试：`gemini -p "Return exactly: GEMINI_CLI_OK"` 曾触发默认模型不存在（404），但仍输出回显；显式 `-m gemini-2.5-pro` 的简单回显 exit 0。`gemini-2.5-flash` 与 `gemini-2.0-flash` 被本地路由改写到不支持的 `gemini-3.5-flash` 并失败。上述均已在会话命令输出中保留。

## 与当前研究的关系

Gemini 本批次没有新增可引用证据，因此机制 gap 仍以已有核验条目为准：BCNet/OCFusion/ORM/Layering/Relook 等遮挡关系与层建模，Piglet-CclusNet/猪群 panoptic 等动物场景方法，以及 OVIS 等遮挡评测基准。当前项目的 7,374 GT、19.84% 失败率、`SPLIT_TYPE` 50 个样本、完整/局部候选共同存活 31/50 仍是项目内部描述性证据；文献尚未证明其为主导因果机制。

## 后续处理

- 不把本批次的模型记忆候选写入 `literature_registry.jsonl`。
- 若要获得 Gemini 贡献，需要修复 CLI 的 `scholar-megasearch` skill 路径/安装，或提供可用的 Gemini 搜索 MCP；修复后按同四组查询重跑并保留原始输出。
- 在修复前继续使用上一批已成功的 arXiv、Semantic Scholar、DeepXiv、OpenAlex 结果；Exa、Scholar Feed、Zotero、WebSearch 的不可用状态按既有日志处理。
