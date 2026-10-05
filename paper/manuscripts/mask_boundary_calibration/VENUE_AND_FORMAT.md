# 目标期刊与格式来源

本稿选择 **Pattern Recognition Letters（PRL，Elsevier / IAPR）** 作为格式和论述长度的目标。选择依据是它接收模式识别领域的简洁方法研究与实证研究，当前工作具有明确的实例分割对象、受控干预、轻量监督方法及跨模型尺度验证，适合组织成方法与诊断并重的短文。IAPR 的[刊物介绍](https://iapr.org/publications/)及[PRL 介绍](https://old.iapr.org/publications/prletters.php)支持这一范围判断。

这是一项基于稿件内容的适配判断，不是录用概率预测。当前方法对固定平滑对照的额外 AP 收益较小，且有推理成本，编辑和审稿人仍可能要求更广泛的比较。稿件通过明确增量、报告误伤控制和解释可修正机制来建立价值。

## 分区信息的口径

截至本次查询，[釜山大学 Scholar 的期刊目录](https://scholar.pusan.ac.kr/journals/17020/)转引的 **2025 JCR** 数据将 PRL 列为 Computer Science, Artificial Intelligence 的 **Q2**。这是高校目录转引，不是本项目直接访问 Clarivate 的查询；Scopus 分区与 JCR 分区不能混用。本次没有核实中科院分区，因而不把它表述为“中科院二区”。最终采用哪一年、哪套分区，以作者单位的认定口径为准。

## 已核对并采用的格式

出版方的 [Authorship and Formatting Guidelines](https://legacyfileshare.elsevier.com/promis_misc/patrec-authorship-and-formatting.pdf) 已保存为 [`vendor/patrec-authorship-and-formatting.pdf`](vendor/patrec-authorship-and-formatting.pdf)。按该文件的可核对要求组织：

| 要求 | 当前稿件 |
|---|---|
| 初投稿正文最多 7 页，包含图表与参考文献 | 6 页 |
| 摘要最多 200 词 | 176 词，按空白分词计数 |
| Highlights 为 3–5 条，每条不超过 85 字符 | 4 条，分别 74、65、78、78 字符 |
| 双栏，作者—年份引用 | 已采用 |
| 补充材料另交 | 独立 7 页 PDF |
| 作者与单位信息 | 独立标题页，未知信息留空 |

最新网页 [Guide for Authors](https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors) 在本次访问中返回 403，无法逐项读取其动态页面。因此这里明确列出已实际核对的出版方 PDF 要求，不声称核实了不可访问网页中的每一项。

## 模板版本与改动

- 基础类文件是从 [CTAN elsarticle 官方分发](https://ctan.org/pkg/elsarticle)取得的 `elsarticle` 3.5（2026-01-09）；原始分发包保留在 `vendor/elsarticle.zip`。
- 上述出版方格式 PDF 指向 `prletter.sty`。本次未取得该文件，采用的是带 Elsevier 版权和 LPPL 许可的 **2013 年 `prletters.sty`** 公开副本，其来源和版权见 [`vendor/README.md`](vendor/README.md)。两者名称不同，不能声称拿到了最新期刊样式。
- 双栏正文的尺寸、Times 字体与图表样式来自该期刊旧样式；`draft_format.tex` 单独适配现代类文件的标题、摘要和关键词，去掉未发表稿件不应虚构的收稿日期、出版商标识和版权行。旧样式文件保留原文。
- 因此，本交付是**满足已核对篇幅要求的 PRL 双栏研究稿**，不是出版商生成的最终出版版。正式投稿时可用编辑部提供的最新模板重新排版；正文、图表、BibTeX 均已分离，便于直接迁移。

生成式 AI 使用按 [Elsevier 当前政策](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) 在方法、示意图说明和声明中披露。作者最终责任声明尚未代签。文件准备与排版不构成正式投稿。

资料查阅及稿件整理：2026-09-15 至 2026-09-16。
