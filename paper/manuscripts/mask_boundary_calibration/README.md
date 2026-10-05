# RCMC 论文稿件

**Selective mask readout calibration through predicted correction benefit**  
中文意译：通过预测修正收益，选择性校准实例掩码的读出。

已完成英文研究稿，按 **Pattern Recognition Letters** 的双栏短文要求排版。当前是写作交付版，尚未投稿。作者、单位和通讯信息保留在独立标题页，未代填身份或声明。

## 阅读入口

| 文件 | 内容 |
|---|---|
| [main.pdf](main.pdf) | 英文正文，6 页，含图表与参考文献 |
| [main.tex](main.tex) | 可编辑 LaTeX 正文 |
| [supplement.pdf](supplement.pdf) / [源码](supplement.tex) | 7 页补充材料：推导、完整协议、全部 COCO 指标、误伤分析、实例图和速度 |
| [title_page.pdf](title_page.pdf) / [源码](title_page.tex) | 独立标题页及待填写的作者信息 |
| [highlights.txt](highlights.txt) | 4 条论文亮点 |
| [RCMC_PRL_manuscript_package.zip](RCMC_PRL_manuscript_package.zip) | PDF、可编辑源码、图表、参考文献及小型复现材料的完整包 |
| [VENUE_AND_FORMAT.md](VENUE_AND_FORMAT.md) | 期刊选择理由、格式来源与模板版本说明 |
| [EVIDENCE_MAP.md](EVIDENCE_MAP.md) | 论文主张与已完成实验的对应关系 |
| [reproducibility/README.md](reproducibility/README.md) | 冻结门控模型、相关源码与数据清单的使用方法 |

## 后续机制探索

[探索计划（2026-09-20）](MECHANISM_EXPLORATION_PLAN_20260920.md)：先区分错误前景的来源和阈值可分性，再按证据选择监督区域、邻居输入或 P/c 路径干预；计划的前两步已于 2026-09-20 在笔记本按固定环境启动；阈值面板已有结果，见[执行与首批发现](../../../experiments/mask_boundary_route_20260914/EXECUTION_MECHANISM_20260920.md)。新诊断不替代既有论文结果。

[固定边界的方法验证结果](../../../experiments/mask_boundary_route_20260914/RESULT_COVERAGE_CALIBRATION_20260920.md)：单 seed 的收益／覆盖代价分解对照已完成，未优于匹配直接回归与冻结 RCMC，按预定条件停止该方案扩展。

## 论文讲述什么

从“框已经找对、掩码也覆盖了自身，却因多余前景而达不到严格 IoU”的失败对象出发，我们冻结模型输出，检验现有 logits 的读出是否仍有改进空间。提高阈值能够修复一部分错误，但也会删除正确目标像素，破坏原本成功的预测。因此，方法的关键是判断一次具体修正的收益。

RCMC 先试着收紧掩码，利用原掩码形状和这次操作删掉的前景比例，预测该操作带来的 IoU 变化。训练时用 GT 监督这项收益；推理时只使用预测结果，收益非正则保留原掩码。本文贡献围绕受控诊断、选择性修正和修复／误伤的权衡展开。

主要结果来自 **4,500 张 COCO val2017 图像**：m 模型 Mask AP 43.38 → 43.94；冻结门控迁移到 s 模型，39.42 → 39.90。相较统一采用同一收紧操作，在固定匹配的 IoU 0.75 诊断中，误伤分别减少 22.74% 和 30.04%。这一比例是误伤数量的相对变化，不能当作 AP 增益；具体分母与限制见证据表。

## 修改和编译

在本目录执行 `./build.ps1` 可重新生成三份 PDF。脚本默认使用本机 MiKTeX，也可通过 `-TexBin` 指定其可执行文件目录。跨平台可依次运行 `pdflatex main`、`bibtex main`、两次 `pdflatex main`；补充材料和标题页各运行两次 `pdflatex`。应在本目录编译，保留 `vendor/`、`figures/`、`tables/`。

表格与图来自 `scripts/prepare_assets.py`，默认读取包内冻结的实验摘要，也能在项目内引用原始 Run。使用带 Matplotlib、NumPy 的 Python 运行即可重新生成。重新编译论文不需要 COCO 图片、模型权重或 GPU。`build/` 是本机编译和检查缓存，不作为论文原文，也不进入交付压缩包。

源实验保持独立：[`experiments/mask_boundary_route_20260914/REPORT.md`](../../../experiments/mask_boundary_route_20260914/REPORT.md)。本稿不改写历史运行，也不继承旧 CCL 稿件的主张。

