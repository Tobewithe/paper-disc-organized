# 机制论文稿件计划（PRL 方向，已定）

日期：2026-10-10。状态：**方向已定，不再摇摆**——投 *Pattern Recognition Letters*（短文），
复用 `paper/manuscripts/mask_boundary_calibration/` 已建好的 PRL LaTeX 模板与构建脚本
（双栏、`build.ps1`、`vendor/`）；若被拒，整体转投 *Pattern Recognition* 正刊（扩写版），
不做“顶会/期刊”之间的反复比较。方法线已按证据关闭，本文只写证据支持的结论。

## 稿件主张（一句话）

> 在原型式实例分割器（YOLO26m-Seg / COCO）中，严格 argmax 条件 mask 失败的
> **66.0% 可在冻结原型空间内由系数修正达到 Mask75**（可恢复），但原生无 GT 预测
> **没有实现**它；我们排除七种候选机制（位置选择、监督量、归属读出、归属/solver、
> 共享末层、外挂全局修正、oracle 蒸馏、受约束功能投影），把瓶颈定位于系数实现的
> 函数约束/参数化，而非信息可用性。

标题候选：*Recoverable but unrealized: the mask-quality gap of prototype-based instance segmentation*。

## 结构（PRL 短文，约 8 页双栏，3 表 1–2 图）

| 节 | 内容 | 绑定证据 | 图表 |
|---|---|---|---|
| Abstract | 可恢复 66% vs 未实现；七项排除；含义 | B1, C1–C7 | — |
| 1. Introduction | 现象与重要性；本文问题；三层记账法的引出 | A1–A4 | — |
| 2. Related work | 原型-系数范式（YOLACT/CondInst）、错误分析（TIDE）、边界细化（PointRend/RefineMask/Mask Transfiner）、修复诊断；新颖性边界只声明“记账+排除”，不宣称结构首创 | — | — |
| 3. Diagnostic framework | 四层评价单位与身份规范（raw/argmax/输出/实例）；Box75/Mask75 定义；strict Mask-fail 定义 | A1–A3, 评价体系总览口径 | 表 1（记账总表） |
| 4. Recoverability | 6,459 严格失败；66.0%（复导数）；规模分层；位置选择 1.293pp；oracle 定义与口径披露 | B1–B4（B1 复导落地后更新） | 表 2（可恢复性分层） |
| 5. Exclusion chain | 逐项：位置选择（B3）、监督量（C1）、归属读出（C2）、外部空间经 solver（C3）、共享末层 46.8% + Phase B 实掩码（C4）、外挂修正 +0.092pp（C5）、oracle 蒸馏不迁移（C6）、受约束投影 30%<随机 59%（C7） | C1–C7 | 表 3（排除链汇总：结论/关键数字/边界） |
| 6. What this implies | 修复空间上界（只删误报 +10.999 AP）、前景占比分层（+30.38pp）、解码对照（−0.901）；指向表示/参数化；对后续工作的判别价值 | A5–A7 | 图 1（排除链示意，可选） |
| 7. Limitations | 单模型单数据集；诊断非 AP；复用 val 非盲测；诊断关联≠COCO 匹配；oracle 口径 | — | — |
| Conclusion | 收敛到“可恢复但未实现”与研究指向 | — | — |

## 直接可用的写作资产（均已核验/已落盘）

- 证据表：`EVIDENCE_MAP_MECHANISM_20261009.md`（27 条编号主张，每条带来源与口径）；
- PRL 模板与构建：`paper/manuscripts/mask_boundary_calibration`（main.tex/supplement.tex/build.ps1/vendor）；
- 表 1/2/3 的数字全部来自已冻结证据，图表由既有 SUMMARY/报告数字直接生成（无需新计算）；
- 复现附录：每个数字的 Run 路径已在证据表内，稿件附录原样引用。

## 写作顺序执行状态（2026-10-10，全部完成）

1. ✅ main.tex 完整初稿：PRL 模板、7 节 + 复现声明、式 1、三表、引用 7 篇齐备；
2. ✅ 表 1 三层记账（A1–A4 全量：raw 五类、argmax 五类、输出保留、基线）已填，分母口径已注明；
3. ✅ 表 2 可恢复性（B1 双口径：复导 4,843/6,456=75.0% 为主、记录 4,263/6,459=66.0% 披露）已填；
4. ✅ 表 3 排除链（C1–C7）已填；第 5 节七项正文与表 3 数字口径一致；
5. ✅ B1 复导完成并核验（7/7），差异按协议两值报告；
6. ✅ 全部证据占位标记清零；终版编译通过（含 bibtex，4 页）；
7. ✅ 复现声明成文（Run 追溯、证据表机制）。

## 剩余（投稿前，非科研）

- ✅ 内部评审轮完成：`INTERNAL_REVIEW_20261010.md`（16 项数字零误差、主张无越界、5 项审稿人风险预案）；
- ✅ 投稿材料：`cover_letter_draft.md`（附函草稿，待作者信息）、`title_page.tex/.pdf`（独立标题页，编译通过）、`build/`（main.pdf + title_page.pdf）；
- ⬜ 作者/单位/通讯信息填写（不代填）；
- ⬜ 系统提交。

## 未决项（不影响开写）

- B1 复导最终数字（RUN_REDERIVE_S0_R1 运行中，约 90 分钟）；
- 图 1 是否需要（视篇幅，PRL 短文可只用 3 表）；
- 作者信息（沿用 RCMC 稿的独立标题页约定，不代填）。

## 明确的“不做”

- 不追加方法候选、不补新实验来“加强”本文（除 B1 复导外）；
- 不为凑 CVPR/ICCV 叙事重排主张；
- 不混合旧 P3 干预稿或 RCMC 线的主张与数字。
