# 固定原型可恢复性：定义完备的复导（协议先行）

日期：2026-10-10。研究线：`neighbor_sensitive_p3`。本协议在执行前冻结。
动机：证据表 B1 的核心数字（6,459 个严格 Mask fail 中 4,263 个可恢复）其**语义与聚合已精确核验**，
但原始求解脚本未随回传保留、oracle 目标的确切形式未能复算（见
`experiments/argmax_maskfail_position_oracle_20261005/VERIFICATION_20261010.md`）。
本 Study 用**项目已记录的 7D 定义**完整重解该面板，使该数字成为无披露负担的可复现事实。

## 1. 面板

- 对象：`argmax_maskfail_position_oracle_20261005` 的 6,459 个严格 Mask fail GT
  （argmax-correct + Box75 + 整个 argmax-correct 集无 Mask75）及其 28,446 个 raw 事件。
- 候选身份沿用原面板：`(image_id, annotation_id, raw_id)`；每 GT 的 raw 清单与 letterbox 几何
  取自 `RUN_full_local_cache` 的 MANIFEST.json 与 per-image NPZ（proto 32×112×160、
  coefficient 32、box_input（448×640 输入像素坐标）、input_shape、original_shape）。
- GT mask：COCO val2017 官方 annotation RLE 解码（原始尺寸），与面板身份一一对应。

## 2. Oracle 定义（与 7D 完全一致，零自由度）

对每个候选，解

    Δc*_λ = argmin_d [ BCE(P(c0+d), y)/area + λ‖d‖²/2 ]， λ = 0.003

- ROI：**GT 框**在解码网格上的 crop（letterbox 几何由 NPZ 的 input_shape/original_shape 重建：
  gain=min(ih/oh, iw/ow)，pad top=round((ih−nh)/2−0.1)、left=round((iw−nw)/2−0.1)；
  GT 以 nearest 插值到解码网格）；
- 求解：LBFGS 120 次迭代、strong_wolfe、double 精度、从 d=0 出发（7D `finite_oracle.py` 惯例）；
- 记录每候选：Δc*、‖Δc*‖、目标值、stationarity 残差、迭代次数；
- **不做**支撑截断、支持约束或任何原运行特有的变体——本复导只实现已记录的 7D 定义。

## 3. 派生量

- 每候选：`oracle_iou`（用 c0+Δc* 在**预测框**内官方 >0 阈值解码与 GT 的 IoU）；
- 每 GT：`oracle_best = max_raw oracle_iou`；
- **可恢复定义（与原面板语义一致）**：`oracle_best ≥ 0.75`；
- 另报：每候选 `baseline_iou`（c0 同法解码）用于内部对照；
- 分层：small/medium/large（按 GT area <1024 / 1024–9216 / ≥9216）、pyramid_level。

## 4. 对照与判读（不调参）

- 主对照：本复导的可恢复 GT 数 / 6,459 与记录的 4,263 / 66.00% 之差；
- 同时报 per-stratum 可恢复率与原面板对照（small 2,285 / medium 1,237 / large 741）；
- 若复导数 ≥ 记录数：记录差异并披露（更完备定义下界更高）；
  若复导数 < 记录数：记录差异并披露（原运行含未记录约束/或上界混入，见 §2 披露的
  309 个 fixed_support_limited 行的 oracle 列记支撑上界一事）；
- **不**为使两者一致而调整 λ、ROI、迭代或阈值；差异本身如实报告。

## 5. 核验

独立核验（verify 阶段）：每第 7 个候选换随机起点重解，核对解与目标值；
聚合（max/best/分层计数）从逐候选产物独立重算；身份集合与 MANIFEST 核对
（无丢失、无重复）；bootstrap 不需要（本 Study 是确定性重解）。

## 6. 边界

- 复导值是“已记录 7D 定义下”的可恢复性，不是原运行的确切目标；两者差异如实披露。
- oracle 是 GT 辅助诊断，不是方法结果；面板为历史 val2017（非盲测）。
- 无任何 AP 声明。

## 7. 执行安排

笔记本 `28358lan`；执行根 `D:/coco_wire/experiments/coefficient_recoverability_rederive_20261010`；
缓存 tar 从本地 `RUN_full_local_cache.tar`（5.02 GB）传入 `cache/` 后解包。
Run 链：smoke（8 图）→ 正式（2,462 图 / 28,446 候选）→ 独立核验 → 报告与对照。
每 Run 保存 run.json、源码快照、环境快照、manifest.sha256，完成后回传本项目并附 transfer.json。
