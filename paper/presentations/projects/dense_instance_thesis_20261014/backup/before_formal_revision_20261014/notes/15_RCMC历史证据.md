这页只说明已有原型可以作为实施起点。在 COCO val2017 相同的四千五百张图和三万二千八百三十一个普通 GT 的旧协议下，m 模型 Mask AP 从基线四十三点三八，到固定平滑四十三点八三，再到选择性响应门控四十三点九四；s 模型对应为三十九点四二、三十九点八零和三十九点九零。

选择性方法同时减少了修复和误伤，不能只报告其中一项。m 的修复从九百零七降到八百零一，误伤从五百七十六降到四百四十五，净修复由三百三十一升至三百五十六；s 的对应净修复由二百八十五升至三百四十。这些是固定匹配的 Mask IoU 零点七五诊断，不是 AP 的样本分母。

结果来自一次门控拟合，验证数据曾参与早期探索，AP 未给置信区间。因此它只能支持所测条件下的原型，不能证明新增邻居信息、密集子集或动物场景已经获益，也不能与原生五千图的基线串表。

m 的源 Run 为 RUN_e0e0d46defb74e3d9a90094f49ce26fe，s 为 RUN_f9df48632ac742248343184263262429，均在 C:/Dpan/codexproject/paper-disc-organized/experiments/mask_boundary_route_20260914/runs/ 对应目录中；冻结摘要副本位于 paper/manuscripts/mask_boundary_calibration/reproducibility/evidence/ 下同名 Run 目录。