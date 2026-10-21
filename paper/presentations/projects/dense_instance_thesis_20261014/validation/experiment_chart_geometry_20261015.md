# Experimental chart verification

Source values: `experiment_figures_20261015/chart_data.json`, extracted from completed runs with SHA-256 provenance. DrawingML uses editable SVG primitives, not native chart workbooks. Independent standard plotting outputs use the same source values.

The PPT Master position calculator was run after authoring each chart family, using the displayed value-axis range and the declared plot frame. Category spacing for the three bar charts was matched with explicit gap widths. Interval plots verified the mean and both interval endpoints. Values are unchanged; coordinate rounding is below 0.001 px.

verify-charts: 07_失败定位.svg | object=failure-split | type=horizontal-bar | mode=direct-calc | scale=0–8000 from ticks | calc=ran | svg=verified

verify-charts: 08_修正空间.svg | object=oracle-area | type=horizontal-bar | mode=direct-calc | scale=0–100% from ticks | calc=ran | svg=verified

verify-charts: 09_自动修正探索.svg | object=pilot-effects | type=scatter-with-intervals | mode=decomposable-calc | scale=−4..2 pp from ticks | calc=ran for all 9 coordinates | svg=verified

verify-charts: 10_面积分层.svg | object=area-damage | type=horizontal-bar | mode=direct-calc | scale=0–16% from ticks | calc=ran | svg=verified

verify-charts: 11_风险因素对比.svg | object=adjusted-effects | type=scatter-with-intervals | mode=decomposable-calc | scale=−5..20 pp from ticks | calc=ran for all 6 coordinates | svg=verified

Bars start at zero. The one-instance baseline category retains its true 0.0675 px bar width and an explicit label; it is not inflated for visibility. The oracle proportions and damage proportions are descriptions of their fixed samples. The pilot intervals are original S2 image-macro intervals; adjusted contrast intervals are the separately specified eight-comparison-corrected intervals. These populations and interval constructions are not pooled.
