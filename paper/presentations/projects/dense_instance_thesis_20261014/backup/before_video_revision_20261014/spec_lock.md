<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 导师与毕业论文汇报评审
- objective: 解释密集感知的现状、两类难点与攻克方向，让评审理解方法如何接入通用系统以及哪些基础可以复用。
- core_message: 实例对应与区域修正是两种衔接的决策；最终形成通用检测分割系统，猪只是应用案例。
- consumption_mode: presentation

## mode
- mode: custom
- mode_references: instructional, pyramid
- mode_behavior: 先讲需求与已有进展，再用具体矛盾推导两个方向，最后落到应用；每页给出可核对的论点，区分历史观察和待验证主张。

## visual_style
- visual_style: custom
- visual_style_references: swiss-minimal
- visual_style_behavior: 白底、深蓝标题和开放留白；蓝色表实例选择，青色表区域修正，橙色表冲突；大型关系图与并排示意组织机制，备用页表格提供证据。

## colors
- background: #FFFFFF
- secondary_bg: #F1F5F8
- primary: #172B46
- accent: #007F82
- secondary_accent: #C36D37
- body_text: #25374C
- secondary_text: #596B80
- divider: #D4DEE7
- instance_selection: #315E9D
- surface: #F7F9FB

## typography
- font_family: Microsoft YaHei, Arial
- title_family: Microsoft YaHei, Arial
- body_family: Microsoft YaHei, Arial
- cover_title: 56
- title: 40
- subtitle: 32
- lead: 32
- body: 28
- annotation: 22
- footnote: 16
- display: 72

## icons
- library: none
- inventory: none

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: breathing
- P06: dense
- P07: breathing
- P08: anchor
- P09: dense
- P10: dense
- P11: dense
- P12: dense
- P13: anchor
- P14: dense
- P15: dense
- P16: dense

## pptx_structure
- mode: flat
- template_reuse_scope: style

## page_visualizations
- P14: table/comparison_matrix
- P15: table/metric_table

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
