<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 导师与毕业论文汇报评审
- objective: 用通俗语言讲清应用需要、现有不足、研究方法和最终系统。技术名称、方法实例和指标只在研究方法章节出现。
- core_message: 围绕认全个体、分清轮廓与连续记录开展研究，最终形成通用系统，猪只作为应用案例。
- consumption_mode: presentation

## mode
- mode: custom
- mode_references: instructional, pyramid
- mode_behavior: 背景只讲需求与不足，方法章节展开诊断指标、真实实验和结论，系统章节讲使用功能，末页简述基础与安排。

## visual_style
- visual_style: custom
- visual_style_references: swiss-minimal
- visual_style_behavior: 白底、深蓝标题和较大字号，开放留白；方法章节使用蓝色与青色，实验数据图集中于方法章节，不用文献时间线或结果大表。

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
- P02: breathing
- P03: breathing
- P04: breathing
- P05: anchor
- P06: dense
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: anchor
- P13: breathing
- P14: breathing
- P15: anchor

## pptx_structure
- mode: flat
- template_reuse_scope: style

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
