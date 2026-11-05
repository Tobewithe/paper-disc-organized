---
name: visual-grounding-and-evidence
description: >-
  Understand and evaluate visual grounding, instance discovery, part ownership,
  coordinate transforms and uncertainty in VLM or specialist model outputs.
  Use for RGB instance parsing, visual tool use and turning model proposals into evidence.
  视觉定位、部位归属、局部观察与模型证据边界。
metadata:
  version: "1.0.0"
---

# 视觉定位与证据

指定对象定位、找全实例、部位识别、owner绑定和隐藏状态推断需要不同信息。读取[方法与反例](references/methods.md)，来源与实际阅读范围见[SOURCES.md](SOURCES.md)。

全图保留关系，原RGB局部裁剪保留细节；两者不能自行保证语义正确。工具坐标、持久实例身份、未知状态和可观察证据应服务于解释输出，不将合法JSON、自一致或精确变换误当准确率。

需要生成几何真值时转`synthetic-scene-methods`；需要时序身份时转`occlusion-video-identity`。本Skill保存方法与经验，不包含原项目候选流程、调用预算或恢复授权。
