# 来源与阅读范围

2026-11-05整理既有`qwen_rgb_instance_parsing_2026-10-28`的19项来源及阅读笔记、专用模型接口记录，以及统一理论记录D43–D54中已经形成的数学/工程认识。本轮没有新外部检索、模型调用或准确率评价；详细路径、SHA和主题范围见[整理来源](CURATION_SOURCES.json)。

- [Qwen3.5-9B模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[Qwen3-VL报告](https://arxiv.org/abs/2511.21631)、[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)：模型家族、具体权重与工具宿主的能力分开。
- [V*](https://arxiv.org/abs/2312.14135)、[Visual Sketchpad](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb82011040977c7712409fbdb5456647-Abstract-Conference.html)、[VISPROG](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html)、[ViperGPT](https://openaccess.thecvf.com/content/ICCV2023/html/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.html)、[DeepEyes](https://arxiv.org/abs/2505.14362)：各自包含的训练和专用工具条件保留。
- [Vision language models are blind](https://arxiv.org/abs/2407.06581)、[POPE](https://aclanthology.org/2023.emnlp-main.20/)、[CAPTURe](https://arxiv.org/abs/2504.15485)、[ROD-MLLM](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_ROD-MLLM_Towards_More_Reliable_Object_Detection_in_Multimodal_Large_Language_CVPR_2025_paper.html)：感知/解码、幻觉、计数与实例发现的区别。
- [MAMMAL](https://www.nature.com/articles/s41467-023-43483-w)、[CClusnet](https://arxiv.org/abs/2206.01942)、[Multi-Pig Part Detection and Association](https://doi.org/10.3390/s19040852)、[SwinePose](https://github.com/AryaanVedak/SwinePose)：猪表面、部位关系、关键点和观测条件。
- [GQA](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)、[PACO](https://openaccess.thecvf.com/content/CVPR2023/html/Ramanathan_PACO_Parts_and_Attributes_of_Common_Objects_CVPR_2023_paper.html)：对象—属性—关系与部位绑定。

原记录逐项区分全文获取、指定章节已读和未核部分。作者README没有被升级为论文全文或数据标签审计，原T07提取失败及修复保留。小集项目反例仅支持所测输入/接口范围，不估计整体模型准确率。
