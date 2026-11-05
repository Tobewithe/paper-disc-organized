# 全发布汇总与共同日程：最终数据资格

仅在7套来源（原2组probe+6个33组V2分片）的build/render/decode/saved-ray真实完成且逐SHA回传后执行。0GPU/4CPU/16GiB/≤30min，最多2次明确工程纠错，无自动后继。

逐文件核各真实Run自身SHA，包括实际BODY/Blend/EXR/RGB/owner网格；核200组/600图/1100顶点身份、固定160/40同组隔离及全部像素/射线条件。原V1尝试中已完成的同名BODY顶点身份必须与最终一致，重复生成不当新样本。不足完整记账，不凑点、不缩分母。

932的1272 train RGB和GT缓存只重新核字节SHA，不在此解码RGB/GT数组，不将GT传回生成。元数据只用train-only清单，禁止混合DATA_INDEX、dev/test文件。2000步日程分别用NumPy seed20261101无放回real流、20261102每80步遍历160合成组的25轮流；每步对应CPU Torch seed20261101+step，唯一坐标/每owner-pair≤32。一次生成后核≥95% batch有≥128点，阴性不得重抽。

只有全部必要项通过才写accepted release，冻结MANIFEST/SCHEDULE/COORDINATES及所有来源SHA。正式训练还须把该实际release、347源码/版本环境和已通过954的程序固定到正式机器合同，不能由本阶段或临时文件自动开放。
