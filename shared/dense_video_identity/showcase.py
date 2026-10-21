from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import read_json,write_json

NAMES={"bytetrack":"ByteTrack","deepocsort":"Deep OC-SORT","overlap_freeze":"重叠冻结","utility_memory":"收益记忆","freeze_recovery":"冻结 + 重现确认","utility_recovery":"收益记忆 + 重现确认"}


def make_charts(root,summary,out):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#d9e1ed","text.color":"#24364e","axes.labelcolor":"#64748b","xtick.color":"#64748b","ytick.color":"#64748b","figure.facecolor":"white","savefig.facecolor":"white"})
    rows=summary["combined"];arms=list(rows);vals=[rows[a]["IDF1"] for a in arms]
    colors=["#b9c9dc","#7b96bd","#a4bcd4","#328a90","#7abdb2","#2863dd"]
    fig,ax=plt.subplots(figsize=(8.2,5.2));y=np.arange(len(arms))
    ax.barh(y,vals,color=colors,height=.56);ax.set_yticks(y,[NAMES[a] for a in arms]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel("IDF1（0–100，越高越好）");ax.set_title("身份保持：同检测、同数据对照",loc="left",fontweight="bold",pad=24)
    for j,v in enumerate(vals):ax.text(v+1,j,f"{v:.2f}",va="center",fontsize=11,fontweight="bold")
    ax.set_axisbelow(True);ax.xaxis.grid(True,color="#eef2f7");fig.tight_layout(pad=2);fig.savefig(out/"metrics.png",dpi=180);plt.close(fig)
    table=summary["by_sequence"];seqs=list(dict.fromkeys(t["sequence"] for t in table))
    diffs=[]
    for s in seqs:
        m=next(t for t in table if t["sequence"]==s and t["arm"]=="utility_recovery")["IDF1"]
        r=next(t for t in table if t["sequence"]==s and t["arm"]=="deepocsort")["IDF1"]
        diffs.append(m-r)
    fig,ax=plt.subplots(figsize=(8.2,5.2));ax.bar(np.arange(len(seqs)),diffs,width=.55,color=["#278b7d" if d>=0 else "#d48758" for d in diffs]);ax.axhline(0,color="#95a3b5",lw=.8);ax.set_xticks(np.arange(len(seqs)),seqs);ax.set_ylabel("候选完整方案 − Deep OC-SORT（IDF1 点）");ax.set_title("逐视频结果：改善与退化都保留",loc="left",fontweight="bold",pad=24)
    lo=min(min(diffs)-.7,-.5);hi=max(max(diffs)+.7,.5);ax.set_ylim(lo,hi)
    for i,d in enumerate(diffs):ax.text(i,d+(.08 if d>=0 else -.08),f"{d:+.2f}",ha="center",va="bottom" if d>=0 else "top",fontsize=11)
    ax.set_axisbelow(True);ax.yaxis.grid(True,color="#eef2f7");fig.tight_layout(pad=2);fig.savefig(out/"sequence.png",dpi=180);plt.close(fig)
    curves=[]
    for key in ["identity","utility"]:
        p=read_json(root/(key+"_pointer.json"));curves.append(read_json(root/"runs"/p["run_id"]/"loss_curve.json"))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,curve,title,c in zip(axes,curves,["身份投影：监督对比训练","收益门：更新效用监督"],["#2863dd","#278b7d"]):
        y=np.asarray([r["loss"] for r in curve]);ax.plot(np.arange(1,len(y)+1),y,color=c,alpha=.25,lw=.8)
        if len(y)>10:ax.plot(np.arange(10,len(y)+1),np.convolve(y,np.ones(10)/10,mode="valid"),color=c,lw=2)
        ax.set_title(title,loc="left",fontweight="bold");ax.set_xlabel("优化步");ax.set_ylabel("训练损失");ax.grid(color="#eef2f7")
    fig.tight_layout(pad=2);fig.savefig(out/"training.png",dpi=170);plt.close(fig)


def architecture(out):
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="640" viewBox="0 0 1440 640"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#7592b6"/></marker></defs><rect width="1440" height="640" fill="#f8fbff"/><g font-family="Microsoft YaHei, sans-serif" fill="#1f3550"><text x="50" y="55" font-size="25" font-weight="bold">从单帧识别到跨帧身份连续</text><text x="50" y="88" font-size="15" fill="#6b7d93">固定分割前端 · 学习可靠记忆 · 以短轨迹确认重现身份</text>
<g stroke="#d6e1ee" fill="white"><rect x="50" y="140" width="270" height="350" rx="18"/><rect x="380" y="140" width="290" height="350" rx="18"/><rect x="730" y="140" width="290" height="350" rx="18"/><rect x="1080" y="140" width="310" height="350" rx="18"/></g>
<g font-size="12" fill="#2863dd"><text x="75" y="177">01 / 可见实例</text><text x="405" y="177">02 / 身份表示</text><text x="755" y="177">03 / 记忆与恢复</text><text x="1105" y="177">SYSTEM / 透明输出</text></g>
<g font-size="21" font-weight="bold"><text x="75" y="213">YOLO26 分割</text><text x="405" y="213">掩码池化 + 投影头</text><text x="755" y="213">收益记忆 + 竞争确认</text><text x="1105" y="213">个体跟踪与事件展示</text></g>
<g font-size="16"><text x="75" y="267">每帧检测框与可见掩码</text><text x="75" y="302">P3 / P4 特征图</text><text x="75" y="337">既有权重冻结</text><text x="405" y="267">个体区域加权聚合</text><text x="405" y="302">1024 → 128 → 64</text><text x="405" y="337">跨帧同身份对比监督</text><text x="755" y="260">收益门：8 → 32 → 1</text><text x="755" y="293">决定是否写入身份记忆</text><text x="755" y="341">旧身份 × 重现短轨迹</text><text x="755" y="374">一对一竞争 + 有限等待</text><text x="1105" y="267">稳定 ID 与短时轨迹</text><text x="1105" y="302">待确认 / 恢复 / 新身份</text><text x="1105" y="337">同步视频与实测数据</text></g>
<g font-size="13" fill="#6b7d93"><text x="75" y="445">所有对照复用同一检测缓存</text><text x="405" y="445">仅在官方训练视频拟合</text><text x="755" y="445">部署只读当前与过去帧</text><text x="1105" y="445">保留不确定性和误恢复代价</text></g>
<g stroke="#7592b6" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M323 310H370"/><path d="M673 310H720"/><path d="M1023 310H1070"/></g>
<rect x="50" y="530" width="1340" height="65" rx="12" fill="#eaf1fc"/><text x="75" y="569" font-size="16">训练监督：真实跨帧 ID / 未来区分收益标签　　评价：IDF1、HOTA、身份切换与重现事件　　应用案例：猪与马</text></g></svg>'''
    (out/"architecture.svg").write_text(svg,encoding="utf-8")


def build(root:Path):
    out=root/"showcase";out.mkdir(exist_ok=True)
    ptr=read_json(root/"evaluation_test_pointer.json");testpath=root/ptr["path"]
    summary=read_json(testpath/"SUMMARY.json");decision=read_json(testpath/"DECISION.json")
    manifest=read_json(root/"input_manifest.json");cache=read_json(root/"features_pointer.json")["cache"]
    make_charts(root,summary,out);architecture(out)
    m=summary["combined"]["utility_recovery"];r=summary["combined"]["deepocsort"];n=summary["combined"]["bytetrack"]
    best=max(summary["combined"],key=lambda a:summary["combined"][a]["IDF1"])
    headline="小样本完整方案达到预设推进门槛" if decision["status"]=="passed" else "系统与对照已跑通，完整方案尚未通过推进门槛"
    text=f"完整方案 IDF1 {m['IDF1']:.2f}，相对 ByteTrack {m['IDF1']-n['IDF1']:+.2f} 点，相对 Deep OC-SORT {m['IDF1']-r['IDF1']:+.2f} 点。本轮 IDF1 最高的方法为 {NAMES[best]}（{summary['combined'][best]['IDF1']:.2f}）。"
    total_frames=sum(s["frames"] for s in manifest["sequences"] if s["split"]=="test")
    scope=f"保留测试为 5 个完整视频、{total_frames} 帧，涵盖猪和马；不是完整 AnimalTrack 榜单。自然重现事件 {m['events']} 个；可见度字段未标注物理遮挡原因。现场视频无 ID GT，不纳入量化成绩。"
    sequences=[]
    for s in sorted([s for s in manifest["sequences"] if s["split"] in {"demo","test"}],key=lambda s:(s["split"]!="demo",s["name"])):
        ep=read_json(root/("evaluation_"+s["split"]+"_pointer.json"))["path"]
        seq_cache=read_json(root/"demo_features_pointer.json")["cache"] if s["split"]=="demo" and (root/"demo_features_pointer.json").exists() else cache
        seq=dict(s,title=s["name"].replace("_"," · ") if s["split"]=="demo" else "公开基准 · "+s["name"],
                 description="现场连续视频中的密集交互。可点击个体查看轨迹和记忆状态。" if s["split"]=="demo" else "公开 AnimalTrack 视频，使用全部原生帧与真实个体 ID 进行评价。",
                 detections=f"{seq_cache}/{s['name']}/detections.json",
                 tracks={a:f"{ep}/{a}/{s['name']}/tracks.json" for a in NAMES},events={a:f"{ep}/{a}/{s['name']}/policy_events.json" for a in NAMES})
        sequences.append(seq)
    split_info="；".join(k+"："+"、".join(s["name"] for s in manifest["sequences"] if s["split"]==v) for k,v in [("训练","train"),("开发","dev"),("保留测试","test")])
    catalog=dict(sequences=sequences,default_sequence=sequences[0]["name"],scores=summary["combined"],conclusion=dict(headline=headline,text=text,scope=scope),
                 charts=dict(metrics="showcase/metrics.png",sequence="showcase/sequence.png",architecture="showcase/architecture.svg",training="showcase/training.png"),
                 split_info=split_info,reproduce="执行目录：D:\\thesis-video-20261018\n启动方式：双击 START_DEMO.cmd，打开 http://127.0.0.1:8871\n数据、模型、源代码与真实运行记录均保留在该目录；不需要互联网。",
                 report="showcase/实验报告.md",run_id=ptr["run_id"],generated_from="actual laptop runs")
    write_json(out/"catalog.json",catalog)
    lines=["# 密集视频个体跟踪与遮挡恢复：小样本原型结果", "",headline+"。"+text,"",scope,"",
           "| 维度 | 本次实验的整体研究链路 |","|---|---|",
           "| 问题 | 既有密集片段产生较多临时 ID，缺少真实身份评价。本次建立公开标注视频的可复现对照，检验记忆污染与过早重接是否影响身份连续。 |",
           "| 机制假说 | 当前检测分数不能充分表示身份记忆写入价值；未来区分收益监督与短轨迹竞争确认可能减少误接。竞争解释是检测缺失和外观不可分主导错误。 |",
           "| 针对性方法 | 固定 YOLO26 检测缓存，训练共享 64 维身份投影和 321 参数收益门；六组对照含 ByteTrack、Deep OC-SORT、普通重叠冻结及记忆/恢复消融。所有训练仅用四个官方训练视频。修复等待期限实现错误后，保持模型、阈值、数据和预算不变，重新评价并保留旧 Run。 |",
           f"| 整体结果 | {text} 全部逐视频结果与阴性保留。自然重现事件数为 {m['events']}，完整方案正确接回 {m['correct_recoveries']}、错误复用 {m['wrong_recoveries']}。事件由 GT 间断定义，并非人工确认遮挡。 |",
           "| 研究决策 | "+("优先推进可靠记忆与短轨迹恢复，依据为同检测保留评价通过。唯一下一步是增加独立场景及人工遮挡事件标注，再验证收益稳定性。" if decision["status"]=="passed" else "停止本轮完整配置的继续调参；优先解决‘相似个体的外观何时值得用于身份关联’。依据是采用外观的 Deep OC-SORT 已低于 ByteTrack，而新增记忆/恢复未挽回差距。两种跟踪器还存在其它差异，该结果只支持优先检查外观可靠性，并非完成原因归属。唯一下一步是在独立现场片段补齐跨帧 ID 与遮挡起止标注，冻结同一关联器的外观开启/关闭比较；只有外观可分性及关联收益得到支持才继续记忆机制。此下一步尚未启动。")+" 本轮未追加训练或事后更换测试样本。 |","",
           "## 同口径结果","","| 方法 | IDF1 | HOTA | AssA | DetA | IDSW |","|---|---:|---:|---:|---:|---:|"]
    for a,s in summary["combined"].items():lines.append(f"| {NAMES[a]} | {s['IDF1']:.3f} | {s['HOTA']:.3f} | {s['AssA']:.3f} | {s['DetA']:.3f} | {s['IDSW']} |")
    lines += ["", "### 差值与不确定性", "", "所有分数来自相同的 5 个测试视频、41,875 个 GT 实例观测。IDF1 由合并的 IDTP/IDFP/IDFN 计算，差值单位为百分点。95% 区间按视频进行 500 次成对 bootstrap；仅 5 个视频，区间只作小样本描述。", "", "| 完整候选相对参照 | IDF1 差值 | 95% 描述区间 |", "|---|---:|---:|"]
    for a,c in summary["comparisons"].items():
        lo,hi=c["sequence_bootstrap_95"]
        lines.append(f"| {NAMES[a]} | {c['delta_IDF1']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "完整候选相对冻结恢复对照仅 +0.293 点，区间跨零；相对不带恢复的收益记忆退化 1.058 点。相对 Deep OC-SORT，身份切换由 406 降至 389，但漏检由 13,212 增至 13,757，IDF1 与 HOTA 均下降，因此不能用切换次数下降宣称整体改善。预设推进门槛未通过；未得到收益门或恢复机制成立的证据。", "", "### 工程修复的范围", "", "现场演示视频方向修复只影响两个无 GT 演示片段；正式 train/dev/test 内容未更改。等待期限修复将候选再次消失时的等待也限制到 6 帧，原始 Run 与修复记录均保留。最终采用修复后的评价；未重新训练。交付完整性核验检查四个不受影响对照的逐帧输出哈希不变、编号唯一、输出对应原检测、等待期限和视频完整解码。"]
    lines += ["","## 证据边界","",split_info,"","- 方法框架可以接入其它动物检测模型，本轮只实测猪与马，不能声称所有动物泛化。","- 分割掩码用于身份特征与显示；AnimalTrack 没有分割 GT，本轮没有新的分割 AP 成绩。","- 旧检测器训练集与公开视频来源的潜在预训练重叠未做逐图去重审计；跟踪头严格按官方视频拆分。","- 等待确认的帧计入漏检，旧帧 ID 不回写。暂不确认不是已经解决身份。","- 训练损失下降只说明拟合完成，不能代替跟踪效用和机制消融。","- 学习式更新、重叠冻结、外观关联均有先前研究；本候选创新仍需要更充分对照和文献定位。","",
              "## 真实执行与复现","",f"- 保留评价 Run：`{ptr['run_id']}`",f"- 身份投影 Run：`{read_json(root/'identity_pointer.json')['run_id']}`",f"- 收益门 Run：`{read_json(root/'utility_pointer.json')['run_id']}`",f"- 固定检测缓存 Run：`{read_json(root/'features_pointer.json')['run_id']}`","- 笔记本工作目录：`D:\\thesis-video-20261018`；源码：`code/shared/dense_video_identity`。","- 冻结协议、实现合同、输入和源码散列、run.json、原始逐帧结果、官方评价分量均在该目录。","- 启动：`START_DEMO.cmd`；界面提供同屏播放、方法切换、个体聚焦、事件定位和实测图表。","",
              "## 主要来源","","- [AnimalTrack 官方数据与划分](https://hengfan2010.github.io/projects/AnimalTrack/download.html)","- [TrackEval 官方评价源码](https://github.com/JonathonLuiten/TrackEval/tree/12c8791b303e0a0b50f753af204249e622d0281a)","- [Deep OC-SORT](https://arxiv.org/abs/2302.11813)","- [UpdateNet](https://arxiv.org/abs/1908.00855)","- [MASA-OA](https://github.com/gonzalocarreteroh/MASA-OA)"]
    (out/"实验报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return catalog


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();build(a.root)
