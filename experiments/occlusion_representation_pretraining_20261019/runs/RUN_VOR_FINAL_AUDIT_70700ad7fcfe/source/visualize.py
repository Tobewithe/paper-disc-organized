"""Fixed-index qualitative delivery using real E1 predictions, including failures."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .common import ROOT,read,save


def main():
    result=read(ROOT/"E1_RESULT.json")
    folder=Path(result["path"])
    summary=read(folder/"SUMMARY.json")
    dataset=Path(read(ROOT/"features_pilot.json")["dataset"])
    output=folder/"visualizations"
    output.mkdir(exist_ok=True)
    palette=["#19b5fe","#ffb342","#44c58a"]
    font=ImageFont.load_default(size=17)
    rows=[]
    for index in range(4):
        family=f"test_{index:04d}"
        variant=0
        data=dataset/family/str(variant)
        tracks=np.load(data/"tracks.npz")
        predictions={a:dict(np.load(folder/"predictions"/a/family/"0.npz"))
                     for a in ["N","B","P","I","Hpos"]}
        path=output/(family+"_comparison.mp4")
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),12,(1536,738))
        assert writer.isOpened()
        for t in range(48):
            image=Image.open(data/"rgb"/f"{t:03d}.png").convert("RGB")
            canvas=Image.new("RGB",(1536,738),"#102238")
            draw=ImageDraw.Draw(canvas)
            draw.text((14,10),f"{family} | Frame {t:02d}/47 | conditional video correspondence | red ring: incorrect accepted match",
                      font=font,fill="white")
            for panel,arm in enumerate(["GT","N","B","P","I","Hpos"]):
                x,y=(panel%3)*512,42+(panel//3)*348
                canvas.paste(image,(x,y+26))
                label="Visible surface ground truth" if arm=="GT" else {
                    "N":"N: frozen initial feature matching","B":"B: ordinary video supervision",
                    "P":"P: physical-pair supervision","I":"I: initial + current only",
                    "Hpos":"Hpos: P + hidden-position target"}[arm]
                draw.text((x+10,y+3),label,font=font,fill="white")
                if arm=="GT":
                    xy=tracks["xy"][t]
                    active=(tracks["state"][t]==0) & tracks["active"]
                    wrong=np.zeros(24,bool)
                else:
                    pred=predictions[arm]
                    xy=pred["xy"][t]
                    active=(pred["eligible"][t] &
                            (pred["confidence"][t]>=summary["thresholds"][arm]["threshold"]) &
                            tracks["active"])
                    wrong=(np.linalg.norm((xy-tracks["xy"][t])*256,axis=-1)>4) | (tracks["state"][t]!=0)
                for q in range(24):
                    px,py=xy[q]*[512,320]
                    if not active[q] or not(0<=px<512 and 0<=py<320):continue
                    color=palette[int(tracks["owners"][q])-1]
                    pos=(x+px,y+26+py)
                    draw.ellipse((pos[0]-3,pos[1]-3,pos[0]+3,pos[1]+3),fill=color)
                    if wrong[q]:
                        draw.ellipse((pos[0]-6,pos[1]-6,pos[0]+6,pos[1]+6),outline="#fa3248",width=2)
            writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
            if t==24:
                canvas.save(output/(family+"_frame24.png"))
        writer.release()
        ffmpeg=shutil.which("ffmpeg")
        if ffmpeg:
            converted=path.with_name(path.stem+"_h264.mp4")
            command=[ffmpeg,"-y","-i",str(path),"-c:v","libx264","-pix_fmt","yuv420p",
                     "-movflags","+faststart",str(converted)]
            done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (output/(family+"_ffmpeg.log")).write_bytes(done.stderr)
            if done.returncode==0:
                path=converted
        rows.append(dict(family=family,selection="first_four_fixed_test_indices_not_selected_by_outcome",video=str(path)))
    save(folder/"VISUALIZATIONS.json",rows)
    means=summary["means"]
    table="\n".join("| "+a+" | "+" | ".join("未测" if means[a][m] is None else f"{100*means[a][m]:.3f}"
               for m in ("R1","W","W_other_instance","coverage","R_object_conditional"))+" |"
               for a in ["N","B","P","I","Hpos"])
    report=f"""# 四臂视频遮挡表征探索实验

本轮已完成同渲染器、留出动物资产的受控评价。以下是实际测试输出；真实视频迁移和 YOLO/BoT-SORT 完整跟踪尚未测量。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| 问题 | 旧状态分类收益没有稳定转成真实遮挡恢复；本轮直接训练对应、无对应与可见证据。 |
| 机制假说 | 物理配对相对普通视频监督有额外恢复效用；竞争解释是普通对应训练或隐藏位置监督已解释收益。 |
| 针对性方法 | 同初始权重、参数、数据、8,000总步预算比较 B/P/I/Hpos；独立测试80家族，另含冻结N和P记忆干预。 |
| 整体结果 | 见下表和配对95%区间；总收益 P−N、普通训练 B−N 与新增收益 P−B 分别保留。 |
| 研究决策 | 冻结裁决：{summary['decision']['status']}；唯一后继：{summary['decision']['next_action']}。不自动追加训练。 |

| 方法 | 首重现R1 % | 错误接受W % | 接到另一实例 % | 接受覆盖 % | 条件实例恢复 % |
|---|---:|---:|---:|---:|---:|
{table}

四臂结论只涉及本次合成、单训练种子、冻结末步模型；条件实例定位不等于完整检测跟踪系统。详细逐家族分母、差值区间及阈值在 `SUMMARY.json`、`FAMILY_METRICS.json`、`VALIDATION_THRESHOLDS.json`；源 Run、冻结协议和所有预测在同目录。

可视化固定选择 test_0000–test_0003，不按成功率挑选；隐藏位置不会画成可见检测。
"""
    (folder/"REPORT.md").write_text(report,encoding="utf-8")
    print(rows,flush=True)


if __name__=="__main__":
    main()
