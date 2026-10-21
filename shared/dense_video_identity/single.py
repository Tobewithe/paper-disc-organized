"""Analyze a user-imported video using fixed trained checkpoints, without fitting."""
from __future__ import annotations
import shutil
from pathlib import Path
import cv2
from .common import read_json,write_json,sha256,now


def analyze(root:Path,job_id:str,kind:str):
    job=root/"jobs"/job_id
    status=job/"job.json"
    try:
        write_json(status,dict(status="running",phase="准备已训练模型",started_at=now()))
        cfg=read_json(root/"config.json");manifest=read_json(root/"input_manifest.json")
        shutil.copytree(root/"vendor",job/"vendor",ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
        shutil.copytree(root/"deps",job/"deps",dirs_exist_ok=True)
        (job/"models").mkdir(exist_ok=True)
        source=root/manifest["models"][kind];model=job/"models"/f"{kind}.pt"
        shutil.copy2(source,model)
        for key in ["identity","utility"]:
            ptr=read_json(root/(key+"_pointer.json"));dst=job/"models"/(key+".pt")
            shutil.copy2(root/ptr["checkpoint"],dst)
            write_json(job/(key+"_pointer.json"),dict(ptr,checkpoint=dst.relative_to(job).as_posix()))
        cap=cv2.VideoCapture(str(job/"input.mp4"))
        if not cap.isOpened():raise ValueError("无法读取此视频，请使用 MP4/H.264 文件。")
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not 1<=fps<=120:raise ValueError("无法识别视频帧率，或超出原型支持的 1–120 fps。")
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=min(total,int(fps*60))
        seq=dict(name="imported_"+job_id[:8],split="demo",model=kind,classes=[0] if kind=="pig" else [17],video="input.mp4",gt=None,
                 frames=frames,fps=fps,width=int(cap.get(3)),height=int(cap.get(4)),gt_kind="unannotated",gt_sha256=None,
                 video_sha256=sha256(job/"input.mp4"),visibility_labels=False)
        cap.release()
        write_json(job/"input_manifest.json",dict(sequences=[seq],models={kind:f"models/{kind}.pt"},model_sha256={kind:sha256(model)}))
        write_json(job/"config.json",cfg)
        for name in ["PROTOCOL.md","IMPLEMENTATION_CONTRACT.md","source_manifest.json"]:
            shutil.copy2(root/name,job/name)
        write_json(status,dict(status="running",phase=f"检测与特征提取 · {frames} 帧",started_at=now()))
        from .infer import extract
        extract(job,cfg["detector"])
        write_json(status,dict(status="running",phase="个体关联与重现确认",started_at=now()))
        from .evaluate import evaluate
        evaluate(job,cfg,"demo")
        fp=read_json(job/"features_pointer.json");ep=read_json(job/"evaluation_demo_pointer.json")
        prefix=f"jobs/{job_id}/"
        case=dict(seq,title=f"导入视频 · {frames/fps:.1f} 秒",video=prefix+"input.mp4",
                  description=f"新视频使用已训练模型完成推理，分析前 {frames/fps:.1f} 秒。未进行新训练，没有真实 ID 标注。",
                  detections=prefix+fp["cache"]+"/"+seq["name"]+"/detections.json",
                  tracks={a:prefix+ep["path"]+"/"+a+"/"+seq["name"]+"/tracks.json" for a in cfg["arms"]},
                  events={a:prefix+ep["path"]+"/"+a+"/"+seq["name"]+"/policy_events.json" for a in cfg["arms"]})
        write_json(job/"case.json",case)
        write_json(status,dict(status="completed",phase="分析完成",case=seq["name"],ended_at=now(),frames=frames))
    except Exception as exc:
        write_json(status,dict(status="failed",phase="分析失败",error=str(exc),ended_at=now()))
        raise


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--job",required=True);p.add_argument("--kind",choices=["pig","horse"],required=True);a=p.parse_args();analyze(a.root,a.job,a.kind)
