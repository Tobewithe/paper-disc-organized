"""Build the explicit, hashed Public transfer package (desktop preparation only)."""
from __future__ import annotations
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
import cv2
from .common import sha256, write_json


def package(repo, stage):
    (stage / "models").mkdir(exist_ok=True)
    models = {
        "pig": Path("C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt"),
        "horse": Path("C:/Dpan/document/model_datasets/models/source-assets/pigcv-task05/yolo26m-seg.pt"),
    }
    for k, p in models.items():
        shutil.copy2(p, stage / "models" / f"{k}.pt")
    shutil.copytree(repo / "shared/dense_video_identity", stage / "code/shared/dense_video_identity", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "code/shared/__init__.py").write_text("", encoding="utf-8")
    shutil.copytree(repo / "shared/vendor/ultralytics_8_4_100/ultralytics", stage / "vendor/ultralytics_8_4_100/ultralytics", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ["PROTOCOL.md", "config.json", "study.json"]:
        shutil.copy2(repo / "experiments/dense_video_identity_20261018" / f, stage / f)
    splits = {"train": ["pig_3", "pig_5", "horse_4", "horse_6"], "dev": ["pig_4", "horse_5"],
              "test": ["pig_1", "pig_2", "horse_1", "horse_2", "horse_3"]}
    seqs = []
    for split, names in splits.items():
        for name in names:
            video = stage / "data/animaltrack/videos" / f"{name}.mp4"
            gt = stage / "data/animaltrack/gt" / f"{name}_gt.txt"
            cap = cv2.VideoCapture(str(video))
            assert cap.isOpened(), video
            seqs.append(dict(name=name, split=split, model=name.split("_")[0],
                             classes=[0] if name.startswith("pig") else [17],
                             video=video.relative_to(stage).as_posix(), gt=gt.relative_to(stage).as_posix(),
                             frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                             width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                             video_sha256=sha256(video), gt_sha256=sha256(gt),
                             gt_kind="official_AnimalTrack_bbox_identity", visibility_labels=False))
            cap.release()
    own = stage / "data/own"
    own.mkdir(exist_ok=True)
    source = Path("C:/Dpan/document/model_datasets/datasets/自拍猪视频/VID20260511133439.mp4")
    for name, start in [("现场_群体活动", 15), ("现场_遮挡交互", 6650)]:
        dst = own / f"{name}.mp4"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(source),
                            "-t", "15", "-vf", "transpose=2,scale=960:540,fps=15", "-an", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "20", "-movflags", "+faststart", str(dst)], check=True)
        cap = cv2.VideoCapture(str(dst))
        seqs.append(dict(name=name, split="demo", model="pig", classes=[0], video=dst.relative_to(stage).as_posix(),
                         gt=None, frames=int(cap.get(7)), fps=cap.get(5), width=int(cap.get(3)), height=int(cap.get(4)),
                         video_sha256=sha256(dst), gt_sha256=None, gt_kind="unannotated", source=str(source),
                         source_start_seconds=start, transform="90deg counterclockwise;960x540;15fps", visibility_labels=False))
        cap.release()
    write_json(stage / "input_manifest.json", dict(source="https://hengfan2010.github.io/projects/AnimalTrack/download.html",
               sequences=seqs, models={k:f"models/{k}.pt" for k in models}, model_sources={k:str(p) for k,p in models.items()},
               model_sha256={k:sha256(stage / "models" / f"{k}.pt") for k in models},
               claim_scope="small subset; not whole AnimalTrack benchmark; own videos qualitative"))
    files = []
    for group in ["code", "vendor"]:
        for f in sorted((stage / group).rglob("*")):
            if f.is_file() and f.suffix in {".py", ".yaml", ".yml"}:
                files.append(dict(path=f.relative_to(stage).as_posix(), sha256=sha256(f)))
    write_json(stage / "source_manifest.json", dict(files=files))
    archive = stage / "pilot_bundle_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted(stage.rglob("*")):
            rel = f.relative_to(stage)
            if f.is_file() and rel.parts[0] in {"code", "vendor", "models", "data"} and "__pycache__" not in rel.parts:
                if "horse_7" not in f.name:
                    z.write(f, rel.as_posix())
        for f in ["PROTOCOL.md", "config.json", "input_manifest.json", "source_manifest.json", "study.json"]:
            z.write(stage / f, f)
    print(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha256(archive), sequences=len(seqs)))
    print([(s["name"],s["split"],s["frames"]) for s in seqs])


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--stage",type=Path,required=True);a=p.parse_args();package(a.repo,a.stage)
