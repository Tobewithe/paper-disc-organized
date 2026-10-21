"""Inspect existing material without treating interpolated or predicted IDs as GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def font(size=18):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def inventory(root: Path, out: Path):
    rows, tiles = [], []
    for p in sorted(root.glob("*/*/output.json")):
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        video = p.parent / "color.mp4"
        if not video.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        objects = data.get("objects", [])
        points = [f for o in objects for f in o.get("frames", [])]
        row = dict(sequence=f"{p.parent.parent.name}_{p.parent.name}", video=str(video),
                   annotation=str(p), frames=n, fps=fps, ids=len(objects),
                   keyframes=len(points), gt_points=sum(f.get("isGroundTruth", False) for f in points),
                   annotation_frame_max=max((f["frameNumber"] for f in points), default=-1),
                   stepSize=data.get("stepSize"))
        rows.append(row)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, n // 2)))
        ok, frame = cap.read()
        cap.release()
        if ok and len(tiles) < 42:
            pic = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pic.thumbnail((360, 200))
            tile = Image.new("RGB", (380, 250), "#101b29")
            tile.paste(pic, ((380 - pic.width) // 2, 8))
            d = ImageDraw.Draw(tile)
            d.text((10, 211), f"{row['sequence']} | {len(objects)} IDs", font=font(16), fill="white")
            tiles.append(tile)
    out.mkdir(parents=True, exist_ok=True)
    (out / "existing_annotation_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for start in range(0, len(tiles), 12):
        group = tiles[start:start + 12]
        sheet = Image.new("RGB", (380 * 3, 250 * ((len(group) + 2) // 3)), "#101b29")
        for i, tile in enumerate(group):
            sheet.paste(tile, (380 * (i % 3), 250 * (i // 3)))
        sheet.save(out / f"existing_annotations_{start // 12:02d}.jpg", quality=88)
    print(json.dumps(dict(sequences=len(rows), examples=rows[:3]), ensure_ascii=False, indent=2))


def sample(video: Path, out: Path, times: list[float]):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        pic = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pic.thumbnail((480, 320))
        tile = Image.new("RGB", (500, 365), "#101b29")
        tile.paste(pic, ((500-pic.width)//2, 5))
        ImageDraw.Draw(tile).text((12, 333), f"t={t:.1f}s | fps={fps:.3f}", font=font(18), fill="white")
        tiles.append(tile)
    cap.release()
    sheet = Image.new("RGB", (1000, 365 * ((len(tiles)+1)//2)), "#101b29")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2)*500, (i//2)*365))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inventory", "sample"])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--times", type=float, nargs="+", default=[0, 5, 10, 15])
    a = ap.parse_args()
    if a.mode == "inventory":
        inventory(a.input, a.output)
    else:
        sample(a.input, a.output, a.times)
