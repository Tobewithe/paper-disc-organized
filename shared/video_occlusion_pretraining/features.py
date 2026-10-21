from pathlib import Path
import os
import time

import numpy as np
from PIL import Image
import torch

from .common import ROOT, Run, read, save, sha


def load_backbone():
    os.environ["XFORMERS_DISABLED"] = "1"
    backbone = torch.hub.load(str(ROOT/"dinov2-main"),"dinov2_vits14",source="local",pretrained=False)
    weights = ROOT/"dinov2_vits14_pretrain.pth"
    download = read(ROOT/"BACKBONE_DOWNLOAD.json")
    expected = next(x["sha256"] for x in download if x["file"] == weights.name)
    assert sha(weights) == expected
    backbone.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True),strict=True)
    return backbone.eval().requires_grad_(False).cuda()


def main(args):
    with Run("features_"+args.mode,dict(dataset=args.dataset,training_steps=0,
         backbone_manifest_sha256=sha(ROOT/"BACKBONE_DOWNLOAD.json"),
         source_size=[512,320],input_size=[448,280],grid=[32,20],
         views=["original","horizontal_flip"],feature_dtype="float16")) as run:
        torch.set_num_threads(4)
        backbone = load_backbone()
        mean = torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
        std = torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
        root = Path(args.dataset)
        results = []
        for folder in sorted(root.glob("*")):
            for variant in range(3):
                clip = folder/str(variant)
                frames = [np.asarray(Image.open(clip/"rgb"/f"{f:03d}.png").convert("RGB").resize(
                    (448,280),Image.Resampling.BICUBIC)).copy() for f in range(48)]
                rgb = torch.from_numpy(np.stack(frames)).permute(0,3,1,2).cuda().float()/255
                views = []
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    for flip in (False,True):
                        x = rgb.flip(-1) if flip else rgb
                        features = [backbone.forward_features((x[j:j+8]-mean)/std)["x_norm_patchtokens"].half().cpu()
                                    for j in range(0,48,8)]
                        views.append(torch.cat(features))
                out = run.path/"features"/folder.name
                out.mkdir(parents=True,exist_ok=True)
                target = out/f"{variant}.pt"
                torch.save(torch.stack(views),target)
                results.append(dict(family=folder.name,variant=variant,path=str(target),
                                    sha256=sha(target),shape=list(torch.stack(views).shape)))
                print(f"features {folder.name}/{variant}",flush=True)
        save(run.path/"FEATURE_MANIFEST.json",results)
        save(ROOT/("features_"+args.mode+".json"),dict(run_id=run.id,dataset=str(root),
            features=str(run.path/"features"),manifest=str(run.path/"FEATURE_MANIFEST.json"),
            manifest_sha256=sha(run.path/"FEATURE_MANIFEST.json")))
        save(run.path/"SUMMARY.json",dict(clips=len(results),rgb_views=len(results)*96,
            maximum_memory_bytes=torch.cuda.max_memory_allocated(),gradient_steps=0))

