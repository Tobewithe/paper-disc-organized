"""CPU-only configuration and manifest preparation; does not submit jobs."""
import argparse
import json
from pathlib import Path
import random

from .common import ROOT, save, sha


def manifest(split, count, offset):
    assets = dict(engineering=["Horse", "Pig", "Cow", "Sheep", "Zebra", "Llama", "Pug", "Fox"],
                  train=["Horse", "Pig", "Cow", "Sheep", "Zebra"],
                  validation=["Llama"], test=["Pug", "Fox"])[split]
    kinds = ["wall_short", "wall_long", "repeated", "censored",
             "exit_return", "turn", "no_occlusion", "wall_short"]
    rows = []
    for i in range(count):
        seed = 20261020 + offset + i
        rng = random.Random(seed)
        kind = kinds[i % len(kinds)]
        barriers = [] if kind in ["exit_return", "turn", "no_occlusion"] else (
            [-1.2, 1.2] if kind == "repeated" else [rng.uniform(-.4, .4)])
        rows.append(dict(id=f"{split}_{i:04d}", split=split, seed=seed,
            asset=assets[i % len(assets)], asset_family=assets[i % len(assets)],
            event_type=kind, barrier_x=barriers,
            barrier_width=(2.8 if kind == "wall_long" else 1.25 if kind == "repeated" else 1.8),
            camera_x=rng.uniform(-.8, .8), camera_height=rng.uniform(4.8, 6),
            background_color=[rng.uniform(.13, .35) for _ in range(3)],
            occluder_color=[rng.uniform(.1, .8) for _ in range(3)],
            animation_offset=rng.randrange(30), direction=rng.randrange(2),
            accel=rng.uniform(-.8, .8),
            intervention="camera" if i % 2 else "occluder_motion",
            background_source="procedural_constant_material",
            template_family=kind,
            domain_scope="asset_heldout_same_renderer_shared_event_templates"))
    return rows


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT/"engineering_manifest.json", manifest("engineering", 8, 0))
    rows = (manifest("train", 160, 10000) + manifest("validation", 40, 20000)
            + manifest("test", 80, 30000))
    save(ROOT/"pilot_manifest.json", rows)
    save(ROOT/"EXECUTION_CONFIG.json", dict(
        design="v0.2-e0-implementation", seed=0, resolution_rgb=[512,320],
        resolution_geometry=[128,80], continuous_point_labels=True,
        model_input_wh=[448,280], feature_grid=[32,20],
        backbone="dinov2_vits14", source="official_facebookresearch",
        backbone_frozen=True, adapter_hidden=64, width=256, temporal_layers=2,
        temporal_heads=4, temporal_buffer=8, frames=48, fps=12,
        point_queries=24, region_queries=3,
        arms=["B","P","I","Hpos"], steps_per_arm=2000, effective_batch_families=4,
        optimizer="AdamW", lr=0.0001, weight_decay=0.0001,
        loss_weights=dict(corr=1,evidence=.5,cf=.1,cal=.1,hidden_Hpos=.2),
        temperature=.07, lambda_xy=1, max_grad_norm=1,
        checkpoint_selection="last_step", formal_training_allowed=False,
        engineering_max_updates_per_arm=4,
        engineering_manifest_sha256=sha(ROOT/"engineering_manifest.json"),
        pilot_manifest_sha256=sha(ROOT/"pilot_manifest.json"),
        source_commit="f70c9a0af6c0f64cf0c51873b03110c37a3633e0"))
    print(json.dumps(dict(root=str(ROOT), engineering_families=8, pilot_families=280)))


if __name__ == "__main__":
    main()
