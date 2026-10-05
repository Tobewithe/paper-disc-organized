"""Generate the single-use queue from the already reviewed runner wrapper."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=(root.parent/'mask_supervision_support_20261004/scripts/pipeline_support.py').read_text(encoding='utf-8')
src=src.replace('mask_supervision_support_20261004',root.name).replace('pipeline_support.py','pipeline_bridge.py')
start=src.index('STAGES =');end=src.index('\n\n\ndef now',start)
src=src[:start]+'''STAGES = (("prepare", "prepare_features.py"), ("smoke", "smoke_bridge.py"),
          ("train_T", "train_bridge.py"), ("train_M", "train_bridge.py"),
          ("train_R", "train_bridge.py"), ("evaluation", "evaluate_bridge.py"))'''+src[end:]
start=src.index('def input_paths(');end=src.index('\n\ndef main',start)
src=src[:start]+'''def input_paths(root, cfg, ids, stage):
    cache=Path(cfg['cache']);control=Path(cfg['control_run'])
    paths=[root/'RUN_CONFIG.json',root/'RUN_IDS.json',root/'SPLIT.json',root/'CODE_READY.json',
           Path(cfg['weights']),Path(cfg['source_root'])/'runner.py',cache/'INDEX.json',cache/'CACHE_IDENTITY.json',
           Path(cfg['control_config']),Path(cfg['control_checkpoint']),control/'COMPLETE.json',control/'MODEL.json',
           control/'RESOLVED_CONFIG.json',Path(cfg['annotations_train']),Path(cfg['prior_results'])]
    if stage!='prepare': paths += [Path(cfg['feature_cache']),Path(cfg['prepare_run'])/'COMPLETE.json']
    if stage.startswith('train_') or stage=='evaluation':paths += [Path(cfg['smoke_run'])/'COMPLETE.json']
    if stage=='evaluation':
        for ck in cfg['checkpoints'].values():
            p=Path(ck);paths += [p,p.parent/'COMPLETE.json',p.parent/'MODEL.json',p.parent/'RESOLVED_CONFIG.json']
    for p in paths:
        if not p.is_file():raise FileNotFoundError(str(p))
    return list(dict.fromkeys(paths))
'''+src[end:]
needle='            state.update(stage=stage, current_run=ids["runs"][stage], stage_started_at=now(), command=command)'
src=src.replace(needle,"            if stage.startswith('train_'): command += ['--mode',stage.split('_',1)[1]]\n"+needle)
src=src.replace('fixed four-stage queue','fixed six-stage queue')
(root/'scripts/pipeline_bridge.py').write_text(src,encoding='utf-8')
print('pipeline ready')
