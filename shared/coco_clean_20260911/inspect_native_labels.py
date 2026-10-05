"""Archive the actual installed native label path and persisted training evidence."""
import inspect,json,hashlib,shutil
from pathlib import Path
import ultralytics
from ultralytics.data import augment,utils,converter,base,dataset,build
from ultralytics.utils import ops
from ultralytics.cfg import DEFAULT_CFG_DICT
R=Path('/root/autodl-tmp/coco_clean_20260911');out=R/'diagnostics/native_label_source_20260912'
out.mkdir(exist_ok=False);(out/'source').mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
modules={'augment':augment,'data_utils':utils,'converter':converter,'base':base,'dataset':dataset,'build':build,'ops':ops}
files={}
for name,module in modules.items():
    source=Path(inspect.getfile(module));shutil.copy2(source,out/'source'/f'{name}.py');files[name]=dict(path=str(source),sha256=sha(source))
functions={
    'polygon2mask':utils.polygon2mask,'polygons2masks':utils.polygons2masks,'polygons2masks_overlap':utils.polygons2masks_overlap,
    'Format_call':augment.Format.__call__,'Format_masks':augment.Format._format_segments,
    'LetterBox_call':augment.LetterBox.__call__,'dataset_update_labels_info':dataset.YOLODataset.update_labels_info,
    'dataset_build_transforms':dataset.YOLODataset.build_transforms,'base_load_image':base.BaseDataset.load_image,
    'base_get_image_and_label':base.BaseDataset.get_image_and_label,'verify_image_label':utils.verify_image_label,
    'merge_multi_segment':converter.merge_multi_segment,'convert_coco':converter.convert_coco,
    'build_yolo_dataset':build.build_yolo_dataset}
for name,fn in functions.items():(out/f'{name}.txt').write_text(inspect.getsource(fn))
run_evidence={}
for p in sorted((R/'runs').glob('*/launch_receipt.json')):
    obj=json.loads(p.read_text());run_evidence[str(p.relative_to(R))]=dict(sha256=sha(p),
        subset={k:obj.get(k) for k in ['seed','ccl_weight','train_images','val_images','train_instances','val_instances','criterion_branches']},
        resolved_args={k:obj.get('resolved_args',{}).get(k) for k in ['data','mask_ratio','overlap_mask','imgsz','mosaic','augment','rect','epochs']})
receipt=dict(version=ultralytics.__version__,module_files=files,
    defaults={k:DEFAULT_CFG_DICT.get(k) for k in ['mask_ratio','overlap_mask','imgsz','rect','mosaic']},run_evidence=run_evidence,
    training_config=json.loads((R/'train_config.json').read_text()),training_script_sha256=sha(R/'train_pair.py'),
    note='Installed source plus persisted local-run receipts. Official asset original pretraining recipe not reconstructed.')
(out/'SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
