import sys
import os
from pathlib import Path

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

def main():
    test_yaml = Path('experiments/ccl_finetune/faro_test.yaml')
    test_yaml.write_text(
        "path: C:/Dpan/document/model_datasets/datasets/FaroPigSeg\n"
        "train: test/images\n"
        "val: test/images\n"
        "names:\n"
        "  0: pig\n"
    )
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Control Group (Finetuned without CCL)
    control_weights = Path('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt')
    print("Loading Control Model (Standard Finetuned)...")
    control_model = load_model(control_weights, 'cuda')
    print("Evaluating Control Model on Test Set...")
    control_metrics = control_model.val(data=str(test_yaml), split='val', imgsz=640, device='cuda', verbose=False)
    
    # CCL Model (Finetuned with CCL)
    ccl_weights = Path('runs/segment/experiments/ccl_finetune/run_ccl_15e/weights/best.pt')
    print("Loading CCL Model...")
    ccl_model = load_model(ccl_weights, 'cuda')
    print("Evaluating CCL Model on Test Set...")
    ccl_metrics = ccl_model.val(data=str(test_yaml), split='val', imgsz=640, device='cuda', verbose=False)
    
    print("\n" + "="*50)
    print("    RIGOROUS END-TO-END EVALUATION (Faro Test Set)    ")
    print("="*50)
    print(f"{'Metric':<15} | {'Control':<10} | {'CCL Model':<10} | {'Pure Delta':<10}")
    print("-" * 50)
    
    b_box_mAP50 = control_metrics.box.map50
    c_box_mAP50 = ccl_metrics.box.map50
    print(f"{'Box mAP@50':<15} | {b_box_mAP50:.4f}     | {c_box_mAP50:.4f}     | {c_box_mAP50 - b_box_mAP50:+.4f}")
    
    b_box_mAP = control_metrics.box.map
    c_box_mAP = ccl_metrics.box.map
    print(f"{'Box mAP@50-95':<15} | {b_box_mAP:.4f}     | {c_box_mAP:.4f}     | {c_box_mAP - b_box_mAP:+.4f}")
    
    b_seg_mAP50 = control_metrics.seg.map50
    c_seg_mAP50 = ccl_metrics.seg.map50
    print(f"{'Mask mAP@50':<15} | {b_seg_mAP50:.4f}     | {c_seg_mAP50:.4f}     | {c_seg_mAP50 - b_seg_mAP50:+.4f}")
    
    b_seg_mAP = control_metrics.seg.map
    c_seg_mAP = ccl_metrics.seg.map
    print(f"{'Mask mAP@50-95':<15} | {b_seg_mAP:.4f}     | {c_seg_mAP:.4f}     | {c_seg_mAP - b_seg_mAP:+.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
