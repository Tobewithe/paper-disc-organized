import sys
import os
from pathlib import Path
sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    test_yaml = 'experiments/ccl_true_finetune/faro_test.yaml'
    
    print("Evaluating True CCL Model on Test Set...")
    ccl = load_model(Path('runs/segment/experiments/ccl_true_finetune/run_ccl_true_15e/weights/best.pt'), 'cuda')
    metrics = ccl.val(data=test_yaml, split='val', imgsz=640, device='cuda', verbose=False)
    
    print(f"CCL Box mAP50: {metrics.box.map50:.4f}")
    print(f"CCL Box mAP50-95: {metrics.box.map:.4f}")
    print(f"CCL Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"CCL Mask mAP50-95: {metrics.seg.map:.4f}")

if __name__ == '__main__':
    main()
