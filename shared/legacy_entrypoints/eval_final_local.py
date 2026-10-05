import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    print("Loading Baseline Model...")
    baseline_path = r"C:\Dpan\codexproject\paper-disc\runs\segment\experiments\baseline_finetune\run_baseline_15e\weights\best.pt"
    baseline_model = YOLO(baseline_path)
    print("Evaluating Baseline Model on Test Set...")
    baseline_metrics = baseline_model.val(data=r"C:\Dpan\document\model_datasets\datasets\FaroPigSeg\data.yaml", split="test", imgsz=640)
    
    print("\nLoading CCL Model...")
    ccl_path = r"C:\Dpan\codexproject\paper-disc\runs\segment\experiments\ccl_true_finetune\run_ccl_true_15e\weights\best.pt"
    ccl_model = YOLO(ccl_path)
    print("Evaluating CCL Model on Test Set...")
    ccl_metrics = ccl_model.val(data=r"C:\Dpan\document\model_datasets\datasets\FaroPigSeg\data.yaml", split="test", imgsz=640)
    
    print("\n" + "="*50)
    print("    LOCAL END-TO-END EVALUATION (Faro Test Set)    ")
    print("="*50)
    print(f"{'Metric':<15} | {'Baseline':<10} | {'CCL Model':<10} | {'Delta':<10}")
    print("-" * 50)
    
    def print_metric(name, b_val, c_val):
        delta = c_val - b_val
        print(f"{name:<15} | {b_val:<10.4f} | {c_val:<10.4f} | {delta:>+10.4f}")
    
    # Extract metrics
    # baseline_metrics.box.map50, baseline_metrics.seg.map50
    print_metric("Box mAP@50", baseline_metrics.box.map50, ccl_metrics.box.map50)
    print_metric("Box mAP@50-95", baseline_metrics.box.map, ccl_metrics.box.map)
    print_metric("Mask mAP@50", baseline_metrics.seg.map50, ccl_metrics.seg.map50)
    print_metric("Mask mAP@50-95", baseline_metrics.seg.map, ccl_metrics.seg.map)
    print("="*50)

if __name__ == '__main__':
    main()
