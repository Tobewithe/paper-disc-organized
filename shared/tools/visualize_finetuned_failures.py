import json
from pathlib import Path
import cv2
import sys

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

def main():
    # Target images from the carousel
    target_img_ids = [33, 7, 92, 86, 79, 49]
    
    manifest = json.loads(Path('C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/manifest.json').read_text(encoding='utf-8'))
    images_by_id = {img['id']: img for img in manifest['images']}
    
    img_root = Path('C:/Dpan/document/model_datasets/datasets/FaroPigSeg/test/images')
    
    # Load the standard fine-tuned model (which has the "facade" 95% mAP)
    weights = Path('runs/segment/experiments/baseline_finetune/run_baseline_15e/weights/best.pt')
    model = load_model(weights, 'cuda')
    
    out_dir = Path('C:/Users/Administrator/.gemini/antigravity/brain/08659641-44b7-4544-8b51-2274a96336ad/images/finetuned_e2e_preds')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id in target_img_ids:
        img_info = images_by_id[img_id]
        img_path = img_root / Path(img_info['file_name']).name
        
        # Real end-to-end prediction (NO ORACLE)
        results = model.predict(source=str(img_path), conf=0.25, retina_masks=True, save=False, verbose=False)
        
        # We can just plot the result directly
        res = results[0]
        plotted = res.plot()
        
        out_path = out_dir / f"pred_img{img_id}.jpg"
        cv2.imwrite(str(out_path), plotted)
        print(f"Saved E2E prediction for img{img_id} to {out_path}")

if __name__ == '__main__':
    main()
