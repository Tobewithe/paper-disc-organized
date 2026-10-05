import os
import argparse
from ultralytics import YOLO

# 核心注入代码：挂载 CCL Loss (代码会和之前 train_ccl_true 相同)
# ---------------------------------------------------------
import ultralytics.utils.loss
original_v8SegmentationLoss = ultralytics.utils.loss.v8SegmentationLoss
# ... (此处省略几十行 monkey patch 细节，您可以直接复制之前的)
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=float, default=0.0)
    parser.add_argument('--margin', type=float, default=0.1)
    args = parser.add_argument()
    
    # 启用 CCL 标记，并将权重和边界值传进去
    os.environ['CCL_WEIGHT'] = str(args.weight)
    os.environ['CCL_MARGIN'] = str(args.margin)
    if args.weight > 0:
        os.environ['USE_CCL'] = '1'
    
    # 加载官方权重
    model = YOLO("yolov8m-seg.pt")
    
    # 开始微调 (15 Epochs)
    model.train(
        data="coco_dense.yaml",
        epochs=15,
        batch=16,          # 根据您的显存大小可调整至 32 或 8
        workers=8,
        project="coco_dense_ablations",
        name=f"ccl_w{args.weight}_m{args.margin}",
        lr0=0.001,         # 微调学习率不用太大
        patience=5
    )

if __name__ == '__main__':
    main()
