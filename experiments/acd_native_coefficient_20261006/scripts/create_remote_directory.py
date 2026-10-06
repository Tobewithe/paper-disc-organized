from pathlib import Path
p=Path('D:/coco_wire/experiments/acd_native_coefficient_20261006')
(p/'scripts').mkdir(parents=True,exist_ok=True)
(p/'runs').mkdir(exist_ok=True)
print(p)
