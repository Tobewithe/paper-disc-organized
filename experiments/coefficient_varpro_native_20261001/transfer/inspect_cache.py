import torch, json
p=r'D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache\images\000000000113.pt'
x=torch.load(p,map_location='cpu',weights_only=False)
print(sorted(x.keys()))
print('seg_gain', x.get('segmentation_gain'))
print('proto', tuple(x['proto'].shape), 'masks', tuple(x['masks'].shape), 'rows', len(x['rows']))
print('row0', x['rows'][0])
print('target_box0', x['target_boxes'][0].tolist(), 'owner0', int(x['owners'][0]))
print('coeff0', tuple(x['coeff'].shape), 'h', tuple(x['h'].shape))
