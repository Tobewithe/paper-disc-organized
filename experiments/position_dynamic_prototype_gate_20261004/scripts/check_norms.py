import torch
for mode,path in [('G','/root/position_dynamic_prototype_gate_20261004/runs/RUN_a3b080cfa2cd40bd92b6b94bdcbc4bbe/final.pt'),('P','/root/position_dynamic_prototype_gate_20261004/runs/RUN_b98febfe0a464c70890e8d32abbd9e81/final.pt')]:
 s=torch.load(path,map_location='cpu',weights_only=False)['state_dict']; print(mode,{k:float(v.norm()) for k,v in s.items() if not k.startswith('native_cv4.')})
