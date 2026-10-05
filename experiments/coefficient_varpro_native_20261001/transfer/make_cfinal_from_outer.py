import torch, sys
src=sys.argv[1]; dst=sys.argv[2]
x=torch.load(src,map_location='cpu',weights_only=False)
torch.save({'cv4':x['cv4']},dst)
print(dst)
