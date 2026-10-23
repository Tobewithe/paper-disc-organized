"""Usable inference entry point: RGB + known camera -> learned complete object."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from object_model import ObjectModel
from runtime import read,dump,sha


@torch.no_grad()
def infer(model_dir,image,camera_file,output,arm):
    manifest=read(model_dir/'MODEL_MANIFEST.json');record=manifest['models'][arm]
    checkpoint=model_dir/record['file'];template_path=model_dir/manifest['inference_template']['file']
    assert sha(checkpoint)==record['sha256']
    assert sha(template_path)==manifest['inference_template']['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['template_sha256']==manifest['source_template_sha256']
    template={k:v.copy() for k,v in np.load(template_path).items()}
    model=ObjectModel(template,saved['config'],structured=arm=='structured')
    model.load_state_dict(saved['model']);model.eval()
    rgb=np.asarray(Image.open(image).convert('RGB')).copy()
    size=saved['config']['image_size']
    assert rgb.shape==(size,size,3),'Provide the calibrated 128x128 input; no silent resize/crop is applied.'
    camera=read(camera_file)
    matrix=np.asarray(camera.get('matrix',camera.get('camera')) if isinstance(camera,dict) else camera,dtype=np.float32)
    assert matrix.shape==(3,4) and np.allclose(matrix[:,:3]@matrix[:,:3].T,np.eye(3),atol=1e-5)
    tensor=torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
    result=model(tensor,torch.from_numpy(matrix)[None])
    output.mkdir(parents=True,exist_ok=False)
    arrays={k:result[k][0].numpy() for k in ['vertices','shape','pose','features','log_sigma']}
    np.savez_compressed(output/'object_prediction.npz',**arrays)
    with open(output/'object_prediction.obj','w',encoding='utf-8') as f:
        f.write('# Learned object prediction from RGB and known camera.\n')
        for v in arrays['vertices']:f.write('v '+' '.join(f'{x:.8f}' for x in v)+'\n')
        for face in template['faces']:f.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
    dump(output/'INFERENCE.json',dict(arm=arm,model_sha256=sha(checkpoint),image_sha256=sha(image),camera_sha256=sha(camera_file),
                                     input_target_shape_or_pose=False,vertices=len(arrays['vertices']),feature_dim=arrays['features'].shape[-1],
                                     scope=manifest['scope']))
    print(json.dumps({'output':str(output),'vertices':len(arrays['vertices']),'feature_dim':arrays['features'].shape[-1]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--camera',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['structured','free'],default='structured')
    a=p.parse_args();torch.set_num_threads(2);infer(a.model_dir,a.image,a.camera,a.output,a.arm)
