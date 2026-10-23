import numpy as np
import torch
from torch.utils.data import Dataset

from geometry import skin
from runtime import read, sha


class Objects(Dataset):
    def __init__(self, folder, split, verify_hashes=True):
        self.folder = folder
        self.manifest = read(folder/'MANIFEST.json')
        assert sha(folder/'template.npz') == self.manifest['template_sha256']
        self.template = {k: v.copy() for k, v in np.load(folder/'template.npz').items()}
        self.rows = [r for r in self.manifest['samples'] if r['split'] == split]
        self.lookup = {r['sample_id']: i for i, r in enumerate(self.rows)}
        assert self.rows
        if verify_hashes:
            for row in self.rows:
                assert sha(folder/row['file']) == row['sha256'], row['file']
        self.camera_lookup = {(r['family'], r['pose'], r['view']): r['camera'] for r in self.rows}

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.folder/row['file']) as data:
            rgb = torch.from_numpy(data['rgb'].copy()).permute(2, 0, 1).float()/255
            canonical = torch.from_numpy(data['canonical'].astype(np.float32))
            visible = torch.from_numpy(data['visible'].copy())
            parts = torch.from_numpy(data['parts'].astype(np.int64))
        pair_id = f"f{row['family']:03d}_p{row['pose']}_v{row['view']}_a{1-row['appearance']}_o{1-row['occluded']}"
        pair = self.rows[self.lookup[pair_id]]
        with np.load(self.folder/pair['file']) as data:
            pair_rgb = torch.from_numpy(data['rgb'].copy()).permute(2, 0, 1).float()/255
        z = torch.tensor(row['shape'], dtype=torch.float32)
        rest = torch.from_numpy(self.template['canonical'])+torch.einsum('d,dvc->vc', z, torch.from_numpy(self.template['shape_basis']))
        pose = torch.from_numpy(self.template['poses'][row['pose']])
        vertices = skin(rest[None], pose[None], torch.from_numpy(self.template['weights']))[0]
        return dict(rgb=rgb, pair_rgb=pair_rgb, camera=torch.tensor(row['camera']), shape=z, pose=pose, vertices=vertices,
                    target_camera=torch.tensor(self.camera_lookup[(row['family'], row['pose'], (row['view']+1)%3)]),
                    focal=torch.tensor(row['focal']), canonical=canonical, visible=visible, parts=parts,
                    index=index, family=row['family'], occluded=row['occluded'])
