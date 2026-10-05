"""S063: density-weighted raw-COCO coefficient readout pilot.

Uses the audited S032 frozen-output pipeline. The only intervention is a
pre-registered weight based on same-class GT boundary exposure at the 640
input scale. The model, head capacity, labels, samples and evaluation remain
the same as the raw-COCO control.
"""
from pathlib import Path

base = Path(__file__).with_name('train_readout_label_controls.py').read_text(encoding='utf-8')
base = base.replace("MODES=['native_overlap','native_independent','raw_coco']", "MODES=['raw_coco','density_raw']")
base = base.replace("data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}", "data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}\n    data['density_raw'] = data['raw_coco']")
base = base.replace("prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)['x']", "prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)")
base = base.replace("method='Existing global73D(h+level+box geometry)->128->128->128(mean4x32) coefficient residual. Same S026 architecture.',", "method='S063 same frozen global73D coefficient head; raw COCO control versus boundary-exposure weighted raw COCO supervision.',")
base = base.replace("loss=loss_value(z,data[mode][ix],data['factor'][ix])", "loss=(loss_value(z,data[mode][ix],data['factor'][ix]) if mode=='raw_coco' else loss_value(z,data[mode][ix],data['factor'][ix]*data['density_weight'][ix]))")
start = base.find("            if mode=='native_overlap':\n                old=")
if start >= 0:
    end = base.find("            trained.append", start)
    base = base[:start] + base[end:]
# Insert deterministic boundary exposure weights before records are appended.
needle = "        hfeat=cuda(item['h']).float();lv=cuda(item['level']).long();boxes=cuda(item['boxes']).float()\n"
insert = "        # Same-class boundary exposure at input scale: fraction of target\n        # boundary pixels within 4 input pixels of another same-class GT.\n        # This is a training weight only; no GT is read at inference.\n        e4=[]\n        for aid in item['annotation_ids']:\n            ann0=gt.anns[int(aid)]; m0=gt.annToMask(ann0).astype('uint8')\n            same=np.zeros_like(m0)\n            for oo in gt.imgToAnns[int(iid)]:\n                if int(oo['id'])!=int(aid) and int(oo['category_id'])==int(ann0['category_id']) and not oo.get('iscrowd',0): same |= gt.annToMask(oo).astype('uint8')\n            er=cv2.erode(m0,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0); bd=(m0>0)&(er==0)\n            dist=cv2.distanceTransform((same==0).astype('uint8'),cv2.DIST_L2,3)\n            e4.append(float(((dist <= 4.0/max(gain,1e-6)) & bd).sum()/max(int(bd.sum()),1)))\n        density_weight=torch.tensor(1.0+3.0*np.asarray(e4,dtype=np.float32),device='cuda')\n"
if needle not in base: raise RuntimeError('runner insertion point missing')
base = base.replace(needle, insert + needle)
base = base.replace("factor=torch.tensor(item['loss_factor']).float(),", "factor=torch.tensor(item['loss_factor']).float(),density_weight=density_weight.float(),")
generated = Path(__file__).with_name('_generated_density_weighted_runner.py')
generated.write_text(base, encoding='utf-8')
exec(compile(base, str(generated), 'exec'), globals(), globals())
