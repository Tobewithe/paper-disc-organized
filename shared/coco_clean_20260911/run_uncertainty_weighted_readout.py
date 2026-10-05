"""S065: prediction-uncertainty weighted coefficient readout.

The weight is computed from the frozen mask logit uncertainty on the same
training samples used by the audited S032 pipeline. It is available at
inference because it depends only on frozen prototype responses and the
predicted coefficient, never on GT or ICI. The raw-COCO arm is the direct
same-capacity control.
"""
from pathlib import Path
base = Path(__file__).with_name('train_readout_label_controls.py').read_text(encoding='utf-8')
base = base.replace("MODES=['native_overlap','native_independent','raw_coco']", "MODES=['raw_coco','uncertainty_raw']")
base = base.replace("data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}", "data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}\n    # Prediction-only uncertainty: maximum near 0.5, computed from the\n    # frozen coefficient/prototype response at sampled pixels. Normalize its\n    # mean to one so the two arms have the same effective optimization scale.\n    z0=(data['p']*data['c'][:,None]).sum(-1)\n    u=4.0*z0.sigmoid()*(1.0-z0.sigmoid())\n    data['uncertainty_weight']=(1.0+4.0*u.mean(1))\n    data['uncertainty_weight']=data['uncertainty_weight']/data['uncertainty_weight'].mean()\n    data['uncertainty_raw']=data['raw_coco']")
base = base.replace("prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)['x']", "prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)")
base = base.replace("method='Existing global73D(h+level+box geometry)->128->128->128(mean4x32) coefficient residual. Same S026 architecture.',", "method='S065 same frozen global73D coefficient head; raw COCO control versus prediction-uncertainty weighted supervision.',")
base = base.replace("loss=loss_value(z,data[mode][ix],data['factor'][ix])", "loss=(loss_value(z,data[mode][ix],data['factor'][ix]) if mode=='raw_coco' else loss_value(z,data[mode][ix],data['factor'][ix]*data['uncertainty_weight'][ix]))")
start = base.find("            if mode=='native_overlap':\n                old=")
if start >= 0:
    end = base.find("            trained.append", start)
    base = base[:start] + base[end:]
generated = Path(__file__).with_name('_generated_uncertainty_weighted_runner.py')
generated.write_text(base, encoding='utf-8')
exec(compile(base, str(generated), 'exec'), globals(), globals())
