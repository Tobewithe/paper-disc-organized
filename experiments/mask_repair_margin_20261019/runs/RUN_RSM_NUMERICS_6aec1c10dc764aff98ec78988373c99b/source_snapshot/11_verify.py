"""Independent RLE/pressure/center checks, optionally full native replay."""
from common import *
import argparse


def verify(run_path,replay=False):
    import numpy as np
    from pycocotools import mask as m
    from margin_stats import analyze
    path=Path(run_path);cfg=read_json(path/'CONFIG.json');rows=lines(path/'PAIR_RESULTS.jsonl')
    for row in rows:
        require(row['endpoints']['C']['rle']==row['endpoints']['W']['rle'],'nominal mask changed')
        require(abs(row['W_logit_rms']/row['C_logit_rms']-1)<=cfg['numerics']['rms_relative_tolerance'],'center scale')
        require(abs(row['pressure_amplitude']/row['C_logit_rms']-cfg['pressure_ratio'])<1e-12,'pressure ratio')
        measured=list(row['endpoints'].values())+[p[k] for p in row['pressure'] for k in ['C','W']]
        gt=m.decode(row['GT_rle']).astype(bool)
        for item in measured:
            if item is None:continue
            binary=m.decode(item['rle']).astype(bool)
            intersection=int(np.count_nonzero(binary & gt));union=int(np.count_nonzero(binary | gt))
            require(intersection==item['intersection'] and union==item['union'],'independent RLE intersection/union differs')
            require(abs(item['iou']-(intersection/union if union else 0.))<1e-14,'RLE IoU differs')
        for p in row['pressure']:
            for key in ['C_actual_rms','W_actual_rms']:
                require(abs(p[key]/row['pressure_amplitude']-1)<=cfg['numerics']['rms_relative_tolerance'],'pressure scale mismatch')
    statistics=read_json(path/'STATISTICS.json')
    recalculated=analyze(rows,cfg,read_json(path/'run.json')['scope'])
    require(all(statistics[k]==v for k,v in recalculated.items()),'saved statistics differ')
    decoded=0
    if replay:
        import torch
        from adapter_data import FrozenCache
        from native_output_bridge import decode_native,encode_rle
        from native_geometry import roi_for
        from margin_math import draw
        from native_ops import Decoder
        configure_native();library=np.load(FIT/'FIT_ERRORS.npz')
        source=FrozenCache(BASE/'trace',sorted({r['image_id'] for r in rows}))
        for iid in source.image_ids:
            cache,_=source.load(iid)
            for row in [r for r in rows if r['image_id']==iid]:
                rank=row['rank'];dec=Decoder(cache,rank)
                vectors,labels=draw(library['errors'],library['mean'],library['image_ids'],iid,rank,cfg)
                require(len(labels)==len(row['pressure']),'pressure count')
                c=torch.tensor(row['C_coefficients'],dtype=torch.float32);w=torch.tensor(row['W_coefficients'],dtype=torch.float32)
                for name,value in [('N',cache['c0'][rank]),('C',c),('W',w)]:
                    full=cache['c0'].clone();full[rank]=value
                    actual=decode_native(cache,full)[rank]
                    require(encode_rle(actual)==row['endpoints'][name]['rle'],'independent full-shape endpoint replay differs')
                    decoded+=1
                require(abs(dec.rms(c.numpy())[0]/row['C_logit_rms']-1)<=cfg['numerics']['rms_relative_tolerance'],'C RMS recheck')
                require(abs(dec.rms(w.numpy())[0]/row['W_logit_rms']-1)<=cfg['numerics']['rms_relative_tolerance'],'W RMS recheck')
                for i,p in enumerate(row['pressure']):
                    require(all(p[k]==v for k,v in labels[i].items()) and np.array_equal(vectors[i],p['vector']),'donor/permutation replay differs')
                    for center,value in [('C',c),('W',w)]:
                        coeff=np.array(p[center+'_coefficients'],dtype=np.float32)
                        expected=(value.numpy().astype(float)+vectors[i]*row['pressure_amplitude']/dec.rms(vectors[i])[0]).astype(np.float32)
                        require(np.array_equal(coeff,expected),'pressure construction differs')
                        amplitude=dec.rms(coeff.astype(float)-value.numpy().astype(float))[0]
                        require(abs(amplitude/row['pressure_amplitude']-1)<=cfg['numerics']['rms_relative_tolerance'],'independent actual pressure RMS')
                        # Full original coefficient matrix and original shared decoder, not selected shortcut.
                        full=cache['c0'].clone();full[rank]=torch.from_numpy(coeff)
                        actual=decode_native(cache,full)[rank]
                        require(encode_rle(actual)==p[center]['rle'],'independent full-shape pressure replay differs')
                        decoded+=1
                del dec
            torch.cuda.empty_cache()
        source.verify()
    return {'status':'passed','pairs':len(rows),'independent_RLE_checks':sum(3+int(r['endpoints']['G_old'] is not None)+2*len(r['pressure']) for r in rows),
            'full_native_replays':decoded,'statistics_recomputed':True,'new_training_steps':0,'G_budget_same_panel_available':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_path');p.add_argument('--replay',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    write_new(Path(a.output),verify(a.run_path,a.replay))
