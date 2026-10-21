"""Pure-stdlib independent bootstrap/RLE verifier using saved sampling indices."""
from pathlib import Path
import argparse,ast,hashlib,json,math,struct,zipfile


def npy(data):
    if data[:6]!=b'\x93NUMPY':raise ValueError('not npy')
    v=data[6];start=10 if v==1 else 12
    length=int.from_bytes(data[8:start],'little');header=ast.literal_eval(data[start:start+length].decode('latin1').strip())
    if header['fortran_order'] or header['descr'] not in ['<i8','<i4']:
        raise ValueError('unsupported saved integer indices')
    code='q' if header['descr']=='<i8' else 'i';size=struct.calcsize(code)
    values=[x[0] for x in struct.iter_unpack('<'+code,data[start+length:])]
    shape=header['shape']
    if len(shape)==1:return values
    if len(shape)==2:return [values[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]
    raise ValueError('unsupported shape')


def rle_counts(rle):
    source=rle['counts'];values=[];p=0
    while p<len(source):
        x=0;k=0
        while True:
            c=ord(source[p])-48;p+=1;x|=(c&31)<<(5*k);k+=1
            if not c&32:
                if c&16:x|=(-1)<<(5*k)
                break
        if len(values)>2:x+=values[-2]
        if x<0:raise AssertionError('negative RLE run')
        values.append(x)
    if sum(values)!=math.prod(rle['size']):raise AssertionError('RLE extent differs')
    return values


def intersection_union(left,right):
    a=rle_counts(left);b=rle_counts(right);i=j=0;ar=a[0];br=b[0];inter=union=0
    while i<len(a) and j<len(b):
        n=min(ar,br);pa=i%2;pb=j%2
        inter+=n if pa and pb else 0;union+=n if pa or pb else 0
        ar-=n;br-=n
        if ar==0:
            i+=1
            if i<len(a):ar=a[i]
        if br==0:
            j+=1
            if j<len(b):br=b[j]
    return inter,union


def mean(xs):return math.fsum(xs)/len(xs)


def quantile(xs,q):
    xs=sorted(xs);x=(len(xs)-1)*q;i=int(x);f=x-i
    return xs[i]*(1-f)+xs[min(i+1,len(xs)-1)]*f


def main():
    p=argparse.ArgumentParser();p.add_argument('run_path');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.run_path);rows=[json.loads(x) for x in (root/'PAIR_RESULTS.jsonl').read_text(encoding='utf8').splitlines()]
    summary=json.loads((root/'STATISTICS.json').read_text(encoding='utf8'))
    with zipfile.ZipFile(root/'BOOTSTRAP_INDICES.npz') as z:
        ids=npy(z.read('image_ids.npy'));samples=npy(z.read('samples.npy'))
    groups=[[r for r in rows if r['image_id']==i] for i in ids]
    checks=0
    for r in rows:
        if r['endpoints']['C']['rle']!=r['endpoints']['W']['rle']:raise AssertionError('nominal mask')
        for item in list(r['endpoints'].values())+[p[k] for p in r['pressure'] for k in ['C','W']]:
            if item is None:continue
            intersection,union=intersection_union(item['rle'],r['GT_rle'])
            if (intersection,union)!=(item['intersection'],item['union']):raise AssertionError('independent RLE counts')
            if abs(intersection/union-item['iou'])>1e-14:raise AssertionError('independent IoU')
            checks+=1
    verified=[]
    def test(values,record,label):
        draws=[mean([values[i] for i in sample]) for sample in samples]
        expected=[mean(values),quantile(draws,.025),quantile(draws,.975)]
        actual=[record['mean'],*record['ci95']]
        if max(abs(x-y) for x,y in zip(expected,actual))>2e-12:raise AssertionError(label)
        verified.append(label)
    def stress(r,c,arm=None):return mean([v[c]['iou'] for v in r['pressure'] if arm is None or v['arm']==arm])
    test([mean([stress(r,'W')-stress(r,'C') for r in g]) for g in groups],summary['stress_iou_gain'],'main_gain')
    for arm in ['S','R','M']:
        test([mean([stress(r,'W',arm)-stress(r,'C',arm) for r in g]) for g in groups],summary['per_direction_gain'][arm],arm+'_gain')
    if summary['geometric_margin_gain'] is not None:
        test([mean([r['margin_W']-r['margin_C'] for r in g]) for g in groups],summary['geometric_margin_gain'],'margin_gain')
    for center in ['C','W']:
        for ref in ['N','G_old']:
            for pressured in [False,True]:
                key=f'{center}{"_stress" if pressured else ""}-{ref}'
                if summary['reference_comparisons'].get(key) is not None:
                    test([mean([(stress(r,center) if pressured else r['endpoints'][center]['iou'])-r['endpoints'][ref]['iou'] for r in g]) for g in groups],summary['reference_comparisons'][key],key)
    wr=summary['witness_retention']
    if wr is not None:
        den=[sum(r['endpoints']['C']['iou']>=.75 for r in g) for g in groups]
        numer={c:[math.fsum(mean([p[c]['iou']>=.75 for p in r['pressure']]) for r in g if r['endpoints']['C']['iou']>=.75) for g in groups] for c in ['C','W']}
        delta=[x-y for x,y in zip(numer['W'],numer['C'])]
        draws=[math.fsum(delta[i] for i in s)/sum(den[i] for i in s) for s in samples]
        computed=[math.fsum(delta)/sum(den),quantile(draws,.025),quantile(draws,.975)]
        if max(abs(x-y) for x,y in zip(computed,[wr['gain'],*wr['ci95']]))>2e-12:raise AssertionError('witness bootstrap')
        verified.append('witness_bootstrap')
    report={'status':'passed','RLE_checks':checks,'independent_statistics':verified,'pairs':len(rows),
        'GPU_replayed_locally':False,'saved_sampling_indices_used':True,'same_panel_G_budget_available':False}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':main()
