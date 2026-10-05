"""Validate census aggregates and produce a Chinese report and data-only figure."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'census'
BINS=['I0_zero','I1_(0,0.5]','I2_(0.5,1]','I3_>1']

def n(x):return f'{x:,}'
def pct(x):return f'{x*100:.2f}%'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    r=json.loads((OUT/'COCO2017_CROWDING_CENSUS.json').read_text(encoding='utf-8'))
    assert r['status']=='PASS'
    full={s:r['splits'][s]['scopes']['full'] for s in ['train2017','val2017']}
    dense={s:r['splits'][s]['scopes']['legacy_dense'] for s in full}
    validation={}
    for split in full:
        with (OUT/f'{split}_images.csv').open(encoding='utf-8-sig') as f:
            images=list(csv.DictReader(f))
        assert len(images)==full[split]['images']
        assert sum(int(x['n_instances']) for x in images)==full[split]['valid_noncrowd_instances']
        assert sum(int(x['n_high_same']) for x in images)==full[split]['high_same']
        assert sum(int(x['legacy_selected']) for x in images)==dense[split]['images']
        assert all(x['legacy_selected']==x['historical_manifest_selected'] for x in images)
        bins=Counter();ids=set();high=0;highall=0;valid=0;near_half=near_one=0
        with (OUT/f'{split}_instances.csv').open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                aid=int(row['annotation_id']);assert aid not in ids;ids.add(aid)
                valid+=1;bins[row['ici_bin']]+=1
                v=float(row['ici_same']);high+=v>.5;highall+=float(row['ici_all'])>.5
                near_half+=abs(v-.5)<1e-12;near_one+=abs(v-1)<1e-12
        assert valid==full[split]['valid_noncrowd_instances'] and high==full[split]['high_same']
        assert highall==full[split]['high_all'] and dict(bins)==full[split]['same_class_bins']
        validation[split]=dict(image_rows=len(images),unique_annotation_rows=valid,aggregates_match=True,
            near_0_5_within_1e_12=near_half,near_1_within_1e_12=near_one,
            note='Canonical bins snap within absolute 1e-10 of 0.5 or 1; raw per-instance values and historical selection remain available.')
    lines=[
        '# COCO 2017 全量拥挤统计', '',
        '日期：2026-09-11。依据服务器公共 COCO2017 原始实例标注与图片压缩包重新计算；未读取模型预测。', '',
        '**主要结论：旧训练子集的 ICI 筛选没有选错，重算图像 ID 与历史清单完全一致。它覆盖 99.26% 的同类高拥挤实例，但子集内只有 24.55% 的实例满足高拥挤定义；图片入选不等于图中所有目标都拥挤。**', '',
        '## 1. 覆盖范围与计数口径', '',
        '本次覆盖服务器现有 COCO 2017 的 train2017、val2017、test2017。不是对 COCO 2014、unlabeled、panoptic 等其他发布包的盘点。训练和验证采用原始 instances JSON；每个 annotation ID 只计一个实例，不按多边形数量拆分。', '',
        '| 划分 | 图片 | 有效普通实例 | crowd 标注 | 同类高拥挤实例 | 高拥挤占比 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for split in full:
        f=full[split];s=r['splits'][split]
        lines.append(f"| {split} | {n(f['images'])} | {n(f['valid_noncrowd_instances'])} | {n(s['crowd'])} | {n(f['high_same'])} | {pct(f['high_same_fraction'])} |")
    lines += [f"| test2017 | {n(r['archives']['test2017']['image_count'])} | 当前无 GT，不能计算 | — | — | — |", '',
        'train2017 原有 849,949 条非 crowd 标注，其中 2 条 bbox 高度为 0（annotation ID 918、2206849），无法定义 ICI 分母，已单列；表中实例分母为 849,947。val2017 无此问题。crowd 标注不作为能一对一分割评分的普通实例，未混入分母。', '',
        '## 2. ICI 究竟测什么', '',
        r'对实例 $i$，$d_i=\sum_{j\ne i,\,y_j=y_i}|B_i\cap B_j|/|B_i|$，以原始 bbox 计算，排除 crowd。', '',
        '- 实例 ICI 衡量与其他同类别 bbox 的累计重叠程度，不是拥挤目标数量，也不是实际可见掩码的遮挡率。',
        '- 同一像素与多个邻框的重叠可能重复累计，因此 ICI 可以大于 1。',
        '- 图片统计另外保存最大 ICI、平均 ICI、ICI>0.5 的实例数，以及这些实例占图中普通实例的比例。',
        '- 历史图片筛选：仅让 bbox 面积≥100 像素的实例作为参考目标，再判断 max ICI>0.5；邻居仍可为小目标。新的实例统计保留所有 bbox 有效的普通实例。', '',
        '## 3. 完整数据中的实例分层', '',
        '| 实例层 | train2017 | val2017 |', '|---|---:|---:|',
    ]
    labels=['ICI=0','0<ICI≤0.5','0.5<ICI≤1','ICI>1']
    for label,b in zip(labels,BINS):lines.append(f"| {label} | {n(full['train2017']['same_class_bins'].get(b,0))} | {n(full['val2017']['same_class_bins'].get(b,0))} |")
    lines += ['', '## 4. 一张图里到底有多少高拥挤实例', '',
        '| 高拥挤实例数/图 | 全 train2017 图片 | 全 val2017 图片 |', '|---|---:|---:|']
    for b in ['0','1','2-4','5-9','10+']:
        lines.append(f"| {b} | {n(full['train2017']['high_same_count_distribution'].get(b,0))} | {n(full['val2017']['high_same_count_distribution'].get(b,0))} |")
    lines += ['', '这张表与 max ICI 互补：只出现一个严重重叠目标和同时出现十个严重重叠目标，可能都满足同一个图片筛选阈值，但场景组成不同。', '',
        '| 高拥挤实例占一张图普通实例的比例 | train2017 图片 | val2017 图片 |', '|---|---:|---:|']
    for b in ['0','(0,25%]','(25%,50%]','(50%,75%]','(75%,100%]']:
        lines.append(f"| {b} | {n(full['train2017']['high_same_ratio_distribution'].get(b,0))} | {n(full['val2017']['high_same_ratio_distribution'].get(b,0))} |")
    lines += ['', '没有普通 GT 的图片记为 0 比例并单独计数：train2017 有 1,021 张，val2017 有 48 张。', '',
        '## 5. 旧子集核验', '',
        '| 旧子集 | 图片 | 有效普通实例 | 同类高拥挤实例 | 子集内高拥挤占比 | 覆盖全量高拥挤实例 |', '|---|---:|---:|---:|---:|---:|']
    for split in full:
        d=dense[split]
        lines.append(f"| {split} Dense | {n(d['images'])} | {n(d['valid_noncrowd_instances'])} | {n(d['high_same'])} | {pct(d['high_same_fraction'])} | {pct(r['splits'][split]['legacy_dense_high_same_coverage'])} |")
    lines += ['',
        '训练清单 37,433 个 ID 和验证清单 1,576 个 ID 均与原规则重算结果完全一致。旧子集遗漏的同类高拥挤实例是 889 个训练实例、39 个验证实例，全部属于 bbox 面积<100 像素的参考目标。', '',
        '注意三个计数层次：旧训练子集有 488,877 条原始非 crowd 标注；排除 2 条零高度框后，拥挤统计使用 488,875 个实例；官方转换器还按 class+bbox 去重 5 条，实际生成 488,870 条训练标签。差异逐 ID 留档，不把官方去重说成多边形拆分。', '',
        '## 6. 同类别与跨类别不能混称', '',
        '| 定义 | 全 train2017 高拥挤实例 | 全 val2017 高拥挤实例 |', '|---|---:|---:|']
    for key,label in [('same','只计算同类别 bbox 重叠'),('all','计算所有类别 bbox 重叠')]:
        lines.append(f"| {label} | {n(full['train2017']['high_'+key])}（{pct(full['train2017']['high_'+key+'_fraction'])}） | {n(full['val2017']['high_'+key])}（{pct(full['val2017']['high_'+key+'_fraction'])}） |")
    lines += ['',
        '旧 ICI 筛选针对同类拥挤；原 CCL 的 IoU>0.05 配对允许跨类别，二者并非同一集合。跨类别 bbox 重叠还可能来自人与自行车等正常空间关系，所以主分析保留同类 ICI，所有类别 ICI 作为另行标注的次分析，不能依效果挑选定义。', '',
        '## 7. 对训练与论文的建议', '',
        '1. 原密集训练子集可继续用作修复标签后的原方法对照，它已经覆盖绝大多数同类高拥挤实例。正式训练前把它准确称为“含同类拥挤实例的图片子集”。',
        '2. 每张训练图保留官方流程允许的全部普通 GT，正常有监督训练；不能删除低拥挤 GT 后将它们隐式当作背景。',
        '3. 主评估使用完整 val2017。先做全部 GT 的官方匹配，再按每个 GT 的 ICI 分层，比较高拥挤召回与低拥挤召回的变化。',
        '4. 将“每图高拥挤实例数”和“占比”作为场景层面的补充描述；不再仅用 max ICI 把图片内所有实例统称为高拥挤。',
        '5. 本统计能证明高拥挤实例在集合中的占比，不能单独证明此前 AP 差异由稀释导致。AP 还涉及类别平均、置信度排序、误检和匹配，必须等公平训练后的分层结果。', '',
        '## 8. 验证和可复现文件', '',
        '逐实例 CSV 的 annotation ID 无重复，实例/图片/类别表与汇总完全对齐。独立标量公式核验 238 张训练图和 11 张验证图，最大数值误差分别为 4.44e-16 和 0。正式训练前另发现边界浮点误差：原始值极接近 1 却略大于 1，会错误进入 ICI>1 组。现将距 0.5 或 1 不超过 1e-10 的值归到精确边界，原始 ICI 列保留，旧图片筛选清单不变。', '',
        '修正前后：训练高拥挤实例 120,923→120,921；验证高拥挤仍为 5,245。ICI>1 组由训练 38,770→34,496、验证 1,708→1,523。原始 float64 版本在服务器 census_raw_float_archive 中留档；不能继续引用其极高拥挤组计数。', '',
        '- `COCO2017_CROWDING_CENSUS.json`：全量汇总、源 JSON 哈希、原筛选核验。',
        '- `train2017_instances.csv` / `val2017_instances.csv`：每个实例的原始 ID、类别、尺度、同类/跨类 ICI。',
        '- `train2017_images.csv` / `val2017_images.csv`：每张图片的拥挤数量、占比、最大值和旧子集归属。',
        '- `train2017_categories.csv` / `val2017_categories.csv`：80 类分别统计。',
        '- `CENSUS_VALIDATION.json`：本地逐表汇总复核。',
        '- `crowding_overview.png` / `.pdf`：由完整统计生成的图。', '',
        '当前正式训练尚未启动；本次先完成全量数据分布核验。', '',
    ]
    (OUT/'COCO2017_CROWDING_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'CENSUS_VALIDATION.json').write_text(json.dumps(validation,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    scopes=[('Train full',full['train2017']),('Train dense',dense['train2017']),('Val full',full['val2017']),('Val dense',dense['val2017'])]
    bottom=np.zeros(4)
    colors=['#c8d9ec','#82abc9','#eea150','#be4c49']
    for b,label,color in zip(BINS,labels,colors):
        label={'ICI=0':'ICI = 0','0<ICI≤0.5':'0 < ICI ≤ 0.5','0.5<ICI≤1':'0.5 < ICI ≤ 1','ICI>1':'ICI > 1'}[label]
        vals=np.array([s['same_class_bins'].get(b,0)/s['valid_noncrowd_instances']*100 for _,s in scopes])
        ax[0].bar(range(4),vals,bottom=bottom,color=color,label=label)
        bottom+=vals
    ax[0].set_xticks(range(4),[s[0] for s in scopes]);ax[0].set_ylim(0,100)
    ax[0].set_ylabel('Share of valid non-crowd GT (%)');ax[0].set_title('Instance crowding within each image set')
    ax[0].legend(loc='lower left',fontsize=9)
    bands=['0','1','2-4','5-9','10+'];x=np.arange(len(bands))
    for shift,split,color in [(-.19,'train2017','#417da7'),(.19,'val2017','#d27636')]:
        vals=[full[split]['high_same_count_distribution'].get(b,0)/full[split]['images']*100 for b in bands]
        bars=ax[1].bar(x+shift,vals,.38,label=split,color=color)
        ax[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax[1].set_xticks(x,bands);ax[1].set_ylim(0,80);ax[1].set_xlabel('Number of same-class high-crowding instances / image')
    ax[1].set_ylabel('Share of all images (%)');ax[1].set_title('Crowding count is different from maximum ICI');ax[1].legend()
    fig.savefig(OUT/'crowding_overview.png',dpi=180)
    fig.savefig(OUT/'crowding_overview.pdf')
    plt.close(fig)
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='ARTIFACT_SHA256.json']
    (OUT/'ARTIFACT_SHA256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(files)},indent=2))
    print('CENSUS_REPORT_PASS',validation,flush=True)

if __name__=='__main__':main()
