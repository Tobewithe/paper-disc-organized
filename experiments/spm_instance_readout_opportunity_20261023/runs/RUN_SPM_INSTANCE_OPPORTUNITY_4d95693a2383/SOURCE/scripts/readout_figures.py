from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from readout_runtime import dump


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def overlay(rgb, mask, color=(50, 210, 80)):
    image = rgb[:, :, ::-1].copy()
    value = image.astype(float)
    value[mask.astype(bool)] = .45*value[mask.astype(bool)]+.55*np.asarray(color)
    return np.clip(value, 0, 255).astype(np.uint8)


def figures_and_report(root, run, metrics, opportunity, coverage, ledger, chosen):
    out = run/'figures'; out.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    names = ['T', 'U', 'R']
    labels = ['Trained 128D', 'Random 128D', 'Raw 512D*']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    for axis, field, title, threshold in zip(ax, ['mean_iou_delta', 'mask75_fraction', 'prototype_coverage'],
                                           ['GT-only failure mean IoU gain', 'GT-only failure repair to IoU75', 'GT-only prototype coverage'],
                                           [.05, .20, .90]):
        values = [opportunity[n]['failure'][field] for n in names]
        axis.bar(labels, values, color=['#0072B2', '#E69F00', '#999999'])
        axis.axhline(threshold, c='k', ls='--', label=f'Frozen requirement {threshold:g}')
        axis.axhline(0, c='gray', lw=.5)
        axis.set_title(title); axis.tick_params(axis='x', labelrotation=15)
        for i, value in enumerate(values):
            axis.annotate(f'{value:.3f}', (i, value), xytext=(0, 4 if value >= 0 else -13), textcoords='offset points', ha='center')
        axis.legend(fontsize=8)
    n = opportunity['T']['failure']['objects']
    fig.suptitle(f'Can each GT instance mean decode a real mask? | {n} fixed Box75 / Mask75 failures\nGT support is an opportunity diagnostic; not deployable AP. *Raw dimension differs.')
    fig.tight_layout()
    save(fig, out/'01_GT_only_readout_opportunity.png')

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
    sizes = ['S', 'M', 'L']
    totals = [coverage['by_size'][s]['total'] for s in sizes]
    sampled = [coverage['by_size'][s]['sampled'] for s in sizes]
    ax[0].bar(sizes, totals, color='#cccccc', label='All real train instances')
    ax[0].bar(sizes, sampled, color='#0072B2', label='In fixed prototype episodes')
    for i, (a, b) in enumerate(zip(totals, sampled)):
        ax[0].text(i, a, f'{b}/{a}\n{b/max(1,a):.1%}', ha='center', va='bottom')
    ax[0].set_ylim(0, max(totals)*1.20); ax[0].legend()
    ax[0].set_title('Actual real training support coverage')
    for frame in coverage['frames']:
        ax[1].scatter(frame['instances'], frame['sampled_instances'], s=8, alpha=.12, c='#0072B2')
    ax[1].plot([0, 30], [0, 30], '--', c='gray', label='All instances sampled')
    ax[1].axhline(8, c='#D55E00', ls=':', label='Frozen per-image cap 8')
    ax[1].set(xlabel='GT instances in an image', ylabel='Instances in its fixed episode', ylim=(0, 31))
    ax[1].legend(fontsize=8); ax[1].set_title('Dense images and the sampling cap')
    fig.suptitle('Extraction coverage, not optimizer revisit frequency | 1272 real train images')
    fig.tight_layout(); save(fig, out/'02_actual_training_instance_coverage.png')

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
    tags = ['native', 'T', 'U', 'R', 'T_wrong']
    values = [metrics[t]['mask_ap'] for t in tags]
    ax[0].bar(['Native B', 'Trained', 'Random', 'Raw*', 'Wrong instance'], values,
              color=['#555555', '#0072B2', '#E69F00', '#999999', '#D55E00'])
    ax[0].axhline(metrics['native']['mask_ap']+5, ls='--', c='k', label='User goal: native +5pp')
    for i, value in enumerate(values):
        ax[0].text(i, value, f'{value:.2f}', ha='center', va='bottom')
    ax[0].set(ylabel='Mask AP (COCO 50:95, percentage points)', ylim=(0, max(values+[metrics['native']['mask_ap']+5])+5))
    ax[0].tick_params(axis='x', labelrotation=15); ax[0].legend(fontsize=8)
    ax[0].set_title('GT-free complete development evaluation')
    x = np.arange(3)
    for k, tag in enumerate(['T', 'U', 'R']):
        ax[1].bar(x+(k-1)*.24, [metrics[tag]['segm_stats'][j]-metrics['native']['segm_stats'][j] for j in [3,4,5]],
                  width=.24, label=labels[k])
    ax[1].set_xticks(x, sizes); ax[1].axhline(0, c='gray', lw=.7)
    ax[1].set(ylabel='Mask AP minus native (absolute pp)', title='GT-free size breakdown')
    ax[1].legend(fontsize=8)
    fig.suptitle('Same fixed native boxes / scores / row identities | 432 real dev images, 2863 GT\nChanged empty masks may be removed; original row identities remain in the audit.')
    fig.tight_layout(); save(fig, out/'03_GT_free_AP_and_size_controls.png')

    samples = sorted(run.glob('example_*.npz'))
    fig, axes = plt.subplots(len(samples), 6, figsize=(17, 3.1*len(samples)), squeeze=False)
    example_records = []
    from readout_core import gt_supports, support_scores
    for index, path in enumerate(samples):
        frame_id = int(path.stem.split('_')[-1])
        rows = [r for r in ledger if r['image_id'] == frame_id]
        chosen_row = min(rows, key=lambda r: (r['native_mask_iou'], r['annotation_id']))
        with np.load(path) as data:
            output_id = chosen_row['native_output_detection_id']
            pair_index = int(np.flatnonzero(data['pairs'][:, 0] == output_id)[0])
            gt_local = int(data['pairs'][pair_index, 1])
            mask = data['gt_masks'][gt_local]
            rgb = data['rgb']
            native = data['native'][output_id]
            deploy = data['gt_free'][output_id]
            oracle = data['oracle_T'][pair_index]
            # Only this GT-labelled diagnostic visual uses GT support; normal AP was fixed earlier.
            supports = gt_supports(data['gt_masks'], data['valid_rect'], 'cpu')
            query = torch.from_numpy(data['T_query'].astype(np.float32)).flatten(1)
            scores, available, counts = support_scores(query, supports)
            others = torch.nonzero(available)[:, 0]
            others = others[others != gt_local]
            margin = (scores[gt_local]-scores[others].amax(0)).reshape(160, 160).numpy() if len(others) else np.zeros((160,160))
            panels = [rgb[:, :, ::-1], overlay(rgb, mask), overlay(rgb, native), overlay(rgb, oracle), overlay(rgb, deploy)]
            titles = [f'Image {frame_id}', f'GT ann {chosen_row["annotation_id"]}',
                      f'Native IoU {chosen_row["native_mask_iou"]:.3f}',
                      f'GT-only T IoU {chosen_row["oracle"]["T"]["mask_iou"]:.3f}',
                      f'GT-free T IoU {chosen_row["GT_free"]["T"]["mask_iou"]:.3f}']
            for j, (panel, title) in enumerate(zip(panels, titles)):
                axes[index, j].imshow(panel); axes[index, j].set_title(title, fontsize=10)
            im = axes[index, 5].imshow(margin, cmap='coolwarm', vmin=-.3, vmax=.3)
            axes[index, 5].set_title('GT-only prototype margin', fontsize=10)
            for axis in axes[index]:
                axis.axis('off')
            example_records.append({k: chosen_row[k] for k in ['image_id','group','native_output_detection_id','annotation_id','native_mask_iou']})
    fig.suptitle('First usable frame in each filename family; lowest native IoU matched instance within that frame\nIdentical green denotes the same fixed instance. GT-only columns are upper-bound diagnostics.', fontsize=12)
    fig.tight_layout(); save(fig, out/'04_same_instance_readout_cases.png')
    dump(out/'CASE_SELECTION.json', example_records)

    text = f'''# 实例均值原型稠密读出机会：完整实验

本实验冻结已有权重，在432张真实开发图上完成无GT读出和GT辅助机会核验，训练0步。原生Mask AP为{metrics['native']['mask_ap']:.4f}，训练原型的无GT读出为{metrics['T']['mask_ap']:.4f}；本实验不将GT辅助结果计作部署收益。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| **问题** | V4.1全量训练的原型对同预算B仅+0.0417pp，关闭原型仅损失0.0565pp，小目标低于B。要区分实例信息没有被全局字典正确读出，还是现有投影在真实图上没有学出可用均值原型。 |
| **机制假说** | H_readout预测现有T嵌入经每实例GT支持能改善固定好框差Mask；H_target预测T在真实稠密图上仍无有效机会，随机/原始表征及实际采样覆盖可定位训练与压缩缺口。 |
| **针对性方法** | 同B官方Box75最终检测匹配与同一Mask，T训练128D/U同结构随机128D/R原始512D均采用单位均值、支持点leave-one-out与其他实例/bg竞争。无GT分支仅使用RGB和原生预测，先锁定全量输出，再读GT；同T循环错配实例、删除整个读出回到B。R维数不同有混杂，GT上界不作无GT AP。 |
| **整体结果** | 固定好框差Mask {n} 条。T/U/R失败组均值IoU差分别为{opportunity['T']['failure']['mean_iou_delta']:.4f}/{opportunity['U']['failure']['mean_iou_delta']:.4f}/{opportunity['R']['failure']['mean_iou_delta']:.4f}；恢复Mask75比例{opportunity['T']['failure']['mask75_fraction']:.2%}/{opportunity['U']['failure']['mask75_fraction']:.2%}/{opportunity['R']['failure']['mask75_fraction']:.2%}。无GT T/U/R AP为{metrics['T']['mask_ap']:.4f}/{metrics['U']['mask_ap']:.4f}/{metrics['R']['mask_ap']:.4f}，T对原生{metrics['T']['mask_ap']-metrics['native']['mask_ap']:+.4f}pp，错配T为{metrics['T_wrong']['mask_ap']:.4f}。真实train {coverage['sampled_instances']}/{coverage['total_instances']}实例进入固定原型episode，详见分层覆盖。 |
| **研究决策** | 机器预冻结裁决：`{chosen['status']}`。我选定的正向重点为`{chosen['chosen_research_direction']}`。本次结果决定后继原型训练方向，正式新训练尚未启动，必须另冻匹配容量/信息/步数及定向干预；+5目标仍未完成。 |

同一432图/2863可见GT、COCO百分制mAP50–95；差值单位为绝对pp。参照已有G/R正常AP57.2036/57.2395，新读出T分别差{metrics['T']['mask_ap']-57.203617484741144:+.4f}/{metrics['T']['mask_ap']-57.23953569089551:+.4f}pp，因后处理与信息形式不完全匹配，不作独立机制因果归因。GT-only上界不是最优oracle，不排除其他读出形式；负结果只关闭本冻结坐标/单均值/固定margin配置。

证据：[协议](../../PROTOCOL.md)、[运行](run.json)、[重放](REPLAY.json)、[指标](METRICS.json)、[机会分层](OPPORTUNITY_SUMMARY.json)、[逐对象](OBJECT_LEDGER.json)、[训练覆盖](TRAINING_INSTANCE_COVERAGE.json)、[裁决](DECISION.json)、[源码/输入冻结](SOURCE/FREEZE.json)。图集在figures/，实际检测身份在OUTPUT_IDENTITIES.json；原test未使用，无阈值或混合幅度搜索。
'''
    (run/'REPORT.md').write_text(text, encoding='utf-8')
    (root/'REPORT.md').write_text(f'# 实例读出机会\n\n完整报告见[runs/{run.name}/REPORT.md](runs/{run.name}/REPORT.md)。\n', encoding='utf-8')
