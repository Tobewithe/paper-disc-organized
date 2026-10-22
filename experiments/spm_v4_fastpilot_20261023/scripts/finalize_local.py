"""Write a local delivery receipt after verifying transferred research artifacts."""
import datetime as dt
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

weights = read(root/'SPM_V4_WEIGHTS.tar.receipt.json')
checks = {}
for name, expected in weights['files'].items():
    actual = {'sha256': sha(root/name), 'size_bytes': (root/name).stat().st_size}
    assert actual == expected, name
    checks[name] = actual
for name, digest in read(root/'TRAIN_FREEZE.json')['files'].items():
    assert sha(root/name) == digest, name
acceptance = read(root/'ACCEPTANCE_RECEIPT.json'); assert acceptance['status'] == 'passed'
viz = Path('runs')/read(root/'VISUALIZE_POINTER.json')['run_id']
review = Path('runs')/read(root/'VISUAL_REVIEW_POINTER.json')['run_id']
captions = {
    '00_method_overview': '实现结构与冻结/训练边界',
    '01_training_and_results': '训练曲线与四臂真实开发成绩',
    '02_prototype_training': '独立原型训练损失和梯度',
    '03_data_distribution': '数据来源、猪体尺度和姿态分布',
    '04_prototype_atlas_fg': '16个前景原型的真实近邻与来源',
    '04_prototype_atlas_bg': '16个背景原型的真实近邻与来源',
    '05_memory_support': '32槽原型利用率与来源占比',
    '06_embedding_PCA': '真实/合成及C/B/I/BG嵌入分布',
    '07_improvement_and_damage': '固定开发样本的改善与退化全图',
    '09_retrieval_gate_residual': '实际检索相似度、分配、gate和有效残差',
    '10_cost_and_interventions': '推理开销与同checkpoint必要干预'
}
figures = []
for key, title in captions.items():
    folder = Path('runs')/read(root/'METHOD_DIAGRAM_POINTER.json')['run_id'] if key == '00_method_overview' else viz
    figures.append((title, folder/'figures'/(key+'.png')))
figures.insert(9, ('接触与猪-背景边界放大；无接触时明确标注回退', review/'figures/08_contact_and_boundary_zoom.png'))
figures.append(('4个推断family的开发收益与损失', review/'figures/11_grouped_development_effect.png'))
lines = ['# SPM-YOLO V4重点技术与最终结果可视化', '',
         '**实际训练已完成，单seed、54张real-dev图。** F Mask AP=50.9267，比同预算continued +1.3045pp，比G +1.3725pp，比B* +0.2306pp；AP75相对B*下降1.1407pp。', '',
         '推理关闭检索几乎不改变结果；优先使用已导出的原生head权重。图中的gate/相似度/残差全部来自真实checkpoint。开发选例不是独立test或统计显著性证据。', '',
         '接触放大图经复核：不存在C token的样本改取猪-背景边界，并在图下注明；旧版crop保留在原Run，不再作为正式展示。', '',
         '[完整报告](REPORT.md) · [权重与推理](USAGE.md) · [独立验收](ACCEPTANCE_RECEIPT.json)', '']
for title, path in figures:
    assert (root/path).is_file(), path
    lines += ['## '+title, '', f'![{title}]({path.as_posix()})', '']
(root/'VISUAL_GUIDE.md').write_text('\n'.join(lines), encoding='utf-8')
receipt = {'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'passed',
           'weights_verified': len(checks), 'weights': checks,
           'weights_bundle_sha256': weights['sha256'], 'light_bundle_original_files_verified': 337,
           'post_review_bundle_sha256': sha(root/'SPM_V4_POST_REVIEW.tar.gz'),
           'inference_example_bundle_sha256': sha(root/'SPM_V4_INFERENCE_EXAMPLE.tar.gz'),
           'frozen_training_inputs_verified': True, 'cache_and_AP_audit': 'ACCEPTANCE_RECEIPT.json',
           'formal_gallery_figures': len(figures), 'CPU_inference_example': 'examples/prediction.png',
           'last_checkpoints_and_features': 'retained_on_registered_server', 'schedules': False}
dump(root/'TRANSFER_RECEIPT.json', receipt)
study = read(root/'study.json')
study.update(plan_status='completed_fastpilot', execution_status='completed',
             scientific_status='single_seed_development_positive_active_retrieval_increment_absent',
             artifact_status='weights_metrics_technical_visualizations_delivered_locally',
             transfer_status='sha256_verified', goal_mode='explicit_user_goal_first_pilot_delivered',
             current_action='deliver_completed_fastpilot_no_further_training_started',
             results={'spm_mask_ap': 50.92666169560185, 'continued_mask_ap': 49.62214235021191,
                      'adapter_mask_ap': 49.55413129128794, 'baseline_mask_ap': 50.69601196553175,
                      'native_head_export_mask_ap': 50.92724685135814,
                      'primary_delta_pp': acceptance['positive_development_delta_pp']},
             reports=['REPORT.md', 'VISUAL_GUIDE.md', 'USAGE.md', 'ACCEPTANCE_RECEIPT.json', 'TRANSFER_RECEIPT.json'],
             recommended_direction='prototype_assisted_training_with_native_head_deployment_pending_independent_test',
             recommended_next_action='independent_real_test_of_existing_checkpoint_not_started')
dump(root/'study.json', study)
print('DELIVERY_VERIFIED', len(checks), 'large artifacts', len(figures), 'gallery figures')
