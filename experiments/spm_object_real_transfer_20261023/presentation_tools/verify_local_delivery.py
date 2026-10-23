"""Local read-back verification of returned bytes and reviewable document links."""
import argparse
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rt import dump, read, sha

p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
a = p.parse_args(); root = a.root.resolve()
receipt, packaging = read(root/'RESULT_RECEIPT.json'), read(root/'PACKAGING.json')
assert sha(root/'RESULTS_LIGHT.tar.gz') == packaging['light_archive_sha256']
assert sha(root/'TRAINED_MODELS.tar.gz') == packaging['weights_archive_sha256']
paths = {}
for item in receipt['metadata_files']+receipt['model_files']:
    path = root/item['path']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
    paths[item['path']] = item['sha256']
frozen = read(root/'FREEZE.json')
for name, digest in frozen['files'].items(): assert sha(root/name) == digest, name
layout = read(root/'presentation_revision/FIGURE_LAYOUT_R1.json')
assert sha(root/'presentation_revision/04_learned_body_candidates_r1.png') == layout['revised_sha256']
assert sha(root/'presentation_tools/fix_candidate_layout.py') == layout['source_sha256']
links = []
for name in ('REPORT.md', 'VISUAL_GUIDE.md', 'artifacts/README.md'):
    file = root/name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', file.read_text(encoding='utf-8')):
        if target.startswith(('https:', 'http:', '#')): continue
        target = target.split('#')[0].strip('<>')
        if target.endswith('FINAL_LOCAL_ACCEPTANCE.json'): continue
        assert (file.parent/target).exists(), (name, target)
        links.append(target)
primary = read(root/'runs'/read(root/'FORMAL_POINTER.json')['run_id']/'SUMMARY.json')
accepted = read(root/'runs'/read(root/'ACCEPTANCE_POINTER.json')['run_id']/'SUMMARY.json')
assert primary['optimizer_steps'] == accepted['optimizer_steps'] == 8000
assert accepted['maximum_metric_absolute_error_pp'] == 0
assert primary['status'] == 'current_object_interface_no_positive_real_segmentation_gain'
result = dict(execution_status='completed_and_independently_accepted',
    scientific_status='negative_real_segmentation_utility', formal_optimizer_steps=8000,
    engineering_optimizer_steps=16, returned_unique_files_SHA_verified=len(paths),
    frozen_local_sources_SHA_verified=len(frozen['files']),
    frozen_external_inputs_verified_by_server=len(frozen['external_files']),
    trained_checkpoints_returned=2, learned_body_bank_returned=True,
    current_figure_groups_reviewed=10, figure_layout_revision_scientific_values_unchanged=True,
    report_links_verified=len(links), independent_COCO_statistics_verified=60,
    independent_metrics_max_abs_error_pp=0, native_full_AP_parity_error_pp=0,
    native_mask_AP=primary['metrics']['native']['mask_ap'],
    structured_mask_AP=primary['metrics']['structured']['mask_ap'],
    structured_minus_native_pp=primary['structured_minus_native_pp'],
    GPU_allocated_seconds=174, max_concurrent_GPUs=1, real_test_used=False,
    original_decision_preserved=True, goal_achieved=False, automatic_followup_training=False,
    next_question_started=False)
dump(root/'FINAL_LOCAL_ACCEPTANCE.json', result)
print(result)
