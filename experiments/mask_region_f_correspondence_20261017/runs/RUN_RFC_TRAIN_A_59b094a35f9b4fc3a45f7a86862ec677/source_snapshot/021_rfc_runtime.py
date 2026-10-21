"""Registered laptop execution, immutable inputs, and bounded resources for RFC-v1."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import time

from rfc_paths import ROOT, STUDY, ORIGINAL, SOFT
from rfc_contract import check_source_references, check_training_design, require_no_drift
from posttrain_run import (RegisteredRun, configure_native, tensor_state_hash, read_json,
                           require, sha256, write_new, save, now)

CACHE = ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
BASE = ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
B = ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2'
G = ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb'
WEIGHT = ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt'
GiB = 2**30


def completed(path):
    require(read_json(path/'run.json')['execution_status'] == 'completed', 'incomplete prerequisite: '+str(path))
    require(read_json(path/'CONFIG_AUDIT.json')['status'] == 'passed', 'configuration audit failed: '+str(path))
    require(read_json(path/'DECISION.json')['status'] == 'passed', 'prerequisite failed: '+str(path))


def pointer(name):
    value = read_json(STUDY/(name+'_POINTER.json'))
    path = STUDY/'runs'/value['run_id']
    require(path.resolve().is_relative_to((STUDY/'runs').resolve()), 'pointer escapes Study')
    completed(path)
    return path


def set_pointer(name, run):
    save(STUDY/(name+'_POINTER.json'), {'run_id': run.run_id, 'completed_at': now()})


def inspect_inputs():
    check_execution_references()
    check_training_design()
    profile, environment = configure_native()
    spec = read_json(STUDY/'DESIGN_SPEC.json')
    require_no_drift(spec['training']['numeric_environment_exact_reference'], environment, 'frozen laptop runtime')
    require(sha256(WEIGHT) == profile['model_sha256'], 'base weights changed')
    contract = read_json(CACHE/'EVAL_CONTRACT.json')
    require(sha256(CACHE/'dataset/training_annotations.json') == contract['gt_subset_sha256'], 'training GT changed')
    for path in (CACHE, B, G):
        completed(path)
    return dict(profile=profile, environment=environment, spec=spec,
                split=read_json(CACHE/'SPLIT_CONTRACT.json'), old_contract=read_json(G/'EVAL_CONTRACT.json'))


def check_execution_references():
    """Pin a historical documentation version without overwriting another checkout's norm."""
    refs=read_json(STUDY/'INPUT_REFERENCES.json')
    overrides=read_json(STUDY/'REFERENCE_LOCATION_OVERRIDES.json')
    require(len(overrides)==1,'only the reviewed documentation relocation is permitted')
    for item in refs['existing_files']:
        if item['path'] not in overrides:
            continue
        change=overrides[item['path']]
        require(item['path'].startswith('project/') and item['path'].endswith('.md'), 'runtime source cannot be relocated')
        require(sha256(ROOT/item['path']) in (item['sha256'],change['existing_laptop_sha256']), 'canonical norm unexpectedly changed')
        replacement=(ROOT/change['snapshot_path']).resolve()
        require(replacement.is_relative_to((STUDY/'reference_snapshots').resolve()),'reference snapshot escapes Study')
        item['path']=change['snapshot_path']
    return check_source_references(references=refs)


class Run(RegisteredRun):
    def __init__(self, kind, parent, budget, smoke_deadline=None):
        super().__init__('RFC_'+kind, 'RFC', parent, [], budget, protocol='EXECUTION_PROTOCOL.md')
        self.path = STUDY/'runs'/self.run_id
        self.lock = STUDY/'.RFC.lock'
        self.smoke_deadline = smoke_deadline
        self.peak_rss = 0
        self.peak_cuda_allocated = self.peak_cuda_reserved = 0
        self.meta.update(study_id='STUDY_MASK_REGION_F_CORRESPONDENCE_20261017',
                         baseline_run_id=BASE.name, optimization_started=False,
                         optimizer_steps=0, new_F_cache_exports=0, base_model_forward_images=0)

    def __enter__(self):
        sequence=STUDY/'.sequence.lock'
        other=[p for p in ROOT.glob('experiments/*/.*.lock') if p != sequence]
        require(not other, 'an existing stage lock requires ownership inspection: '+str(other))
        if sequence.exists():
            require(read_json(sequence)['pid']==os.getppid(), 'stage is not a child of the owning RFC sequence')
        return super().__enter__()

    def freeze(self, contract):
        context = read_json(STUDY/'EXECUTION_SOURCE.json')
        self.sources = context['files']
        for rel, digest in self.sources.items():
            require(sha256(ROOT/rel) == digest, 'execution source drift: '+rel)
        for i, rel in enumerate(self.sources):
            target = self.path/'source_snapshot'/(f'{i:03d}_'+Path(rel).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/rel, target)
        shutil.copyfile(STUDY/'EXECUTION_PROTOCOL.md', self.path/'PROTOCOL.md')
        shutil.copyfile(STUDY/'EXECUTION_DIRTY.patch', self.path/'dirty.patch')
        write_new(self.path/'SOURCE_MANIFEST.json', context)
        write_new(self.path/'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', code_commit=context['code_commit'],
                    dirty_patch_sha256=sha256(self.path/'dirty.patch'),
                    protocol_path='PROTOCOL.md', protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))

    def check_budget(self):
        super().check_budget()
        if self.smoke_deadline is not None:
            require(time.time() < self.smoke_deadline, 'combined 1800-second smoke budget exhausted')
        import psutil
        import torch
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        require(self.peak_rss <= 8*GiB, 'CPU RSS exceeds frozen 8 GiB')
        if torch.cuda.is_initialized():
            self.peak_cuda_allocated = max(self.peak_cuda_allocated, torch.cuda.max_memory_allocated())
            self.peak_cuda_reserved = max(self.peak_cuda_reserved, torch.cuda.max_memory_reserved())
            require(max(self.peak_cuda_allocated, self.peak_cuda_reserved) <= 7*GiB,
                    'CUDA allocated/reserved exceeds frozen 7 GiB')

    def resource_receipt(self):
        self.check_budget()
        F_bytes=all_F_bytes()
        total=sum(p.stat().st_size for p in STUDY.rglob('*') if p.is_file())
        require(F_bytes <= 48*GiB and total-F_bytes <= 8*GiB, 'Study disk allocation exceeded')
        return dict(peak_RSS_bytes=self.peak_rss, peak_cuda_allocated_bytes=self.peak_cuda_allocated,
                    peak_cuda_reserved_bytes=self.peak_cuda_reserved,
                    free_disk_bytes=shutil.disk_usage(STUDY).free,
                    regional_F_bytes=F_bytes,other_study_bytes=total-F_bytes)


def initial_resources():
    import psutil
    owned = sum(p.stat().st_size for p in STUDY.rglob('*') if p.is_file())
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(STUDY).free
    require(available >= 3*GiB, 'need 3 GiB available system RAM')
    # The initial 64-GiB reservation is consumed by this Study's outputs only.
    require(free+owned >= 64*GiB, 'need 64 GiB free plus this Study already-written outputs')
    return dict(system_available_bytes=available, disk_free_bytes=free, study_bytes=owned,
                reservation_bytes=64*GiB)


def audit(run, data, **checks):
    from native_output_bridge import native_environment
    check_execution_references()
    require_no_drift(data['environment'], native_environment(), 'post-run runtime')
    require(sha256(WEIGHT) == data['profile']['model_sha256'], 'base weight drift after run')
    write_new(run.path/'CONFIG_AUDIT.json', dict(status='passed', native_environment_parity=True,
              environment=data['environment'], resources=run.resource_receipt(), **checks))


def all_F_bytes():
    return sum(p.stat().st_size for p in (STUDY/'runs').glob('*/regions/*/*.pt'))
