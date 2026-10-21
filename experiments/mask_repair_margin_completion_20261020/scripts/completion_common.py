"""New observation lifecycle; parent frozen scientific files stay read-only."""
from pathlib import Path
import sys

NEW_STUDY = Path(__file__).resolve().parents[1]
OLD_STUDY = NEW_STUDY.parent / 'mask_repair_margin_20261019'
sys.path.append(str(OLD_STUDY / 'scripts'))
from common import *


def check_completion_freeze():
    frozen = read_json(NEW_STUDY / 'FREEZE_MANIFEST.json')
    for relative, digest in frozen['files'].items():
        require(sha256(ROOT / relative) == digest, 'frozen source drift: ' + relative)
    parent = read_json(OLD_STUDY / 'FREEZE_MANIFEST_V3.json')
    require(science_files() == parent['files'], 'parent V3 science changed')
    require(sha256(OLD_STUDY / 'INPUT_MANIFEST.json') == parent['input_manifest_sha256'], 'input manifest changed')
    require(sha256(OLD_STUDY / 'PREPARATION.json') == parent['preparation_sha256'], 'preparation changed')
    return frozen


class CompletionRun(RegisteredRun):
    def __init__(self, scope, parent):
        frozen = check_completion_freeze()
        cfg = read_json(OLD_STUDY / 'CONFIG.json')
        super().__init__('RSMC_' + scope.upper(), 'RSMC', parent, list(frozen['files']), cfg['budget'][scope+'_seconds'])
        self.path = NEW_STUDY / 'runs' / self.run_id
        self.lock = OLD_STUDY / '.RSM.lock'
        self.meta.update(study_id='STUDY_MASK_REPAIR_MARGIN_COMPLETION_20261020', scope=scope,
            new_training_steps=0, base_model_forward_calls=0, adapter_forward_calls=0,
            launch_mode='attached_SSH', original_precision_unchanged=True,
            pressure_failure_policy='record_failure_and_continue_descriptive_evaluation')

    def __enter__(self):
        require(socket.gethostname().upper() == 'YANGLEI', 'wrong executor')
        require(Path(sys.executable).resolve() == Path('C:/Users/28358/anaconda3/envs/pytorch/python.exe').resolve(), 'wrong Python')
        self.resources = resource_check()
        return super().__enter__()

    def freeze(self, contract):
        frozen = check_completion_freeze()
        self.sources = frozen['files']
        context = read_json(NEW_STUDY / 'EXECUTION_CONTEXT.json')
        self.meta.update(code_commit=context['code_commit'], dirty_patch_sha256=context['tracked_diff_sha256'])
        mapping = {}
        for i, relative in enumerate(self.sources):
            dst = self.path / 'source_snapshot' / f'{i:02d}_{Path(relative).name}'
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / relative, dst)
            mapping[relative] = dst.relative_to(self.path).as_posix()
        for name in ['CONFIG.json', 'INPUT_MANIFEST.json', 'PREPARATION.json']:
            shutil.copyfile(OLD_STUDY / name, self.path / name)
        shutil.copyfile(NEW_STUDY / 'PROTOCOL.md', self.path / 'PROTOCOL.md')
        shutil.copyfile(OLD_STUDY / 'PROTOCOL.md', self.path / 'ORIGINAL_PROTOCOL.md')
        write_new(self.path / 'SOURCE_MANIFEST.json', frozen)
        write_new(self.path / 'SOURCE_SNAPSHOT_MAP.json', mapping)
        write_new(self.path / 'RESOURCES.json', self.resources)
        contract.update(continuation_protocol=str(NEW_STUDY / 'PROTOCOL.md'),
            pressure_failure_policy=self.meta['pressure_failure_policy'], scientific_thresholds_unchanged=True)
        write_new(self.path / 'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path / 'PROTOCOL.md'),
            protocol_sha256=sha256(self.path / 'PROTOCOL.md'), contract_sha256=sha256(self.path / 'EVAL_CONTRACT.json'),
            input_manifest_sha256=sha256(OLD_STUDY / 'INPUT_MANIFEST.json'))
