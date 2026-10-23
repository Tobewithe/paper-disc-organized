from contextlib import contextmanager
from common import dump, read, run_context, sha

@contextmanager
def task_context(root, stage, freeze='TASK_FREEZE.json'):
    with run_context(root, stage, contract='TASK_PROTOCOL_V3.md', freeze=freeze) as (run, meta):
        meta['TASK_CONFIG_V3.json_sha256'] = sha(root/'TASK_CONFIG_V3.json')
        dump(run/'EVAL_CONTRACT.json', read(root/'TASK_CONFIG_V3.json'))
        yield run, meta
