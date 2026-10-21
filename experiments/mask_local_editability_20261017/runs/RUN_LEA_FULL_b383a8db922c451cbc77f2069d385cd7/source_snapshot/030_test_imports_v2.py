from lea_common_v2 import *
import inspect
import lea_run_v2
import lea_verify_v2

require(Path(lea_verify_v2.__file__).resolve() == (STUDY/'scripts/lea_verify_v2.py').resolve(), 'Verifier resolved outside Study')
require('replay' in inspect.signature(lea_verify_v2.verify).parameters, 'Verifier signature')
old = (STUDY/'scripts/run_audit.py').read_text(encoding='utf8')
expected = old.replace('from lea_common import *','from lea_common_v2 import *').replace('from verify import verify','from lea_verify_v2 import verify')
require(expected == (STUDY/'scripts/lea_run_v2.py').read_text(encoding='utf8'), 'Scientific runner changed beyond imports')
expected = (STUDY/'scripts/verify.py').read_text(encoding='utf8').replace('from lea_common import *','from lea_common_v2 import *')
require(expected == (STUDY/'scripts/lea_verify_v2.py').read_text(encoding='utf8'), 'Verifier calculation changed')
verify_files(read_json(STUDY/'EXECUTION_FREEZE.json')['files'])
verify_files(read_json(STUDY/'EXECUTION_FREEZE_V2.json')['files'])
write_new(STUDY/'IMPORT_TESTS_V2.json', dict(status='passed', at=now(), exact_verifier_module=lea_verify_v2.__file__,
          replay_argument=True, runner_import_only_change=True, verifier_common_import_only_change=True,
          original_scientific_sources_and_protocol_unchanged=True, original_freeze_sha256=sha256(STUDY/'EXECUTION_FREEZE.json'),
          v2_freeze_sha256=sha256(STUDY/'EXECUTION_FREEZE_V2.json'), new_scientific_outcomes=0))
print('PASS: isolated verifier; signature; import-only code changes; both frozen manifests')
