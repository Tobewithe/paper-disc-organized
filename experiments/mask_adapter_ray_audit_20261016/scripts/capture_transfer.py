"""Capture original completed executor artifacts before local closeout additions."""
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FULL = 'RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b'


def main():
    meta = json.loads((STUDY / 'runs' / FULL / 'run.json').read_text(encoding='utf-8-sig'))
    assert meta['execution_status'] == 'completed' and meta['exit_code'] == 0
    target = STUDY / 'EXECUTOR_TRANSFER_MANIFEST.json'
    assert not target.exists()
    entries = []
    for path in sorted(STUDY.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            entries.append(dict(relative_path=path.relative_to(STUDY).as_posix(),
                                size_bytes=path.stat().st_size, sha256=digest.hexdigest()))
    result = dict(captured_at=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                  root=str(STUDY), latest_run_id=FULL, files=len(entries),
                  total_bytes=sum(e['size_bytes'] for e in entries), artifacts=entries,
                  scope='All existing Study files at capture time, before local closeout; manifest excludes itself')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}))


if __name__ == '__main__':
    main()
