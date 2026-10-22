from pathlib import Path
from common import dump, run_context, sha
from visualize import method_figure

root = Path(__file__).resolve().parents[1]
with run_context(root, 'method_diagram') as (run, meta):
    out = run/'figures'; out.mkdir(); method_figure(out)
    meta['diagram_source_sha256'] = sha(root/'scripts/visualize.py')
    (run/'DIAGRAM_SOURCE.py').write_bytes((root/'scripts/visualize.py').read_bytes())
    dump(run/'SUMMARY.json', {'diagram': 'implemented_frozen_P3_query_and_memory_path', 'training_steps': 0})
    dump(run/'DECISION.json', {'status': 'method_diagram_rendered', 'next_action': 'deliver_existing_results'})
