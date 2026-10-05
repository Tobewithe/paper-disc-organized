"""Assemble the manuscript's small, versioned reproduction materials, then zip.

No fitting, inference, remote access, or modification of source experiments.
Run --collect in the project before finalizing documentation, then --zip.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parents[2]
STUDY = PROJECT / 'experiments/mask_boundary_route_20260914'
REPRO = PAPER / 'reproducibility'


def collect():
    entries = []

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append({'source': source.relative_to(PROJECT).as_posix(),
                        'copy': destination.relative_to(PAPER).as_posix(),
                        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})

    def script(name, run=None):
        source = STUDY / 'scripts' / name
        if run:
            directory = STUDY / 'runs' / run
            record = json.loads((directory / 'run.json').read_text(encoding='utf-8-sig'))
            snapshots = [s for s in record.get('snapshots', [])
                         if s['path'].replace('\\', '/').split('/')[-1] == name]
            if snapshots:
                source = directory / snapshots[0]['snapshot']
        copy(source, REPRO / 'code' / name)

    timing = 'RUN_e7df55bcc56541ca8c449e5065c885a4'
    for name in ('mask_calibration.py', 'risk_calibration.py', 'portable_risk.py',
                 'benchmark_risk_inference.py'):
        script(name, timing)
    for name, run in (
        ('fit_risk_calibration.py', 'RUN_d01ffdef9f3c498fa3529d96cea875eb'),
        ('decoder_calibration_experiment.py', 'RUN_2ae67d6556a14848bceb5783a72e5790'),
        ('prepare_train_calibration.py', 'RUN_dc58df6f08f64b9f932b84a812ad75d8'),
        ('verify_risk_adapter.py', 'RUN_12944cf151014819b4b34a7f6aca63c3'),
        ('score_exported_variant.py', None), ('compare_completed_metrics.py', None),
        ('export_portable_risk.py', 'RUN_140f19c618fe44f1a52006d56ce046a5')):
        script(name, run)

    for mode in ('area', 'shape', 'response'):
        copy(STUDY / 'runs/RUN_140f19c618fe44f1a52006d56ce046a5' / f'{mode}.json',
             REPRO / 'models' / f'{mode}.json')
    for run, name, destination in (
        ('RUN_d01ffdef9f3c498fa3529d96cea875eb', 'frozen_models.json', 'frozen_models.json'),
        ('RUN_d01ffdef9f3c498fa3529d96cea875eb', 'selection.json', 'selection.json'),
        ('RUN_dc58df6f08f64b9f932b84a812ad75d8', 'split.json', 'calibration_split.json'),
        ('RUN_2ae67d6556a14848bceb5783a72e5790', 'image_ids.json', 'validation_image_ids.json')):
        copy(STUDY / 'runs' / run / name, REPRO / destination)
    for protocol in ('PROTOCOL_DECODER_V2.json', 'PROTOCOL_TRANSFER_V5.json'):
        copy(STUDY / protocol, REPRO / protocol)

    evidence = json.loads((PAPER / 'EVIDENCE_SOURCES.json').read_text())
    for key, info in evidence.items():
        source = PROJECT / info['path']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != info['sha256']:
            raise ValueError(f'Evidence changed since manuscript generation: {key}')
        copy(source, REPRO / 'evidence' / key)
    # These supporting checks document deployment and portable estimator parity.
    for run in ('RUN_140f19c618fe44f1a52006d56ce046a5',
                'RUN_12944cf151014819b4b34a7f6aca63c3'):
        copy(STUDY / 'runs' / run / 'SUMMARY.json', REPRO / 'evidence' / run / 'SUMMARY.json')
    (REPRO / 'COPY_MANIFEST.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    print(f'Collected {len(entries)} small source artifacts; no prediction banks or images copied.')


def archive():
    required = ['main.pdf', 'main.tex', 'supplement.pdf', 'supplement.tex',
                'title_page.pdf', 'title_page.tex', 'highlights.txt', 'references.bib',
                'elsarticle.cls', 'elsarticle-harv.bst', 'draft_format.tex', 'build.ps1',
                'README.md', 'VENUE_AND_FORMAT.md', 'EVIDENCE_MAP.md',
                'EVIDENCE_SOURCES.json', 'study.json', 'reproducibility/README.md',
                'vendor/README.md', 'vendor/prletters.sty', 'vendor/elsarticle.zip',
                'vendor/patrec-authorship-and-formatting.pdf']
    files = [PAPER / name for name in required]
    for folder in ('scripts', 'figures', 'tables', 'reproducibility'):
        files.extend(p for p in (PAPER / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    output = PAPER / 'RCMC_PRL_manuscript_package.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):
            z.write(path, path.relative_to(PAPER).as_posix())
        quality = PAPER / 'build/quality_check.json'
        if quality.exists():
            z.write(quality, 'PDF_QUALITY_CHECK.json')
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        print(f'Package: {output.name}; {len(z.namelist())} files; {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args()
    if not (args.collect or args.zip):
        parser.error('Choose --collect or --zip.')
    if args.collect:
        collect()
    if args.zip:
        archive()
