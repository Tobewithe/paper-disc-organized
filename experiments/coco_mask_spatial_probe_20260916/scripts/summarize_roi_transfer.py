"""Report where fixed prediction-GT pair damage appears in the decoder."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result = {}
    for source in a.source:
        rows = [json.loads(s) for s in (source / 'instances.jsonl').read_text().splitlines()]
        groups = {}
        for subset in ('fit', 'selection'):
            for box_min in (.5, .75):
                rs = [r for r in rows if r['split'] == subset and r['box_iou'] >= box_min]
                methods = {}
                for name in ('plain_half', 'guard_full', 'safe_full'):
                    damaged = [r for r in rs if r['values']['baseline']['full'][0] >= .75
                               and r['values'][name]['full'][0] < .75]
                    methods[name] = dict(
                        damaged75=len(damaged),
                        damaged_images=len({r['image_id'] for r in damaged}),
                        roi_iou_worse=sum(r['values'][name]['roi'][0] < r['values']['baseline']['roi'][0]
                                          for r in damaged),
                        input_iou_worse=sum(r['values'][name]['input'][0] < r['values']['baseline']['input'][0]
                                            for r in damaged),
                        roi_loss_worse=sum(r['values'][name]['roi_loss'] > r['values']['baseline']['roi_loss']
                                           for r in damaged))
                groups[f'{subset}:Box{box_min}'] = dict(
                    pairs=len(rs), images=len({r['image_id'] for r in rs}),
                    unique_gt=len({(r['image_id'], r['annotation_id']) for r in rs}), methods=methods)
        result[source.name] = dict(groups=groups, source=str(source),
                                  selection=json.loads((source / 'SPLIT.json').read_text()))
    (a.out / 'ANALYSIS.json').write_text(json.dumps(result, indent=2))
    (a.out / 'COMPLETE.json').write_text(json.dumps({'sources': len(result)}))
    for rid, value in result.items():
        print(rid, json.dumps(value['groups']['selection:Box0.75']))


if __name__ == '__main__':
    main()
