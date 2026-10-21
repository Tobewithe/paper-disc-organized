"""Register the preselected held-out sequence; no tracking or outcome selection."""
from pathlib import Path
import cv2
from .common import Run, read_json, write_json, sha256
from .infer import extract


def main(root):
    manifest = read_json(root / 'input_manifest.json')
    assert not any(s['name'] == 'horse_7' for s in manifest['sequences'])
    video = root / 'data/horse_7.mp4'
    gt = root / 'data/horse_7_gt.txt'
    with Run(root, 'confirmation_registration', dict(sequence='horse_7', split='confirm',
             selection='preselected in METHOD_PROTOCOL before fitting', metrics_read=False)) as run:
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        seq = dict(name='horse_7', split='confirm', model='horse', classes=[17],
                   video=str(video), gt=str(gt), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                   fps=cap.get(cv2.CAP_PROP_FPS), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), video_sha256=sha256(video),
                   gt_sha256=sha256(gt), gt_kind='official_AnimalTrack_bbox_identity', visibility_labels=False)
        cap.release()
        write_json(run.path/'input_manifest_before.json', manifest)
        manifest['sequences'].append(seq)
        write_json(root/'input_manifest.json', manifest)
        write_json(run.path/'input_manifest_after.json', manifest)
        write_json(run.path/'SUMMARY.json', seq)
        print('CONFIRMATION_FROZEN', seq, flush=True)
    extract(root, read_json(root/'config.json')['detector'], only_names=['horse_7'],
            pointer_name='confirmation_features_pointer.json')
    pointer = read_json(root/'confirmation_features_pointer.json')
    manifest['sequences'][-1]['cache_path'] = str(root / pointer['cache'] / 'horse_7')
    write_json(root/'input_manifest.json', manifest)
    write_json(root/'confirmation_manifest.json', manifest['sequences'][-1])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
