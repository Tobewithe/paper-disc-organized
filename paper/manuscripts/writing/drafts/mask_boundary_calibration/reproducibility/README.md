# Reproducibility materials for RCMC

This directory packages small artifacts from completed experiments. It does not contain COCO images, pretrained segmenter weights, or full prediction banks. Source paths and content versions are recorded in `COPY_MANIFEST.json`; the original experiment Study retains its identity and full Run directories.

## Included files

| Path | Purpose |
|---|---|
| `models/response.json` | Frozen five-feature response gate used for both m and s |
| `models/area.json`, `models/shape.json` | Frozen geometry controls |
| `frozen_models.json`, `selection.json` | Fit parameters, feature definitions, image identities and selection results; joblib filenames refer to the original fitting Run |
| `calibration_split.json` | Actual train2017 fit/selection split |
| `validation_image_ids.json` | Exact 4,500-image validation list |
| `code/mask_calibration.py` | Input-grid logits, binary export and fixed threshold rules |
| `code/risk_calibration.py`, `code/portable_risk.py` | Evaluated predictor adapter and portable gate inference |
| `code/decoder_calibration_experiment.py` | Candidate export, annotation matching and fixed intervention diagnostics |
| `code/fit_risk_calibration.py` | Gate fitting, model selection and bank-based evaluation |
| `code/benchmark_risk_inference.py` | Timing implementation |
| `evidence/` | Frozen small result summaries used in tables and supporting interface checks |

These scripts are preserved research sources. Some command defaults point to the original project layout; pass explicit input/output paths. For full evaluation, use original COCO instance annotations and the exact image list. One COCO annotation remains one instance even when its segmentation contains several polygons.

## Reading and rebuilding the manuscript

From the manuscript directory, run `python scripts/prepare_assets.py` in an environment with NumPy and Matplotlib, then compile the LaTeX files. Table generation prefers these archived summaries over mutable experiment paths. Existing development-example figures are included. The manuscript can therefore be rebuilt without inference or access to the original project.

PDF inspection uses `scripts/check_pdf.py` with PyMuPDF and Pillow. Its local optional dependency directory is not distributed. It checks page count, abstract length, highlight length and LaTeX diagnostics, and produces page previews; visual inspection remains separate.

## Frozen-gate inference

Use Ultralytics **8.4.100**, compatible PyTorch, NumPy and OpenCV. The recorded laptop deployment used Python 3.12.3 and Torch 2.5.1. The runner's installed-distribution inventory can list Ultralytics 8.4.27 because the actual scripts prepend the 8.4.100 source directory; the timing script asserts and records the imported version as 8.4.100. Follow the imported runtime version, not that generic inventory field. The portable JSON gate does not require scikit-learn or joblib for inference.

From this directory, a minimal use of the evaluated adapter is:

```python
import sys
from pathlib import Path
import ultralytics
from ultralytics import YOLO

assert ultralytics.__version__ == '8.4.100'
sys.path.insert(0, str(Path('code').resolve()))
from risk_calibration import make_risk_predictor

model = YOLO('/path/to/yolo26m-seg.pt')
model.model.model[-1].end2end = True  # evaluated one-to-one branch
predictor = make_risk_predictor('models/response.json', mode='response', chunk=24)
results = model.predict(
    source='/path/to/image.jpg', predictor=predictor,
    imgsz=640, conf=0.001, max_det=300, batch=1,
    half=False, retina_masks=False, device=0, save=False,
)
```

The same gate is used with `yolo26s-seg.pt` for transfer. This example returns ordinary predictor results. The reported COCO tables come from the archived export and COCOeval pipeline; simply timing this demonstration is not a reproduction of their AP. In particular, preserve binary-export order, byte conversion, candidate identity and the empty-output fallback described in the supplement.

## Refitting or reproducing evaluation

1. Obtain COCO train2017/val2017 images and official instance annotations, and official COCO-pretrained YOLO26m-seg/s-seg weights from their providers. The included image lists define the experiment; do not substitute a dense-image subset or a random validation sample.
2. The fitting bank was constructed on the 2,000-image calibration subset with `decoder_calibration_experiment.py --branch one2one --export-only`. The subset annotation file comes from the recorded calibration preparation; candidate and instance CSV files and variant RLE predictions are the inputs expected by the fitter. Invoke each script with `--help` for its exact required paths. The full banks remain in the original experimental Runs, not this compact package.
3. Fit using `fit_risk_calibration.py --phase fit --input <train-bank> --annotations <instances_train2017.json> --split calibration_split.json --output <new-fit-run>`. This additionally requires scikit-learn, joblib and pycocotools. Source code and saved parameters define the estimator; portable inference avoids cross-version pickle loading. A fresh refit is a new result, not assumed bitwise identical to the supplied frozen model.
4. Export m and s validation banks using the exact included validation image IDs, the one-to-one branch, and the stated decoder. With the original complete COCO annotation file, these IDs correspond to sorted positions 500 through 4999. Run the fitter's `--phase evaluate` using a fitting Run containing its joblib models. For direct deployment, use the included portable gate JSON instead.
5. Official COCO segmentation AP and the fixed-box-matched R75 diagnostic are different evaluations. Preserve both definitions and report repaired and damaged instances, rather than only the subset that improved.

Original experiment entry: `experiments/mask_boundary_route_20260914/REPORT.md`. The paper's `EVIDENCE_MAP.md` identifies the exact completed Runs. No new model training or inference was performed while assembling this manuscript package.
