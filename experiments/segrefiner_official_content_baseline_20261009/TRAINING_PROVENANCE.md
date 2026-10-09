# SegRefiner LR training provenance and evaluation-overlap audit

Status at 2026-10-09: source/configuration facts are verified from the
repository archive locked by this Study.  The declared LVIS-v1-train versus
project COCO-val2017 image-ID audit is complete and found zero shared IDs.
The checkpoint's own training manifest remains unavailable, so this is not
proof that the released checkpoint trained on exactly that declared image set.

## Primary sources and locked artifacts

| Item | Source / local artifact | Verification |
|---|---|---|
| SegRefiner source | [official repository](https://github.com/MengyuWang826/SegRefiner), locked commit `53419a2d38ea3da0b6e2be77e5b45e139195a0b3`; archive `assets/upstream/SegRefiner-53419a2d38ea.zip` | archive SHA-256 `fb8171134419c0a264bb8eb168fdaf404b4674b4e067065f90202167d55315e1` |
| LR training config | `configs/segrefiner/segrefiner_lr.py` in that archive | UTF-8 SHA-256 `ee0797eb75d075cd01b8cb78600fedcc4ce6d4428942d52954144c5fdea2ff40` |
| COCO evaluation config | `configs/segrefiner/segrefiner_coco.py` in that archive | UTF-8 SHA-256 `20bd9fa015632deaee8c3f988d35f1a7a7984f603fb3486f4168d06bcbbece1c` |
| README / official LR checkpoint link | `README.md` in the locked archive; it links LR weight file ID `1FrhbdwNyTlQYNbF9IgFF2tY3iQXHFcau`; downloaded file `assets/models/segrefiner_lr_latest.pth` | README UTF-8 SHA-256 `b1f0b2eae621971427c6f2bae751fd6bc26196646c279dffcdc307efe6b43be6`; weight 474,603,251 bytes, SHA-256 `d5806a41a4bff6f1e313d1a9dcf18d11dab28e3a8e1497871a156214cd8fe07a` |
| LVIS v1 train annotation | `https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip`; retained at `assets/training_provenance/lvis_v1_train.json.zip` | downloaded ZIP SHA-256 `334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba`, 350,264,821 bytes; central directory has one entry, `lvis_v1_train.json`, advertised uncompressed length 1,097,154,875 bytes |
| Project COCO val annotation | `C:/Dpan/document/model_datasets/datasets/coco/annotations/instances_val2017.json` | supplied and locally recomputed SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`, 19,987,840 bytes |

The LVIS API's official repository describes LVIS v1.0 as having a 100k-image
train split and directs users to obtain the release annotations from the LVIS
site ([LVIS API](https://github.com/lvis-dataset/lvis-api)).  The exact S3 URL
above is the publicly documented LVIS v1 train distribution URL; no publisher
checksum or signed manifest was available in the sources inspected.  The local
file hash is therefore a receipt for the retrieved bytes, not independent
proof that the checkpoint used those exact bytes.

## What the locked author materials say

These are author/source claims, not reconstructed checkpoint history:

* The README says LR-SegRefiner is trained on LVIS and that its coarse masks are
  generated online.  It supplies the LR Google Drive weight link and instructs
  COCO evaluation through `segrefiner_coco.py` with
  `segrefiner_lr_latest.pth`.
* `segrefiner_lr.py` sets `dataset_type='LVISRefine'`,
  `ann_file=...lvis_v1_train.json`, object size 256, online coarse-mask loading,
  AdamW (`lr=4e-4`), and `max_iters=120000`.
* `segrefiner_coco.py` inherits `segrefiner_lr.py` and defines COCO test input
  as `instances_val2017.json`; its test settings are `pad_width=20`,
  `model_size=256`, `batch_max=32`, and `area_thr=512`.

This establishes the released configuration's declared training corpus and
evaluation route.  It does **not** establish the actual samples, number of
iterations completed, random seed, data version, or any extra/pretraining data
for the released checkpoint.

## Related paper (bibliographically verified)

| Field | Record |
|---|---|
| Title | *SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process* |
| Authors | Mengyu Wang, Henghui Ding, Jun Hao Liew, Jiajun Liu, Yao Zhao, Yunchao Wei |
| Year / venue | 2023, *Advances in Neural Information Processing Systems* 36 (NeurIPS 2023), Main Conference Track |
| Primary URLs | [NeurIPS proceedings abstract](https://papers.neurips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html); [arXiv:2312.12425](https://arxiv.org/abs/2312.12425) |
| Problem and method (author claim) | Refine coarse segmentation masks from prior models. The authors frame refinement as a conditional discrete diffusion/denoising process, using image content and coarse masks. |
| Reported result (author claim) | The paper reports improvements in segmentation and boundary metrics across semantic, instance, and dichotomous segmentation tasks. These published results are not measurements on this Study's frozen predictions. |
| Limitation for this Study | The paper/repository configuration supports a relevant official content-refinement comparison. Neither publication metadata nor the released state-dict-only checkpoint establishes this checkpoint's complete training manifest or a project-specific performance claim. |

## Checkpoint metadata audit

`runs/RUN_SEGREFINER_CHECKPOINT_METADATA_AUDIT_S0_R3/audit_results.json` used
PyTorch 2.9.1 on CPU only with `weights_only=True`, `mmap=True`, and
`map_location='cpu'`; it did not use a GPU or attempt an unsafe deserialization
fallback.  The official weight safely loaded as a top-level `OrderedDict` of
455 float tensors (453 keys beginning `denoise_model`, two beginning
`loss_texture`; 118,604,691 stored tensor elements).

There is no top-level non-tensor `meta`, `metadata`, `config`, `cfg`, `runner`,
or `data` container. Consequently the file itself does not expose an
`ann_file`, completed iteration, seed, source revision, optimizer history, or
extra-data/training manifest. It neither confirms nor contradicts the locked
configuration's `lvis_v1_train.json` and 120,000-iteration declarations; all
such checkpoint-history facts remain **unknown**.

## Completed image-level audit

`runs/RUN_SEGREFINER_LVIS_COCO_OVERLAP_AUDIT_S0_R1/audit_results.json` records
the completed stdlib streaming audit (the earlier `..._S0` is retained as the
pre-metadata-receipt run).  It extracted the one ZIP member and decoded only
the top-level `images` arrays; it did not traverse LVIS `annotations` or decode
any polygon/RLE mask.

| Audited set | Distinct image IDs | Sorted-ID-list SHA-256 |
|---|---:|---|
| Downloaded LVIS `lvis_v1_train.json` | 100,170 | `03a93da106befd9b3bdc99b1968082c11247c44d945256bf31c2efbd5e866bd7` |
| Project `instances_val2017.json` | 5,000 | `ea37cee007069bde6d45a67186cc6c2591d0c6d7effd3c481996b6b367e060b6` |
| Exact ID intersection | **0** | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The extracted JSON is 1,097,154,875 bytes with SHA-256
`3741bead824ff960d4f3780a33988588289a58fa3505ee8a6fb7d6a7b1629487`.
The run retains both source ID lists, the empty overlap list, and an empty
per-overlap `coco_url`/file-name audit file.  Its all-record metadata receipt
also finds that all 100,170 LVIS image records have a `coco_url` with a
`/train2017/` segment and that LVIS records have no `file_name` field; all 5,000
COCO val records have `/val2017/` URLs ending in their `file_name`, with numeric
file stem equal to `id`.  Thus the source metadata also separates the declared
COCO image split.  Since no IDs overlap, there are no overlapping records for
which a `coco_url` or file-name equivalence test can be evaluated; that is a
zero-record result, not a skipped test.

Thus, for the downloaded LVIS v1 train annotation named by the locked released
configuration, this project COCO val5k has no image-level overlap.  This does
not exclude a different unpublished/revised training set, extra training data,
or an unrecorded checkpoint history.  Evaluation scope remains a decision for
B/root; this audit did not select examples based on ground truth or model
result.

## B review: declared training scope and frozen evaluation images

On 2026-10-09, B accepted evaluation under the author's declared LVIS-v1-train provenance, with the checkpoint-history limitation above retained. B independently rechecked the two sorted, unique ID lists and their hashes and reproduced the empty intersection; this does not reconstruct the checkpoint's actual training history. The authorized path remains the fixed engineering panel followed by the registered 5k evaluation when engineering and cost feasibility pass, without training, threshold selection, or seed selection.

B also rehashed all 5,000 desktop evaluation JPEGs (814,705,164 bytes), matching the returned notebook source manifest, the original native `BASELINE_PARITY_IMAGES.jsonl`, and the desktop image metadata in identical image order. Native input/identity bindings and the image-list, metadata and source-manifest hashes matched. The source Run `RUN_SEGREFINER_RGB_SOURCE_MANIFEST_20261009_02` and local preparation Run `RUN_SEGREFINER_RGB_CACHE_PREPARATION_20261009_01` both completed with exit 0; their SUMMARY hashes are `42fb4f92c97461b248486f00ff8925d7df4771b05d5624f9b9170d51a15bb65c` and `5078253edfc4e55914af7c5c7fa14ca2c1c8c6ed1bdc7287736d703b5ce693f7`. This review establishes frozen JPEG/source identity; it did not freshly decode RGB, rerun a model, or establish refiner quality or runtime parity. Those remain separate engineering and evaluation evidence.
