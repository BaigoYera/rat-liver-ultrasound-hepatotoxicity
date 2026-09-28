# Early Detection of Hepatotoxicity from Ultrasound and Biochemical Data

This repository contains the de-identified experimental data, PyTorch analysis code, and derived results supporting the manuscript **“Early Detection of Hepatotoxicity via Deep Learning of Ultrasound Imagery and Biochemical Profiling with Mechanistic Evidence.”**

## Repository contents

- `data/ultrasound/` — 16 B-mode liver ultrasound images used in the local rat analysis.
- `data/biochemistry/` — model-ready ALT, AST, and MDA tables extracted from the study records, plus a coverage report.
- `manifests/` — subject-level ultrasound manifest and severity labels.
- `scripts/run_rat_loocv.py` — ResNet-18 leave-one-rat-out cross-validation.
- `scripts/make_enhanced_paper_outputs.py` — generation of manuscript figures and result tables.
- `scripts/build_ultrasound_manifests.py` — reproducible manifest construction.
- `multimodal/src/` — late-fusion ultrasound and biochemical modeling scaffold.
- `results/rat_loocv/` — fold metrics, per-subject predictions, and aggregate metrics.
- `results/figures/` and `results/tables/` — derived manuscript-ready outputs.

The trained model checkpoint is not included because it is a generated binary artifact. It can be recreated from the supplied images, manifest, and training script.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the rat-level validation

Run from the repository root:

```bash
python scripts/run_rat_loocv.py \
  --manifest manifests/local_rat_ultrasound_manifest.csv \
  --imagenet \
  --output-dir outputs/rat_loocv
```

To regenerate the paper figures and tables after producing predictions:

```bash
python scripts/make_enhanced_paper_outputs.py
```

## Important limitations

This is a small proof-of-concept animal study. The independent evaluation unit is the animal (`n = 16`), not the individual image. Group-level biomarker correlations are exploratory because they are based on three injury-severity groups. The supplied multimodal scaffold should not be interpreted as a clinically validated diagnostic system.

## Data notes

- The ultrasound images contain no human participant identifiers.
- Histology microphotographs in original resolution are not included and remain available from the corresponding author on reasonable request.
- External pretraining datasets are not redistributed here; consult their original sources and licenses.

## Citation

Please cite the accompanying manuscript when using this repository. Full publication metadata will be added after publication.
