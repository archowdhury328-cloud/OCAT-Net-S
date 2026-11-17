# OCAT-Net-S — documented reconstruction

Research software reconstructing the pipeline of

> Rijvi M.H., Momotaj P.A., Uddin S.M.M., Saha U.S., Hossain S., Chy M.A.R., Manik M.M.T.G., Basit A.
> *A compact texture frequency spatial gating network for oral cancer classification from clinical oral images.*
> Scientific Reports 16:31071 (2026). doi:10.1038/s41598-026-59480-0

Maintainer: **Md Asikur Rahman Chy** (Department of Business Administration, International American University, Los Angeles, CA, USA).

> **Status: partial implementation / documented reconstruction — not the authors' code, not an exact reproduction.**
> No dataset was available in the build environment, so **no model has been trained on real data and no AUC/F1 is claimed.**
> Research software for an image-level Normal-vs-Oral-Cancer task. Not a medical device; no clinical validity is implied.
> See [docs/reproduction_status.md](docs/reproduction_status.md).

## What is implemented
OCAT-Net-S (stem, MBConv, MBConv-SE, CPE, windowed attention with a differentiable DCT-guided frequency gate, TFSG,
stochastic depth) with shape/param tests; AdamW + layer-wise LR decay, warm-up+cosine, soft-target weighted focal loss,
MixUp/CutMix with correct soft targets, EMA, AMP, resumable checkpoints; grouped 5-fold CV with inner validation and
leakage audits (pHash/SSIM, CLIP hook); validation-only temperature scaling and thresholds; metrics, fold-by-seed
statistics; synthetic corruptions; Grad-CAM; 10 timm baselines + ConvNeXt-B; 8 ablations; 475-unit experiment grid with
resumable status; cost benchmarks; hash-verified inference bundle; CLI. Every unspecified detail is a labelled assumption:
[docs/assumptions_and_deviations.md](docs/assumptions_and_deviations.md).

**Notable measured finding:** the literal architecture measures **4.73 M parameters** (paper: 3.55 M); an MLP expansion
of 2 instead of the text's 4 gives 3.53 M (evidence only; not adopted).

## Install
```bash
python -m venv .venv && . .venv/Scripts/activate        # Python 3.10-3.12
# GPU: install a CUDA torch build first, e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
pytest --cov && ocat smoke
```

## Data
Kaggle "Oral Cancer Images for Classification" (muhammadatef). Accept its terms, download with your own Kaggle credentials
and place it under `data/raw/` (class folders `Normal/`, `Oral Cancer/` or similar; mapping in `data/manifest.py`).
Images and manifests are never committed ([docs/artifact_policy.md](docs/artifact_policy.md)). Groups are
**filename-sequential proxies, not patients or sessions**. External sets (SMART-OM, Oral Images Dataset) are **pending**.

## Use
[docs/command_reference.md](docs/command_reference.md): `ocat audit-data`, `ocat summary`, `ocat run`, `ocat status`,
`ocat benchmark`, `ocat predict`.

## Documentation
[paper spec](docs/paper_specification.md) · [assumptions/deviations](docs/assumptions_and_deviations.md) ·
[traceability](docs/implementation_traceability.md) · [experiment plan](docs/experiment_plan.md) ·
[reproduction status](docs/reproduction_status.md) · [model card](docs/model_card.md) · [data statement](docs/data_statement.md)

## Licence
Code MIT. Does not license datasets, third-party pretrained weights, or the paper (CC BY 4.0). Cite via `CITATION.cff`.
