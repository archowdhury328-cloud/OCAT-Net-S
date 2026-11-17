# Reproduction status

**Classification: partial implementation / documented reconstruction. Not an exact reproduction.**
No dataset was available (no Kaggle credentials; no network research performed), so **no training, no baseline runs, no
ablation runs, no calibration/threshold/external/robustness results exist**. Nothing is claimed beyond what is listed below.

## Executed and verified (this session, CPU-only PyTorch 2.13.0+cpu, Windows 11, Python 3.10.11, timm 1.0.28)
| Check | Result |
|---|---|
| `pytest --cov` (synthetic data only) | **59 passed**, **96 %** line coverage of `ocat_net_s` (cli excluded), 74 s |
| `ruff check` / `ruff format --check` | clean |
| Stage shapes vs paper Table 3 | exact match (112/56/28/14/7, channels 32/64/96/160/256) |
| TFSG optimized vs loop reference (float64) | agree (atol 1e-4); lambda=0 reduces to ungated attention; gate/lambda gradients non-zero |
| DCT-II vs `scipy.fft.dct(norm='ortho')` | agree to 1e-10; matrix orthonormal |
| Every parameter in exactly one AdamW group; LLRD ratios .75 | verified |
| Synthetic end-to-end unit (train->EMA->temperature->thresholds->predictions->bundle->reload) | passes; bundle probability equals pipeline probability |
| Checkpoint resume, tamper detection, bad-input rejection | pass |

## Measured model size and cost (`costs.py`; fp32; this machine)
| Model | Params (M) | Note |
|---|---|---|
| OCAT-Net-S (literal spec) | **4.730** (paper 3.55) | 18.1 MB state_dict; MLP ratio 2 variant: 3.531 M |
| CNN-only ablation (A-ABL1) | 5.973 | assumption |
| no-SE / no-CPE / MHSA | 4.508 / 4.726 / 4.730 | |
| ResNet-50 / MobileNetV3-L / EfficientNet-B0 / DenseNet-121 | 23.51 / 4.20 / 4.01 / 6.96 | 2-class heads; paper Table 11: 25.6 / 5.4 / 5.3 / 8.0 (those are 1000-class heads) |
| DeiT-S / ConvNeXt-T / Swin-T / MaxViT-T / EfficientViT-M4 / TinyViT-5M | 21.67 / 27.82 / 27.52 / 30.40 / 8.42 / 5.07 | |

fvcore MACs for OCAT-Net-S: 0.967 G; **unsupported ops (not counted)**: gelu, add, mean, sigmoid, mul, DCT transpose, abs,
softmax. Not comparable to the paper's 1.478 GFLOPs (unknown counter/convention). Latency / peak GPU memory: **not measured**
(CPU torch only).

## Pilot resource estimate (measured)
One batch-32 training step on this CPU: ~6.7 s (4 steps). With ~28 steps/epoch x 200 epochs ~ **10.5 h per unit on CPU**; the
475-unit grid ~ 5,000 CPU-hours before the heavier baselines. An RTX 3050 6 GB is present but the installed torch has no CUDA
build; even on GPU the plan needs a cluster-scale budget. No epochs/seeds/folds/model sizes were reduced; all units are pending.

## Pending (blockers)
| Item | Blocker |
|---|---|
| Real-data audit, folds, leakage audit (CLIP part) | dataset download needs the user's Kaggle credentials; CLIP checkpoint tag not pinned |
| 25 proposed + 250 baseline + 200 ablation runs, ConvNeXt-B | compute (see above) and dataset |
| Calibration/threshold/robustness/statistics tables and figures from real runs | depend on runs |
| External evaluation (SMART-OM, Oral Images Dataset) | exact dataset versions/licences not verified; adapters pending |
| Exact reproduction of TFSG / parameter count | original code/checkpoint unavailable; see deviations D5 |
