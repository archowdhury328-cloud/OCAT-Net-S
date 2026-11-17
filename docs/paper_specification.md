# Paper specification (audited from the attached PDF)

Rijvi MH, Momotaj PA, Uddin SMM, Saha US, Hossain S, Chy MAR, Manik MMTG, Basit A. *A compact texture frequency
spatial gating network for oral cancer classification from clinical oral images.* Scientific Reports 16:31071 (2026).
doi:10.1038/s41598-026-59480-0. CC BY 4.0. Version note: only the attached PDF was available; the publisher version
was not separately compared (no network research performed) -> version differences **unverified**.

Legend: **S** stated · **I** inferred from figure/context · **U** unavailable (assumption in this repo).

## Data and protocol
| Item | Paper (location) | Status |
|---|---|---|
| Internal data | Kaggle "Oral Cancer Images for Classification" (M. Atef), 1,238 RGB images: 553 Normal / 685 Oral Cancer (Dataset description, Table 1) | S |
| Folder / filename layout, release, licence | not given | U (must be verified at download) |
| Outer CV | 5-fold stratified group CV x 5 seeds = 25 fold-seed evaluations (Data partitioning) | S |
| Proxy group rule | "filename-sequential grouping", group size / ordering not given | U -> A-D1 (natural sort per class, blocks of 5, configurable) |
| Fold sizes | train 990-991 (442-443 N / 548 C), test 247-248 (110-111 N / 137 C) (Fig. 1) | S (note: 137 C in every fold x 5 = 685 and 110-111 N x 5 = 553: consistent; *train* 548 C = 685-137 ok) |
| Inner validation | "validation operations restricted to the training fold"; fraction, selection rule not given | U -> A-D2 (10 %, group-disjoint, StratifiedGroupKFold) |
| Steps/epoch 31, 6,200 steps (Table 4) | 990/32 = 31 -> implies **no** reserved validation data | **C** (conflicts with reserving validation); we use actual steps (E) |
| Normalisation | train-fold mean/std, applied to val/test (Table 1, Preprocessing) | S; here computed on the optimisation-train partition only |
| Resize | variable -> 224x224; train RandomResizedCrop; eval Resize 256 + CenterCrop 224 | S |

## Augmentation (Table 2)
RRC scale .70-1.00, ratio .75-1.33, bicubic; HFlip .5; Rotation +-15 (p=1, fill 0); ColorJitter (.4,.4,.4,.1);
GaussianBlur 3x3 sigma .1-2.0 p=.5; AutoAugment ImageNet; RandomErasing p=.25 scale .02-.33 ratio .3-3.3 random fill;
MixUp alpha .4 p=.5; CutMix alpha 1.0 p=.5; label smoothing .10. **U**: operation order, MixUp/CutMix exclusivity
(-> A-MIX1: independent draws, never stacked), eval interpolation (bicubic assumed).

## Architecture (Fig. 3, Table 3, text)
Verified hierarchy 224x224x3 -> stem 112x112x32 (conv3x3 s2, BN, GELU) -> 56x56x64 -> S1 2xMBConv (r=4, DP 0) ->
28x28x96 -> S2 3xMBConv-SE (r=4, SE red. 4, DP .05) -> 14x14x160 (+CPE) -> S3 4xTFSG (5 heads, 7x7 windows, DP .15) ->
7x7x256 (+CPE) -> S4 3xTFSG (8 heads, global 7x7, DP .20) -> GAP -> Linear(256,2). **S** (all shapes reproduced by
the code, `tests/unit/test_model.py`).

Internal inconsistencies found: (1) SE text says W2 maps rC -> rC/4; Fig. 5 and channel-wise scaling require rC/4 -> rC
(typo; implemented as restore). (2) Fig. 5 caption is titled "TFSG block with DCT-guided attention" but shows MBConv-SE;
Fig. 6 caption says "MBConv-SE block" but shows TFSG (captions swapped). (3) Table 3 stem row lists 3x3 conv + BN + GELU
but transitions list conv only (no norm) - followed literally. (4) The A_w equation omits the sigmoid that the text
defines for G_w and shows G_w[p,q] = (e_p + e_q) without sigma; text/definition (sigmoid) used. (5) Parameter count
3.55 M vs measured 4.73 M under the literal spec (see reproduction_status).

## TFSG (Eq. for A_w, Fig. 6)
A_w = Softmax((QK^T/sqrt d_k) (.) (1 + lambda G_w) + B) V; DCT-1D of the window representation, C_HF = 16 high-frequency
magnitudes, W_C projection to token-level gate e, G[p,q] = sigma(e_p + e_q). **U**: DCT axis/type/normalisation,
coefficient selection/order, W_C output dims, head sharing, lambda (fixed/learned/init), CPE per stage vs block,
relative-bias form. All resolved as configurable assumptions T1-T6 in `models/tfsg.py`.

## Training (Tables 4, 5)
AdamW (.9,.999,1e-8), wd .05 (no bias/norm), grad clip L2<=1, batch 32, 200 epochs, warm-up 10 epochs linear 0->base,
cosine to 1e-6 (no restart), LLRD .75 with LR head 2e-4 / S4 1.5e-4 / S3 1.125e-4 / S2 8.438e-5 / S1 6.328e-5 / stem
4.746e-5, DropPath per stage, label smoothing .1, EMA .9998, weighted focal loss (gamma 2, w_cancer .6, w_normal .4),
soft targets, ImageNet-1K initialisation "through timm", full fine-tuning, PyTorch 2.0+, RTX 3090. **U**: timm model id
for OCAT-Net-S (none exists, custom architecture); checkpoint-selection rule; EMA update cadence; which weights (live/EMA)
are evaluated ("EMA stabilises evaluation weights" -> EMA assumed).

## Evaluation outputs to reproduce (published values live ONLY in `reporting/paper_reference.py`-style reference data)
Table 6 (10 baselines + stats), Table 7 (9 ablations A1-A10 incl. CNN-only), Tables 8-10 (robustness, 3 models, 9
conditions), Table 11 (efficiency), Tables 12-13 (calibration, thresholds), Tables 14-15 (fold-seed matrix, variance),
Table 16 (leakage audit), Tables 17-18 (external cohorts SMART-OM n=2,165 [2,145 N/20 OSCC], Oral Images Dataset n=323
[165 benign/158 malignant]), Figs. 1-10. ConvNeXt-B is the Grad-CAM comparator (Fig. 10), distinct from the quantitative
ConvNeXt-Tiny.

## Arithmetic cross-checks of the paper's tables (recomputed by hand/script; all passed)
* Table 12 means (T 1.34, Brier .0419) = means of the 5 fold rows; Table 14 grand mean .956 = mean of fold means and of seed means;
  fold-1 mean .942 = mean of its five seeds; ICC_fold .704 = 1.6325e-4 / 2.3175e-4.
* Table 13/18 PPV, NPV, LR+/LR- and FN per 1,000 follow from the stated sensitivity/specificity/prevalence
  (e.g. pi=.03: PPV .288, NPV .9981, 1.74 FN/1000; pi=.553: PPV .942, NPV .928).
* Table 6 deltas = AUC differences; Table 9 degradation % = relative AUC drop; Table 11 AUC/Mparam .269 and AUC/GFLOP .647.
* Fold counts in Fig. 1 add up (553 N, 685 C).

## Real caveats a reproducer must carry
1. Steps/epoch 31 (= 990/32) implies no reserved validation data -> conflicts with the stated inner-validation protocol (D1).
2. Table 6 / 7 "AUC z", p-values, Cohen d use sigma_AUC over 25 dependent fold-seed runs (overlapping training sets, repeated
   test images): not independent. The paper itself reports fold-level variance (ICC .704) and states this limitation.
3. Table 17: 6,190 fold-seed predictions = 1,238 x 5 seeds (each image predicted once per seed), not 25 independent samples.
4. Parameter count (3.55 M) is not reproduced by the literal architecture (D5).
5. Fig. 5/6 captions are swapped relative to their images (D3).
