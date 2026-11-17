# Assumptions and deviations

Every item is configurable in code. "A-" = assumption where the paper is silent; "D" = deviation / resolved conflict.

| ID | Item | Choice | Where |
|---|---|---|---|
| A-D1 | Proxy groups | per class, natural filename sort, blocks of 5 consecutive images; exact decoded duplicates merged | `data/manifest.py` |
| A-D2 | Inner validation | 10 % group-disjoint StratifiedGroupKFold split of each outer training fold (seed = fold) | `inner_val_split` |
| A-D3 | Outer folds | StratifiedGroupKFold(5, shuffle, seed 0), fixed across training seeds {42,123,2024,7,1337} | `assign_outer_folds` |
| A-TR1 | Checkpoint selection | best EMA-model validation AUC over epochs (live AUC also logged) | `training/trainer.py` |
| A-TR2 | EMA | updated after every optimiser step; BN buffers copied | `training/optim.py` |
| A-TR3 | Warm-up | 10 epochs of **actual** steps/epoch (not 310) | `trainer.fit` |
| A-TR4 | Baselines | single LR 2e-4 (no LLRD), same schedule/loss/aug, optional ImageNet init | `runner.py` |
| A-MIX1 | MixUp/CutMix | per batch, each fires with p=.5 independently; if both fire, one chosen at random (never stacked) | `training/mix.py` |
| A-AUG1 | Aug order | RRC, flip, rotation, jitter, blur, AutoAugment, ToTensor, Normalize, RandomErasing | `data/transforms.py` |
| T1-T6 | TFSG details | DCT-II ortho along channels; top-16 coefficient magnitudes; W_C: Linear(16 -> heads); sigmoid on (e_p+e_q); lambda learnable init 0.1; Swin-style relative bias | `models/tfsg.py` |
| A-CPE1 | CPE | residual 3x3 depthwise conv once at each TFSG stage entry (`cpe_mode=block` switches to per-block) | `models/ocat.py` |
| A-ABL1 | CNN-only ablation | S3/S4 TFSG blocks replaced by MBConv-SE blocks with identical widths/depths (the paper does not define it; this makes it 5.97 M params) | `registry.py` |
| A-CAL1 | Thresholds | F1-opt; sens>=.95 (max spec); spec>=.95 (max sens); ties as documented; decision prob >= thr; ECE 15 equal-width bins on top-label confidence | `calibration/calibrate.py` |
| A-COR1 | Corruptions | applied on uint8 RGB after resize+crop, before normalisation; blur sigma from OpenCV kernel rule (paper gives kernel only); noise sigma on [0,1] scale, clipped; JPEG via PIL/libjpeg | `data/transforms.py` |
| A-PRE1 | Pretraining | **no pretrained OCAT-Net-S checkpoint exists** (custom architecture, not in timm). Default = from scratch, clearly labelled. `load_partial_checkpoint` exists for technically valid partial transfer and logs missing/unexpected/shape-mismatched keys | `registry.py` |
| A-BL1 | Baseline ids (timm 1.0.28) | resnet50, mobilenetv3_large_100, efficientnet_b0, densenet121, deit_small_patch16_224, convnext_tiny, swin_tiny_patch4_window7_224, maxvit_tiny_tf_224, efficientvit_m4, tiny_vit_5m_224; ConvNeXt-B -> convnext_base | `registry.py` |
| D1 | Steps/epoch | paper's 31 steps (=990/32) implies training on all 990 fold images; we reserve validation, so ~28 steps/epoch and total steps differ; validation data is never trained on to match the count | |
| D2 | SE projection | W2: rC/4 -> rC (paper prose typo) | `models/ocat.py` |
| D3 | Captions | Fig. 5/6 captions are swapped relative to content; followed content | |
| D4 | Gate equation | sigmoid applied to e_p+e_q per text | `tfsg.py` |
| D5 | **Model size** | literal spec measures **4.73 M** (reported 3.55 M). Not forced. Decomposition: MLP expansion 4C accounts for the excess - with `mlp_ratio=2` the same code measures **3.53 M** (evidence only; the paper text states C->4C->C). Other variants: no-SE 4.51 M, no-CPE 4.73 M, shared gate 4.73 M | `OCATConfig.mlp_ratio` |
| D6 | Stats | z-tests / CIs on 25 dependent fold-seed runs are re-done as fold-level paired comparisons (df = 4) with Holm correction; interaction variance not separately identified | `evaluation/stats.py` |
| D7 | Repeated predictions | pooled out-of-fold summaries use one prediction per image (seed-averaged) and are labelled; per-seed metrics are never pooled as independent | docs |

## Not resolvable offline (kept as blockers)
* Exact timm id / checkpoint for OCAT-Net-S; original TFSG implementation; group size; inner-val fraction; checkpoint rule.
* Verification of SMART-OM / "Oral Images Dataset" record versions, licences and counts (no dataset access in this session).
* CLIP ViT-B/32 provenance (open_clip checkpoint tag) - hook provided (`clip_embed`), not executed.
