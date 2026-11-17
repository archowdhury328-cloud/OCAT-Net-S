# Implementation traceability

Status: **S** stated by paper · **I** inferred · **U** unavailable (assumption). Tests are synthetic-data software checks.

| Requirement | Source | Module / config | Verification | Spec status |
|---|---|---|---|---|
| Stage shapes 112/56/28/14/7 | Table 3 | `models/ocat.py` | `test_stage_shapes_match_paper_table3` | S |
| Blocks 2/3/4/3, heads 5/8, windows, DropPath | Table 3, text | `OCATConfig` | `test_block_counts_heads_windows_droppath` | S |
| MBConv r=4, SE red. 4 | Figs. 4-5, text | `MBConv`, `SE` | `test_se_restores_expanded_channels` | S (typo D2) |
| TFSG attention + DCT gate | Eq. A_w, Fig. 6 | `models/tfsg.py` | reference-loop agreement, lambda-0, gradients, shapes | S/U (T1-T6) |
| C_HF = 16 | text | `FreqGate` | `test_gate_changes_output...` | S |
| CPE | Table 3/Fig. 6 | `CPE`, `cpe_mode` | `test_ablation_param_counts` | I (A-CPE1) |
| Relative position bias | Fig. 6 | `WindowFreqAttention.rel_bias` | shape/ref tests | I |
| AdamW, wd .05 w/ exclusions | Table 4 | `build_param_groups` | `test_param_groups_cover_every_param_exactly_once...` | S |
| LLRD .75 + per-group LRs | Table 4 | `GROUP_LR` | `test_llrd_ratio_is_0_75` | S |
| Warm-up + cosine, min 1e-6 | Table 4 | `lr_at` | `test_lr_schedule_...` | S |
| Grad clip L2 <= 1 after unscale | Table 4 | `trainer.fit` | code path in e2e tests | S |
| Weighted focal, gamma 2, w .4/.6, soft targets | Table 5 | `training/losses.py` | hand-calculated test | S/I |
| Label smoothing once, MixUp/CutMix soft targets, CutMix area | Table 2 | `training/mix.py` | `test_mixup_*`, `test_cutmix_*`, `test_apply_batch_mix_*` | S/U |
| EMA .9998 | Table 5 | `EMA` | `test_ema_update_and_buffer_copy` | S |
| Train transforms | Table 2 | `build_train_transform` | `test_train_transform_composition_matches_table2` | S |
| Eval: resize 256, center crop 224, no TTA | text | `build_eval_transform` | `test_eval_transform_deterministic...` | S |
| Train-fold-only normalisation | Table 1 | `compute_norm_stats`, `run_unit` | `test_norm_stats_from_given_images_only` | S |
| 5 outer folds, group-disjoint, same across seeds | Data partitioning | `data/manifest.py` | `test_manifest_groups_folds_and_hash` | S/U |
| Inner val, test fold untouched | Data partitioning | `run_unit` | `test_run_unit_outputs_and_no_test_fold_leakage` | S/U |
| Temperature (val-only), thresholds | Calibration | `calibration/calibrate.py` | calibration tests | S/U |
| ECE-15, Brier, NLL | Table 12 | same | `test_ece_*` | S |
| Prevalence-adjusted PPV/NPV (scenario) | Table 13 | `prevalence_adjusted` | known-answer test | S |
| Fold-by-seed matrix, variance, fold-level inference | Tables 14-15 | `evaluation/stats.py` | stats tests | S/I |
| Leakage audit pHash/SSIM/CLIP | Table 16 | `evaluation/leakage.py` | `test_phash_ssim_...` (CLIP hook only) | S |
| Corruptions | Tables 9-10 | `data/transforms.py::corrupt` | `test_corruptions_reproducible...` | S/U |
| Grad-CAM, predefined example rule | Fig. 10 | `explainability/gradcam.py` | `test_gradcam_on_ocat_stage4` | S |
| Params / MACs / latency measured | Table 11 | `reporting/costs.py`, `ocat benchmark` | `test_stage_report_and_costs` | S |
| 10 baselines + ConvNeXt-B | Table 6, Fig. 10 | `models/registry.py` | `test_baselines_instantiate_with_two_classes` | S/U |
| 8 ablations (+ CNN-only) | Table 7 | `ABLATIONS`, `TRAINING_ABLATIONS` | `test_ablation_param_counts_and_configs` | S/U |
| 475-unit grid, status | plan | `make_grid`, `status` | `test_grid_counts_match_workload_estimate` | - |
| Inference bundle | - | `inference/bundle.py` | round-trip + tamper tests | - |
| External adapters (SMART-OM, Oral Images Dataset) | Table 17 | **pending** (dataset versions unverified) | - | U |
