# Experiment plan

Order: env/paper audit -> dataset+grouping audit -> tests -> smoke run -> ONE proposed-model fold (pipeline check) ->
**freeze config** -> proposed 5x5 -> 10 baselines 5x5 -> ablations 5x5 -> ConvNeXt-B (qualitative, selected folds) ->
calibration/statistics/robustness/external/benchmarks -> report.

| Block | Units | Command stem |
|---|---|---|
| Proposed OCAT-Net-S | 5 folds x 5 seeds = 25 | `ocat run --model ocat` |
| 10 baselines | 250 (total primary = **275**) | `ocat run --model <key> --pretrained` |
| 8 ablations (CNN-only, MHSA, no-SE, no-CPE, no-MixUp/CutMix, CE loss, no label smoothing, no EMA) | **200** | `ocat run --model ocat --ablation <key>` |
| ConvNeXt-B Grad-CAM comparator | as needed | `ocat run --model convnext_b` |

`ocat status --ablations` reports done/pending out of 475 units (`DONE` markers; every unit resumable via `last.pt`).
Seeds {42, 123, 2024, 7, 1337}; no seed or fold is dropped. Test folds are read once per unit, after selection.

**Pilot / resource estimate:** see `docs/reproduction_status.md` (measured on this machine, CPU-only torch).
No epochs/seeds/folds/model sizes are reduced silently; if resources are insufficient the remainder is reported as pending.
Ablation note: Table 7 has 9 rows (A1 full + 8 variants listed above + CNN-only = A10); the "ablations" count here is the
8 non-reference variants.
