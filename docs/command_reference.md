# Command reference

```bash
pip install -e ".[dev]"
ocat summary [--ablation full|cnn_only|mhsa|no_se|no_cpe]     # per-stage shapes + params from a real forward pass
ocat benchmark [--flops] [--device cuda]                      # params / MACs / latency of OCAT-Net-S + 10 baselines

# data (after downloading the Kaggle dataset yourself)
ocat audit-data --root data/raw --out manifests/private --group-size 5 --seed 0   # manifest, exclusions, 5 fixed folds

# one experiment unit (model, fold, seed); resumable (re-run the same command)
ocat run --manifest manifests/private/manifest.csv --data-root data/raw --out runs \
         --model ocat --ablation full --fold 0 --seed 42 --device cuda
ocat run ... --model tinyvit_5m --pretrained --fold 0 --seed 42          # baselines (keys in models/registry.py)
ocat run ... --model ocat --ablation no_ema                              # training ablations: no_mixup_cutmix, ce_loss, no_label_smoothing, no_ema
ocat status --runs runs --ablations                                      # done/pending over the 475-unit grid

ocat predict --bundle runs/ocat-full-f0-s42/bundle --input path/to/image_or_dir

pytest --cov ; ruff check src tests ; ruff format --check src tests ; ocat smoke   # smoke = SYNTHETIC data only
```
Outputs per unit: `run_config.json` (split hash, norm stats, git commit, env), `history.jsonl` (live and EMA validation AUC,
per-group LRs, mix counts), `best.pt`/`last.pt`, `predictions/test.csv` (sample-level logits/probabilities/fold/seed/
checkpoint/temperature/threshold), `metrics.json`, `bundle/`.
