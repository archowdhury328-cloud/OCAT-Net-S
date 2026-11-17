"""ocat command line. See docs/command_reference.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _cmd_audit_data(a):
    from ocat_net_s.data.manifest import (
        assign_outer_folds,
        assign_proxy_groups,
        build_manifest,
        folds_hash,
        verify_folds,
    )

    df, excl = build_manifest(a.root)
    if df.empty:
        sys.exit(f"no labelled images under {a.root} (expected class folders such as Normal/ and Oral Cancer/)")
    df = assign_outer_folds(assign_proxy_groups(df, a.group_size), seed=a.seed)
    verify_folds(df)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest.csv", index=False)
    excl.to_csv(out / "exclusions.csv", index=False)
    s = {"n_images": len(df), "n_excluded": len(excl), "folds_hash": folds_hash(df), "group_size": a.group_size,
         "per_fold_class": df.groupby(["fold", "label"]).size().unstack(fill_value=0).to_dict(),
         "note": "filename-sequential proxy groups; NOT patients or sessions"}  # fmt: skip
    (out / "audit_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))


def _cmd_summary(a):
    from ocat_net_s.models.registry import build_model
    from ocat_net_s.reporting import costs

    m = build_model("ocat", ablation=a.ablation)
    print(pd.DataFrame(costs.stage_report(m)).to_string(index=False))
    print("params:", costs.param_counts(m))


def _cmd_run(a):
    import yaml

    from ocat_net_s.training.runner import RunSpec, run_unit
    from ocat_net_s.training.trainer import TrainConfig

    cfg = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    spec = RunSpec(a.manifest, a.data_root, a.out, a.model, a.ablation, a.fold, a.seed, device=a.device,
                   pretrained=a.pretrained, train=TrainConfig(**cfg.get("train", {})))  # fmt: skip
    print(json.dumps(run_unit(spec), indent=2, default=str))


def _cmd_status(a):
    from ocat_net_s.models.registry import ABLATIONS, BASELINES
    from ocat_net_s.training.runner import make_grid, status

    models = [("ocat", "full")] + [(b, "base") for b in BASELINES]
    if a.ablations:
        models += [("ocat", k) for k in list(ABLATIONS) if k != "full"] + [
            ("ocat", k) for k in ("no_mixup_cutmix", "ce_loss", "no_label_smoothing", "no_ema")
        ]
    st = status(make_grid(models), a.runs)
    print(f"{int(st.done.sum())}/{len(st)} runs done")
    st.to_csv(Path(a.runs) / "status.csv" if Path(a.runs).exists() else "status.csv", index=False)


def _cmd_benchmark(a):
    from ocat_net_s.models.registry import BASELINES, build_model
    from ocat_net_s.reporting import costs

    models = {"ocat_full": build_model("ocat")} | {k: build_model(k) for k in BASELINES}
    rows = []
    for k, m in models.items():
        r = {"model": k, **costs.param_counts(m), "size_mb": round(costs.serialized_mb(m), 1)}
        if a.flops:
            r["macs_g"] = round(costs.macs_fvcore(m)["macs_g"], 3)
        lat = costs.latency(m, a.device, 1)
        r.update(b1_median_ms=lat["median_ms"], b1_p95_ms=lat["p95_ms"])
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def _cmd_predict(a):
    from ocat_net_s.inference.bundle import load_bundle, predict_files

    b = load_bundle(a.bundle, device=a.device)
    p = Path(a.input)
    print(json.dumps(predict_files(b, sorted(p.iterdir()) if p.is_dir() else [p]), indent=2))


def _cmd_smoke(a):
    import tempfile

    from ocat_net_s.data.manifest import assign_outer_folds, assign_proxy_groups, build_manifest
    from ocat_net_s.training.runner import RunSpec, run_unit
    from ocat_net_s.training.trainer import TrainConfig
    from ocat_net_s.utils.synthetic import make_synthetic_dataset

    with tempfile.TemporaryDirectory() as t:
        make_synthetic_dataset(Path(t) / "d")
        df, _ = build_manifest(Path(t) / "d")
        assign_outer_folds(assign_proxy_groups(df, 3)).to_csv(Path(t) / "m.csv", index=False)
        tc = TrainConfig(epochs=1, batch_size=8, warmup_epochs=0, mix=True)
        r = run_unit(RunSpec(str(Path(t) / "m.csv"), str(Path(t) / "d"), str(Path(t) / "o"), train=tc))
        print("SYNTHETIC SMOKE OK (metrics meaningless):", json.dumps(r["val_calibration"]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="ocat", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit-data")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)  # noqa: E702
    s.add_argument("--group-size", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=_cmd_audit_data)  # noqa: E702
    s = sub.add_parser("summary")
    s.add_argument("--ablation", default="full")
    s.set_defaults(f=_cmd_summary)  # noqa: E702
    s = sub.add_parser("run", help="one (model, fold, seed) unit")
    for k in ("--manifest", "--data-root", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--model", default="ocat")
    s.add_argument("--ablation", default="full")  # noqa: E702
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--seed", type=int, default=42)  # noqa: E702
    s.add_argument("--config")
    s.add_argument("--device", default="cpu")
    s.add_argument("--pretrained", action="store_true")  # noqa: E702
    s.set_defaults(f=_cmd_run)
    s = sub.add_parser("status")
    s.add_argument("--runs", default="runs")
    s.add_argument("--ablations", action="store_true")
    s.set_defaults(f=_cmd_status)  # noqa: E702
    s = sub.add_parser("benchmark")
    s.add_argument("--device", default="cpu")
    s.add_argument("--flops", action="store_true")
    s.set_defaults(f=_cmd_benchmark)  # noqa: E702
    s = sub.add_parser("predict")
    s.add_argument("--bundle", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--device", default="cpu")
    s.set_defaults(f=_cmd_predict)  # noqa: E702
    s = sub.add_parser("smoke")
    s.set_defaults(f=_cmd_smoke)  # noqa: E702
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
