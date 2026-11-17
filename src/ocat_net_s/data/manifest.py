"""Manifest, filename-sequential proxy groups, grouped folds, inner validation split.

Proxy group rule (ASSUMPTION A-D1; the paper gives no group size / sort rule): within each class, files are
sorted by natural filename order and chunked into consecutive blocks of ``group_size`` (default 5). Groups are
NOT patients or sessions. Folds: 5 fixed outer StratifiedGroupKFold (seed 0), identical across training seeds.
Exact-duplicate images (decoded-pixel checksum) are forced into one group; label-conflicting duplicates are excluded.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from ocat_net_s.utils.common import CANCER, NORMAL, sha256_file, sha256_json

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_LABEL_DIRS = {"normal": NORMAL, "non-cancer": NORMAL, "noncancer": NORMAL, "cancer": CANCER,
                      "oral cancer": CANCER, "oral_cancer": CANCER, "oscc": CANCER}  # fmt: skip
COLUMNS = ["image_id", "rel_path", "filename", "label", "source", "width", "height", "file_sha256",
           "decoded_sha256", "group_id", "fold"]  # fmt: skip


def natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _inspect(p: Path) -> tuple[str, int, int]:
    with Image.open(p) as im:
        im.verify()
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest(), a.shape[1], a.shape[0]


def build_manifest(root: str | Path, source: str = "atef_kaggle", label_dirs: dict | None = None):
    """Return (manifest, exclusions). Unreadable files, unmapped dirs and conflicting duplicates are reported."""
    root = Path(root).resolve()
    label_dirs = label_dirs or DEFAULT_LABEL_DIRS
    rows, excl = [], []
    for p in sorted(root.rglob("*"), key=lambda q: natural_key(q.as_posix())):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        rel = p.relative_to(root)
        lab = next((label_dirs[x.lower()] for x in rel.parts[:-1] if x.lower() in label_dirs), None)
        if lab is None:
            excl.append({"rel_path": rel.as_posix(), "reason": "unmapped_label_dir"})
            continue
        try:
            dec, w, h = _inspect(p)
        except Exception as e:
            excl.append({"rel_path": rel.as_posix(), "reason": f"corrupt:{type(e).__name__}"})
            continue
        f = sha256_file(p)
        rows.append({"image_id": f[:16], "rel_path": rel.as_posix(), "filename": p.name, "label": lab,
                     "source": source, "width": w, "height": h, "file_sha256": f, "decoded_sha256": dec,
                     "group_id": "", "fold": -1})  # fmt: skip
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df, pd.DataFrame(excl, columns=["rel_path", "reason"])
    nl = df.groupby("decoded_sha256")["label"].nunique()
    bad = df["decoded_sha256"].isin(nl[nl > 1].index)
    excl += [{"rel_path": r, "reason": "label_conflict_duplicate"} for r in df.loc[bad, "rel_path"]]
    df = df[~bad].reset_index(drop=True)
    dup = df["image_id"].duplicated(keep=False)
    df.loc[dup, "image_id"] = df.loc[dup, "image_id"] + "_" + df.loc[dup].index.astype(str)
    return df, pd.DataFrame(excl, columns=["rel_path", "reason"])


def assign_proxy_groups(df: pd.DataFrame, group_size: int = 5) -> pd.DataFrame:
    """Filename-sequential groups per class; exact duplicates merged into the earliest member's group."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    out = df.copy()
    out["_k"] = out["filename"].map(natural_key)
    gid = pd.Series("", index=out.index, dtype=object)
    for lab, sub in out.groupby("label"):
        order = sub.sort_values("_k", key=lambda s: s.map(lambda k: [(isinstance(t, str), t) for t in k])).index
        for i, idx in enumerate(order):
            gid[idx] = f"c{lab}-g{i // group_size}"
    out["group_id"] = gid
    first = out.groupby("decoded_sha256")["group_id"].transform("first")
    out["group_id"] = first
    return out.drop(columns="_k")


def assign_outer_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(out, out["label"], out["group_id"])):
        out.iloc[te, out.columns.get_loc("fold")] = k
    return out


def inner_val_split(train_df: pd.DataFrame, frac: float = 0.10, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-disjoint stratified optimisation-train / validation split inside an outer training fold."""
    n = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(train_df, train_df["label"], train_df["group_id"]))
    return train_df.iloc[tr], train_df.iloc[va]


def verify_folds(df: pd.DataFrame) -> None:
    if (df["fold"] < 0).any():
        raise ValueError("unassigned fold")
    if (df.groupby("group_id")["fold"].nunique() > 1).any():
        raise ValueError("a proxy group spans multiple folds")


def folds_hash(df: pd.DataFrame) -> str:
    return sha256_json(sorted(zip(df["image_id"], df["fold"].astype(int), strict=True)))
