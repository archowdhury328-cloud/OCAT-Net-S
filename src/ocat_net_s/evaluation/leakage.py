"""Image-similarity leakage audit: pHash, SSIM, CLIP ViT-B/32 cosine. A screening aid, NOT proof of patient identity.

Pair populations: within-group (adjacent filenames / same proxy group), cross-group (different outer folds),
random (any two images), cross-split (different outer folds, = cross-group by construction of the folds).
Sampling is seeded; thresholds: pHash Hamming <= 5, SSIM >= 0.90, CLIP cosine >= 0.95 (paper Table 16).
CLIP: pass ``clip_embed`` (callable PIL->unit vector) built from open_clip ViT-B/32 with its checkpoint tag recorded;
no network download happens here.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import uniform_filter


def phash(img: Image.Image, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    n = hash_size * highfreq
    a = np.asarray(img.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float64)
    d = dctn(a, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())


def ssim(a: Image.Image, b: Image.Image, size: int = 128, win: int = 7) -> float:
    """Mean SSIM on grayscale resized images (uniform window, standard C1/C2 for 8-bit)."""
    x = np.asarray(a.convert("L").resize((size, size)), dtype=np.float64)
    y = np.asarray(b.convert("L").resize((size, size)), dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = uniform_filter(x, win), uniform_filter(y, win)
    vx, vy = uniform_filter(x * x, win) - mx**2, uniform_filter(y * y, win) - my**2
    cxy = uniform_filter(x * y, win) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s.mean())


def sample_pairs(groups: np.ndarray, folds: np.ndarray, n: int, seed: int = 0) -> dict[str, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    N = len(groups)
    within = [(i, j) for g in np.unique(groups) for i, j in combinations(np.flatnonzero(groups == g), 2)]
    out = {"within_group": [within[k] for k in rng.permutation(len(within))[:n]] if within else []}
    cross, rand = [], []
    while len(cross) < n or len(rand) < n:
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
        if len(rand) < n:
            rand.append((int(i), int(j)))
        if folds[i] != folds[j] and len(cross) < n:
            cross.append((int(i), int(j)))
    out["cross_group_different_folds"], out["random"] = cross, rand
    return out


def audit(images: list[Image.Image], pairs: dict, clip_embed=None) -> dict:
    hashes = [phash(im) for im in images]
    emb = [clip_embed(im) for im in images] if clip_embed else None
    res = {}
    for name, ps in pairs.items():
        if not ps:
            res[name] = {"n_pairs": 0}
            continue
        h = np.array([hamming(hashes[i], hashes[j]) for i, j in ps])
        s = np.array([ssim(images[i], images[j]) for i, j in ps])
        r = {"n_pairs": len(ps), "mean_phash": float(h.mean()), "frac_phash_le5": float((h <= 5).mean()),
             "mean_ssim": float(s.mean()), "frac_ssim_ge90": float((s >= 0.90).mean())}  # fmt: skip
        if emb is not None:
            c = np.array([float(np.dot(emb[i], emb[j])) for i, j in ps])
            r.update(mean_clip=float(c.mean()), frac_clip_ge95=float((c >= 0.95).mean()))
        else:
            r["clip"] = "not computed (no embedding function supplied)"
        res[name] = r
    return res
