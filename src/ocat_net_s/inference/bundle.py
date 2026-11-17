"""Inference bundle: weights (state_dict only), model config, preprocessing/normalisation, temperature, threshold.

Security: weights load with ``weights_only=True`` (tensors only; no pickle objects) and are verified against the
SHA-256 in bundle.json (pass ``expected_sha256`` obtained out-of-band for untrusted bundles). No downloads,
no training, no logging of images at inference.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from PIL import Image

from ocat_net_s.calibration.calibrate import positive_prob
from ocat_net_s.data.transforms import build_eval_transform
from ocat_net_s.models.ocat import OCATConfig, OCATNetS
from ocat_net_s.utils.common import LABEL_NAMES, sha256_file

VERSION = "1"
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_PIXELS = 64_000_000


def save_bundle(
    path, model: OCATNetS, mean, std, temperature: float, threshold: float, extra: dict | None = None
) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "weights.pt")
    cfg = {k: v for k, v in dataclasses.asdict(model.cfg).items() if k != "extras"}
    meta = {"bundle_version": VERSION, "model_config": cfg, "norm_mean": list(mean), "norm_std": list(std),
            "temperature": float(temperature), "threshold": float(threshold), "label_map": LABEL_NAMES,
            "preprocess": "Resize256-bicubic, CenterCrop224, ToTensor, Normalize(train-fold stats)",
            "sha256": {"weights.pt": sha256_file(d / "weights.pt")}, "research_use_only": True, **(extra or {})}  # fmt: skip
    (d / "bundle.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d


def load_bundle(path, expected_sha256: str | None = None, device="cpu") -> dict:
    d = Path(path)
    meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    if meta.get("bundle_version") != VERSION:
        raise ValueError(f"unsupported bundle version {meta.get('bundle_version')}")
    h = sha256_file(d / "weights.pt")
    if h != meta["sha256"]["weights.pt"] or (expected_sha256 and h != expected_sha256):
        raise ValueError("weights.pt checksum mismatch")
    c = meta["model_config"]
    cfg = OCATConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.items()})
    model = OCATNetS(cfg)
    model.load_state_dict(torch.load(d / "weights.pt", map_location=device, weights_only=True))
    return {"meta": meta, "model": model.eval().to(device), "device": device,
            "tf": build_eval_transform(meta["norm_mean"], meta["norm_std"])}  # fmt: skip


@torch.no_grad()
def predict_files(b: dict, paths: list) -> list[dict]:
    out = []
    for p in map(Path, paths):
        if p.suffix.lower() not in ALLOWED or not p.is_file():
            raise ValueError(f"unsupported or missing image: {p.name}")
        with Image.open(p) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError(f"image too large: {p.name}")
            x = b["tf"](im.convert("RGB")).unsqueeze(0).to(b["device"])
        z = b["model"](x).cpu().numpy()
        prob = float(positive_prob(z, b["meta"]["temperature"])[0])
        thr = b["meta"]["threshold"]
        out.append({"file": p.name, "predicted_class": LABEL_NAMES[int(prob >= thr)], "prob_oral_cancer": prob,
                    "threshold": thr, "model_version": b["meta"]["bundle_version"],
                    "note": "research software; not a diagnostic device"})  # fmt: skip
    return out
