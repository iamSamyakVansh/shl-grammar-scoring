"""Inference on new .wav files using the artifacts written by `src.train`.

Run:
    python -m src.inference --audio_dir path/to/wavs --out predictions.csv

Each clip goes through exactly the same stages as training (transcribe ->
features -> base models -> blend). Base models were refit on all training
data by `train.py`, and their parameters are reloaded from `artifacts/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CFG, Config
from .features import build_embeddings, build_tabular_features
from .models import clip_scores
from .transcribe import transcribe_split


def predict_frame(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns id, path. Returns id, prediction."""
    tx = transcribe_split(cfg, df)
    F = build_tabular_features(cfg, df, "infer", tx)
    blend = joblib.load(cfg.model_dir / "blend.joblib")
    base = joblib.load(cfg.model_dir / "base_models.joblib")

    preds = {}
    for name, (mk_block, model) in base.items():
        X = F.to_numpy(float) if mk_block == "tab" else build_embeddings(cfg, df, "infer", mk_block, tx)
        preds[name] = clip_scores(model.predict(X), cfg)

    P = np.column_stack([preds[n] for n in blend["names"]] + [np.ones(len(df))])
    return pd.DataFrame({"id": df["id"].values, "prediction": clip_scores(P @ blend["weights"], cfg)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=CFG.submission_path)
    args = ap.parse_args()

    wavs = sorted(args.audio_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files under {args.audio_dir}")
    df = pd.DataFrame({"id": [p.stem for p in wavs], "path": [str(p) for p in wavs]})
    out = predict_frame(CFG, df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
