"""Dataset discovery and loading.

The loader is deliberately tolerant of the exact Kaggle layout: it finds the
CSVs and .wav files recursively and auto-detects the filename / label columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

FILENAME_CANDIDATES = ("filename", "file_name", "file", "audio", "audio_id", "id", "fname")
LABEL_CANDIDATES = ("label", "score", "grammar_score", "target", "grammar")


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find '{name}' anywhere under {root}")
    return hits[0]


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"Could not detect the {what} column in {list(df.columns)}")


def _audio_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in root.rglob("*.wav"):
        index.setdefault(p.stem, []).append(p)
    return index


def _resolve(stem: str, split: str, index: dict[str, list[Path]]) -> Path | None:
    paths = index.get(stem)
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    # Same filename in train/ and test/ folders -> prefer the one under the split folder
    for p in paths:
        if split in (part.lower() for part in p.parts):
            return p
    return paths[0]


def load_split(cfg: Config, split: str) -> pd.DataFrame:
    """Return a DataFrame with columns: id, path, [label]."""
    csv_name = cfg.train_csv_name if split == "train" else cfg.test_csv_name
    df = pd.read_csv(_find_one(cfg.data_dir, csv_name))

    fcol = _detect_column(df, FILENAME_CANDIDATES, "filename")
    # Keep the original filename (with .wav) for the submission; `id` is the stem used for caching
    out = pd.DataFrame({"filename": df[fcol].astype(str)})
    # Filenames repeat across train and test (e.g. audio_12.wav in both), so the id
    # must include the split or caches collide and test rows reuse train transcripts.
    stem = out["filename"].str.replace(r"\.wav$", "", regex=True)
    out["id"] = f"{split}_" + stem

    if split == "train":
        lcol = _detect_column(df, LABEL_CANDIDATES, "label")
        out["label"] = df[lcol].astype(float).values

    index = _audio_index(cfg.data_dir)
    out["path"] = [_resolve(s, split, index) for s in stem]
    missing = out["path"].isna().sum()
    if missing:
        raise FileNotFoundError(f"{missing} {split} audio files not found under {cfg.data_dir}")
    out["path"] = out["path"].astype(str)
    return out.reset_index(drop=True)


def load_audio(path: str | Path, sr: int = 16_000) -> np.ndarray:
    """Load mono float32 audio resampled to `sr`."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)
