"""Feature orchestration with per-stage caching.

Every stage is unsupervised (it never sees the label), so computing it on
train and test independently cannot leak target information. All supervised
steps (scaling, PCA, models) happen inside CV folds in `src/models.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import Config
from ..utils import get_logger

log = get_logger()

# Stage name -> needs heavy optional deps?
TABULAR_STAGES = ("basic", "fluency", "asr", "acoustic", "spacy", "languagetool", "perplexity", "gec")
EMBEDDING_KINDS = ("sent", "ssl")


def _make_featurizer(stage: str, cfg: Config) -> Callable[[str, dict], dict]:
    """Return f(audio_path, transcript) -> feature dict."""
    from . import acoustic, text

    if stage == "basic":
        return lambda path, tr: text.basic_text_features(tr["text"])
    if stage == "fluency":
        return lambda path, tr: acoustic.fluency_features(tr, cfg)
    if stage == "asr":
        return lambda path, tr: acoustic.asr_confidence_features(tr)
    if stage == "acoustic":
        return lambda path, tr: acoustic.acoustic_features(path, cfg)
    if stage == "spacy":
        f = text.SpacyFeaturizer(cfg.spacy_model)
        return lambda path, tr: f(tr["text"])
    if stage == "languagetool":
        f = text.LanguageToolFeaturizer()
        return lambda path, tr: f(tr["text"])
    if stage == "perplexity":
        f = text.PerplexityFeaturizer(cfg.lm_model)
        return lambda path, tr: f(tr["text"])
    if stage == "gec":
        f = text.GECFeaturizer(cfg.gec_model)
        return lambda path, tr: f(tr["text"])
    raise ValueError(stage)


def _stage_cache(cfg: Config, split: str, stage: str) -> Path:
    return cfg.cache_dir / "features" / f"{split}_{stage}.parquet"


def build_tabular_features(
    cfg: Config, df: pd.DataFrame, split: str, transcripts: dict[str, dict],
    stages: tuple[str, ...] = TABULAR_STAGES,
) -> pd.DataFrame:
    """Return a DataFrame indexed by id with one column per feature."""
    parts = []
    for stage in stages:
        path = _stage_cache(cfg, split, stage)
        if path.exists():
            cached = pd.read_parquet(path)
            if set(cached.index) >= set(df["id"]):
                parts.append(cached.loc[df["id"]])
                continue
        try:
            fn = _make_featurizer(stage, cfg)
        except Exception as e:  # missing optional dependency / model download failure
            log.warning("Skipping stage '%s': %s: %s", stage, type(e).__name__, e)
            continue
        rows = [fn(r.path, transcripts[r.id]) for r in tqdm(df.itertuples(), total=len(df), desc=f"{split}:{stage}")]
        feat = pd.DataFrame(rows, index=pd.Index(df["id"], name="id")).astype(float)
        path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(path)
        parts.append(feat)
    if not parts:
        raise RuntimeError("No feature stage could be computed")
    return pd.concat(parts, axis=1)


def build_embeddings(cfg: Config, df: pd.DataFrame, split: str, kind: str,
                     transcripts: dict[str, dict] | None = None) -> np.ndarray | None:
    from . import embeddings

    tag = (cfg.sentence_model if kind == "sent" else cfg.ssl_model).replace("/", "_")
    path = cfg.cache_dir / "embeddings" / f"{split}_{kind}_{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        pos = {k: i for i, k in enumerate(z["ids"])}
        if all(i in pos for i in df["id"]):
            return z["X"][[pos[i] for i in df["id"]]]
    try:
        if kind == "sent":
            X = embeddings.sentence_embeddings([transcripts[i]["text"] for i in df["id"]], cfg)
        elif kind == "ssl":
            X = embeddings.ssl_embeddings(df["path"].tolist(), cfg)
        else:
            raise ValueError(kind)
    except Exception as e:
        log.warning("Skipping embeddings '%s': %s: %s", kind, type(e).__name__, e)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, ids=df["id"].to_numpy())
    return X
