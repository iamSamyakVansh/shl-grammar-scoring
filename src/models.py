"""Model definitions, repeated stratified CV, and metrics.

Leakage controls:
* Unsupervised feature stages are computed once per split (no labels used).
* Scalers, PCA and every regressor are fitted inside each CV fold via sklearn
  Pipelines, so validation folds never influence preprocessing.
* Stratification is on the binned score, so every fold sees the full 1-5 range.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import pearsonr

from .config import Config


# ----------------------------------------------------------------- metrics
def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def pearson(y, p) -> float:
    return float(pearsonr(np.asarray(y, float), np.asarray(p, float))[0])


def clip_scores(p, cfg: Config) -> np.ndarray:
    return np.clip(np.asarray(p, float), cfg.min_score, cfg.max_score)


def score_report(y, p, cfg: Config) -> dict[str, float]:
    pc = clip_scores(p, cfg)
    return {"rmse": rmse(y, pc), "pearson": pearson(y, pc)}


# ------------------------------------------------------------------ CV
def stratify_bins(y: np.ndarray) -> np.ndarray:
    """Half-point bins of the score; used only to stratify folds."""
    return np.clip(np.round(np.asarray(y) * 2).astype(int), 0, None)


def make_cv(cfg: Config) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=cfg.n_folds, n_repeats=cfg.n_repeats, random_state=cfg.seed)


# -------------------------------------------------------------- estimators
def ridge_pipe() -> object:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def svr_pipe() -> object:
    return make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.1, gamma="scale"))


def embedding_pipe(n_components: int = 64) -> object:
    """Dense embeddings are high-dimensional and 769 rows is small: reduce, then regress."""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=0),
                         RidgeCV(alphas=np.logspace(-2, 4, 30)))


def gbdt_models(seed: int = 42) -> dict[str, object]:
    """Lazy import: the notebook still runs if a GBDT library is missing."""
    models: dict[str, object] = {}
    try:
        from catboost import CatBoostRegressor

        models["catboost"] = CatBoostRegressor(
            iterations=1500, learning_rate=0.03, depth=5, l2_leaf_reg=5,
            loss_function="RMSE", random_seed=seed, verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
            random_state=seed, verbose=-1,
        )
    except ImportError:
        pass
    return models


def oof_predict(model_fn, X: np.ndarray, y: np.ndarray, cfg: Config):
    """Repeated-CV out-of-fold predictions.

    Returns (oof_mean, oof_std, fold_rmse, fold_pearson). `oof_mean` averages
    the predictions each sample received across repeats: it is an honest
    out-of-sample prediction for every training row.
    """
    cv = make_cv(cfg)
    bins = stratify_bins(y)
    n = len(y)
    sums, sq, counts = np.zeros(n), np.zeros(n), np.zeros(n)
    fold_rmse, fold_r = [], []
    for tr, va in cv.split(X, bins):
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = clip_scores(m.predict(X[va]), cfg)
        sums[va] += p
        sq[va] += p ** 2
        counts[va] += 1
        fold_rmse.append(rmse(y[va], p))
        fold_r.append(pearson(y[va], p))
    mean = sums / counts
    std = np.sqrt(np.maximum(sq / counts - mean ** 2, 0))
    return mean, std, np.array(fold_rmse), np.array(fold_r)


def fit_full(model_fn, X: np.ndarray, y: np.ndarray):
    m = model_fn()
    m.fit(X, y)
    return m


# ------------------------------------------------------------ ensembling
def fit_stacker(oof: np.ndarray, y: np.ndarray, cfg: Config):
    """Non-negative linear blend of base-model OOF predictions.

    Fitted on OOF predictions only, so the blend weights are not tuned on
    rows the base models were trained on. Weights are then applied to
    test-time predictions from models refit on all training data.
    """
    from scipy.optimize import nnls

    A = np.column_stack([oof, np.ones(len(y))])
    w, _ = nnls(A, y)
    return w  # last entry is the intercept
