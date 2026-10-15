"""Plotting helpers used by the notebook and saved to reports/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _save(fig, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_score_distribution(y, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals, counts = np.unique(np.round(y, 1), return_counts=True)
    ax.hist(y, bins=np.arange(0.75, 5.5, 0.25), edgecolor="k", alpha=0.8)
    ax.set_title(f"Grammar score distribution (n={len(y)}, mean={np.mean(y):.2f}, sd={np.std(y):.2f})")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    _save(fig, path)
    return fig


def plot_pred_vs_actual(y, p, title="Out-of-fold prediction vs actual", path: Path | None = None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, s=14, alpha=0.5)
    lims = [0.8, 5.2]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    r = pearsonr(y, p)[0]
    rmse = np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2))
    ax.set_title(f"{title}\nRMSE={rmse:.3f}  Pearson r={r:.3f}")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    _save(fig, path)
    return fig


def plot_residuals(y, p, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    res = np.asarray(y) - np.asarray(p)
    ax.scatter(p, res, s=14, alpha=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("residual (actual - predicted)")
    ax.set_title("Residuals: check for systematic bias across the score range")
    _save(fig, path)
    return fig


def plot_correlation_with_target(F: pd.DataFrame, y, top: int = 25, path: Path | None = None):
    corr = F.apply(lambda c: pearsonr(c, y)[0] if c.std() > 0 else 0.0).sort_values(key=np.abs, ascending=False)
    top_c = corr.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * top + 1))
    ax.barh(top_c.index, top_c.values, color=np.where(top_c.values >= 0, "#2a6fdb", "#d1495b"))
    ax.set_title(f"Top {top} features by |Pearson r| with the grammar score")
    ax.set_xlabel("Pearson r")
    _save(fig, path)
    return fig


def plot_feature_correlation_heatmap(F: pd.DataFrame, top: int = 20, path: Path | None = None):
    cols = F.corr().abs().sum().sort_values(ascending=False).head(top).index
    C = F[cols].corr()
    fig, ax = plt.subplots(figsize=(0.4 * top + 3, 0.4 * top + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Feature inter-correlation (multicollinearity check)")
    _save(fig, path)
    return fig


def plot_feature_importance(names, importances, top: int = 25, path: Path | None = None):
    s = pd.Series(importances, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.3 * top + 1))
    ax.barh(s.index, s.values, color="#2a6fdb")
    ax.set_title(f"Top {top} feature importances (CatBoost)")
    _save(fig, path)
    return fig
