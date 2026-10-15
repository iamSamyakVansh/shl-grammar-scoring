"""End-to-end training: features -> base models (repeated CV) -> blend -> artifacts.

Run from the project root:
    python -m src.train            # full pipeline
    python -m src.train --quick    # tabular features only, no GPU models
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import CFG, Config
from .data import load_split
from .features import build_embeddings, build_tabular_features
from .models import (embedding_pipe, fit_full, fit_stacker, gbdt_models, oof_predict,
                     ridge_pipe, score_report, svr_pipe, clip_scores, rmse, pearson)
from .transcribe import transcribe_split
from .utils import get_logger, seed_everything

log = get_logger()


def build_feature_matrices(cfg: Config, train: pd.DataFrame, test: pd.DataFrame, quick: bool):
    tr_tx = transcribe_split(cfg, train)
    te_tx = transcribe_split(cfg, test)
    stages = ("basic", "fluency", "asr", "acoustic") if quick else None
    kw = {"stages": stages} if stages else {}
    Ftr = build_tabular_features(cfg, train, "train", tr_tx, **kw)
    Fte = build_tabular_features(cfg, test, "test", te_tx, **kw)
    blocks = {"tab": (Ftr.to_numpy(float), Fte.to_numpy(float))}
    if not quick:
        for kind in ("sent", "ssl"):
            a = build_embeddings(cfg, train, "train", kind, tr_tx)
            b = build_embeddings(cfg, test, "test", kind, te_tx)
            if a is not None and b is not None:
                blocks[kind] = (a, b)
    return Ftr, Fte, blocks


def main(quick: bool = False, cfg: Config = CFG) -> None:
    seed_everything(cfg.seed)
    cfg.make_dirs()
    train, test = load_split(cfg, "train"), load_split(cfg, "test")
    y = train["label"].to_numpy(float)
    log.info("train=%d test=%d  score mean=%.3f std=%.3f", len(train), len(test), y.mean(), y.std())

    Ftr, Fte, blocks = build_feature_matrices(cfg, train, test, quick)
    log.info("feature blocks: %s", {k: v[0].shape[1] for k, v in blocks.items()})

    # Base learners: each is a (model factory, feature block) pair
    # name -> (factory, block key, train matrix, test matrix)
    specs = {}
    Xtab_tr, Xtab_te = blocks["tab"]
    specs["ridge_tab"] = (ridge_pipe, "tab", Xtab_tr, Xtab_te)
    specs["svr_tab"] = (svr_pipe, "tab", Xtab_tr, Xtab_te)
    if not quick:
        for name, proto in gbdt_models(cfg.seed).items():
            specs[f"{name}_tab"] = (lambda p=proto: clone(p), "tab", Xtab_tr, Xtab_te)
        for kind in ("sent", "ssl"):
            if kind in blocks:
                specs[f"ridge_{kind}"] = (embedding_pipe, kind, *blocks[kind])

    oof, test_pred, report, fitted = {}, {}, {}, {}
    for name, (mk, block, Xtr, Xte) in specs.items():
        mean, std, f_rmse, f_r = oof_predict(mk, Xtr, y, cfg)
        oof[name] = mean
        report[name] = {
            "oof_rmse": rmse(y, mean), "oof_pearson": pearson(y, mean),
            "fold_rmse_mean": float(f_rmse.mean()), "fold_rmse_std": float(f_rmse.std()),
            "fold_pearson_mean": float(f_r.mean()),
        }
        log.info("%-14s OOF RMSE=%.4f  r=%.4f  (fold RMSE %.3f±%.3f)", name,
                 report[name]["oof_rmse"], report[name]["oof_pearson"],
                 report[name]["fold_rmse_mean"], report[name]["fold_rmse_std"])
        full = fit_full(mk, Xtr, y)
        fitted[name] = (block, full)
        test_pred[name] = clip_scores(full.predict(Xte), cfg)

    names = list(oof)
    O = np.column_stack([oof[n] for n in names])
    T = np.column_stack([test_pred[n] for n in names])
    w = fit_stacker(O, y, cfg)
    blend_oof = clip_scores(np.column_stack([O, np.ones(len(y))]) @ w, cfg)
    blend_test = clip_scores(np.column_stack([T, np.ones(len(T))]) @ w, cfg)

    report["blend"] = score_report(y, blend_oof, cfg)
    report["blend"]["weights"] = dict(zip(names + ["intercept"], np.round(w, 4).tolist()))
    log.info("BLEND OOF RMSE=%.4f  Pearson=%.4f", report["blend"]["rmse"], report["blend"]["pearson"])

    # Persist everything the inference script and the notebook need
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"names": names, "weights": w}, cfg.model_dir / "blend.joblib")
    joblib.dump({n: fitted[n] for n in names}, cfg.model_dir / "base_models.joblib")
    (cfg.artifact_dir / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"id": train["id"], "label": y, **{f"oof_{n}": oof[n] for n in names},
                  "oof_blend": blend_oof}).to_csv(cfg.artifact_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({"filename": test["filename"], "label": blend_test}).to_csv(cfg.submission_path, index=False)
    pd.DataFrame({"id": test["id"], **{n: test_pred[n] for n in names}}).to_csv(
        cfg.artifact_dir / "test_base_predictions.csv", index=False)
    Ftr.to_parquet(cfg.artifact_dir / "train_features.parquet")
    Fte.to_parquet(cfg.artifact_dir / "test_features.parquet")
    log.info("Wrote submission -> %s", cfg.submission_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tabular features and CPU models only")
    main(quick=ap.parse_args().quick)
