# SHL Hiring Assessment 2026: Spoken Grammar Scoring

Predicts a continuous grammar score in **[1, 5]** from a 45–60 s spoken English answer.
Metrics: **RMSE** (lower is better) and **Pearson r** (higher is better).

## Approach

```
.wav ──► Whisper (verbatim-biased, word timestamps) ──► transcript + timings
  │                                                        │
  │                 ┌──────────────────────────────────────┤
  │                 ▼                                      ▼
  │   tabular features                          sentence embeddings (mpnet)
  │   text · spaCy · LanguageTool · GPT-2 surprisal · GEC edit rate
  │   fluency · ASR confidence                              │
  ▼                                                         │
  acoustic (MFCC, F0, energy)  ──►  WavLM hidden states ────┤
                                                            ▼
              repeated stratified 5-fold x 3 (leakage-free pipelines)
                                                            ▼
          Ridge · SVR · CatBoost · LightGBM · Ridge-on-embeddings
                                                            ▼
                  non-negative linear blend (fit on out-of-fold preds)
                                                            ▼
                                       clip to [1, 5] → submission.csv
```

## Project layout

```
shl-grammar-scoring/
├── notebooks/shl_grammar_scoring.ipynb   # mandatory notebook: EDA → features → CV → blend → plots → submission
├── src/
│   ├── config.py          # every path and hyper-parameter
│   ├── data.py            # recursive CSV / wav discovery, tolerant column detection
│   ├── transcribe.py      # Whisper with JSON caching
│   ├── features/
│   │   ├── text.py        # grammar / linguistic features
│   │   ├── acoustic.py    # fluency, ASR confidence, acoustic
│   │   ├── embeddings.py  # sentence + WavLM embeddings
│   │   └── __init__.py    # per-stage caching, graceful skipping
│   ├── models.py          # estimators, metrics, repeated CV, stacking
│   ├── train.py           # python -m src.train [--quick]
│   ├── inference.py       # python -m src.inference --audio_dir ...
│   ├── viz.py             # figures
│   └── utils.py
├── artifacts/             # caches, fitted models, CV report (generated)
├── reports/figures/       # generated plots
├── submissions/           # submission.csv (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# LanguageTool needs Java 17+ on PATH; if absent that stage is skipped automatically.
```

Put the competition files under `data/` (or set `SHL_DATA_DIR`). The loader searches recursively for `train.csv`, `test.csv` and `*.wav`.

## Run

```bash
python -m src.train            # full pipeline, writes submissions/submission.csv and artifacts/cv_report.json
python -m src.train --quick    # tabular features and CPU models only, for a fast smoke test
python -m src.inference --audio_dir path/to/wavs --out predictions.csv
```

Or open `notebooks/shl_grammar_scoring.ipynb` and run all cells.

## Validation results

Results are out-of-fold (OOF): every training clip is predicted by models that never saw it.

| Model | OOF RMSE | OOF Pearson r |
|---|---|---|
| Constant (train mean) baseline | ~1.238 | n/a |
| Ridge, tabular | 0.704 | 0.830 |
| SVR, tabular | 0.760 | 0.795 |
| CatBoost, tabular | 0.674 | 0.851 |
| LightGBM, tabular | 0.676 | 0.849 |
| Ridge, sentence embeddings | 0.949 | 0.645 |
| Ridge, WavLM embeddings | 0.642 | 0.865 |
| **Blend** | **0.616** | **0.887** |

Run with Whisper `base.en` on CPU and greedy decoding (see "CPU vs GPU" below); a `large-v3` GPU run should do better still. The blend weights (non-negative, sum excludes an unused intercept) put most of its weight on WavLM (0.59) and CatBoost (0.23), with Ridge and LightGBM contributing smaller amounts and SVR and sentence embeddings weighted to zero. Numbers are written fresh to `artifacts/cv_report.json` by every run.

## Design decisions

* **Frozen encoders, not fine-tuning.** 769 labelled clips is too few to fine-tune a 95M-parameter speech encoder without overfitting. Frozen embeddings plus a regularised regressor are stable under CV.
* **Whisper prompted towards disfluent English.** Standard Whisper silently corrects grammar, which removes the errors being scored. Word timestamps are kept because pauses are fluency features.
* **Unsupervised stages are computed once per split.** No label is used, so this cannot leak the target. All supervised steps (scaling, PCA, models, blend) sit inside CV folds or are fit on OOF predictions.
* **Repeated stratified CV on half-point bins.** Repetition stabilises the estimate on 769 rows; stratification keeps every fold across the full score range.
* **Averaged OOF predictions for blending.** Each row gets one prediction averaged over repeats, which is an honest out-of-sample prediction for stacking.
* **Non-negative blend.** Weights are non-negative and fit on OOF predictions, which limits overfitting the blend and keeps the output interpretable.
* **Clipping to [1, 5].** Regressors can overshoot the range, and clipping never hurts RMSE when the true label lies in the range.
* **Caching at every expensive stage.** Transcription and GEC/LM inference are the slow steps; caching makes reruns cheap and deterministic.
* **Graceful degradation.** Missing optional dependencies (LanguageTool, GPU models) skip their stage instead of crashing the run.

## Leaderboard ideas (in priority order)

1. **Feature selection inside CV.** Keep the top features by correlation computed within each training fold, not on the full set.
2. **Multiple Whisper transcripts.** Use `medium.en` and `large-v3` and average the GEC/LM features. ASR errors are noise; averaging reduces it.
3. **Target engineering.** Try training on the rank-transformed score or on the half-point bin, and compare by OOF RMSE.
4. **Stronger embeddings.** Compare HuBERT-large and WavLM-large, and use a learned attention pooling head over frozen frames.
5. **Pairwise / ordinal models.** Model score differences between clips from the same rater pool, if rater IDs are available.
6. **Multi-seed GBDT averaging.** Average 5 seeds of CatBoost; this is usually a free small gain on tiny data.
7. **Stacking with a second-level CatBoost** on OOF predictions plus a few key features, validated with nested CV.
8. **Pseudo-labelling** the 216 test clips only if the blend is confident, and report the effect on OOF.

## Notes

* `inference.py` caches features under the split name `infer`. If you run inference on several different folders, delete `artifacts/cache/features/infer_*` between runs to avoid reusing stale features.
* Transcripts are cached per Whisper model name. Changing `whisper_model` or the prompt does not invalidate old caches. Delete `artifacts/cache/transcripts/` after changing them.
