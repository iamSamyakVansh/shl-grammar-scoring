"""Fluency (from Whisper word timings), ASR-confidence and acoustic features.

Grammar is the target, but human raters are influenced by fluency, and
disfluent speech correlates with grammatical errors, so these features carry
real signal. ASR confidence also tells the model when the transcript (and thus
every text feature) is unreliable.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..data import load_audio


def _stats(prefix: str, x, fns=("mean", "std", "min", "max")) -> dict[str, float]:
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if x.size == 0:
        return {f"{prefix}_{f}": 0.0 for f in fns}
    return {f"{prefix}_{f}": float(getattr(np, f)(x)) for f in fns}


def fluency_features(tr: dict, cfg: Config) -> dict[str, float]:
    words = [w for w in tr.get("words", []) if w.get("start") is not None and w.get("end") is not None]
    feats: dict[str, float] = {}
    if len(words) < 2:
        keys = ("flu_speech_rate", "flu_articulation_rate", "flu_pause_rate", "flu_pause_mean", "flu_pause_std",
                "flu_pause_max", "flu_long_pause_rate", "flu_pause_time_ratio", "flu_mean_run_len")
        return {k: 0.0 for k in keys}

    span = max(words[-1]["end"] - words[0]["start"], 1e-3)
    gaps = np.array([b["start"] - a["end"] for a, b in zip(words, words[1:])])
    gaps = np.clip(gaps, 0, None)
    pauses = gaps[gaps > cfg.pause_threshold_s]
    phonation = span - pauses.sum()
    # Mean length of run: words spoken between pauses (classic L2 fluency measure)
    runs = np.diff(np.r_[-1, np.where(gaps > cfg.pause_threshold_s)[0], len(words) - 1])

    minutes = span / 60
    feats.update({
        "flu_speech_rate": len(words) / span,
        "flu_articulation_rate": len(words) / max(phonation, 1e-3),
        "flu_pause_rate": len(pauses) / minutes,
        "flu_pause_mean": float(pauses.mean()) if pauses.size else 0.0,
        "flu_pause_std": float(pauses.std()) if pauses.size else 0.0,
        "flu_pause_max": float(pauses.max()) if pauses.size else 0.0,
        "flu_long_pause_rate": float((gaps > cfg.long_pause_threshold_s).sum()) / minutes,
        "flu_pause_time_ratio": float(pauses.sum()) / span,
        "flu_mean_run_len": float(runs.mean()),
    })
    return feats


def asr_confidence_features(tr: dict) -> dict[str, float]:
    probs = [w.get("prob") for w in tr.get("words", [])]
    probs = [p for p in probs if p is not None]
    segs = tr.get("segments", [])
    feats = _stats("asr_word_prob", probs, ("mean", "std", "min"))
    feats["asr_low_conf_ratio"] = float(np.mean(np.array(probs) < 0.5)) if probs else 0.0
    feats.update(_stats("asr_seg_logprob", [s.get("avg_logprob") for s in segs], ("mean", "min")))
    feats.update(_stats("asr_compression", [s.get("compression_ratio") for s in segs], ("mean", "max")))
    feats.update(_stats("asr_no_speech", [s.get("no_speech_prob") for s in segs], ("mean", "max")))
    return feats


def acoustic_features(path: str, cfg: Config) -> dict[str, float]:
    import librosa

    y = load_audio(path, cfg.sample_rate)
    sr = cfg.sample_rate
    feats: dict[str, float] = {"ac_duration": len(y) / sr}
    if len(y) < sr // 2:
        return feats

    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum(e - s for s, e in intervals)
    feats["ac_silence_ratio"] = 1 - voiced / len(y)
    feats["ac_n_voiced_segments_per_min"] = len(intervals) / (len(y) / sr / 60)

    rms = librosa.feature.rms(y=y)[0]
    feats.update(_stats("ac_rms", rms, ("mean", "std")))

    # YIN is ~50x faster than pYIN; we mask unvoiced frames using energy.
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr)
    n = min(len(f0), len(rms))
    f0 = f0[:n][rms[:n] > np.percentile(rms, 40)]
    feats.update(_stats("ac_f0", f0, ("mean", "std")))
    feats["ac_f0_range"] = float(np.percentile(f0, 95) - np.percentile(f0, 5)) if f0.size else 0.0

    feats.update(_stats("ac_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)[0], ("mean", "std")))
    feats.update(_stats("ac_zcr", librosa.feature.zero_crossing_rate(y)[0], ("mean",)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"ac_mfcc{i}_mean"] = float(mfcc[i].mean())
        feats[f"ac_mfcc{i}_std"] = float(mfcc[i].std())
    return feats
