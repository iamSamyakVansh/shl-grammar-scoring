"""Whisper transcription with word timestamps and ASR confidence.

Uses faster-whisper (CTranslate2) when available: ~4x faster than the reference
implementation on a T4 and it returns word-level timestamps + log-probs, which
we turn into fluency/pause features. Falls back to HF transformers.

Each result is cached as JSON so transcription (the slowest step) runs once.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import Config
from .data import load_audio
from .utils import get_device, get_logger

log = get_logger()


def _cache_path(cfg: Config, file_id: str) -> Path:
    model_tag = cfg.whisper_model.replace("/", "_")
    return cfg.cache_dir / "transcripts" / model_tag / f"{file_id}.json"


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device()
        try:
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(cfg.whisper_model, device=self.device, compute_type=compute_type)
            self.backend = "faster-whisper"
        except ImportError:
            from transformers import pipeline

            name = cfg.whisper_model if "/" in cfg.whisper_model else f"openai/whisper-{cfg.whisper_model}"
            self.model = pipeline(
                "automatic-speech-recognition", model=name, device=0 if self.device == "cuda" else -1,
                chunk_length_s=30, return_timestamps="word",
            )
            self.backend = "transformers"
        log.info("Whisper backend=%s model=%s device=%s", self.backend, cfg.whisper_model, self.device)

    def __call__(self, path: str) -> dict:
        if self.backend == "faster-whisper":
            # Pass a decoded array instead of a path: faster-whisper's built-in
            # decoder is incompatible with some installed `av` versions.
            audio = load_audio(path, sr=16_000)
            segments, info = self.model.transcribe(
                audio,
                language="en",
                # Greedy decoding: ~30% faster on CPU, and beam search tended to drop
                # articles ("a school" -> "school"), which erases grammar signal.
                beam_size=1,
                temperature=0.0,
                word_timestamps=True,
                initial_prompt=self.cfg.whisper_prompt,
                condition_on_previous_text=False,  # avoids hallucinated repetition loops
                vad_filter=False,                  # keep pauses: they are features
            )
            segs, words = [], []
            for s in segments:
                segs.append({
                    "start": s.start, "end": s.end, "text": s.text,
                    "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                    "compression_ratio": s.compression_ratio,
                })
                words += [{"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                          for w in (s.words or [])]
            return {"text": " ".join(s["text"].strip() for s in segs).strip(),
                    "segments": segs, "words": words, "duration": info.duration}

        out = self.model(path, generate_kwargs={"language": "en", "task": "transcribe"})
        words = [{"word": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1], "prob": None}
                 for c in out.get("chunks", [])]
        return {"text": out["text"].strip(), "segments": [], "words": words, "duration": None}


def transcribe_split(cfg: Config, df: pd.DataFrame) -> dict[str, dict]:
    """Transcribe every file in df (id, path); returns {id: transcript dict}."""
    todo = [r for r in df.itertuples() if not _cache_path(cfg, r.id).exists()]
    if todo:
        model = Transcriber(cfg)
        for r in tqdm(todo, desc="whisper"):
            res = model(r.path)
            p = _cache_path(cfg, r.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(res), encoding="utf-8")
    return load_transcripts(cfg, df)


def load_transcripts(cfg: Config, df: pd.DataFrame) -> dict[str, dict]:
    return {fid: json.loads(_cache_path(cfg, fid).read_text(encoding="utf-8")) for fid in df["id"]}
