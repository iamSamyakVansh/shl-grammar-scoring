"""Central configuration.

Every path and hyper-parameter lives here so the notebook, the training script
and the inference script are guaranteed to use identical settings.
Paths can be overridden with environment variables (useful on Kaggle/Colab).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    # Root of the raw competition data. Files are discovered recursively, so
    # the exact Kaggle layout (csvs/, audios/train, audios/test ...) does not
    # need to be hard-coded.
    data_dir: Path = field(default_factory=lambda: _env_path("SHL_DATA_DIR", ROOT / "data"))
    train_csv_name: str = "train.csv"
    test_csv_name: str = "test.csv"

    artifact_dir: Path = field(default_factory=lambda: _env_path("SHL_ARTIFACT_DIR", ROOT / "artifacts"))
    figure_dir: Path = field(default_factory=lambda: _env_path("SHL_FIGURE_DIR", ROOT / "reports" / "figures"))
    submission_path: Path = field(
        default_factory=lambda: _env_path("SHL_SUBMISSION_PATH", ROOT / "submissions" / "submission.csv")
    )

    # ------------------------------------------------------------ reproducibility
    seed: int = 42
    n_folds: int = 5
    n_repeats: int = 3          # repeated CV -> more stable estimate on 769 samples

    # ------------------------------------------------------------------ audio
    sample_rate: int = 16_000
    pause_threshold_s: float = 0.30   # gap between words counted as a pause
    long_pause_threshold_s: float = 1.0

    # ---------------------------------------------------------------- models
    # Whisper: large-v3 is the most accurate but too slow on CPU for ~1000 clips.
    # small.en is the CPU default here; switch to "large-v3" on a GPU machine.
    # base.en is ~2x faster than small.en on CPU and keeps disfluencies; use
    # SHL_WHISPER_MODEL=large-v3 on a GPU for maximum accuracy.
    whisper_model: str = field(default_factory=lambda: os.environ.get("SHL_WHISPER_MODEL", "base.en"))
    # Verbatim prompt: Whisper silently "fixes" grammar and drops disfluencies.
    # Priming it with a disfluent example nudges it towards verbatim output,
    # which matters because grammar errors ARE the signal we want to measure.
    whisper_prompt: str = (
        "Umm, let me think like, hmm... Okay, so I goes to the market yesterday "
        "and, uh, I buyed some, some vegetables."
    )
    sentence_model: str = "sentence-transformers/all-mpnet-base-v2"
    ssl_model: str = "microsoft/wavlm-base-plus"
    ssl_layers: tuple[int, ...] = (4, 5, 6, 7, 8)   # middle layers: phonetic/linguistic info
    ssl_chunk_s: float = 20.0
    lm_model: str = "gpt2"                          # perplexity as a fluency/grammaticality proxy
    gec_model: str = "vennify/t5-base-grammar-correction"
    spacy_model: str = "en_core_web_sm"

    # Score range of the competition
    min_score: float = 1.0
    max_score: float = 5.0

    # ------------------------------------------------------------- derived
    @property
    def cache_dir(self) -> Path:
        return self.artifact_dir / "cache"

    @property
    def model_dir(self) -> Path:
        return self.artifact_dir / "models"

    def make_dirs(self) -> None:
        for p in (self.cache_dir, self.model_dir, self.figure_dir, self.submission_path.parent):
            p.mkdir(parents=True, exist_ok=True)


CFG = Config()
