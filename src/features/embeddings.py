"""Dense representations.

* Sentence embeddings of the transcript (what was said / how well-formed it is).
* Self-supervised speech embeddings (WavLM / HuBERT / Wav2Vec2): capture
  pronunciation, prosody and fluency that the transcript loses.

Both are frozen feature extractors: with only 769 labelled clips, fine-tuning
a 95M-parameter encoder end-to-end overfits easily, while frozen embeddings +
a regularised regressor are stable and cheap to cross-validate.
"""
from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from ..config import Config
from ..data import load_audio
from ..utils import get_device


def sentence_embeddings(texts: list[str], cfg: Config, batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.sentence_model, device=get_device())
    # mpnet truncates at 384 tokens; a 60 s answer (~150 words) fits comfortably.
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def ssl_embeddings(paths: list[str], cfg: Config) -> np.ndarray:
    """Mean-pooled hidden states averaged over `cfg.ssl_layers`.

    Audio is processed in `cfg.ssl_chunk_s` chunks to bound GPU memory;
    chunk outputs are concatenated in time before pooling. We return
    [mean || std] pooling, so the dimension is 2 * hidden_size.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = get_device()
    fe = AutoFeatureExtractor.from_pretrained(cfg.ssl_model)
    model = AutoModel.from_pretrained(cfg.ssl_model).to(device).eval()
    chunk = int(cfg.ssl_chunk_s * cfg.sample_rate)

    out = []
    for p in tqdm(paths, desc="ssl-embed"):
        y = load_audio(p, cfg.sample_rate)
        frames = []
        for s in range(0, len(y), chunk):
            seg = y[s : s + chunk]
            if len(seg) < cfg.sample_rate // 4:  # skip <250 ms tail
                continue
            inp = fe(seg, sampling_rate=cfg.sample_rate, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                hs = model(inp, output_hidden_states=True).hidden_states
            frames.append(torch.stack([hs[i][0] for i in cfg.ssl_layers]).mean(0).float().cpu())
        h = torch.cat(frames, 0) if frames else torch.zeros(1, model.config.hidden_size)
        out.append(torch.cat([h.mean(0), h.std(0, unbiased=False)]).numpy())
    return np.stack(out)
