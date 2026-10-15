"""Grammar / linguistic features computed from the Whisper transcript.

Feature families (each is a separate class so heavy dependencies are optional):

* basic_text_features  - lexical stats, disfluencies, repetitions (pure Python)
* SpacyFeaturizer      - syntactic complexity, POS profile, fragments,
                         subject-verb agreement heuristics
* LanguageToolFeaturizer - rule-based grammar error counts per 100 words
* PerplexityFeaturizer - GPT-2 token surprisal: ungrammatical text is "surprising"
* GECFeaturizer        - edit rate between transcript and a grammar-corrected
                         version (seq2seq GEC model). Usually the single most
                         predictive feature for grammar scoring.

All features are normalised by length where it makes sense, because response
length varies a lot and raw counts would mostly measure "how much was said".
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

import numpy as np

from ..utils import get_device, get_logger

log = get_logger()

WORD_RE = re.compile(r"[a-zA-Z']+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "hm", "ah", "eh", "mm", "mhm"}
DISCOURSE_MARKERS = ("you know", "i mean", "kind of", "sort of", "like")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# --------------------------------------------------------------------- basic
def basic_text_features(text: str) -> dict[str, float]:
    toks = tokenize(text)
    n = len(toks)
    sents = split_sentences(text)
    sent_lens = [len(tokenize(s)) for s in sents] or [0]
    content = [t for t in toks if t not in FILLERS]
    types = set(content)
    lower = " " + " ".join(toks) + " "

    immediate_reps = sum(1 for a, b in zip(toks, toks[1:]) if a == b and a not in FILLERS)
    bigrams = list(zip(toks, toks[1:]))
    bigram_reps = sum(1 for i in range(2, len(bigrams)) if bigrams[i] == bigrams[i - 2])

    return {
        "txt_n_words": n,
        "txt_n_sentences": len(sents),
        "txt_words_per_sent_mean": float(np.mean(sent_lens)),
        "txt_words_per_sent_std": float(np.std(sent_lens)),
        "txt_words_per_sent_max": float(np.max(sent_lens)),
        "txt_ttr": _safe_div(len(types), len(content)),
        "txt_root_ttr": _safe_div(len(types), np.sqrt(len(content))),  # Guiraud: length-robust
        "txt_mean_word_len": float(np.mean([len(t) for t in content])) if content else 0.0,
        "txt_long_word_ratio": _safe_div(sum(len(t) > 6 for t in content), len(content)),
        "txt_filler_rate": _safe_div(sum(t in FILLERS for t in toks), n),
        "txt_discourse_marker_rate": _safe_div(sum(lower.count(f" {m} ") for m in DISCOURSE_MARKERS), n),
        "txt_repetition_rate": _safe_div(immediate_reps, n),
        "txt_bigram_repetition_rate": _safe_div(bigram_reps, n),
        "txt_empty": float(n < 5),
    }


# --------------------------------------------------------------------- spaCy
POS_TAGS = ("NOUN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "PROPN", "NUM")
SUBORD_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass"}
SUBJ_DEPS = {"nsubj", "nsubjpass", "expl", "csubj"}
SING_PRON = {"he", "she", "it", "this", "that", "everyone", "everybody", "someone", "nobody"}
PLUR_PRON = {"i", "you", "we", "they", "these", "those"}


def _depth(tok) -> int:
    # Compare token indices, not objects: spaCy builds a new Python Token on each
    # access, so `is` never matches the root and the loop would never terminate.
    d = 0
    while tok.head.i != tok.i:
        tok = tok.head
        d += 1
    return d


class SpacyFeaturizer:
    def __init__(self, model: str = "en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(model, disable=["ner"])

    def __call__(self, text: str) -> dict[str, float]:
        doc = self.nlp(text)
        toks = [t for t in doc if not t.is_punct and not t.is_space]
        n = len(toks)
        sents = list(doc.sents)
        feats: dict[str, float] = {}

        pos_counts = Counter(t.pos_ for t in toks)
        for p in POS_TAGS:
            feats[f"sp_pos_{p.lower()}"] = _safe_div(pos_counts[p], n)

        subord = sum(t.dep_ in SUBORD_DEPS for t in toks)
        coord = sum(t.dep_ == "conj" for t in toks)
        dep_dist = [abs(t.i - t.head.i) for t in toks if t.head.i != t.i]
        depths = [max((_depth(t) for t in s), default=0) for s in sents]

        fragments = 0
        agreement_errors = 0
        det_noun_errors = 0
        for s in sents:
            has_finite = any("Fin" in t.morph.get("VerbForm") for t in s)
            has_subj = any(t.dep_ in SUBJ_DEPS for t in s)
            fragments += int(not (has_finite and has_subj))
            for t in s:
                # Subject-verb agreement heuristic on present-tense verbs
                if t.dep_ == "nsubj" and t.head.tag_ in ("VBZ", "VBP"):
                    w = t.lower_
                    subj_sing = t.tag_ in ("NN", "NNP") or w in SING_PRON
                    subj_plur = t.tag_ in ("NNS", "NNPS") or w in PLUR_PRON
                    if (subj_sing and t.head.tag_ == "VBP") or (subj_plur and t.head.tag_ == "VBZ"):
                        agreement_errors += 1
                # "a/an" + plural noun ("a books")
                if t.lower_ in ("a", "an") and t.dep_ == "det" and t.head.tag_ == "NNS":
                    det_noun_errors += 1

        tenses = Counter(m for t in toks for m in t.morph.get("Tense"))
        n_s = max(len(sents), 1)
        feats.update({
            "sp_subord_per_sent": subord / n_s,
            "sp_coord_per_sent": coord / n_s,
            "sp_subord_ratio": _safe_div(subord, n),
            "sp_mean_dep_dist": float(np.mean(dep_dist)) if dep_dist else 0.0,
            "sp_tree_depth_mean": float(np.mean(depths)) if depths else 0.0,
            "sp_tree_depth_max": float(np.max(depths)) if depths else 0.0,
            "sp_fragment_ratio": fragments / n_s,
            "sp_agreement_err_rate": _safe_div(agreement_errors, n),
            "sp_det_noun_err_rate": _safe_div(det_noun_errors, n),
            "sp_past_ratio": _safe_div(tenses["Past"], sum(tenses.values())),
            "sp_verbs_per_sent": sum(t.pos_ in ("VERB", "AUX") for t in toks) / n_s,
        })
        return feats


# -------------------------------------------------------------- LanguageTool
# ASR punctuation/casing is machine-generated, so those rule categories are noise.
LT_IGNORE_CATEGORIES = {"TYPOGRAPHY", "CASING", "PUNCTUATION", "TYPOS"}
LT_CATEGORIES = ("GRAMMAR", "CONFUSED_WORDS", "COLLOCATIONS", "REDUNDANCY", "STYLE", "MISC", "SEMANTICS")


class LanguageToolFeaturizer:
    def __init__(self):
        import language_tool_python

        self.tool = language_tool_python.LanguageTool("en-US")

    def __call__(self, text: str) -> dict[str, float]:
        n = max(len(tokenize(text)), 1)
        matches = [m for m in self.tool.check(text) if m.category not in LT_IGNORE_CATEGORIES]
        cats = Counter(m.category for m in matches)
        feats = {f"lt_{c.lower()}_per100": 100 * cats[c] / n for c in LT_CATEGORIES}
        feats["lt_total_per100"] = 100 * len(matches) / n
        return feats


# ---------------------------------------------------------------- perplexity
class PerplexityFeaturizer:
    """Token-level surprisal under a causal LM.

    Text is lower-cased and stripped of punctuation first, so the LM judges
    word order and morphology rather than ASR punctuation choices.
    """

    def __init__(self, model: str = "gpt2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device).eval()
        self.max_len = min(getattr(self.model.config, "n_positions", 1024), 1024)

    def __call__(self, text: str) -> dict[str, float]:
        clean = " ".join(tokenize(text))
        ids = self.tok(self.tok.bos_token + clean, return_tensors="pt").input_ids[:, : self.max_len].to(self.device)
        if ids.shape[1] < 3:
            return {k: 0.0 for k in ("lm_nll_mean", "lm_nll_std", "lm_nll_p90", "lm_nll_max", "lm_high_surprisal_ratio")}
        with self.torch.no_grad():
            logits = self.model(ids).logits[0, :-1]
        nll = self.torch.nn.functional.cross_entropy(logits.float(), ids[0, 1:], reduction="none").cpu().numpy()
        return {
            "lm_nll_mean": float(nll.mean()),
            "lm_nll_std": float(nll.std()),
            "lm_nll_p90": float(np.percentile(nll, 90)),
            "lm_nll_max": float(nll.max()),
            "lm_high_surprisal_ratio": float((nll > 10).mean()),
        }


# ----------------------------------------------------------------------- GEC
def edit_ops(src: list[str], tgt: list[str]) -> dict[str, int]:
    ops = Counter()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=src, b=tgt, autojunk=False).get_opcodes():
        if tag == "replace":
            ops["sub"] += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            ops["del"] += i2 - i1
        elif tag == "insert":
            ops["ins"] += j2 - j1
    return ops


class GECFeaturizer:
    """Grammatical-error-correction edit rate.

    Idea: a proficient speaker's sentence needs few edits to become
    grammatical. We correct each sentence with a seq2seq GEC model and count
    word-level insertions / deletions / substitutions (fillers removed first so
    disfluencies are not counted as grammar errors).
    """

    def __init__(self, model: str = "vennify/t5-base-grammar-correction", batch_size: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device).eval()
        self.batch_size = batch_size
        self.prefix = "grammar: " if "vennify" in model else ""

    def correct(self, sentences: list[str]) -> list[str]:
        out = []
        for i in range(0, len(sentences), self.batch_size):
            batch = [self.prefix + s for s in sentences[i : i + self.batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=128, num_beams=4)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out

    def __call__(self, text: str) -> dict[str, float]:
        sents = [" ".join(t for t in tokenize(s) if t not in FILLERS) for s in split_sentences(text)]
        sents = [s for s in sents if s]
        if not sents:
            return {"gec_edit_rate": 0.0, "gec_sub_rate": 0.0, "gec_ins_rate": 0.0,
                    "gec_del_rate": 0.0, "gec_changed_sent_ratio": 0.0}
        corrected = self.correct(sents)
        ops, changed, n = Counter(), 0, 0
        for s, c in zip(sents, corrected):
            src, tgt = tokenize(s), tokenize(c)
            e = edit_ops(src, tgt)
            ops.update(e)
            changed += int(sum(e.values()) > 0)
            n += len(src)
        return {
            "gec_edit_rate": _safe_div(sum(ops.values()), n),
            "gec_sub_rate": _safe_div(ops["sub"], n),
            "gec_ins_rate": _safe_div(ops["ins"], n),
            "gec_del_rate": _safe_div(ops["del"], n),
            "gec_changed_sent_ratio": changed / len(sents),
        }
