"""
BERTScore for BLP-2027 Task 2, computed with the official `bert_score` library
(Zhang et al., 2020; https://github.com/Tiiiger/bert_score) and a language-specific encoder.

Settings (fixed for the leaderboard): idf=False, no baseline rescaling, embeddings from the
layer given below, inputs truncated to 512 tokens.

Two small safeguards around the library, neither of which changes a score:
  * Truncation: bert_score truncates to `tokenizer.model_max_length`. IndicBERTv2's tokenizer
    leaves this unset (a huge sentinel value), so answers longer than 512 tokens would crash
    Hindi/Assamese scoring. We set it to the model's real limit (512).
  * Empty strings: an empty answer or reference gets P = R = F = 0 and is not sent to the
    library (some bert_score/transformers versions fail on empty input).

Bangla text is normalised with the csebuetnlp normalizer, as recommended for BanglaBERT.
"""

import logging
import os

import torch
from bert_score import BERTScorer as _LibBERTScorer

# ---------------------------------------------------------------------------
# Official scorer configuration. Do NOT change for leaderboard submissions.
# `layer` is bert_score's `num_layers`: embeddings are taken from that encoder layer.
# ---------------------------------------------------------------------------
LANG_MODEL_CONFIG = {
    "bn": {"model": "csebuetnlp/banglabert", "layer": 9, "max_length": 512, "normalize_bangla": True},
    "en": {"model": "bert-base-uncased", "layer": 9, "max_length": 512, "normalize_bangla": False},
    "hi": {"model": "ai4bharat/IndicBERTv2-MLM-only", "layer": 9, "max_length": 512, "normalize_bangla": False},
    "as": {"model": "ai4bharat/IndicBERTv2-MLM-only", "layer": 9, "max_length": 512, "normalize_bangla": False},
}


def _get_bangla_normalizer():
    try:
        from normalizer import normalize  # https://github.com/csebuetnlp/normalizer
    except ImportError as e:
        raise ImportError(
            "BanglaBERT expects text normalised with the csebuetnlp normalizer. Install it with:\n"
            "    pip install git+https://github.com/csebuetnlp/normalizer"
        ) from e
    return normalize


def resolve_model_path(model_id):
    """Use a local copy if BLP27_MODEL_DIR is set and contains <model_id>/ (e.g. for offline
    clusters); otherwise return the Hugging Face id unchanged. The weights are the same either way."""
    root = os.environ.get("BLP27_MODEL_DIR")
    if root:
        local = os.path.join(root, model_id)
        if os.path.isfile(os.path.join(local, "config.json")):
            return local
        logging.warning(f"BLP27_MODEL_DIR is set but {local}/config.json was not found; "
                        f"falling back to the Hugging Face id '{model_id}'.")
    return model_id


class BERTScorer:
    """Thin wrapper around bert_score.BERTScorer for one language's encoder."""

    def __init__(self, model_name, layer, max_length=512, batch_size=32, device=None,
                 normalize_bangla=False):
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = _get_bangla_normalizer() if normalize_bangla else None

        logging.info(f"Loading BERTScore encoder {model_name} (layer {layer}) on {self.device}")
        self._scorer = _LibBERTScorer(
            model_type=model_name,
            num_layers=layer,
            batch_size=batch_size,
            device=self.device,
            idf=False,
            rescale_with_baseline=False,
        )

        # Safeguard 1: make sure long inputs are truncated to what the model can take.
        tokenizer = self._scorer._tokenizer
        model_limit = getattr(self._scorer._model.config, "max_position_embeddings", max_length)
        limit = min(max_length, model_limit)
        if tokenizer.model_max_length is None or tokenizer.model_max_length > limit:
            tokenizer.model_max_length = limit

    def _prep(self, texts):
        texts = [t.strip() for t in texts]
        if self.normalize is not None:
            texts = [self.normalize(t) if t else t for t in texts]
        return texts

    def score(self, candidates, references, label=""):
        """Return (P, R, F) lists of floats, one value per (candidate, reference) pair."""
        if len(candidates) != len(references):
            raise ValueError("candidates and references must have the same length.")
        cands, refs = self._prep(candidates), self._prep(references)

        # Safeguard 2: empty strings score 0 and are not sent to the library.
        keep = [i for i, (c, r) in enumerate(zip(cands, refs)) if c and r]
        P = [0.0] * len(cands)
        R = [0.0] * len(cands)
        F = [0.0] * len(cands)
        if keep:
            if label:
                n_empty = len(cands) - len(keep)
                logging.info(f"  {label}: computing BERTScore for {len(keep)} pairs"
                             + (f" ({n_empty} empty, scored 0)" if n_empty else ""))
            p, r, f = self._scorer.score([cands[i] for i in keep], [refs[i] for i in keep],
                                         verbose=bool(label), batch_size=self.batch_size)
            for j, i in enumerate(keep):
                P[i], R[i], F[i] = p[j].item(), r[j].item(), f[j].item()
        return P, R, F


def get_scorer(lang, batch_size=32, device=None, config=None):
    cfg = (config or LANG_MODEL_CONFIG)[lang]
    return BERTScorer(resolve_model_path(cfg["model"]), cfg["layer"], cfg.get("max_length", 512),
                      batch_size, device, cfg.get("normalize_bangla", False))