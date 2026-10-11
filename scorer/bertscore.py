"""
Self-contained BERTScore (Zhang et al., 2020) with language-specific encoders.

Why not call `bert_score.score()` directly?
  * BanglaBERT and IndicBERT are not in bert_score's built-in model table, and
    IndicBERTv2's tokenizer config does not set `model_max_length`, so long answers
    would not be truncated safely. Here the maximum length and layer are explicit.
  * Every setting that affects the leaderboard is pinned in LANG_MODEL_CONFIG below.

The computation follows the reference implementation (bert_score v0.3.x) with
idf=False and no baseline rescaling:
  * contextual embeddings from a fixed hidden layer, L2-normalised;
  * greedy matching by cosine similarity over all non-padding tokens;
  * [CLS]/[SEP] get weight 0 when averaging;
  * P = mean over candidate tokens of max-sim to reference,
    R = mean over reference tokens of max-sim to candidate,
    F = 2PR / (P + R).
An empty candidate (or reference) receives P = R = F = 0.
"""

import logging

import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Official scorer configuration. Do NOT change for leaderboard submissions.
# `layer` is the index into `hidden_states` (0 = embeddings, 1..N = encoder layers).
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


class BERTScorer:
    def __init__(self, model_name, layer, max_length=512, batch_size=32, device=None,
                 normalize_bangla=False):
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = _get_bangla_normalizer() if normalize_bangla else None

        logging.info(f"Loading BERTScore encoder {model_name} (layer {layer}) on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        n_layers = self.model.config.num_hidden_layers
        if not 0 <= layer <= n_layers:
            raise ValueError(f"layer={layer} out of range for {model_name} ({n_layers} layers).")

        # Some tokenizers (e.g. IndicBERTv2) leave model_max_length unset (a huge sentinel).
        tok_max = self.tokenizer.model_max_length
        model_max = getattr(self.model.config, "max_position_embeddings", max_length)
        self.max_length = min(max_length, model_max, tok_max if tok_max < 100_000 else max_length)
        # Like bert_score, only [CLS] and [SEP] get zero weight ([UNK] etc. still count).
        self.special_ids = {i for i in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id) if i is not None}

    def _prep(self, texts):
        texts = [t.strip() for t in texts]
        if self.normalize is not None:
            texts = [self.normalize(t) if t else t for t in texts]
        return texts

    @torch.no_grad()
    def _embed(self, texts):
        """Return list of (embeddings [T, H] normalised, weights [T]) per text (T includes specials)."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
            hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for j in range(len(batch)):
                mask = enc["attention_mask"][j].bool()
                ids = enc["input_ids"][j][mask].tolist()
                weights = torch.tensor([0.0 if t in self.special_ids else 1.0 for t in ids],
                                       device=self.device)
                out.append((hidden[j][mask].float(), weights))
        return out

    def score(self, candidates, references):
        """Return (P, R, F) lists of floats, one value per (candidate, reference) pair."""
        if len(candidates) != len(references):
            raise ValueError("candidates and references must have the same length.")
        cands, refs = self._prep(candidates), self._prep(references)

        cand_emb = self._embed(cands)
        ref_emb = self._embed(refs)

        P, R, F = [], [], []
        for (c_e, c_w), (r_e, r_w), c_txt, r_txt in zip(cand_emb, ref_emb, cands, refs):
            if not c_txt or not r_txt or c_w.sum() == 0 or r_w.sum() == 0:
                P.append(0.0); R.append(0.0); F.append(0.0)
                continue
            sim = c_e @ r_e.T                      # [Tc, Tr]
            p = (sim.max(dim=1).values * c_w).sum() / c_w.sum()
            r = (sim.max(dim=0).values * r_w).sum() / r_w.sum()
            f = 2 * p * r / (p + r) if (p + r) > 0 else torch.tensor(0.0)
            P.append(p.item()); R.append(r.item()); F.append(f.item())
        return P, R, F


def get_scorer(lang, batch_size=32, device=None, config=None):
    cfg = (config or LANG_MODEL_CONFIG)[lang]
    return BERTScorer(cfg["model"], cfg["layer"], cfg.get("max_length", 512), batch_size, device,
                      cfg.get("normalize_bangla", False))
