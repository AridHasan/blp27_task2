"""
Lexical overlap metrics for BLP-2027 Task 2: ROUGE-1/2/L, METEOR and BLEU.
These are reported alongside BERTScore; they are NOT used for official ranking.

All metrics share one Unicode-aware tokenizer. This matters for Indic scripts:
  * rouge_score's default tokenizer deletes every non-Latin character, so it gives
    ROUGE = 0 for identical Bangla/Hindi/Assamese strings;
  * Python's built-in \\w does not match vowel signs (Unicode category M), so it
    splits Indic words in the middle.
Tokens are maximal runs of letters, combining marks and digits; every other non-space
character (e.g. "।", "?", ",") is its own token. Text is lower-cased (affects English only).

Score conventions (all reported on a 0-1 scale):
  * ROUGE-1/2/L : F-measure, averaged over items, no stemming (stemmers are English-only).
  * METEOR      : NLTK implementation, averaged over items. WordNet synonym and Porter
                  stem matching only apply to English; other languages use exact matches.
  * BLEU        : corpus-level BLEU-4 (sacrebleu) on the pre-tokenized text, divided by 100.
  Empty predictions score 0 on every item-level metric.
"""

import regex
import sacrebleu
from rouge_score import rouge_scorer

_TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\p{L}\p{M}\p{N}\s]")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class _RougeTokenizer:
    """Adapter so rouge_score uses our tokenizer instead of its Latin-only default."""

    def tokenize(self, text):
        return tokenize(text)


def _get_meteor():
    import nltk
    from nltk.translate.meteor_score import meteor_score
    for res in ("corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res.split("/")[1], quiet=True)
    return meteor_score


def lexical_scores(candidates, references):
    """Return a dict of corpus-level averages for one language."""
    meteor = _get_meteor()
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False,
                                     tokenizer=_RougeTokenizer())

    r1 = r2 = rl = met = 0.0
    cand_tok, ref_tok = [], []
    for c, r in zip(candidates, references):
        ct, rt = tokenize(c), tokenize(r)
        cand_tok.append(" ".join(ct))
        ref_tok.append(" ".join(rt))
        if not ct or not rt:
            continue
        s = rouge.score(r, c)  # (target, prediction)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
        met += meteor([rt], ct)

    n = len(candidates)
    bleu = sacrebleu.corpus_bleu(cand_tok, [ref_tok], tokenize="none").score / 100.0
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n, "meteor": met / n, "bleu": bleu}
