"""
Official scorer for BLP-2027 Task 2: Multilingual Cultural QA.

A submission may contain any combination of (subtask, language) tracks: one language of
one subtask, several languages, both subtasks, or all eight tracks. The scorer evaluates
every track present in the file.

Official ranking metric: BERTScore F1 with a language-specific encoder
    Bangla   -> csebuetnlp/banglabert
    English  -> bert-base-uncased
    Hindi    -> ai4bharat/IndicBERTv2-MLM-only
    Assamese -> ai4bharat/IndicBERTv2-MLM-only

Leaderboards (each ranked by BERTScore F1)
    * one leaderboard per track: 2A-bn, 2A-en, 2A-as, 2A-hi, 2B-bn, 2B-en, 2B-as, 2B-hi
    * 2B-All: macro-average over the four languages, only for 2B submissions that cover
      all four languages

Also reported (not used for ranking): BERTScore precision/recall, ROUGE-1/2/L, METEOR,
BLEU (see scorer/lexical_metrics.py) and "script fidelity", the share of answers written
mainly in the expected script for that language. All scores are on a 0-1 scale.

Usage
    python scorer/task.py --gold-file-path data/dev_test/all.jsonl --pred-file-path task.jsonl
"""

import argparse
import json
import logging
import os
import sys
import unicodedata
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from format_checker.task import (LANGUAGE_NAMES, LANGUAGES, SUBTASK_NAMES, SUBTASKS,  # noqa: E402
                                 check_format, get_tracks, read_jsonl)
from scorer.bertscore import LANG_MODEL_CONFIG, get_scorer  # noqa: E402
from scorer.lexical_metrics import lexical_scores  # noqa: E402

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

# Assamese-specific letters (ৰ, ৱ) distinguish it from Bangla within the Bengali script.
ASSAMESE_ONLY = {"\u09F0", "\u09F1"}
EXPECTED_SCRIPT = {"bn": "BENGALI", "as": "BENGALI", "hi": "DEVANAGARI", "en": "LATIN"}


def dominant_script(text):
    counts = defaultdict(int)
    for ch in text:
        if ch.isalpha() or unicodedata.category(ch).startswith("M"):
            try:
                counts[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
    return max(counts, key=counts.get) if counts else None


def script_ok(text, lang):
    if dominant_script(text) != EXPECTED_SCRIPT[lang]:
        return False
    if lang == "bn":  # Bangla answer should not use Assamese-only letters
        return not any(ch in ASSAMESE_ONLY for ch in text)
    return True


def load_gold(path):
    gold = {}
    for row in read_jsonl(path):
        if "answer" not in row:
            raise ValueError(f"Gold file {path} line {row['_line']} has no 'answer' field.")
        gold[row["id"]] = row
    return gold


def evaluate(gold_path, pred_path, batch_size=32, device=None, model_config=None):
    # Validates fields, model rules, ids/langs against gold, and completeness of each track.
    if not check_format(pred_path, test_file=gold_path):
        raise ValueError("Invalid prediction file. See messages above.")

    gold = load_gold(gold_path)
    preds = read_jsonl(pred_path)
    tracks = get_tracks(preds)
    gold_langs = {g["lang"] for g in gold.values()}

    pred_by_track = defaultdict(dict)
    for r in preds:
        pred_by_track[(r["subtask"], r["lang"])][r["id"]] = r["answer"]

    results = defaultdict(dict)
    for lang in [l for l in LANGUAGES if any(t[1] == l for t in tracks)]:
        ids = [gid for gid, g in gold.items() if g["lang"] == lang]
        refs = [gold[i]["answer"] for i in ids]
        scorer = get_scorer(lang, batch_size=batch_size, device=device, config=model_config)
        for sub in [s for s in SUBTASKS if (s, lang) in tracks]:
            cands = [pred_by_track[(sub, lang)][i] for i in ids]
            P, R, F = scorer.score(cands, refs)
            results[sub][lang] = {
                "n": len(ids),
                "bertscore_f1": sum(F) / len(F),
                "bertscore_precision": sum(P) / len(P),
                "bertscore_recall": sum(R) / len(R),
                **lexical_scores(cands, refs),
                "script_fidelity": sum(script_ok(c, lang) for c in cands) / len(cands),
            }
        del scorer

    results = {s: {l: results[s][l] for l in LANGUAGES if l in results[s]} for s in SUBTASKS if s in results}
    out = {"tracks": [f"{s}-{l}" for s, l in tracks], "results": results,
           "encoders": {l: (model_config or LANG_MODEL_CONFIG)[l]["model"] for l in gold_langs}}
    if "2B" in results and set(results["2B"]) == gold_langs:
        out["2B_all"] = {k: sum(r[k] for r in results["2B"].values()) / len(results["2B"])
                         for k, v in next(iter(results["2B"].values())).items() if isinstance(v, float)}
    return out


def leaderboard_scores(res):
    """Flat {key: value} dict, e.g. 2A_bn_bertscore_f1, 2B_all_bertscore_f1 (for Codabench)."""
    flat = {}
    for sub, by_lang in res["results"].items():
        for lang, metrics in by_lang.items():
            for k, v in metrics.items():
                if isinstance(v, float):
                    flat[f"{sub}_{lang}_{k}"] = v
    for k, v in res.get("2B_all", {}).items():
        flat[f"2B_all_{k}"] = v
    return flat


METRIC_COLUMNS = [("bertscore_f1", "BS-F1*"), ("bertscore_precision", "BS-P"), ("bertscore_recall", "BS-R"),
                  ("rouge1", "R-1"), ("rouge2", "R-2"), ("rougeL", "R-L"), ("meteor", "METEOR"),
                  ("bleu", "BLEU"), ("script_fidelity", "Script")]


def print_report(res):
    header = f"{'Language':<10}{'N':>6}" + "".join(f"{name:>9}" for _, name in METRIC_COLUMNS)
    for sub, by_lang in res["results"].items():
        print(f"\nSubtask {sub} ({SUBTASK_NAMES[sub]})")
        print(header + "\n" + "-" * len(header))
        for lang in [l for l in LANGUAGES if l in by_lang]:
            r = by_lang[lang]
            print(f"{LANGUAGE_NAMES[lang]:<10}{r['n']:>6}" + "".join(f"{r[k]:>9.4f}" for k, _ in METRIC_COLUMNS))
        if sub == "2B":
            print("-" * len(header))
            if "2B_all" in res:
                print(f"{'All (macro)':<16}" + "".join(f"{res['2B_all'][k]:>9.4f}" for k, _ in METRIC_COLUMNS))
            else:
                print("All (macro): not ranked; requires all four languages in 2B.")
    print("\n* BERTScore F1 is the official ranking metric for each track (and for 2B-All). "
          "All scores on a 0-1 scale.\n")


def main():
    parser = argparse.ArgumentParser(description="Scorer for BLP-2027 Task 2.")
    parser.add_argument("--gold-file-path", "-g", required=True, help="Gold .jsonl file (with answers).")
    parser.add_argument("--pred-file-path", "-p", required=True, help="Prediction .jsonl file.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    parser.add_argument("--output-json", default=None, help="Write full results as JSON.")
    parser.add_argument("--leaderboard-json", default=None,
                        help="Write flat leaderboard scores (e.g. 2A_bn_bertscore_f1) as JSON.")
    args = parser.parse_args()

    try:
        res = evaluate(args.gold_file_path, args.pred_file_path, args.batch_size, args.device)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    print_report(res)
    for path, obj in [(args.output_json, res), (args.leaderboard_json, leaderboard_scores(res) if args.leaderboard_json else None)]:
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            logging.info(f"Scores written to {path}")


if __name__ == "__main__":
    main()
