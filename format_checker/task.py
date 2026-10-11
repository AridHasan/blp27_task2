"""
Format checker for BLP-2027 Task 2: Multilingual Cultural QA.

A submission is ONE JSON Lines file (`task.jsonl`). Each line answers one question
for one subtask:

    {"id": "bn_bd_cf92ec1e...", "subtask": "2A", "lang": "bn", "answer": "...", "model": "my-bn-model"}

Participation is flexible. A file may contain any combination of
(subtask, language) "tracks", for example:
    * one language in one subtask          (2A-bn)
    * several languages in one subtask     (2A-bn, 2A-hi)
    * both subtasks                        (2A-as, 2B-as, 2B-en)
    * everything                           (2A-{bn,en,as,hi}, 2B-{bn,en,as,hi})
The same question id may appear once per subtask.

Rules
-----
* Required fields: id, subtask, lang, answer, model (all strings).
* subtask in {2A, 2B}; lang in {bn, en, as, hi}.
* (subtask, id) pairs are unique.
* 2A (monolingual): one model name per language (a separate model per language).
* 2B (multilingual): one model name across ALL 2B rows (a single multilingual model).
* With --test-file: every id must exist in the test file with the same lang, and every
  track you submit must answer ALL questions of that language.

Usage
-----
    python format_checker/task.py -p task.jsonl
    python format_checker/task.py -p task.jsonl --test-file data/test/all.jsonl
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

LANGUAGES = ["bn", "en", "as", "hi"]
LANGUAGE_NAMES = {"bn": "Bangla", "en": "English", "as": "Assamese", "hi": "Hindi"}
SUBTASKS = ["2A", "2B"]
SUBTASK_NAMES = {"2A": "Monolingual QA", "2B": "Multilingual QA"}
REQUIRED_FIELDS = ["id", "subtask", "lang", "answer", "model"]


def read_jsonl(file_path):
    """Read a JSONL file. Raises ValueError with the offending line number."""
    rows = []
    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON ({e.msg}).")
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no}: each line must be a JSON object.")
            obj["_line"] = line_no
            rows.append(obj)
    return rows


def get_tracks(rows):
    """Sorted list of (subtask, lang) tracks present in validated rows."""
    tracks = {(r["subtask"], r["lang"]) for r in rows}
    return sorted(tracks, key=lambda t: (SUBTASKS.index(t[0]), LANGUAGES.index(t[1])))


def format_tracks(tracks):
    by_sub = defaultdict(list)
    for s, l in tracks:
        by_sub[s].append(LANGUAGE_NAMES[l])
    return "; ".join(f"{s} ({SUBTASK_NAMES[s]}): {', '.join(ls)}" for s, ls in by_sub.items())


def check_format(file_path, test_file=None):
    """Return True if `file_path` is a valid submission. If `test_file` is given,
    also check ids/languages against it and that every submitted track is complete."""
    logging.info(f"Checking format of {file_path}")

    if not file_path.endswith(".jsonl"):
        logging.warning("Submission file should have a .jsonl extension (expected name: task.jsonl).")

    try:
        rows = read_jsonl(file_path)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        logging.error(f"Could not read {file_path}: {e}")
        return False
    if not rows:
        logging.error("Submission file is empty.")
        return False

    # ---- row-level checks ----------------------------------------------------------
    ok = True
    seen = set()
    n_empty = Counter()
    for row in rows:
        ln = row["_line"]
        missing = [k for k in REQUIRED_FIELDS if k not in row]
        if missing:
            logging.error(f"Line {ln}: missing field(s) {missing}.")
            ok = False
            continue
        bad_type = [k for k in REQUIRED_FIELDS if not isinstance(row[k], str)]
        if bad_type:
            logging.error(f"Line {ln}: field(s) {bad_type} must be strings.")
            ok = False
            continue
        if row["subtask"] not in SUBTASKS:
            logging.error(f"Line {ln}: invalid subtask '{row['subtask']}'. Must be one of {SUBTASKS}.")
            ok = False
        if row["lang"] not in LANGUAGES:
            logging.error(f"Line {ln}: invalid lang '{row['lang']}'. Must be one of {LANGUAGES}.")
            ok = False
        if not row["id"].strip():
            logging.error(f"Line {ln}: empty id.")
            ok = False
        if not row["model"].strip():
            logging.error(f"Line {ln}: empty model name.")
            ok = False
        key = (row["subtask"], row["id"])
        if key in seen:
            logging.error(f"Line {ln}: duplicate id '{row['id']}' for subtask {row['subtask']}.")
            ok = False
        seen.add(key)
        if not row["answer"].strip():
            n_empty[(row["subtask"], row["lang"])] += 1

    if not ok:
        logging.error(f"{file_path}: format check FAILED.")
        return False

    # ---- subtask-level model rules -----------------------------------------------
    models = defaultdict(set)
    for r in rows:
        models[(r["subtask"], r["lang"])].add(r["model"])
    for (sub, lang), names in models.items():
        if sub == "2A" and len(names) > 1:
            logging.error(f"2A-{lang}: {len(names)} different model names {sorted(names)}; "
                          "use one model per language.")
            ok = False
    models_2b = {r["model"] for r in rows if r["subtask"] == "2B"}
    if len(models_2b) > 1:
        logging.error(f"2B requires a single multilingual model for all its languages, but found "
                      f"{len(models_2b)} model names: {sorted(models_2b)}.")
        ok = False

    tracks = get_tracks(rows)

    # ---- optional checks against the test/gold file --------------------------------
    if test_file and ok:
        try:
            test_rows = read_jsonl(test_file)
        except (OSError, ValueError) as e:
            logging.error(f"Could not read test file {test_file}: {e}")
            return False
        test_lang = {t["id"]: t["lang"] for t in test_rows}
        ids_by_lang = defaultdict(set)
        for i, l in test_lang.items():
            ids_by_lang[l].add(i)

        for r in rows:
            if r["id"] not in test_lang:
                logging.error(f"Line {r['_line']}: id '{r['id']}' is not in the test file.")
                ok = False
            elif r["lang"] != test_lang[r["id"]]:
                logging.error(f"Line {r['_line']}: id '{r['id']}' has lang '{r['lang']}', "
                              f"expected '{test_lang[r['id']]}'.")
                ok = False
        if ok:
            for sub, lang in tracks:
                answered = {r["id"] for r in rows if r["subtask"] == sub and r["lang"] == lang}
                missing = ids_by_lang[lang] - answered
                if missing:
                    logging.error(f"{sub}-{lang}: {len(missing)} of {len(ids_by_lang[lang])} "
                                  f"{LANGUAGE_NAMES[lang]} questions are unanswered "
                                  f"(e.g. {sorted(missing)[:3]}). Each submitted track must be complete.")
                    ok = False

    for (sub, lang), n in sorted(n_empty.items()):
        logging.warning(f"{sub}-{lang}: {n} empty answer(s); they will receive a score of 0.")

    if ok:
        logging.info(f"{file_path}: format check PASSED.")
        logging.info(f"Tracks found -> {format_tracks(tracks)}")
    else:
        logging.error(f"{file_path}: format check FAILED.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Format checker for BLP-2027 Task 2.")
    parser.add_argument("--pred-files-path", "-p", nargs="+", required=True,
                        help="One or more space-separated prediction files (.jsonl).")
    parser.add_argument("--test-file", "-t", default=None,
                        help="Optional test file (.jsonl with id and lang) to check id coverage.")
    args = parser.parse_args()

    results = [check_format(p, args.test_file) for p in args.pred_files_path]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
