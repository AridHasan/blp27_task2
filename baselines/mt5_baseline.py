"""
Official baseline for BLP-2027 Task 2: fine-tuned mT5 (google/mt5-base).

Run any combination of subtasks and languages; all predictions go into ONE task.jsonl.
  Subtask 2A (monolingual): one mT5 model per requested language.
  Subtask 2B (multilingual): one mT5 model trained on ALL four languages (as the task
             defines), used to answer the requested languages.

The model input is "Answer in <Language>: <question>", so the 2B model knows which
language to answer in.

Usage
    python baselines/mt5_baseline.py --subtasks 2A --langs bn                 # one track
    python baselines/mt5_baseline.py --subtasks 2A --langs bn hi              # two languages
    python baselines/mt5_baseline.py --subtasks 2A 2B --langs as              # both subtasks
    python baselines/mt5_baseline.py                                          # all 8 tracks

    python scorer/task.py -g data/dev_test/all.jsonl -p runs/mt5_baseline/task.jsonl

Default hyper-parameters are the ones used for the reported baseline results.
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from format_checker.task import LANGUAGE_NAMES, LANGUAGES, read_jsonl  # noqa: E402


def build_input(row):
    return f"Answer in {LANGUAGE_NAMES[row['lang']]}: {row['question']}"


def load_rows(path, langs):
    return [{k: v for k, v in r.items() if k != "_line"} for r in read_jsonl(path) if r["lang"] in langs]


def train_and_predict(args, train_langs, predict_langs, run_dir):
    """Train one mT5 model on `train_langs`; return its answers for the test rows of `predict_langs`."""
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess(batch):
        rows = [dict(zip(batch, vals)) for vals in zip(*batch.values())]
        enc = tokenizer([build_input(r) for r in rows], max_length=args.max_source_length, truncation=True)
        enc["labels"] = tokenizer(text_target=batch["answer"], max_length=args.max_target_length,
                                  truncation=True)["input_ids"]
        return enc

    train_rows = load_rows(args.train_file, train_langs)
    dev_rows = load_rows(args.dev_file, train_langs)
    test_rows = load_rows(args.test_file, predict_langs)
    cols = list(train_rows[0].keys())
    train_ds = Dataset.from_list(train_rows).map(preprocess, batched=True, remove_columns=cols)
    dev_ds = Dataset.from_list(dev_rows).map(preprocess, batched=True, remove_columns=cols)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # mT5 is unstable in fp16
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(os.path.join(run_dir, "best"))

    model = trainer.model.eval()
    answers = {}
    with torch.no_grad():
        for i in range(0, len(test_rows), args.batch_size):
            batch = test_rows[i:i + args.batch_size]
            enc = tokenizer([build_input(r) for r in batch], max_length=args.max_source_length,
                            truncation=True, padding=True, return_tensors="pt").to(model.device)
            gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=args.max_target_length, num_beams=args.num_beams,
                                 no_repeat_ngram_size=3)
            for r, ans in zip(batch, tokenizer.batch_decode(gen, skip_special_tokens=True)):
                answers[r["id"]] = ans.strip()
    return test_rows, answers


def main():
    p = argparse.ArgumentParser(description="mT5 baseline for BLP-2027 Task 2.")
    p.add_argument("--subtasks", nargs="+", default=["2A", "2B"], choices=["2A", "2B"])
    p.add_argument("--langs", nargs="+", default=LANGUAGES, choices=LANGUAGES,
                   help="Languages to answer (default: all four).")
    p.add_argument("--model", default="google/mt5-base")
    p.add_argument("--train-file", default="data/train/all.jsonl")
    p.add_argument("--dev-file", default="data/dev/all.jsonl")
    p.add_argument("--test-file", default="data/dev_test/all.jsonl")
    p.add_argument("--output-dir", default="runs/mt5_baseline")
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-target-length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    langs = [l for l in LANGUAGES if l in args.langs]
    model_tag = os.path.basename(args.model.rstrip("/"))

    # (subtask, train languages, languages to answer, run dir, model name)
    jobs = []
    if "2A" in args.subtasks:
        jobs += [("2A", [l], [l], os.path.join(args.output_dir, f"2A_{l}"), f"{model_tag}-2A-{l}") for l in langs]
    if "2B" in args.subtasks:
        jobs.append(("2B", LANGUAGES, langs, os.path.join(args.output_dir, "2B_all"), f"{model_tag}-2B"))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "task.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for subtask, train_langs, predict_langs, run_dir, model_name in jobs:
            print(f"=== Subtask {subtask}: train on {train_langs}, answer {predict_langs} ===")
            test_rows, answers = train_and_predict(args, train_langs, predict_langs, run_dir)
            for r in test_rows:
                f.write(json.dumps({"id": r["id"], "subtask": subtask, "lang": r["lang"],
                                    "answer": answers[r["id"]], "model": model_name},
                                   ensure_ascii=False) + "\n")
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
