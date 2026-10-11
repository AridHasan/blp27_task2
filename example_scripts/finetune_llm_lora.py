"""
Starter script: LoRA fine-tuning of an instruction-tuned decoder LLM
(default: Qwen/Qwen2.5-1.5B-Instruct) for BLP-2027 Task 2, then answer generation.

Run any combination of subtasks and languages; all predictions go into ONE task.jsonl.
  Subtask 2A (monolingual): one LoRA adapter per requested language.
  Subtask 2B (multilingual): one LoRA adapter trained on ALL four languages, used to
             answer the requested languages.
Loss is computed on answer tokens only (prompt tokens are masked).

Usage
    python example_scripts/finetune_llm_lora.py --subtasks 2A --langs hi        # one track
    python example_scripts/finetune_llm_lora.py --subtasks 2A 2B --langs bn as  # four tracks
    python example_scripts/finetune_llm_lora.py                                 # all 8 tracks
    python example_scripts/finetune_llm_lora.py --zero-shot                     # no training

Other models worth trying: meta-llama/Llama-3.2-3B-Instruct, google/gemma-3-4b-it,
CohereForAI/aya-expanse-8b, sarvamai/sarvam-1 (check each model's licence).
"""

import argparse
import json
import math
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from format_checker.task import LANGUAGE_NAMES, LANGUAGES, read_jsonl  # noqa: E402

SYSTEM = ("You are a helpful assistant with deep knowledge of South Asian culture. "
          "Answer the user's question concisely and accurately, in the same language as the question.")


def messages_for(row):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[{LANGUAGE_NAMES[row['lang']]}] {row['question']}"}]


def load_rows(path, langs):
    return [{k: v for k, v in r.items() if k != "_line"} for r in read_jsonl(path) if r["lang"] in langs]


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    return tokenizer, model, dtype


def train(args, tokenizer, model, dtype, train_langs, run_dir):
    def encode(row):
        prompt = tokenizer.apply_chat_template(messages_for(row), tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(row["answer"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:args.max_length]
        labels = ([-100] * len(p_ids) + a_ids)[:args.max_length]
        return {"input_ids": ids, "labels": labels}

    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([f["input_ids"] + [pad] * (n - len(f["input_ids"])) for f in features]),
            "labels": torch.tensor([f["labels"] + [-100] * (n - len(f["labels"])) for f in features]),
            "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (n - len(f["input_ids"]))
                                            for f in features]),
        }

    train_rows = load_rows(args.train_file, train_langs)
    dev_rows = load_rows(args.dev_file, train_langs)
    cols = list(train_rows[0].keys())
    train_ds = Dataset.from_list(train_rows).map(encode, remove_columns=cols)
    dev_ds = Dataset.from_list(dev_rows).map(encode, remove_columns=cols)

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    total_steps = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)) * args.epochs

    trainer = Trainer(
        model=model, data_collator=collate, train_dataset=train_ds, eval_dataset=dev_ds,
        args=TrainingArguments(
            output_dir=run_dir, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, num_train_epochs=args.epochs, lr_scheduler_type="cosine",
            warmup_steps=int(0.03 * total_steps), eval_strategy="epoch", save_strategy="epoch",
            save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            logging_steps=20, bf16=dtype == torch.bfloat16, report_to="none", seed=args.seed,
            remove_unused_columns=False),
    )
    trainer.train()
    model.save_pretrained(os.path.join(run_dir, "adapter"))
    return model


@torch.no_grad()
def generate(args, tokenizer, model, rows):
    model.eval()
    tokenizer.padding_side = "left"
    answers = {}
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        prompts = [tokenizer.apply_chat_template(messages_for(r), tokenize=False, add_generation_prompt=True)
                   for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        for r, ans in zip(batch, tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                        skip_special_tokens=True)):
            answers[r["id"]] = ans.strip()
    tokenizer.padding_side = "right"
    return answers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subtasks", nargs="+", default=["2A", "2B"], choices=["2A", "2B"])
    p.add_argument("--langs", nargs="+", default=LANGUAGES, choices=LANGUAGES,
                   help="Languages to answer (default: all four).")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-file", default="data/train/all.jsonl")
    p.add_argument("--dev-file", default="data/dev/all.jsonl")
    p.add_argument("--test-file", default="data/dev_test/all.jsonl")
    p.add_argument("--output-dir", default="runs/llm_lora")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--zero-shot", action="store_true", help="Skip training; prompt the base model.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--team-model-name", default=None, help="Prefix for the 'model' field.")
    args = p.parse_args()

    langs = [l for l in LANGUAGES if l in args.langs]
    tag = args.team_model_name or f"{os.path.basename(args.model.rstrip('/'))}-{'zeroshot' if args.zero_shot else 'lora'}"

    # (subtask, train languages, languages to answer, run dir, model name)
    jobs = []
    if "2A" in args.subtasks:
        jobs += [("2A", [l], [l], os.path.join(args.output_dir, f"2A_{l}"), f"{tag}-2A-{l}") for l in langs]
    if "2B" in args.subtasks:
        jobs.append(("2B", LANGUAGES, langs, os.path.join(args.output_dir, "2B_all"), f"{tag}-2B"))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "task.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for subtask, train_langs, predict_langs, run_dir, model_name in jobs:
            print(f"=== Subtask {subtask}: train on {train_langs}, answer {predict_langs} ===")
            set_seed(args.seed)
            tokenizer, model, dtype = load_model(args)  # fresh base model for each job
            if not args.zero_shot:
                model = train(args, tokenizer, model, dtype, train_langs, run_dir)
            test_rows = load_rows(args.test_file, predict_langs)
            answers = generate(args, tokenizer, model, test_rows)
            for r in test_rows:
                f.write(json.dumps({"id": r["id"], "subtask": subtask, "lang": r["lang"],
                                    "answer": answers[r["id"]], "model": model_name},
                                   ensure_ascii=False) + "\n")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
