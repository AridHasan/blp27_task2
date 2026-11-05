# [Multilingual Cultural QA Shared Task](https://TBA) at [BLP Workshop @NAACL 2027](https://blp-workshop.github.io/)

## Task Website

### [Please visit this site to get updated information on this task.](https://multilingual-qa.github.io/)

[https://multilingual-qa.github.io/](https://multilingual-qa.github.io/)

## Objective

The **Multilingual Cultural QA shared task** (BLP-2027 Task 2) targets culturally and regionally grounded, open-ended question answering in **Bangla, English, Assamese, and Hindi**. Building on the [MultiNativQA](https://huggingface.co/datasets/QCRI/MultiNativQA) dataset ([Hasan et al., 2025](https://aclanthology.org/2025.findings-acl.770/)), whose questions come from real native-speaker queries about everyday life in Dhaka, Kolkata, Assam, Delhi and Doha, participants build systems that generate an appropriate answer in the same language as the question. The task asks two complementary questions: how well can language-specific models answer culturally grounded questions, and can a single multilingual model serve all four languages, including low-resource Assamese? Please see the [Task Description](#task-description) below.

**Table of contents:**

- [Important Dates](#important-dates)
- [Recent Updates](#recent-updates)
- [Contents of the Directory](#contents-of-the-directory)
- [Task Description](#task-description)
- [Participation](#participation)
- [Dataset](#dataset)
- [Example Fine-tuning Scripts](#example-fine-tuning-scripts)
- [Scorer and Official Evaluation Metrics](#scorer-and-official-evaluation-metrics)
- [Baselines](#baselines)
- [Format checker](#format-checker)
- [Submission](#submission)
- [Citation](#citation)
- [Organizers](#organizers)

## Important Dates

- **October 15, 2026:** Registration on EvalAI and beginning of the development cycle
- **January 15, 2027:** Beginning of the evaluation cycle (test set release and run submission)
- **January 24, 2027:** Registration Ends
- **January 31, 2027:** End of the evaluation cycle
- **February 03, 2027:** Publish rank list and share paper submission details
- **February 15, 2027:** Paper Submission Deadline (Shared Task System Papers Due)
- **March 26, 2027:** Notification of acceptance
- **April 16, 2027:** Camera-ready due
- **June 1-5, 2027:** Workshop co-located with NAACL 2027

## Proceedings

### Instructions to Prepare Your Shared Task Paper

The title of the paper should be in the following format: **< Team Name > at BLP-2027 Task 2: < Descriptive title of your paper >**

For example, team **AlphaX** would have their title as follows: **AlphaX at BLP-2027 Task 2: Culturally-Aware LLM Fine-tuning for Multilingual Question Answering**

- **The shared task papers may consist of up to four (4) pages of content.**

**Templates:** Shared task papers must follow the ACL two-column format, using the supplied [official templates](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj). Please do not modify these style files, nor use templates designed for other conferences. Submissions that do not conform to the required styles, including paper size, margin width, and font size restrictions, will be rejected without review. We will verify conformance using the [ACL pubcheck tool](https://github.com/acl-org/aclpubcheck); please run your camera-ready PDF through it before final submission, and we recommend using it at submission time too.

Paper submissions are open only to teams that submitted systems during the evaluation phase and appear on the leaderboard.

## Recent Updates

- **October 15, 2026** Development phase starts
- **October 15, 2026** Training, dev and dev-test data released
- **October 15, 2026** Release of the mT5 baseline and an example LoRA LLM fine-tuning script for Subtask 2A and 2B

## Contents of the Directory

- Main folder: [data](./data)<br/>
  Contains the data preparation script and a description of the data. See [data/README.md](./data/README.md).
- Main folder: [baselines](./baselines)<br/>
  Contains the mT5 baseline for Subtask 2A and 2B.
- Main folder: [example_scripts](./example_scripts)<br/>
  Contains a starter script to fine-tune an instruction-tuned LLM with LoRA for Subtask 2A and 2B.
- Main folder: [format_checker](./format_checker)<br/>
  Contains the script to check the format of a submission file.
- Main folder: [scorer](./scorer)<br/>
  Contains the official scorer (BERTScore, ROUGE, METEOR and BLEU).
- [README.md](./README.md)<br/>
  This file!

## Task Description

Given a culturally or regionally grounded question written by a native speaker, generate an appropriate, accurate answer **in the same language as the question**. Questions span 18 everyday topics (e.g., food & drinks, travel, religion, tradition, education, names & persons, weather). The task has two subtasks:

- **Subtask 2A – Monolingual QA:** Given a question in a target language, generate an answer in the same language using a **separate model for each language**.
- **Subtask 2B – Multilingual QA:** Given a question in any of the four target languages, generate an answer in the same language using a **single multilingual model** trained across all four languages.

Teams may use any pre-trained model (open or closed) and any additional publicly available data, but must describe them in their system paper. The MultiNativQA **test** split is used as our dev-test set and must **not** be used for training.

## Participation

Each combination of a subtask and a language is a **track**, giving eight tracks in total:

| | Bangla | English | Assamese | Hindi |
|---|---|---|---|---|
| **2A** Monolingual | 2A-bn | 2A-en | 2A-as | 2A-hi |
| **2B** Multilingual | 2B-bn | 2B-en | 2B-as | 2B-hi |

Teams may take part in **any combination of tracks**, for example:

- a single language in a single subtask (e.g., 2A-bn only);
- several languages in one subtask (e.g., 2A-bn and 2A-hi);
- the same or different languages across both subtasks (e.g., 2A-as, 2B-as and 2B-en);
- all eight tracks.

All predictions go into **one submission file**. Every track you include must answer **all** test questions of that language. The model rules are:

- **2A:** one model per language. Different 2A languages may use different models.
- **2B:** one multilingual model, trained on all four languages, for every 2B language you submit, even if you submit only some of them.

**Rankings.** Each track has its own leaderboard ranked by BERTScore F1 (eight leaderboards). There is also a **2B-All** leaderboard (macro-average BERTScore F1 over the four languages) for 2B submissions that cover all four languages.

## Dataset

For an overview of the data, please refer to [data/README.md](./data/README.md). In brief, we use the language-specific training subsets of MultiNativQA (~37K QA pairs):

| Language | Regions      | Train | Dev | Dev-Test |
|---|--------------|---:|---:|---:|
| Bangla (`bn`) | Dhaka, Kolkata | 13,909 | 1,883 | 3,667 |
| English (`en`) | Dhaka, Doha  | 12,973 | 1,820 | 3,435 |
| Assamese (`as`) | Assam, India | 1,131 | 157 | 545 |
| Hindi (`hi`) | Delhi, India | 9,288 | 1,286 | 2,745 |
| **Total** |              | **37,301** | **5,146** | **10,392** |


### Input data format

Each file is in JSON Lines format (one JSON object per line). Files are provided per language (`data/<split>/{bn,en,as,hi}.jsonl`) and combined (`data/<split>/all.jsonl`):

```
{
    "id": "...", 
    "lang": "...", 
    "region": "...", 
    "category": "...", 
    "question": "...", 
    "answer": "..."
}
```

Where:

- id: a unique id of the QA pair
- lang: language code, one of *bn*, *en*, *as*, *hi*
- region: region of the native-speaker query (*bd*, *in*, *qa*)
- category: topic of the question (e.g., *food_drinks*, *travel*, *religion*)
- question: the question
- answer: the reference answer (absent in the blind test set)

##### Example

```
{
    "id": "en_qa_cf92ec1e52b4b3071d263a1063b43928", 
    "lang": "en", 
    "region": "qa", 
    "category": "immigration", 
    "question": "Can I extend my tourist visa in Qatar?", 
    "answer": "If you would like to extend your visa, you will need to proceed to immigration headquarters in Doha prior to the expiry of your visa and apply there for an extension."
}
```

## Example Fine-tuning Script

We release an example script to help participants get started with data loading, LoRA fine-tuning of an instruction-tuned LLM (default: Qwen2.5-1.5B-Instruct) and answer generation for both subtasks. It writes a ready-to-submit `task.jsonl` and can also run zero-shot (no training). We encourage you to use and adapt it to build your own systems. The script is available in [example_scripts](./example_scripts/finetune_llm_lora.py).

Choose any combination of tracks with `--subtasks` and `--langs`. The script trains one adapter per language for 2A and one adapter on all four languages for 2B, and writes all predictions into a single `task.jsonl`.

```
# One track
python example_scripts/finetune_llm_lora.py --subtasks 2A --langs hi

# Two languages in both subtasks (four tracks)
python example_scripts/finetune_llm_lora.py --subtasks 2A 2B --langs bn as

# All eight tracks (default)
python example_scripts/finetune_llm_lora.py

# Zero-shot (no training)
python example_scripts/finetune_llm_lora.py --zero-shot
```

## Scorer and Official Evaluation Metrics

### Scorers

The scorer for the task is located in the [scorer](./scorer) module of the project. It reports the official evaluation metric (BERTScore F1) together with ROUGE, METEOR and BLEU for every track found in a prediction file. No subtask flag is needed. The scorer invokes the format checker to verify that the output is properly shaped, checks that every prediction id exists in the gold file with the correct language, and checks that every question of each submitted language has been answered.

You can install all prerequisites through,

```
pip install -r requirements.txt
```

Launch the scorer for the task as follows:

```
python scorer/task.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file>
```

##### Example

```
python scorer/task.py --gold-file-path data/dev_test/all.jsonl --pred-file-path task.jsonl
```

Use `--output-json` to save the full results, or `--leaderboard-json` to save flat scores keyed by track (e.g. `2A_bn_bertscore_f1`, `2B_all_bertscore_f1`).

The scorer downloads the encoder models from Hugging Face on first use. A GPU is recommended (`--device cuda`), though the dev-test set can be scored on CPU.

### Official Evaluation Metrics

The **official evaluation metric** is **BERTScore F1** ([Zhang et al., 2020](https://openreview.net/forum?id=SkeHuCVFDr)) computed with a **language-specific encoder**:

| Language | Encoder | Layer |
|---|---|---:|
| Bangla | [csebuetnlp/banglabert](https://huggingface.co/csebuetnlp/banglabert) | 9 |
| English | [bert-base-uncased](https://huggingface.co/bert-base-uncased) | 9 |
| Hindi | [ai4bharat/IndicBERTv2-MLM-only](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) | 9 |
| Assamese | [ai4bharat/IndicBERTv2-MLM-only](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) | 9 |

BERTScore uses greedy cosine matching of contextual token embeddings, no IDF weighting and no baseline rescaling; inputs are truncated to 512 tokens. Bangla text is normalised with the [csebuetnlp normalizer](https://github.com/csebuetnlp/normalizer), as recommended for BanglaBERT. The implementation in [scorer/bertscore.py](./scorer/bertscore.py) is numerically equivalent to the reference `bert_score` package for the same model and layer.

- **Per track (2A-bn, …, 2B-hi):** ranked by BERTScore F1, giving eight leaderboards.
- **2B-All:** ranked by the macro-average BERTScore F1 over the four languages; only 2B submissions that cover all four languages are ranked.

### Additional Metrics

The scorer also reports the following metrics. They are listed on the leaderboard and in the overview paper, but **official ranking uses BERTScore F1 only**.

| Metric | Details |
|---|---|
| BERTScore P / R | Precision and recall from the same BERTScore computation |
| ROUGE-1 / ROUGE-2 / ROUGE-L | F-measure, averaged over items; no stemming |
| METEOR | NLTK implementation, averaged over items; WordNet synonyms and stemming apply to English only |
| BLEU | Corpus-level BLEU-4 ([sacrebleu](https://github.com/mjpost/sacrebleu)) |
| Script fidelity | Share of answers written mainly in the expected script for the language (diagnostic) |

All scores are reported on a **0–1 scale** (BLEU is divided by 100). The lexical metrics (ROUGE, METEOR, BLEU) share a Unicode-aware tokenizer that keeps Indic vowel signs attached to their letters and splits off punctuation such as the danda (।). This is necessary because the default `rouge_score` tokenizer removes all non-Latin characters, which would give a ROUGE of 0 for Bangla, Hindi and Assamese. English text is lower-cased. Empty answers receive a score of 0.

##### Example output

The scorer prints one table for each subtask present, with one row for each submitted language:

```
Subtask 2A (Monolingual QA)
Language       N   BS-F1*     BS-P     BS-R      R-1      R-2      R-L   METEOR     BLEU   Script
-------------------------------------------------------------------------------------------------
Assamese     ...

Subtask 2B (Multilingual QA)
Language       N   BS-F1*     BS-P     BS-R      R-1      R-2      R-L   METEOR     BLEU   Script
-------------------------------------------------------------------------------------------------
English      ...
Assamese     ...
-------------------------------------------------------------------------------------------------
All (macro): not ranked; requires all four languages in 2B.

* BERTScore F1 is the official ranking metric for each track (and for 2B-All). All scores on a 0-1 scale.
```

## Baselines

The [baselines](./baselines) module contains the official **mT5 baseline** ([google/mt5-base](https://huggingface.co/google/mt5-base)), fine-tuned to generate the answer from the input `Answer in <Language>: <question>`.

- **Subtask 2A:** one mT5 model is trained for each requested language.
- **Subtask 2B:** a single mT5 model is trained on all four languages and answers the requested languages.

All predictions are written to one `task.jsonl`. Choose tracks with `--subtasks` and `--langs` (default: all eight tracks):

```
python baselines/mt5_baseline.py                                   # all eight tracks
python baselines/mt5_baseline.py --subtasks 2A --langs bn          # one track
python baselines/mt5_baseline.py --subtasks 2A 2B --langs as hi    # four tracks
python scorer/task.py -g data/dev_test/all.jsonl -p runs/mt5_baseline/task.jsonl
```

Default settings: 5 epochs, learning rate 5e-4, effective batch size 32, maximum input/output length 256/512 tokens, beam search with 4 beams, best checkpoint selected by dev loss. A GPU with bf16 support is recommended, since mT5 is unstable in fp16.

#### Subtask 2A

Baseline Results for the task on Dev-Test set (mT5-base, one model per language)

| Language | BS-F1 | BS-P | BS-R | ROUGE-L | METEOR | BLEU |
|---|---|---|---|---|---|---|
| Bangla | **0.5081** | 0.5833 | 0.4585 | 0.1738 | 0.1046 | 0.0097 |
| English | **0.5479** | 0.5908 | 0.5153 | 0.2213 | 0.1697 | 0.0443 |
| Assamese | **0.5019** | 0.5736 | 0.4547 | 0.2206 | 0.1396 | 0.0073 |
| Hindi | **0.4719** | 0.4959 | 0.4614 | 0.1565 | 0.1204 | 0.0220 |

#### Subtask 2B

Baseline Results for the task on Dev-Test set (mT5-base, single multilingual model)

| Language | BS-F1 | BS-P | BS-R | ROUGE-L | METEOR | BLEU |
|---|---|---|---|---|---|---|
| Bangla | **0.5086** | 0.5821 | 0.4601 | 0.1751 | 0.1076 | 0.0097 |
| English | **0.5465** | 0.5882 | 0.5151 | 0.2194 | 0.1699 | 0.0441 |
| Assamese | **0.5272** | 0.5897 | 0.4858 | 0.2343 | 0.1524 | 0.0142 |
| Hindi | **0.4699** | 0.4939 | 0.4597 | 0.1548 | 0.1189 | 0.0218 |
| All (macro) | **0.5130** | 0.5635 | 0.4802 | 0.1959 | 0.1372 | 0.0224 |
Results on the blind test set will be added after the evaluation phase.

## Format checker

The format checker for the task is located in the [format_checker](./format_checker) module of the project. It verifies that your generated results file complies with the expected format.

Before running the format checker please install all prerequisites,

```
pip install -r requirements.txt
```

To launch it, please run the following command:

```
python format_checker/task.py -p results_files [--test-file <test_file>]
```

##### Example

```
python format_checker/task.py -p ./task.jsonl --test-file data/test/all.jsonl
```

**results_files**: can be one path or a space-separated list of paths

**--test-file** (optional): the released test file. When given, the checker also verifies that every id exists with the right language and that **each track you submitted is complete**. We strongly recommend using it before uploading. The checker prints the tracks it found, for example:

```
INFO : Tracks found -> 2A (Monolingual QA): Assamese; 2B (Multilingual QA): English, Assamese
```

## Submission

### Guidelines

Evaluation consists of two phases:

1. **Development phase:** This phase involves working on the *dev-test set*.
2. **Evaluation phase:** This phase involves working on the *blind test set*, which will be released during the ***evaluation cycle***.

For each phase, please adhere to the following guidelines:

- We request each team to establish and manage a single account for all submissions. Hence, all runs should be submitted through the same account. Submissions made from multiple accounts by the same team may lead to your system not being ranked in the overview paper.
- The most recently uploaded file on the leaderboard will serve as your final submission.
- Adhere strictly to the naming convention for the output file, which must be named **`task.jsonl`**. Deviation from this standard could trigger an error on the leaderboard.
- Submission protocol requires you to compress the `.jsonl` file into a `.zip` file (for instance, `zip task.zip task.jsonl`) and submit it through the Codabench page.
- With each submission, please include your team name along with a brief explanation of your methodology.
- Each team is allowed a maximum of 100 submissions per day. Please adhere to this limit.

### Submission Format

The submission file format is JSON Lines (one JSON object per line). Each line adheres to the following structure:

```
{
    "id": "...", 
    "subtask": "...", 
    "lang": "...", 
    "answer": "...", 
    "model": "..."
}
```

Where:

- id: the id of the question
- subtask: *2A* or *2B*
- lang: language code of the question, one of *bn*, *en*, *as*, *hi*
- answer: the generated answer (in the same language as the question)
- model: model name (one name per language for 2A; a single name for all 2B rows)

A file may contain any combination of tracks (see [Participation](#participation)). The same question id appears once per subtask: if you take part in both 2A-bn and 2B-bn, each Bangla question appears twice, once with `"subtask": "2A"` and once with `"subtask": "2B"`. Files produced for different tracks can simply be concatenated, e.g. `cat run_2A.jsonl run_2B.jsonl > task.jsonl`.

##### Example

```
{
    "id": "bn_bd_0a1b2c...", 
    "subtask": "2A", 
    "lang": "bn", 
    "answer": "ঢাকার বিখ্যাত খাবারের মধ্যে কাচ্চি বিরিয়ানি অন্যতম।", 
    "model": "banglat5-ft"
}

{
    "id": "bn_bd_0a1b2c...", 
    "subtask": "2B", 
    "lang": "bn", 
    "answer": "কাচ্চি বিরিয়ানি ঢাকার অন্যতম জনপ্রিয় খাবার।", 
    "model": "mt5-multi-ft"
}
```

### Submission Site

All tracks are submitted to a single EvalAI competition, which shows one leaderboard for each track plus 2B-All:

**https://eval.qcri.org/web/challenges/challenge-page/6/overview**

## Citation

There are various papers associated with the task. Details for the papers specific to the task as well as an overall overview will be posted here as they come out. Bib entries for each paper are included here.

```
@inproceedings{hasan-etal-2025-nativqa,
    title = "{N}ativ{QA}: Multilingual Culturally-Aligned Natural Query for {LLM}s",
    author = "Hasan, Md. Arid and Hasanain, Maram and Ahmad, Fatema and Laskar, Sahinur Rahman and Upadhyay, Sunaya and Sukhadia, Vrunda N and Kutlu, Mucahid and Chowdhury, Shammur Absar and Alam, Firoj",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.770/",
    doi = "10.18653/v1/2025.findings-acl.770",
    pages = "14886--14909",
}

@article{alam2025nativqa,
    title = {{NativQA Framework:} Enabling LLMs with Native, Local, and Everyday Knowledge},
    author = {Alam, Firoj and Hasan, Md Arid and Laskar, Sahinur Rahman and Kutlu, Mucahid and Darwish, Kareem and Chowdhury, Shammur Absar},
    journal = {arXiv preprint arXiv:2504.05995},
    year = {2025},
}

@inproceedings{blp2027-overview-task2,
    title = "Overview of BLP 2027 Task 2: Multilingual Cultural Question Answering",
    author = "TBA",
    booktitle = "Proceedings of the Third International Workshop on Bangla Language Processing (BLP-2027)",
    year = "2027",
    publisher = "Association for Computational Linguistics",
}
```

## Leaderboard

Leaderboards for each track (2A-bn, 2A-en, 2A-as, 2A-hi, 2B-bn, 2B-en, 2B-as, 2B-hi) and for 2B-All will be announced after the evaluation phase.

## Communication

Please join us in the Slack channel for discussion and questions:

- [Slack](https://join.slack.com/t/blpworkshop/shared_invite/zt-1ryu9eyac-7fevK9A4_Bt~qN_eCK349g)

## Organizers

- [Md Arid Hasan](https://aridhasan.github.io), PhD Student, University of Toronto, Canada
- [Firoj Alam](https://firojalam.one/), Senior Scientist, Qatar Computing Research Institute, Qatar
- [Shammur Absar Chowdhury](https://shammur.one), Scientist, Qatar Computing Research Institute, Qatar
- [Sahinur Rahman Laskar](https://sahinurlaskar.github.io/), Assistant Professor, UPES, India
- [Nowshin Mahjabin](#), Bangladesh
- [Syed Ishtiaque Ahmed](https://ishtiaque.net), Associate Professor, University of Toronto, Canada

## License

The MultiNativQA data is distributed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. The blind test set will be released under the same license.
