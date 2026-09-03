# Prompt Complexity Classifier

A TF-IDF + Logistic Regression classifier that labels a prompt's complexity as
**simple / moderate / complex / reasoning**. The trained model is exported for
pure-NumPy inference to `bedrock_smart_router/data/ml_classifier.json` and used
by the router's ML complexity path.

## The flow (run in this order)

```
  download/download_all.py          train_tfidf.py
        │                                 │
        ▼                                 ▼
  datasets/*.json  ───────────────►  ml_classifier.json
  (per-source downloads)      (trained model, in bedrock_smart_router/data/)
        ▲
        │
  datasets/generated/*.json  (in-repo synthetic prompts, already committed)
```

**Step 1 — Download the datasets** (one-time, ~2–4 min; needs network access):
```bash
python benchmarks/classifier/download/download_all.py
```
Runs one script per source and writes the raw training files into
`datasets/*.json`. Re-run only when you want to refresh the data. (You can also
run a single source, e.g. `python benchmarks/classifier/download/download_gsm8k.py`.)

**Step 2 — Train and export the model:**
```bash
python benchmarks/classifier/train_tfidf.py
```
Reads everything in `datasets/` (the downloads + the committed
`datasets/generated/` synthetic prompts), trains the classifier, and writes
`bedrock_smart_router/data/ml_classifier.json` — the file the router loads at
runtime. Set `CLASSIFIER_OUTPUT=ml_classifier_new.json` to write a throwaway
test file instead of overwriting the shipped model.

**Prerequisite:** `pip install datasets scikit-learn numpy`.

> You must run **Step 1 before Step 2** — the trainer errors out if `datasets/`
> is empty and tells you to run the downloader first.

## Layout

```
classifier/
├── download/            # one download script per data source
│   ├── _common.py       #   shared helpers (save, z-score, paths)
│   ├── download_all.py  #   runs every source script
│   └── download_<source>.py
├── datasets/            # all training data lives here
│   ├── *.json           #   downloaded datasets (gitignored — fetch fresh)
│   └── generated/       #   in-repo synthetic prompts (committed) + scripts/
├── train_tfidf.py       # the ONE trainer → ml_classifier.json
└── tfidf_model/         # trainer artifacts (classifier.pkl, classifier_data.json)
```

## How to (re)train

```bash
# 1. Download the training datasets into datasets/ (one-time, ~2-4 min)
python benchmarks/classifier/download/download_all.py

# 2. Train and export the model
python benchmarks/classifier/train_tfidf.py
#    → writes bedrock_smart_router/data/ml_classifier.json
#    Set CLASSIFIER_OUTPUT=ml_classifier_new.json to write a test file instead.
```

Requirements: `pip install datasets scikit-learn numpy`.

## Datasets & licenses

Downloaded datasets are **not** committed (each has its own upstream license;
`datasets/*.json` is gitignored). Every source below is permissively licensed
and ungated. Review each dataset's card before use.

| Source (script) | HuggingFace ID | License | Contribution |
|---|---|---|---|
| `download_devquasar.py` | `DevQuasar/llm_router_dataset-synth` | Apache-2.0 | simple/complex routing prompts |
| `download_hellaswag.py` | `Rowan/hellaswag` | MIT | commonsense completion → moderate |
| `download_math.py` | `EleutherAI/hendrycks_math` | MIT | competition math → complex + cross-difficulty (level 1–5) |
| `download_gsm8k.py` | `openai/gsm8k` | MIT | grade-school math → complex + cross-difficulty (step count) |
| `download_arc.py` | `allenai/ai2_arc` | CC-BY-SA-4.0 | science QA → complex |
| `download_mmlu.py` | `cais/mmlu` | MIT | easy-subject QA → simple |
| `download_winogrande.py` | `allenai/winogrande` | Apache-2.0 | pronoun resolution → moderate |
| `download_commonsense_qa.py` | `tau/commonsense_qa` | MIT | commonsense QA → moderate |
| `download_humaneval.py` | `openai/openai_humaneval` | MIT | Python coding → complex |
| `download_mbpp.py` | `google-research-datasets/mbpp` | CC-BY-4.0 | Python coding → complex |
| `download_bbh.py` | `lukaemon/bbh` | MIT | hard reasoning → cross-difficulty |
| `download_ifeval.py` | `google/IFEval` | Apache-2.0 | instruction-following → cross-difficulty |
| `datasets/generated/` | — (in-repo) | Apache-2.0 | synthetic domain prompts (SQL, code, summarization, …) |

**Cross-difficulty files** (`cross_difficulty_*.json`) carry a numeric z-scored
difficulty; the trainer bins them into moderate/complex/reasoning. All other
files carry a final `simple`/`moderate`/`complex` label directly. DevQuasar's
`complex` bucket is split into moderate/complex by keyword heuristics at train
time.

**Not included**, by design:
- **Alpaca** (`tatsu-lab/alpaca`) — CC-BY-NC (non-commercial); excluded so the
  model contains no non-commercial training data.
- **GPQA** — gated (requires accepting terms); excluded for frictionless,
  reproducible downloads.

## Architecture

- **Features:** TF-IDF over word 1–3 grams, sublinear TF, L2-normalized
  (`max_features=25000`).
- **Classifier:** multinomial Logistic Regression (`C=5.0`,
  `class_weight="balanced"`).
- **Inference:** the exported JSON (vocabulary, IDF, coefficients, intercept,
  classes) is consumed by `bedrock_smart_router/ml_classifier.py` via a
  dependency-light NumPy implementation (no scikit-learn at runtime).

Typical accuracy: ~85% cross-validation / ~0.86 test (4-class).
