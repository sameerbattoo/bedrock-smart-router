# Bedrock Smart Router — Benchmark & Evaluation Suite

Comprehensive benchmarking infrastructure to validate the Smart Router's value proposition across cost, latency, accuracy, and resilience.

## What This Proves

| Hypothesis | How benchmark validates |
|---|---|
| **Cost savings** | Router uses cheaper models for simple prompts → lower avg cost |
| **Faster processing** | Simple prompts on smaller models = lower latency |
| **Better accuracy** | Router picks capable models for hard prompts → higher quality |
| **Better fallback** | Under load, router succeeds where single-model boto3 throttles |

## Quick Results (from initial testing)

| Runner | Avg Score | Avg Cost | Avg Latency |
|---|---|---|---|
| Smart Router (Default) | **9.0/10** | $0.00008/prompt | 1129ms |
| Claude Sonnet 4.6 | 8.7/10 | $0.006/prompt | 5090ms |
| Claude Haiku 4.5 | 8.5/10 | $0.001/prompt | 2076ms |
| Amazon Nova Pro | 7.8/10 | $0.001/prompt | 1333ms |

Router Default: **98.7% cheaper** and **78% faster** than Sonnet with comparable quality.

---

## Directory Structure

```
benchmarks/
├── runner/                           # Benchmark execution scripts
│   ├── config.py                    # Models, strategies, judge prompts
│   ├── run_benchmark.py             # Main orchestrator
│   ├── judge.py                     # LLM-as-judge scoring (Sonnet 4.6)
│   ├── burst_test.py                # Concurrency/throttling test
│   ├── analyze_results.py           # Generates REPORT.md from results
│   ├── quick_mix_test.py            # Quick 3-prompt test across runners
│   ├── tune_heuristic.py            # Tunes the keyword-heuristic scorer on labeled prompts
│   └── generate_all.py              # Regenerate the synthetic benchmark prompts
├── classifier/                       # Complexity classifier (TF-IDF + LogReg)
│   ├── download/                    # One download script per data source
│   ├── datasets/                    # Downloaded data + generated/ synthetic prompts
│   │   └── generated/               #   synthetic prompts (also used as benchmark prompts)
│   │       └── scripts/             #   gen_*.py generators (run via runner/generate_all.py)
│   ├── train_tfidf.py               # Trains the classifier → ml_classifier.json
│   ├── tfidf_model/                 # Saved model artifacts
│   └── README.md
└── results/                          # Benchmark run outputs + REPORT.md
```

---

## Runners (7 total)

### Baselines (direct boto3 converse API)
| Runner | Model | Purpose |
|---|---|---|
| `sonnet` | Claude Sonnet 4.6 | High quality baseline |
| `haiku` | Claude Haiku 4.5 | Cost floor |
| `nova-pro` | Amazon Nova Pro | AWS-native mid-tier |
| `opus` | Claude Opus 4.7 | Quality ceiling |

### Smart Router
| Runner | Strategy | Purpose |
|---|---|---|
| `router-default` | balanced | Default routing |
| `router-budget` | cost-optimized | Cost optimization mode |
| `router-quality` | quality-optimized | Quality optimization mode |

---

## Usage

### Run Benchmark
```bash
# Full run (295 prompts × 7 runners)
python benchmarks/runner/run_benchmark.py

# Subset
python benchmarks/runner/run_benchmark.py --category text_to_sql --limit 5
python benchmarks/runner/run_benchmark.py --runner router-default

# Quick test (3 prompts, all runners, with judging)
python benchmarks/runner/quick_mix_test.py
```

### Judge Responses
```bash
python benchmarks/runner/judge.py benchmarks/results/benchmark_TIMESTAMP.json
```

### Generate Report
```bash
# From a single results file
python benchmarks/runner/analyze_results.py benchmarks/results/benchmark_judged.json

# From all results in directory
python benchmarks/runner/analyze_results.py benchmarks/results/
```

### Burst/Throttling Test
```bash
python benchmarks/runner/burst_test.py
python benchmarks/runner/burst_test.py --levels 10,25,50,100
```

### Complexity Classifier
```bash
# 1. Download the training datasets into classifier/datasets/
python benchmarks/classifier/download/download_all.py

# 2. Train and export the model (requires: pip install datasets scikit-learn numpy)
python benchmarks/classifier/train_tfidf.py
#    → bedrock_smart_router/data/ml_classifier.json
```

See `benchmarks/classifier/README.md` for the full dataset/license table and
architecture details.

---

## Classifier Training Data

Datasets are downloaded (not committed) via one script per source under
`classifier/download/`. All sources are permissively licensed and ungated
(DevQuasar, HellaSwag, MATH, GSM8K, ARC, MMLU, WinoGrande, CommonsenseQA,
HumanEval, MBPP, BBH, IFEval) plus in-repo synthetic prompts under
`classifier/datasets/generated/`. Alpaca (non-commercial) and GPQA (gated) are
intentionally excluded. Full table: `classifier/README.md`.

---

## Complexity Classifier Results

TF-IDF (1–3 gram) + Logistic Regression, exported for pure-NumPy runtime
inference (no scikit-learn dependency at inference time):

| Metric | Keyword Heuristic | TF-IDF Classifier |
|---|---|---|
| **Overall Accuracy** | 42.5% | **~86%** |
| Inference Time | <1ms | <1ms |
| Runtime Dependency | none | numpy only |

The classifier keeps strong recall on complex/reasoning prompts, ensuring the
router doesn't send hard tasks to cheap models.

---

## Router Fixes Applied (May 9, 2026)

1. **System prompt extraction bug** — `_extract_text()` now handles system prompt format, fixing complexity scoring
2. **Expanded REASONING_MARKERS** — added "analysis", "build a", "for each", etc.
3. **Added DATA_ANALYSIS_SIGNALS** — SQL analytics keywords (cohort, CTE, window functions)
4. **Reasoning auto-promote threshold** — raised from 2 to 4 markers to reduce false positives
5. **Opus serviceTier fix** — all Opus models (4.1-4.7) reject serviceTier parameter; catalog updated

See `docs/JS_PORT_CHANGES_2026-05-09.md` for full details.

---

## Prompt Format

The synthetic prompts in `classifier/datasets/generated/*.json` serve double duty:
they are the prompts the benchmark runner sends to each runner, and they are also
folded into the classifier's training data. Regenerate them with
`python benchmarks/runner/generate_all.py` (which runs the `gen_*.py` scripts under
`classifier/datasets/generated/scripts/`).

Each prompt object:
```json
{
  "id": "sql_001",
  "category": "text_to_sql",
  "difficulty": "simple|medium|complex",
  "system_prompt": "You are a SQL expert...",
  "user_prompt": "Find all orders from last month",
  "context": "CREATE TABLE orders (...)",
  "expected_answer": "SELECT * FROM orders WHERE..."
}
```

---

## Cost Estimate

Full benchmark run: ~295 prompts × 7 runners = ~2,065 API calls + ~2,065 judge calls
Estimated cost: $20-40 per full run (varies by output length)

Region: **us-west-2**
