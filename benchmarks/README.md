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
├── data/
│   ├── generated/                    # 295 custom prompts (6 categories)
│   │   ├── text_to_sql.json          (50 prompts)
│   │   ├── document_extraction.json  (50 prompts)
│   │   ├── log_analysis.json         (45 prompts)
│   │   ├── anomaly_detection.json    (50 prompts)
│   │   ├── code_generation.json      (50 prompts)
│   │   └── summarization.json        (50 prompts)
│   ├── generated_scripts/            # Scripts to regenerate prompts
│   │   └── gen_*.py (6 files)
│   ├── industry_standard/
│   │   ├── text_datasets/            # BoolQ, TriviaQA, XSum, T-REx, etc. (9 datasets)
│   │   ├── multimodal/              # Invoice/DocVQA/ChartQA images (150 samples)
│   │   ├── pdfs/                    # Real multi-page PDF documents (50 samples)
│   │   ├── reasoning/              # GSM8K, GPQA, MBPP, MATH, ARC (complex)
│   │   └── auxiliary/              # HellaSwag, WinoGrande, MMLU, Alpaca
│   └── download_scripts/            # Scripts to download all datasets
├── runner/                           # Benchmark execution scripts
│   ├── config.py                    # Models, strategies, judge prompts
│   ├── run_benchmark.py             # Main orchestrator
│   ├── judge.py                     # LLM-as-judge scoring (Sonnet 4.6)
│   ├── burst_test.py                # Concurrency/throttling test
│   ├── analyze_results.py           # Generates REPORT.md from results
│   ├── quick_mix_test.py            # Quick 3-prompt test across runners
│   ├── calibrate_thresholds.py      # Threshold calibration using labeled data
│   └── generate_all.py              # Regenerate all prompt files
├── classifier/                       # Complexity classifier (ML-based)
│   ├── prepare_data.py              # Combines all data sources
│   ├── train.py                     # Trains sentence-embedding classifier
│   ├── predict.py                   # Inference / interactive mode
│   ├── training_data.json           # 2,545 labeled samples
│   ├── trained_model/               # Saved model artifacts
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
# Prepare training data (2,545 samples from all sources)
python benchmarks/classifier/prepare_data.py

# Train (requires: pip install sentence-transformers scikit-learn)
python benchmarks/classifier/train.py

# Predict
python benchmarks/classifier/predict.py "Build a cohort analysis with CTEs"
```

---

## Data Sources (2,545 total training samples)

| Source | Samples | Labels | Purpose |
|---|---|---|---|
| Custom prompts (6 categories) | 295 | Hand-labeled simple/medium/complex | Core benchmark |
| Text datasets (BoolQ, TriviaQA, etc.) | 1,050 | By task type mapping | Classifier training |
| Multimodal (invoices, DocVQA, ChartQA, PDFs) | 200 | By task complexity | Vision/document testing |
| Reasoning (GSM8K, GPQA, MBPP, MATH, ARC) | 250 | All complex | Complex class balance |
| Auxiliary (HellaSwag, WinoGrande, MMLU, Alpaca) | 750 | Medium/simple | Class balance |

---

## Complexity Classifier Results

Trained on 2,545 samples using sentence embeddings (all-MiniLM-L6-v2) + logistic regression:

| Metric | Keyword Heuristic | Embedding Classifier |
|---|---|---|
| **Overall Accuracy** | 42.5% | **82.8%** |
| Complex Recall | ~30% | **96%** |
| Inference Time | <1ms | ~5ms |
| Model Size | 0 (rules only) | 22MB |

The classifier almost never misses a complex prompt (96% recall), ensuring the router doesn't send hard tasks to cheap models.

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

Each prompt in `data/generated/*.json`:
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
