# Industry Standard Datasets

This directory contains third-party datasets used for training and evaluating the
prompt complexity classifier. **These datasets are NOT distributed with the project**
due to varying licenses and large file sizes.

## Downloading Datasets

To download all datasets needed for classifier training:

```bash
pip install datasets

# Download routing-specific datasets (Easy2Hard-Bench, LeetCode, MT-Bench, IFEval, etc.)
python benchmarks/data/download_scripts/download_routing_datasets.py

# Download additional reasoning datasets (GSM8K, GPQA, MBPP)
python benchmarks/data/download_scripts/download_complex_reasoning.py

# Download more data (MATH, ARC-Challenge, HumanEval, HellaSwag)
python benchmarks/data/download_scripts/download_more_data.py
```

After downloading, you can retrain the classifier:

```bash
python benchmarks/classifier/train_tfidf.py
```

## Dataset Sources and Licenses

The following datasets are used for training. Each has its own license terms
which you must review and accept before downloading:

| Dataset | HuggingFace ID | License | Used For |
|---------|---------------|---------|----------|
| DevQuasar LLM Router | `DevQuasar/llm-router-dataset` | Apache-2.0 | Binary simple/complex labels |
| Easy2Hard-Bench | `furonghuang-lab/Easy2Hard-Bench` | MIT | Continuous difficulty scores |
| LeetCode | `newfacade/LeetCodeDataset` | See source | Coding problem difficulty |
| MT-Bench | `HuggingFaceH4/mt_bench_prompts` | Apache-2.0 | Multi-turn evaluation prompts |
| IFEval | `google/IFEval` | Apache-2.0 | Instruction-following constraints |
| OpenOrca (subset) | `Open-Orca/OpenOrca` | MIT | System + user prompt pairs |
| Big Bench Hard | Google BIG-Bench | Apache-2.0 | Hard reasoning tasks |
| GSM8K | `openai/gsm8k` | MIT | Multi-step math reasoning |
| MATH | `lighteval/MATH` | MIT | Competition math |
| ShareGPT (sample) | Community re-hosted | See source | Numeric difficulty scores |
| WildChat | `allenai/WildChat` | See source | Real conversations with system prompts |
| ARC-Challenge | `allenai/ai2_arc` | CC-BY-SA-4.0 | Science reasoning |
| AlpacaEval | `tatsu-lab/alpaca_eval` | CC-BY-NC-4.0 | Instruction following |

**Important:** Some datasets have non-commercial licenses (e.g., CC-BY-NC-4.0).
Review each dataset's license on HuggingFace before use. The pre-trained
`ml_classifier.json` shipped with the package contains only learned TF-IDF
vocabulary weights and logistic regression coefficients — not the training data itself.

## What Ships with the Package

Only `bedrock_smart_router/data/ml_classifier.json` ships in the published package.
This file contains:
- A TF-IDF vocabulary (word → index mapping)
- IDF weights (inverse document frequency values)
- Logistic regression coefficients and intercepts
- Class labels: `["simple", "moderate", "complex", "reasoning"]`

No third-party training data is reproduced in the shipped classifier weights.
