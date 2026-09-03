# Model Card — Prompt Complexity Classifier

This card documents the machine-learning model shipped with Smart Router for
Amazon Bedrock at `bedrock_smart_router/data/ml_classifier.json`. The router
uses it to label an incoming prompt's complexity so it can route to an
appropriately capable (and cost-effective) Bedrock model.

## Model overview

| | |
|---|---|
| **Name** | Prompt Complexity Classifier |
| **Version** | Ships with the Smart Router package (`bedrock_smart_router/data/ml_classifier.json`) |
| **Task** | Multi-class text classification of prompt complexity |
| **Classes** | `simple`, `moderate`, `complex`, `reasoning` |
| **Architecture** | TF-IDF (word 1–3 grams, sublinear TF, L2-normalized, `max_features=25000`) + multinomial Logistic Regression (`C=5.0`, `class_weight="balanced"`) |
| **Framework** | Trained with scikit-learn; exported to a dependency-light JSON consumed by a pure-NumPy inference path (`bedrock_smart_router/ml_classifier.py`) — no scikit-learn at runtime |
| **Artifact format** | JSON containing vocabulary, IDF vector, learned coefficients, intercept, and class labels. **Learned weights only — no training data is bundled.** |
| **Size** | ~3 MB |
| **License** | Apache-2.0 (same as the project) |

## Intended use

- **Primary use:** Internal signal for the router to estimate prompt complexity
  and select a Bedrock model that matches the difficulty of the request, in
  order to reduce cost and latency without sacrificing quality on hard prompts.
- **Users:** Developers embedding Smart Router for Amazon Bedrock in their own
  applications.
- **Scope:** English-language prompts typical of developer/assistant workloads
  (Q&A, code, SQL, summarization, extraction, reasoning tasks).

### Out-of-scope / not intended for

- Content moderation, safety classification, or any decision affecting an
  individual's rights, access, or treatment.
- A ground-truth measure of "difficulty" for people or for grading.
- Non-English prompts or domains far outside the training distribution — the
  label may be unreliable. The router degrades gracefully (a heuristic path and
  fallbacks exist), so a wrong label affects model selection, not correctness of
  the final response.

## Training data

The classifier is trained only on **publicly available, permissively licensed,
ungated** datasets, plus a small set of in-repo synthetic prompts. Datasets are
downloaded at training time (via `benchmarks/classifier/download/`) and are
**not** committed to the repository. The shipped artifact contains learned
weights derived from this data, not the data itself.

| Source | HuggingFace ID | License | Contribution to labels |
|---|---|---|---|
| DevQuasar router set | `DevQuasar/llm_router_dataset-synth` | Apache-2.0 | simple / complex routing prompts |
| HellaSwag | `Rowan/hellaswag` | MIT | commonsense completion → moderate |
| Hendrycks MATH | `EleutherAI/hendrycks_math` | MIT | competition math → complex + cross-difficulty |
| GSM8K | `openai/gsm8k` | MIT | grade-school math → complex + cross-difficulty |
| ARC | `allenai/ai2_arc` | CC-BY-SA-4.0 | science QA → complex |
| MMLU | `cais/mmlu` | MIT | easy-subject QA → simple |
| WinoGrande | `allenai/winogrande` | Apache-2.0 | pronoun resolution → moderate |
| CommonsenseQA | `tau/commonsense_qa` | MIT | commonsense QA → moderate |
| HumanEval | `openai/openai_humaneval` | MIT | Python coding → complex |
| MBPP | `google-research-datasets/mbpp` | CC-BY-4.0 | Python coding → complex |
| BBH | `lukaemon/bbh` | MIT | hard reasoning → cross-difficulty |
| IFEval | `google/IFEval` | Apache-2.0 | instruction-following → cross-difficulty |
| In-repo synthetic prompts | — (committed under `benchmarks/classifier/datasets/generated/`) | Apache-2.0 | synthetic domain prompts (SQL, code, summarization, etc.) |

**Label derivation:** Fixed-label sources emit a final
`simple`/`moderate`/`complex` label directly. Cross-difficulty sources carry a
numeric z-scored difficulty that the trainer bins into moderate/complex/
reasoning. DevQuasar's `complex` bucket is split into moderate/complex by
keyword heuristics at train time.

### Datasets deliberately excluded

- **Alpaca** (`tatsu-lab/alpaca`) — CC-BY-NC (non-commercial). Excluded so the
  model contains no non-commercial training data.
- **GPQA** — gated (requires accepting terms). Excluded for frictionless,
  reproducible downloads.

> Users retraining the model should review each dataset's card and license
> before use, as upstream terms can change.

## Evaluation

- **Method:** stratified train/test split with cross-validation during training.
- **Typical results:** ~85% cross-validation accuracy / ~0.86 test accuracy on
  the 4-class task.
- **Per-class behavior:** strongest on `simple`; the model retains high recall on
  `complex`/`reasoning` prompts by design, so hard tasks are less likely to be
  routed to weaker models. `reasoning` is the hardest class and has the lowest
  per-class F1.

Metrics are indicative of the training distribution above and will differ on
other workloads. To reproduce, see `benchmarks/classifier/README.md`.

## Limitations and known risks

- **Distribution shift:** accuracy drops on prompts unlike the training data
  (non-English, niche domains, adversarial phrasing).
- **Not a safety tool:** the label says nothing about whether content is
  harmful, sensitive, or policy-violating.
- **Class imbalance:** `reasoning` is underrepresented relative to `simple`;
  `class_weight="balanced"` mitigates but does not eliminate this.
- **Failure mode is bounded:** a misclassification changes *which* Bedrock model
  is selected, not the correctness of Bedrock's response. The router's fallback
  chain and heuristic path bound the impact.

## Reproducibility

The model is fully reproducible from public data:

```bash
# 1. Download the datasets (not committed; fetched fresh at train time)
python benchmarks/classifier/download/download_all.py

# 2. Train and export the model
python benchmarks/classifier/train_tfidf.py
#    → bedrock_smart_router/data/ml_classifier.json
```

See `benchmarks/classifier/README.md` for the full pipeline, dataset/license
table, and architecture details.

## Maintainers

Maintained as part of Smart Router for Amazon Bedrock. Please file issues and
questions through the project's issue tracker.
