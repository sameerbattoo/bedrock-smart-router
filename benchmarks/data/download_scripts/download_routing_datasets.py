# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Download high-quality datasets specifically designed for prompt complexity/routing classification.

These datasets contain prompts with difficulty labels or scores that can be mapped
to our 4-tier taxonomy: simple, moderate, complex, reasoning.

Datasets:
1. Easy2Hard-Bench (NeurIPS 2024) — continuous difficulty scores from real human/LLM performance
2. RouterBench — 30K+ prompts from 11 benchmarks with multi-model performance data
3. LeetCode Dataset — coding problems with Easy/Medium/Hard labels
4. Arena-Hard — curated hard prompts from Chatbot Arena that separate top models
5. WildChat (additional system+user prompts) — real conversations with system prompts
6. LMSYS Chatbot Arena conversations — real user prompts with complexity signal

Output: benchmarks/data/industry_standard/ (JSON files ready for train_tfidf.py)

Requirements:
    pip install datasets
"""
import json
import os
import random
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)

random.seed(42)

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "industry_standard")
os.makedirs(BASE_DIR, exist_ok=True)


def save_to_industry_standard(filename: str, samples: list[dict]) -> int:
    """Save samples in the format expected by train_tfidf.py: {text, label}."""
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  ✅ Saved {len(samples)} samples to {filename}")
    return len(samples)


# ═══════════════════════════════════════════════════════════════
# 1. Easy2Hard-Bench (NeurIPS 2024)
#    Continuous difficulty scores (0-1) from real performance data.
#    Subsets: AMC math, ARC science, Codeforces, GSM8K, Winogrande
# ═══════════════════════════════════════════════════════════════
def download_easy2hard_bench():
    """Download Easy2Hard-Bench with fine-grained difficulty annotations."""
    print("\n" + "=" * 60)
    print("1. Easy2Hard-Bench (NeurIPS 2024 — difficulty-scored prompts)")
    print("=" * 60)

    samples = []

    # Map continuous difficulty score to our 4-tier taxonomy
    def score_to_label(score: float) -> str:
        if score < 0.30:
            return "simple"
        elif score < 0.55:
            return "moderate"
        elif score < 0.80:
            return "complex"
        else:
            return "reasoning"

    subsets = ["E2H-AMC", "E2H-ARC", "E2H-GSM8K", "E2H-Winogrande"]

    for subset in subsets:
        try:
            # Try loading the eval split (has difficulty scores)
            ds = load_dataset("furonghuang-lab/Easy2Hard-Bench", subset, split="eval")
            print(f"  {subset}: {len(ds)} samples, columns: {ds.column_names}")

            for i, item in enumerate(ds):
                # Extract the problem text
                text = item.get("problem", item.get("question", item.get("text", "")))
                if not text:
                    # Try other column names
                    for key in item.keys():
                        if key not in ("difficulty", "id", "answer", "solution", "label"):
                            val = item[key]
                            if isinstance(val, str) and len(val) > 20:
                                text = val
                                break
                if not text or len(text.strip()) < 10:
                    continue

                # Get difficulty score
                difficulty = item.get("difficulty", None)
                if difficulty is None:
                    continue
                try:
                    score = float(difficulty)
                except (ValueError, TypeError):
                    continue

                # Normalize score to 0-1 if needed
                if score > 1.0:
                    score = score / 100.0
                score = max(0.0, min(1.0, score))

                label = score_to_label(score)
                samples.append({
                    "text": text.strip(),
                    "label": label,
                    "source": f"easy2hard/{subset}",
                    "difficulty_score": round(score, 3),
                })
        except Exception as e:
            print(f"  {subset} failed: {e}")
            continue

    # Also try Codeforces subset (great for complex/reasoning)
    try:
        ds = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-Codeforces", split="eval")
        print(f"  E2H-Codeforces: {len(ds)} samples, columns: {ds.column_names}")
        for i, item in enumerate(ds):
            text = item.get("problem", item.get("description", ""))
            if not text or len(text.strip()) < 10:
                continue
            difficulty = item.get("difficulty", None)
            if difficulty is None:
                continue
            try:
                score = float(difficulty)
            except (ValueError, TypeError):
                continue
            if score > 1.0:
                score = score / 100.0
            score = max(0.0, min(1.0, score))
            label = score_to_label(score)
            samples.append({
                "text": text.strip(),
                "label": label,
                "source": "easy2hard/E2H-Codeforces",
                "difficulty_score": round(score, 3),
            })
    except Exception as e:
        print(f"  E2H-Codeforces failed: {e}")

    if samples:
        # Print distribution
        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        return save_to_industry_standard("easy2hard_bench.json", samples)
    else:
        print("  ⚠️  No samples collected")
        return 0


# ═══════════════════════════════════════════════════════════════
# 2. RouterBench — Multi-LLM performance data for routing
#    30K+ prompts with correctness labels across 11 models.
#    Derive difficulty from how many models get it right.
# ═══════════════════════════════════════════════════════════════
def download_routerbench():
    """Download RouterBench and derive difficulty from model agreement."""
    print("\n" + "=" * 60)
    print("2. RouterBench (30K+ prompts with multi-model performance)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("withmartian/routerbench", split="train")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            # Extract the prompt
            prompt = item.get("prompt", item.get("input", item.get("question", "")))
            if not prompt or len(str(prompt).strip()) < 10:
                continue

            # If prompt is a list (messages format), join them
            if isinstance(prompt, list):
                text_parts = []
                for msg in prompt:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if content:
                            text_parts.append(str(content))
                    elif isinstance(msg, str):
                        text_parts.append(msg)
                prompt = "\n\n".join(text_parts)

            prompt = str(prompt).strip()
            if len(prompt) < 10:
                continue

            # Derive difficulty from model performance columns
            # RouterBench has columns like 'gpt-4_correctness', 'llama-2-7b_correctness', etc.
            # If only big models get it right, it's harder
            correctness_cols = [col for col in item.keys() if "correctness" in col.lower() or "score" in col.lower()]

            if correctness_cols:
                scores = []
                for col in correctness_cols:
                    val = item.get(col)
                    if val is not None:
                        try:
                            scores.append(float(val))
                        except (ValueError, TypeError):
                            pass

                if scores:
                    avg_score = sum(scores) / len(scores)
                    # Lower average score = harder problem
                    if avg_score > 0.8:
                        label = "simple"
                    elif avg_score > 0.5:
                        label = "moderate"
                    elif avg_score > 0.25:
                        label = "complex"
                    else:
                        label = "reasoning"
                else:
                    # Use source benchmark as proxy
                    source = item.get("source", item.get("benchmark", ""))
                    if "gsm" in str(source).lower() or "math" in str(source).lower():
                        label = "complex"
                    elif "hellaswag" in str(source).lower() or "winogrande" in str(source).lower():
                        label = "moderate"
                    elif "mmlu" in str(source).lower():
                        label = "moderate"
                    else:
                        label = "moderate"  # default
            else:
                # Fallback: use benchmark source for labeling
                source = str(item.get("source", item.get("benchmark", "")))
                if "gsm" in source.lower() or "math" in source.lower() or "mbpp" in source.lower():
                    label = "complex"
                elif "mt-bench" in source.lower() or "mt_bench" in source.lower():
                    label = "complex"
                elif "hellaswag" in source.lower() or "winogrande" in source.lower():
                    label = "moderate"
                elif "mmlu" in source.lower():
                    label = "moderate"
                elif "arc" in source.lower():
                    label = "complex"
                else:
                    label = "moderate"

            samples.append({
                "text": prompt[:5000],  # Truncate very long prompts
                "label": label,
                "source": f"routerbench/{item.get('source', item.get('benchmark', 'unknown'))}",
            })

        # Print distribution
        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        return save_to_industry_standard("routerbench.json", samples)

    except Exception as e:
        print(f"  Failed to load RouterBench: {e}")
        print("  Trying alternative loading...")
        try:
            # Try loading specific configs
            for config in ["main", "default"]:
                try:
                    ds = load_dataset("withmartian/routerbench", config, split="train")
                    print(f"  Loaded config '{config}': {len(ds)} samples")
                    break
                except:
                    continue
        except Exception as e2:
            print(f"  All attempts failed: {e2}")
            return 0


# ═══════════════════════════════════════════════════════════════
# 3. LeetCode Dataset — coding problems with Easy/Medium/Hard
# ═══════════════════════════════════════════════════════════════
def download_leetcode():
    """Download LeetCode problems with built-in difficulty labels."""
    print("\n" + "=" * 60)
    print("3. LeetCode Dataset (coding with Easy/Medium/Hard labels)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("newfacade/LeetCodeDataset", split="train")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            # Get problem description
            text = item.get("content", item.get("description", item.get("problem", "")))
            title = item.get("title", "")
            difficulty = item.get("difficulty", "")

            if not text or len(str(text).strip()) < 20:
                continue

            # Prepend title as a user might phrase it
            prompt = f"Solve this coding problem: {title}\n\n{text}" if title else str(text)

            # Map LeetCode difficulty to our labels
            diff_lower = str(difficulty).lower()
            if "easy" in diff_lower:
                label = "simple"
            elif "medium" in diff_lower:
                label = "moderate"
            elif "hard" in diff_lower:
                label = "complex"
            else:
                continue  # Skip unknown difficulty

            samples.append({
                "text": prompt.strip()[:5000],
                "label": label,
                "source": "leetcode",
            })

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        return save_to_industry_standard("leetcode.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")
        # Try alternative LeetCode dataset
        try:
            ds = load_dataset("greengerong/leetcode", split="train")
            print(f"  Alternative: {len(ds)} samples, columns: {ds.column_names}")
            for i, item in enumerate(ds):
                text = item.get("content", item.get("description", ""))
                difficulty = item.get("difficulty", "")
                if not text or not difficulty:
                    continue
                diff_lower = str(difficulty).lower()
                if "easy" in diff_lower:
                    label = "simple"
                elif "medium" in diff_lower:
                    label = "moderate"
                elif "hard" in diff_lower:
                    label = "complex"
                else:
                    continue
                samples.append({
                    "text": str(text).strip()[:5000],
                    "label": label,
                    "source": "leetcode",
                })
            if samples:
                return save_to_industry_standard("leetcode.json", samples)
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
        return 0


# ═══════════════════════════════════════════════════════════════
# 4. Arena-Hard — curated hard prompts from Chatbot Arena
#    These are the prompts that best separate strong from weak models.
#    All labeled as complex/reasoning.
# ═══════════════════════════════════════════════════════════════
def download_arena_hard():
    """Download Arena-Hard prompts (curated difficult prompts)."""
    print("\n" + "=" * 60)
    print("4. Arena-Hard (curated hard prompts separating top models)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("lmarena-ai/arena-hard-auto", split="train")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            # Extract prompt from various possible column names
            prompt = None
            for key in ["prompt", "turns", "question", "input", "messages"]:
                val = item.get(key)
                if val:
                    if isinstance(val, list):
                        # Messages format
                        parts = []
                        for msg in val:
                            if isinstance(msg, dict):
                                content = msg.get("content", "")
                                if content:
                                    parts.append(str(content))
                            elif isinstance(msg, str):
                                parts.append(msg)
                        prompt = "\n\n".join(parts)
                    else:
                        prompt = str(val)
                    break

            if not prompt or len(prompt.strip()) < 20:
                continue

            # Arena-Hard prompts are specifically curated to be difficult
            # Split between complex and reasoning based on length/indicators
            text = prompt.strip()
            reasoning_indicators = [
                "prove", "derive", "step by step", "think carefully",
                "mathematical", "theorem", "induction", "logic",
                "formal proof", "contradict",
            ]
            is_reasoning = any(ind in text.lower() for ind in reasoning_indicators)
            label = "reasoning" if is_reasoning else "complex"

            samples.append({
                "text": text[:5000],
                "label": label,
                "source": "arena_hard",
            })

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("arena_hard.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")

    return 0


# ═══════════════════════════════════════════════════════════════
# 5. GAIR/Preference-Dissection — Chatbot Arena with task taxonomy
#    Real user prompts annotated with task type and complexity.
# ═══════════════════════════════════════════════════════════════
def download_preference_dissection():
    """Download GAIR Preference Dissection (annotated Arena conversations)."""
    print("\n" + "=" * 60)
    print("5. GAIR/Preference-Dissection (annotated Arena conversations)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("GAIR/Preference-Dissection", split="train")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            # Extract the first user prompt from the conversation
            conversation = item.get("conversation", item.get("messages", []))
            prompt = ""
            if isinstance(conversation, list):
                for msg in conversation:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user" and content:
                            prompt = str(content)
                            break
            elif isinstance(conversation, str):
                prompt = conversation

            if not prompt or len(prompt.strip()) < 15:
                # Try other fields
                prompt = str(item.get("prompt", item.get("input", item.get("question", ""))))

            if not prompt or len(prompt.strip()) < 15:
                continue

            # Use task type and difficulty annotations if available
            task_type = str(item.get("task", item.get("category", item.get("task_type", "")))).lower()
            difficulty = item.get("difficulty", item.get("complexity", ""))

            if difficulty:
                diff_str = str(difficulty).lower()
                if "easy" in diff_str or "simple" in diff_str:
                    label = "simple"
                elif "medium" in diff_str or "moderate" in diff_str:
                    label = "moderate"
                elif "hard" in diff_str or "complex" in diff_str:
                    label = "complex"
                elif "expert" in diff_str or "reasoning" in diff_str:
                    label = "reasoning"
                else:
                    # Try numeric
                    try:
                        score = float(difficulty)
                        if score < 0.3:
                            label = "simple"
                        elif score < 0.6:
                            label = "moderate"
                        elif score < 0.85:
                            label = "complex"
                        else:
                            label = "reasoning"
                    except (ValueError, TypeError):
                        # Infer from task type
                        if task_type in ("chat", "greeting", "translation"):
                            label = "simple"
                        elif task_type in ("coding", "math", "reasoning"):
                            label = "complex"
                        else:
                            label = "moderate"
            else:
                # Infer from task category
                if task_type in ("chat", "greeting", "translation", "qa"):
                    label = "simple"
                elif task_type in ("writing", "summarization", "explanation"):
                    label = "moderate"
                elif task_type in ("coding", "code", "math", "analysis"):
                    label = "complex"
                elif task_type in ("reasoning", "proof", "logic"):
                    label = "reasoning"
                else:
                    # Use winner model as proxy: if strong model won, prompt is harder
                    winner = str(item.get("winner", item.get("preference", ""))).lower()
                    if "gpt-4" in winner or "claude" in winner:
                        label = "complex"
                    else:
                        label = "moderate"

            samples.append({
                "text": prompt.strip()[:5000],
                "label": label,
                "source": "preference_dissection",
            })

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("preference_dissection.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")

    return 0


# ═══════════════════════════════════════════════════════════════
# 6. MT-Bench — multi-turn benchmark prompts (complex)
#    Curated by LMSYS for evaluating conversational abilities.
# ═══════════════════════════════════════════════════════════════
def download_mt_bench():
    """Download MT-Bench prompts (curated multi-turn evaluation prompts)."""
    print("\n" + "=" * 60)
    print("6. MT-Bench (multi-turn evaluation prompts)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            # MT-Bench has multi-turn prompts
            prompt = item.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "\n\n".join(str(p) for p in prompt if p)
            prompt = str(prompt).strip()

            if not prompt or len(prompt) < 15:
                continue

            # MT-Bench is designed to test advanced capabilities
            category = str(item.get("category", "")).lower()

            # Map MT-Bench categories to our complexity levels
            if category in ("writing", "roleplay"):
                label = "moderate"
            elif category in ("reasoning", "math"):
                label = "complex"
            elif category in ("coding",):
                label = "complex"
            elif category in ("extraction", "stem"):
                label = "moderate"
            elif category in ("humanities",):
                label = "moderate"
            else:
                label = "complex"  # MT-Bench is generally harder

            samples.append({
                "text": prompt[:5000],
                "label": label,
                "source": f"mt_bench/{category}",
            })

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("mt_bench.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")
        # Try alternative
        try:
            ds = load_dataset("lmsys/mt_bench_human_judgments", split="train")
            print(f"  Alternative (judgments): {len(ds)}, columns: {ds.column_names}")
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")

    return 0


# ═══════════════════════════════════════════════════════════════
# 7. IFEval — Instruction Following Evaluation (moderate/complex)
#    Prompts with explicit verifiable constraints.
# ═══════════════════════════════════════════════════════════════
def download_ifeval():
    """Download IFEval (instruction-following with constraints)."""
    print("\n" + "=" * 60)
    print("7. IFEval (instruction following with constraints)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("google/IFEval", split="train")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            prompt = item.get("prompt", item.get("input", ""))
            if not prompt or len(str(prompt).strip()) < 15:
                continue

            prompt = str(prompt).strip()

            # IFEval prompts have varying numbers of constraints
            # More constraints = harder to follow correctly
            num_constraints = item.get("num_constraints", 0)
            if not num_constraints:
                # Count constraint keywords
                constraint_keywords = ["must", "should", "exactly", "at least", "no more than",
                                       "include", "do not", "avoid", "format", "bullet"]
                num_constraints = sum(1 for kw in constraint_keywords if kw in prompt.lower())

            if num_constraints <= 1:
                label = "simple"
            elif num_constraints <= 3:
                label = "moderate"
            else:
                label = "complex"

            samples.append({
                "text": prompt[:5000],
                "label": label,
                "source": "ifeval",
            })

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("ifeval.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")

    return 0


# ═══════════════════════════════════════════════════════════════
# 8. Alpaca Eval — instruction-following prompts (simple/moderate)
# ═══════════════════════════════════════════════════════════════
def download_alpaca_eval():
    """Download AlpacaEval prompts (instruction-following evaluation)."""
    print("\n" + "=" * 60)
    print("8. AlpacaEval (instruction-following prompts)")
    print("=" * 60)

    samples = []

    try:
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        print(f"  Size: {len(ds)}, columns: {ds.column_names}")

        for i, item in enumerate(ds):
            instruction = item.get("instruction", item.get("prompt", ""))
            if not instruction or len(str(instruction).strip()) < 15:
                continue

            text = str(instruction).strip()

            # Classify based on complexity indicators
            complexity_indicators = {
                "simple": ["what is", "define", "name", "list", "who is", "when was",
                          "translate", "hello", "hi", "thanks"],
                "moderate": ["explain", "compare", "summarize", "describe", "write a",
                           "create a", "how to", "suggest", "recommend"],
                "complex": ["design", "implement", "analyze", "evaluate", "develop",
                          "architect", "optimize", "debug", "refactor", "build a system"],
            }

            text_lower = text.lower()
            label = "moderate"  # default

            if len(text) < 50 and any(kw in text_lower for kw in complexity_indicators["simple"]):
                label = "simple"
            elif any(kw in text_lower for kw in complexity_indicators["complex"]):
                label = "complex"
            elif any(kw in text_lower for kw in complexity_indicators["simple"]):
                label = "simple"

            samples.append({
                "text": text[:5000],
                "label": label,
                "source": "alpaca_eval",
            })

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("alpaca_eval.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")

    return 0


# ═══════════════════════════════════════════════════════════════
# 9. OpenOrca (subset) — diverse instruction+response pairs
#    Uses system prompts + user prompts — directly matches our use case.
# ═══════════════════════════════════════════════════════════════
def download_openorca_subset():
    """Download a balanced subset of OpenOrca with system+user prompts."""
    print("\n" + "=" * 60)
    print("9. OpenOrca subset (system + user prompts)")
    print("=" * 60)

    samples = []

    try:
        # Load a small streaming subset to avoid downloading the full 4M dataset
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        print(f"  Loading streaming subset...")

        count = 0
        max_per_label = 500  # 500 per category = 2000 total max
        label_counts = {"simple": 0, "moderate": 0, "complex": 0, "reasoning": 0}

        for item in ds:
            if count >= 5000:  # Sample from first 5000 then stop
                break
            count += 1

            system_prompt = item.get("system_prompt", "")
            question = item.get("question", item.get("input", ""))

            if not question or len(str(question).strip()) < 15:
                continue

            # Combine system + user for training (matches our inference pattern)
            text_parts = []
            if system_prompt:
                text_parts.append(str(system_prompt))
            text_parts.append(str(question))
            text = "\n\n".join(text_parts)

            # Classify based on source dataset indicators in the system prompt
            sys_lower = str(system_prompt).lower()
            q_lower = str(question).lower()

            if "step-by-step" in sys_lower or "chain of thought" in sys_lower:
                label = "complex"
            elif "prove" in q_lower or "derive" in q_lower or "formal" in q_lower:
                label = "reasoning"
            elif len(question) > 500:
                label = "complex"
            elif len(question) > 200:
                label = "moderate"
            elif any(kw in q_lower for kw in ["what is", "who", "when", "where", "define"]):
                label = "simple"
            else:
                label = "moderate"

            # Balance the labels
            if label_counts[label] >= max_per_label:
                continue

            label_counts[label] += 1
            samples.append({
                "text": text.strip()[:5000],
                "label": label,
                "source": "openorca",
            })

            if all(c >= max_per_label for c in label_counts.values()):
                break

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("openorca_subset.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")

    return 0


# ═══════════════════════════════════════════════════════════════
# 10. Big Bench Hard (BBH) — tasks that are hard for LLMs
# ═══════════════════════════════════════════════════════════════
def download_bbh():
    """Download Big Bench Hard tasks (complex reasoning)."""
    print("\n" + "=" * 60)
    print("10. Big Bench Hard (tasks hard for LLMs)")
    print("=" * 60)

    samples = []

    try:
        # BBH has multiple subtasks
        subtasks = [
            "boolean_expressions", "causal_judgement", "date_understanding",
            "disambiguation_qa", "formal_fallacies", "geometric_shapes",
            "hyperbaton", "logical_deduction_five_objects",
            "logical_deduction_seven_objects", "movie_recommendation",
            "multistep_arithmetic_two", "navigate", "object_counting",
            "penguins_in_a_table", "reasoning_about_colored_objects",
            "ruin_names", "salient_translation_error_detection",
            "snarks", "sports_understanding", "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects", "web_of_lies",
            "word_sorting",
        ]

        for subtask in subtasks:
            try:
                ds = load_dataset("lukaemon/bbh", subtask, split="test")
                for i, item in enumerate(ds):
                    if i >= 20:  # 20 per subtask
                        break
                    text = item.get("input", item.get("question", ""))
                    if not text or len(str(text).strip()) < 15:
                        continue

                    # BBH is specifically designed to be hard
                    # Math/logic subtasks are reasoning, others are complex
                    reasoning_tasks = [
                        "formal_fallacies", "logical_deduction", "boolean_expressions",
                        "multistep_arithmetic", "tracking_shuffled", "navigate",
                        "web_of_lies",
                    ]
                    is_reasoning = any(rt in subtask for rt in reasoning_tasks)
                    label = "reasoning" if is_reasoning else "complex"

                    samples.append({
                        "text": str(text).strip()[:5000],
                        "label": label,
                        "source": f"bbh/{subtask}",
                    })
            except Exception:
                continue

        from collections import Counter
        dist = Counter(s["label"] for s in samples)
        print(f"  Distribution: {dict(dist)}")
        if samples:
            return save_to_industry_standard("bbh.json", samples)

    except Exception as e:
        print(f"  Failed: {e}")

    return 0


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("DOWNLOADING HIGH-QUALITY ROUTING/COMPLEXITY DATASETS")
    print("=" * 60)
    print(f"Output directory: {BASE_DIR}")
    print(f"Target format: {{text, label}} with label in [simple, moderate, complex, reasoning]")

    total = 0
    results = {}

    download_fns = [
        ("Easy2Hard-Bench", download_easy2hard_bench),
        ("RouterBench", download_routerbench),
        ("LeetCode", download_leetcode),
        ("Arena-Hard", download_arena_hard),
        ("Preference-Dissection", download_preference_dissection),
        ("MT-Bench", download_mt_bench),
        ("IFEval", download_ifeval),
        ("AlpacaEval", download_alpaca_eval),
        ("OpenOrca subset", download_openorca_subset),
        ("Big Bench Hard", download_bbh),
    ]

    for name, fn in download_fns:
        try:
            count = fn()
            results[name] = count
            total += count
        except Exception as e:
            print(f"\n  ❌ {name} FAILED: {e}")
            results[name] = 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {name}: {count} samples")
    print(f"\n  TOTAL: {total} new samples")
    print("=" * 60)

    if total > 0:
        print("\nNext steps:")
        print("  1. Update train_tfidf.py to include the new datasets")
        print("  2. Run: python benchmarks/classifier/train_tfidf.py")
        print("  3. Check accuracy improvement in classification report")


if __name__ == "__main__":
    main()
