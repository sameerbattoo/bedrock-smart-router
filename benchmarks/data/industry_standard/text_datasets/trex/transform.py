#!/usr/bin/env python3
"""
Transform the T-REx dataset into ModelEval's built-in evaluation format.

Produces ONE project with TWO evaluations:
  1. Accuracy (Real-World Knowledge) — clean passages, tests factual extraction
  2. Robustness — perturbed passages (typos, synonym swaps, reworded sentences),
     tests whether accuracy holds under noisy input

Each evaluation gets ~200 variable sets with expected answers.

T-REx source: https://hadyelsahar.github.io/t-rex/
Paper: ElSahar et al., LREC 2018
"""
import json
import os
import sys
import random
import re
import string
from collections import defaultdict

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_VARIABLE_SETS = 200       # cap total test cases per evaluation
MAX_PASSAGE_CHARS = 1500      # truncate very long passages
SEED = 42                     # reproducibility
MAX_PER_PREDICATE = 5         # cap per predicate for diversity

# Wikidata property ID -> human-readable label for the most common predicates.
# ~92% of T-REx triples have null predicate surfaceform and only a URI like P17.
WIKIDATA_PROPERTY_LABELS = {
    "P17":   "country",
    "P47":   "shares border with",
    "P31":   "instance of",
    "P279":  "subclass of",
    "P27":   "country of citizenship",
    "P131":  "located in",
    "P530":  "diplomatic relation",
    "P361":  "part of",
    "P106":  "occupation",
    "P527":  "has part",
    "P569":  "date of birth",
    "P570":  "date of death",
    "P463":  "member of",
    "P37":   "official language",
    "P1411": "nominated for",
    "P150":  "contains",
    "P1412": "languages spoken",
    "P136":  "genre",
    "P138":  "named after",
    "P36":   "capital",
    "P19":   "place of birth",
    "P20":   "place of death",
    "P495":  "country of origin",
    "P159":  "headquarters location",
    "P175":  "performer",
    "P264":  "record label",
    "P276":  "location",
    "P30":   "continent",
    "P35":   "head of state",
    "P6":    "head of government",
    "P40":   "child",
    "P22":   "father",
    "P25":   "mother",
    "P26":   "spouse",
    "P39":   "position held",
    "P69":   "educated at",
    "P108":  "employer",
    "P112":  "founded by",
    "P118":  "league",
    "P127":  "owned by",
    "P135":  "movement",
    "P137":  "operator",
    "P140":  "religion",
    "P155":  "follows",
    "P156":  "followed by",
    "P161":  "cast member",
    "P162":  "producer",
    "P166":  "award received",
    "P170":  "creator",
    "P171":  "parent taxon",
    "P176":  "manufacturer",
    "P178":  "developer",
    "P180":  "depicts",
    "P184":  "doctoral advisor",
    "P190":  "twinned with",
    "P194":  "legislative body",
    "P206":  "located next to body of water",
    "P241":  "military branch",
    "P263":  "official residence",
    "P272":  "production company",
    "P355":  "subsidiary",
    "P364":  "original language",
    "P400":  "platform",
    "P403":  "mouth of watercourse",
    "P449":  "original network",
    "P466":  "occupant",
    "P488":  "chairperson",
    "P551":  "residence",
    "P607":  "conflict",
    "P674":  "characters",
    "P706":  "located on terrain feature",
    "P737":  "influenced by",
    "P740":  "location of formation",
    "P749":  "parent organization",
    "P800":  "notable work",
    "P840":  "narrative location",
    "P921":  "main subject",
    "P937":  "work location",
    "P1001": "applies to jurisdiction",
    "P1376": "capital of",
}

# Skip words — pronouns and articles that make bad subjects/objects
_SKIP_WORDS = {
    "he", "she", "it", "they", "him", "her", "his",
    "its", "them", "this", "that", "the", "a", "an",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
ACCURACY_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer factual questions accurately "
    "and concisely based on the provided passage. If the answer is not in the "
    "passage, say so."
)

ACCURACY_USER_PROMPTS = [
    (
        "Given the following passage:\n\n{{passage}}\n\n"
        "Question: What is the {{predicate}} of {{subject}}?\n"
        "Answer concisely."
    ),
]

ROBUSTNESS_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer factual questions accurately "
    "and concisely based on the provided passage. The passage may contain "
    "minor errors, typos, or unusual phrasing — focus on the factual content. "
    "If the answer is not in the passage, say so."
)

ROBUSTNESS_USER_PROMPTS = [
    (
        "Given the following passage:\n\n{{passage}}\n\n"
        "Question: What is the {{predicate}} of {{subject}}?\n"
        "Answer concisely."
    ),
]


# ---------------------------------------------------------------------------
# Text perturbation functions for robustness evaluation
# ---------------------------------------------------------------------------

def _insert_typos(text: str, rate: float = 0.03) -> str:
    """Insert random character-level typos at the given rate."""
    rng = random.Random()  # uses module-level seed
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "insert", "replace"])
            if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "drop":
                chars[i] = ""
            elif op == "insert":
                chars[i] = chars[i] + rng.choice(string.ascii_lowercase)
            elif op == "replace":
                chars[i] = rng.choice(string.ascii_lowercase)
    return "".join(chars)


# Common word-level synonym swaps for robustness perturbation
_SYNONYMS = {
    "located": ["situated", "found", "based"],
    "known": ["recognized", "regarded", "acknowledged"],
    "born": ["birthed", "delivered", "brought into the world"],
    "founded": ["established", "created", "started"],
    "largest": ["biggest", "greatest", "most extensive"],
    "capital": ["main city", "chief city", "seat of government"],
    "country": ["nation", "state", "land"],
    "city": ["town", "municipality", "urban area"],
    "member": ["part", "affiliate", "participant"],
    "official": ["formal", "authorized", "recognized"],
    "border": ["boundary", "frontier", "edge"],
    "approximately": ["roughly", "about", "around"],
    "important": ["significant", "notable", "major"],
    "famous": ["well-known", "renowned", "celebrated"],
    "ancient": ["old", "historic", "age-old"],
    "modern": ["contemporary", "current", "present-day"],
    "small": ["little", "minor", "compact"],
    "large": ["big", "sizable", "substantial"],
}


def _swap_synonyms(text: str, rate: float = 0.15) -> str:
    """Replace some words with synonyms."""
    rng = random.Random()
    words = text.split()
    for i, w in enumerate(words):
        clean = w.strip(".,;:!?()\"'").lower()
        if clean in _SYNONYMS and rng.random() < rate:
            replacement = rng.choice(_SYNONYMS[clean])
            # Preserve original casing of first char
            if w[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            trailing = ""
            while w and w[-1] in ".,;:!?()\"'":
                trailing = w[-1] + trailing
                w = w[:-1]
            words[i] = replacement + trailing
    return " ".join(words)


def _perturb_passage(passage: str) -> str:
    """Apply a combination of perturbations to a passage for robustness testing."""
    text = _swap_synonyms(passage)
    text = _insert_typos(text)
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_surface(surface: str) -> str:
    """Normalize a surface form string."""
    if not surface:
        return ""
    s = surface.strip().replace("_", " ")
    if s.startswith("http"):
        s = s.rsplit("/", 1)[-1].replace("_", " ")
    return s


def _predicate_label(predicate_entity: dict) -> str:
    """Get a human-readable predicate label from the triple's predicate entity."""
    sf = predicate_entity.get("surfaceform", "")
    if sf:
        return _clean_surface(sf)
    uri = predicate_entity.get("uri", "")
    if "/" in uri:
        prop_id = uri.rsplit("/", 1)[-1]
        if prop_id in WIKIDATA_PROPERTY_LABELS:
            return WIKIDATA_PROPERTY_LABELS[prop_id]
        return ""
    return ""


def _build_passage(doc: dict, sentence_id: int) -> str:
    """Build a passage around the target sentence (±1 sentence for context)."""
    text = doc.get("text", "")
    boundaries = doc.get("sentences_boundaries", [])
    if not boundaries:
        return text[:MAX_PASSAGE_CHARS].strip()

    start_idx = max(0, sentence_id - 1)
    end_idx = min(len(boundaries), sentence_id + 2)

    start_char = int(boundaries[start_idx][0])
    end_char = int(boundaries[end_idx - 1][1])
    passage = text[start_char:end_char].strip()

    if len(passage) > MAX_PASSAGE_CHARS:
        passage = passage[:MAX_PASSAGE_CHARS].rstrip() + "…"
    return passage


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def load_raw_documents():
    """Load all T-REx JSON documents from the raw directory."""
    docs = []
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: Raw directory not found at {RAW_DIR}")
        print("Run download.py first.")
        sys.exit(1)

    json_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    if not json_files:
        print(f"ERROR: No JSON files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s)")
    for fpath in sorted(json_files):
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                docs.extend(data)
            elif isinstance(data, dict):
                docs.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Skipping {fpath}: {e}")

    print(f"Loaded {len(docs)} documents total")
    return docs


def extract_triples(docs):
    """Extract high-quality (subject, predicate, object, passage) tuples."""
    candidates = []
    predicate_counts = defaultdict(int)

    for doc in docs:
        for triple in doc.get("triples", []):
            subj = triple.get("subject", {})
            pred = triple.get("predicate", {})
            obj = triple.get("object", {})

            subject_sf = _clean_surface(subj.get("surfaceform", ""))
            predicate_label = _predicate_label(pred)
            object_sf = _clean_surface(obj.get("surfaceform", ""))

            if not subject_sf or not object_sf or not predicate_label:
                continue
            if len(subject_sf) < 2 or len(object_sf) < 2:
                continue
            if object_sf.lower() in _SKIP_WORDS or subject_sf.lower() in _SKIP_WORDS:
                continue
            if predicate_counts[predicate_label] >= MAX_PER_PREDICATE:
                continue

            sentence_id = triple.get("sentence_id", 0)
            passage = _build_passage(doc, sentence_id)
            if len(passage) < 20:
                continue
            if object_sf.lower() not in passage.lower():
                continue

            predicate_counts[predicate_label] += 1
            candidates.append({
                "passage": passage,
                "subject": subject_sf,
                "predicate": predicate_label,
                "expected_answer": object_sf,
                "doc_title": doc.get("title", ""),
            })

    print(f"Extracted {len(candidates)} candidate triples "
          f"across {len(predicate_counts)} unique predicates")
    return candidates


# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

def _build_evaluation(name, system_prompt, user_prompts, candidates, quality_metrics,
                      perturb=False):
    """Build a single evaluation dict from candidates."""
    prompt_variable_sets = []
    expected_answers = []
    num_prompts = len(user_prompts)

    for c in candidates:
        passage = c["passage"]
        if perturb:
            passage = _perturb_passage(passage)

        prompt_variable_sets.append({
            "passage": passage,
            "subject": c["subject"],
            "predicate": c["predicate"],
        })
        expected_answers.append([c["expected_answer"]] * num_prompts)

    return {
        "useCaseName": name,
        "systemPrompt": system_prompt,
        "userPrompts": user_prompts,
        "promptVariableSets": prompt_variable_sets,
        "expectedAnswers": expected_answers,
        "models": [],
        "judgeModels": [],
        "qualityMetrics": quality_metrics,
        "isRAG": False,
    }


def build_modeleval_json(candidates):
    """Build the final output with one project and two evaluations."""
    random.seed(SEED)
    random.shuffle(candidates)

    # Split candidates: first half for accuracy, second half for robustness
    # This ensures no overlap between the two evals
    total_needed = MAX_VARIABLE_SETS * 2
    pool = candidates[:total_needed]
    accuracy_candidates = pool[:MAX_VARIABLE_SETS]
    robustness_candidates = pool[MAX_VARIABLE_SETS:MAX_VARIABLE_SETS * 2]

    print(f"Accuracy eval: {len(accuracy_candidates)} test cases")
    print(f"Robustness eval: {len(robustness_candidates)} test cases")

    # Log predicate diversity
    for label, cands in [("Accuracy", accuracy_candidates), ("Robustness", robustness_candidates)]:
        pred_dist = defaultdict(int)
        for c in cands:
            pred_dist[c["predicate"]] += 1
        top = sorted(pred_dist.items(), key=lambda x: -x[1])[:5]
        print(f"  {label} top predicates: {top}")

    accuracy_eval = _build_evaluation(
        name="Accuracy — Real-World Knowledge (T-REx)",
        system_prompt=ACCURACY_SYSTEM_PROMPT,
        user_prompts=ACCURACY_USER_PROMPTS,
        candidates=accuracy_candidates,
        quality_metrics=["correctness"],
        perturb=False,
    )

    robustness_eval = _build_evaluation(
        name="Robustness — Real-World Knowledge (T-REx)",
        system_prompt=ROBUSTNESS_SYSTEM_PROMPT,
        user_prompts=ROBUSTNESS_USER_PROMPTS,
        candidates=robustness_candidates,
        quality_metrics=["correctness"],
        perturb=True,
    )

    output = {
        "dataset": {
            "id": "trex",
            "name": "T-REx — Real-World Knowledge",
            "description": (
                "Evaluates a model's ability to extract factual knowledge from text. "
                "Based on the T-REx dataset which aligns natural language with "
                "Wikidata/DBpedia knowledge base triples. Includes two evaluations: "
                "(1) Accuracy — tests knowledge extraction from clean passages, "
                "(2) Robustness — tests whether accuracy holds under noisy/perturbed input."
            ),
            "source": "https://hadyelsahar.github.io/t-rex/",
            "citation": (
                "ElSahar et al., 'T-REx: A Large Scale Alignment of Natural Language "
                "with Knowledge Base Triples', LREC 2018"
            ),
            "taskType": "General text generation",
            "metrics": ["Correctness"],
        },
        "project": {
            "customerName": "T-REx — Real-World Knowledge",
            "description": (
                "Built-in benchmark: Evaluates real-world knowledge extraction "
                "using the T-REx dataset (LREC 2018). Contains two evaluations — "
                "Accuracy (clean passages) and Robustness (perturbed passages)."
            ),
        },
        "evaluations": [accuracy_eval, robustness_eval],
    }
    return output


def main():
    print("=" * 60)
    print("T-REx → ModelEval Transform")
    print("=" * 60)

    docs = load_raw_documents()
    candidates = extract_triples(docs)

    if not candidates:
        print("ERROR: No valid triples extracted. Check the raw data.")
        sys.exit(1)

    output = build_modeleval_json(candidates)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "trex_modeleval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\nOutput written to {out_path}")
    for i, ev in enumerate(output["evaluations"]):
        print(f"  Eval {i+1}: {ev['useCaseName']}")
        print(f"    Variable sets: {len(ev['promptVariableSets'])}")
        print(f"    Prompt templates: {len(ev['userPrompts'])}")
        print(f"    Expected answers: {len(ev['expectedAnswers'])}")

    # Preview file with 3 samples from each eval
    preview = {**output}
    preview["evaluations"] = []
    for ev in output["evaluations"]:
        pev = {**ev}
        pev["promptVariableSets"] = ev["promptVariableSets"][:3]
        pev["expectedAnswers"] = ev["expectedAnswers"][:3]
        preview["evaluations"].append(pev)

    preview_path = os.path.join(OUTPUT_DIR, "trex_preview.json")
    with open(preview_path, "w", encoding="utf-8") as fh:
        json.dump(preview, fh, indent=2, ensure_ascii=False)
    print(f"\n  Preview (3 samples each): {preview_path}")


if __name__ == "__main__":
    main()
