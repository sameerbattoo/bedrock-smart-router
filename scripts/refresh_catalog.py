#!/usr/bin/env python3
"""Auto-refresh the model catalog from AWS APIs + Artificial Analysis + LiteLLM.

Discovers all available Bedrock models, probes their capabilities,
fetches pricing, and generates a complete models.json catalog.

Usage:
    # Quick refresh (skip probes, use cached AA data) — ~1s
    python scripts/refresh_catalog.py --skip-probes --aa-cache scripts/_aa_models.json --write

    # Full refresh with probes (~60s, parallel)
    python scripts/refresh_catalog.py --aa-cache scripts/_aa_models.json --write

    # Full refresh with fresh AA data from API
    python scripts/refresh_catalog.py --aa-key YOUR_KEY --write

    # Overwrite production catalog
    python scripts/refresh_catalog.py --aa-cache scripts/_aa_models.json --write --output bedrock_smart_router/data/models.json

Sources:
    - AWS Bedrock ListFoundationModels API
    - AWS Bedrock ListInferenceProfiles API
    - LiteLLM model_prices_and_context_window.json (GitHub, for token limits + pricing)
    - Artificial Analysis Intelligence Index API (for quality baselines)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy urllib3 connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

CATALOG_PATH = Path(__file__).parent.parent / "bedrock_smart_router" / "data" / "models.json"

# ── Geography prefixes for CRIS inference profiles ──────────────────
GEO_PREFIXES = ("us.", "eu.", "ap.", "global.")

# ── Family detection from model ID ──────────────────────────────────
FAMILY_PATTERNS = {
    "anthropic": "anthropic",
    "amazon": "amazon",
    "meta": "meta",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "cohere": "cohere",
    "ai21": "ai21",
    "stability": "stability",
    "google": "google",
    "nvidia": "nvidia",
    "writer": "writer",
    "openai": "openai",
    "minimax": "minimax",
    "zai": "zai",
    "moonshotai": "moonshot",
    "moonshot": "moonshot",
    "qwen": "qwen",
    "twelvelabs": "twelvelabs",
}

# ── Tier derivation from multiple signals ───────────────────────────

# Name-based tier signals (provider naming conventions)
_REASONING_NAME_SIGNALS = ["r1", "thinking", "reasoning"]
_MID_NAME_SIGNALS = ["pro", "large", "sonnet", "maverick"]
_LITE_NAME_SIGNALS = ["lite", "haiku", "scout", "ministral", "mini"]
_MICRO_NAME_SIGNALS = ["micro", "nano"]


def _get_model_size_b(model_id: str, display_name: str) -> int:
    """Extract parameter count in billions from model name/id."""
    text = f"{model_id} {display_name}".lower()
    match = re.search(r"(\d+)b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)x(\d+)b", text)
    if match:
        return int(match.group(1)) * int(match.group(2))
    return 0


def derive_tier(quality_baseline: float, model_id: str, display_name: str = "",
                price_in: float = 0, caps: dict | None = None) -> str:
    """Derive tier from multiple signals: name, size, price, quality, capabilities.

    Rules (in priority order):
    1. Reasoning: quality >= 50 OR name contains reasoning indicators
    2. Heavy: expensive (>= $4/M input) + full capabilities (cache + thinking)
    3. Micro: name says micro/nano OR (small model + cheap + low quality)
    4. Lite: name says lite/haiku/scout/mini OR small model (<=14B) + cheap
    5. Mid: name says pro/large/sonnet/maverick OR large model (>=70B) OR quality >= 15
    6. Default: lite
    """
    caps = caps or {}
    name = display_name.lower()
    name_tokens = name.replace("-", " ").replace("_", " ").split()
    model_size = _get_model_size_b(model_id, display_name)
    has_cache = caps.get("prompt_caching", False)
    has_thinking = caps.get("extended_thinking", False)

    # Rule 1: Reasoning — frontier reasoning models
    if quality_baseline >= 50:
        return "reasoning"
    if any(s in name_tokens for s in _REASONING_NAME_SIGNALS):
        return "reasoning"

    # Rule 2: Heavy — expensive + full capabilities
    if price_in >= 0.004 and has_cache and has_thinking:
        return "heavy"

    # Rule 3: Micro — smallest/cheapest models
    if any(s in name_tokens for s in _MICRO_NAME_SIGNALS):
        return "micro"
    if quality_baseline < 8 and model_size <= 8:
        return "micro"
    if model_size <= 8 and price_in < 0.0003 and quality_baseline < 12:
        return "micro"

    # Rule 4: Lite — lightweight models
    if any(s in name_tokens for s in _LITE_NAME_SIGNALS):
        return "lite"
    if model_size > 0 and model_size <= 14 and price_in < 0.0003:
        return "lite"

    # Rule 5: Mid — mainstream capable models
    if any(s in name_tokens for s in _MID_NAME_SIGNALS):
        return "mid"
    if model_size >= 70:
        return "mid"
    if quality_baseline >= 15 and price_in >= 0.0002:
        return "mid"
    if caps.get("output_per_1k", 0) >= 0.001:
        return "mid"

    # Default: lite
    return "lite"


def strip_geo_prefix(model_id: str) -> str:
    for prefix in GEO_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def detect_family(model_id: str) -> str:
    base = strip_geo_prefix(model_id).lower()
    for pattern, family in FAMILY_PATTERNS.items():
        if base.startswith(pattern):
            return family
    return "unknown"


# ── Step 1: Discover models from Bedrock API ────────────────────────

def discover_models(bedrock_client: Any, region: str) -> list[dict]:
    """List all foundation models available in the region.

    Filters:
    - Only ACTIVE lifecycle (skip LEGACY/deprecated)
    - Only TEXT output models (skip embedding, image-only)
    - Only ON_DEMAND or INFERENCE_PROFILE (skip PROVISIONED-only)
    - Skip context-window variants (e.g., model-v1:0:128k)
    - Skip rerank/safeguard/embedding utility models
    """
    logger.info("Step 1: Discovering models via ListFoundationModels...")
    try:
        response = bedrock_client.list_foundation_models()
    except ClientError as e:
        logger.error(f"Failed to list models: {e}")
        return []

    models = []
    for summary in response.get("modelSummaries", []):
        model_id = summary.get("modelId", "")
        if not model_id:
            continue

        # Filter: Only ACTIVE lifecycle
        status = summary.get("modelLifecycle", {}).get("status", "")
        if status != "ACTIVE":
            continue

        # Filter: Must output TEXT
        output_modalities = summary.get("outputModalities", [])
        if "TEXT" not in output_modalities:
            continue

        # Filter: Must support ON_DEMAND or INFERENCE_PROFILE
        inference_types = summary.get("inferenceTypesSupported", [])
        if not any(t in inference_types for t in ["ON_DEMAND", "INFERENCE_PROFILE"]):
            continue

        # Filter: Skip context-window variants (e.g., model-v1:0:128k)
        parts = model_id.split(":")
        if len(parts) > 2 and parts[-1].endswith("k"):
            continue

        # Filter: Skip rerank/safeguard/embedding utility models
        name_lower = model_id.lower()
        if any(skip in name_lower for skip in ["rerank", "safeguard", "embed"]):
            continue

        # Filter: Skip speech/audio models and video-only understanding models
        # - SPEECH/AUDIO input = audio model (e.g., Voxtral)
        # - VIDEO input without IMAGE = video-specific model (e.g., Pegasus)
        # - TEXT + IMAGE + VIDEO = multimodal LLM (keep, e.g., Nova Pro)
        input_modalities = summary.get("inputModalities", [])
        if "TEXT" not in input_modalities:
            continue
        if "SPEECH" in input_modalities or "AUDIO" in input_modalities:
            continue
        if "VIDEO" in input_modalities and "IMAGE" not in input_modalities:
            continue
        has_vision = "IMAGE" in input_modalities
        has_document = has_vision  # Bedrock Converse handles documents for all vision models
        streaming = summary.get("responseStreamingSupported", True)

        models.append({
            "model_id": model_id,
            "model_name": summary.get("modelName", model_id),
            "provider_name": summary.get("providerName", ""),
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "has_vision": has_vision,
            "has_document": has_document,
            "streaming": streaming,
            "inference_types": inference_types,
            "model_lifecycle": status,
        })

    logger.info(f"  Found {len(models)} active text-capable models")
    return models


# ── Step 2: Discover CRIS inference profiles ────────────────────────

def discover_profiles(bedrock_client: Any) -> dict[str, list[str]]:
    """List inference profiles and map base model → profile IDs."""
    logger.info("Step 2: Discovering inference profiles...")
    profiles: dict[str, list[str]] = {}

    try:
        paginator = bedrock_client.get_paginator("list_inference_profiles")
        for page in paginator.paginate():
            for profile in page.get("inferenceProfileSummaries", []):
                profile_id = profile.get("inferenceProfileId", "")
                # Map to base model
                base = strip_geo_prefix(profile_id)
                if base not in profiles:
                    profiles[base] = []
                profiles[base].append(profile_id)
    except ClientError as e:
        logger.warning(f"Failed to list inference profiles: {e}")
    except Exception as e:
        logger.warning(f"Inference profiles not available: {e}")

    logger.info(f"  Found profiles for {len(profiles)} base models")
    return profiles


# ── Step 3: Fetch token limits from LiteLLM ─────────────────────────

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
LITELLM_CACHE_PATH = Path(__file__).parent / "_litellm_models.json"


def fetch_litellm_data() -> dict[str, dict]:
    """Download the latest LiteLLM model data from GitHub.

    Always downloads fresh and overwrites the local cache.
    Returns a dict keyed by model_id with max_input_tokens, max_output_tokens, pricing.
    """

    logger.info("Step 3: Fetching token limits from LiteLLM (GitHub)...")
    try:
        resp = requests.get(LITELLM_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Save to local cache (overwrite)
        LITELLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LITELLM_CACHE_PATH, "w") as f:
            json.dump(data, f)
        logger.info(f"  Downloaded {len(data)} models, saved to {LITELLM_CACHE_PATH.name}")

    except Exception as e:
        logger.warning(f"  Failed to download from GitHub: {e}")
        # Fall back to local cache if available
        if LITELLM_CACHE_PATH.exists():
            logger.info(f"  Using local cache: {LITELLM_CACHE_PATH}")
            with open(LITELLM_CACHE_PATH) as f:
                data = json.load(f)
        else:
            return {}

    # Extract Bedrock models — LiteLLM uses the model_id directly as key
    # e.g., "us.anthropic.claude-sonnet-4-6", "us.amazon.nova-pro-v1:0"
    result: dict[str, dict] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        max_input = entry.get("max_input_tokens")
        max_output = entry.get("max_output_tokens")
        if max_input is None and max_output is None:
            continue
        result[key] = {
            "max_input_tokens": max_input,
            "max_output_tokens": max_output,
            "input_cost_per_token": entry.get("input_cost_per_token"),
            "output_cost_per_token": entry.get("output_cost_per_token"),
            "cache_read_input_token_cost": entry.get("cache_read_input_token_cost"),
            "cache_creation_input_token_cost": entry.get("cache_creation_input_token_cost"),
        }

    logger.info(f"  Extracted token limits for {len(result)} models")
    return result


# ── Step 4: Probe capabilities ──────────────────────────────────────

def _probe_converse_support(client: Any, model_id: str) -> bool:
    """Test if model supports the Converse API."""
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 5},
        )
        return True
    except ClientError as e:
        msg = str(e).lower()
        if "throttl" in msg:
            return True
        if any(phrase in msg for phrase in [
            "does not support converse", "is not supported with converse",
            "not supported for this model", "model identifier is invalid",
            "provided model identifier is invalid",
        ]):
            return False
        return True  # Other errors (validation, content) = Converse works
    except Exception:
        return False


def _probe_tool_use(client: Any, model_id: str) -> bool:
    """Test if model supports tool use (function calling)."""
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "What is 2+2?"}]}],
            toolConfig={
                "tools": [{
                    "toolSpec": {
                        "name": "calculator",
                        "description": "Performs math",
                        "inputSchema": {"json": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        }},
                    }
                }]
            },
            inferenceConfig={"maxTokens": 20},
        )
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = str(e).lower()
        if "throttl" in msg:
            return True
        if "tool" in msg or "not supported" in msg or "does not support" in msg:
            return False
        if "internal" in code.lower():
            return False  # InternalServerException = can't handle tools
        return False
    except Exception:
        return False


def _probe_streaming_tool_use(client: Any, model_id: str) -> bool:
    """Test if model supports streaming with tool use."""
    try:
        resp = client.converse_stream(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "What is 2+2?"}]}],
            toolConfig={
                "tools": [{
                    "toolSpec": {
                        "name": "calculator",
                        "description": "Performs math",
                        "inputSchema": {"json": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        }},
                    }
                }]
            },
            inferenceConfig={"maxTokens": 20},
        )
        for event in resp.get("stream", []):
            break
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = str(e).lower()
        if "throttl" in msg:
            return True
        if "tool" in msg or "not supported" in msg or "streaming" in msg:
            return False
        if "internal" in code.lower():
            return False
        return False
    except Exception:
        return False


def _probe_extended_thinking(client: Any, model_id: str) -> bool:
    """Test if model supports extended thinking."""
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            additionalModelRequestFields={
                "thinking": {"type": "enabled", "budget_tokens": 100}
            },
            inferenceConfig={"maxTokens": 1024},
        )
        return True
    except ClientError as e:
        msg = str(e).lower()
        if "throttl" in msg:
            return True
        if "budget_tokens" in msg or ("thinking" in msg and "must be" in msg):
            return True  # Model knows about thinking, just param mismatch
        if any(phrase in msg for phrase in [
            "not supported", "unknown field", "does not support", "unrecognized",
        ]):
            return False
        if "validation" in msg:
            return False
        return False
    except Exception:
        return False


def _probe_priority_tier(client: Any, model_id: str) -> bool:
    """Test if model supports priority inference tier."""
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 10},
            performanceConfig={"latency": "optimized"},
        )
        return True
    except ClientError as e:
        msg = str(e).lower()
        if "throttl" in msg:
            return True
        if any(phrase in msg for phrase in [
            "not supported", "performance", "latency", "does not support",
        ]):
            return False
        if "validation" in msg:
            return False
        return False
    except Exception:
        return False


def _probe_guardrail_compatible(client: Any, model_id: str) -> bool:
    """Test if model supports Bedrock Guardrails.

    Sends a request with a fake guardrail ID. If the error is about the
    guardrail not being found (ResourceNotFoundException), the model supports
    guardrails. If it says guardrails are not supported, it doesn't.
    """
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 10},
            guardrailConfig={"guardrailIdentifier": "fake-guardrail-id", "guardrailVersion": "1"},
        )
        return True  # Shouldn't succeed but if it does, guardrails work
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = str(e).lower()
        if "throttl" in msg:
            return True
        # "ResourceNotFoundException" = model tried to find the guardrail = supports it
        if "resourcenotfound" in code.lower() or "not found" in msg:
            return True
        # "ValidationException" with guardrail-related message = supports it
        if "guardrail" in msg and ("not found" in msg or "does not exist" in msg or "invalid" in msg):
            return True
        # "not supported" = model doesn't support guardrails
        if "not supported" in msg or "does not support" in msg:
            return False
        # Generic validation = likely supports it (just bad params)
        if "validation" in code.lower():
            return True
        return True  # Default to true (most models support it)
    except Exception:
        return True


def probe_capabilities(bedrock_runtime: Any, model_id: str, skip_probes: bool = False) -> dict[str, Any]:
    """Probe a model's capabilities via test API calls.

    Returns dict with: converse_supported, tool_use, streaming_tool_use,
    extended_thinking, prompt_caching, supported_tiers.
    """
    if skip_probes:
        return {
            "converse_supported": None,
            "tool_use": None,
            "streaming_tool_use": None,
            "extended_thinking": None,
            "prompt_caching": None,
            "supported_tiers": ["standard"],
            "guardrail_compatible": True,  # Safe default
        }

    # First check Converse API support
    converse_ok = _probe_converse_support(bedrock_runtime, model_id)
    if not converse_ok:
        return {
            "converse_supported": False,
            "tool_use": False,
            "streaming_tool_use": False,
            "extended_thinking": False,
            "prompt_caching": None,
            "supported_tiers": ["standard"],
            "guardrail_compatible": False,
        }

    time.sleep(0.3)
    tool_use = _probe_tool_use(bedrock_runtime, model_id)
    time.sleep(0.3)

    # Only probe streaming tools if tool_use is supported
    if tool_use:
        streaming_tool_use = _probe_streaming_tool_use(bedrock_runtime, model_id)
        time.sleep(0.3)
    else:
        streaming_tool_use = False

    extended_thinking = _probe_extended_thinking(bedrock_runtime, model_id)
    time.sleep(0.3)

    supported_tiers = ["standard"]
    if _probe_priority_tier(bedrock_runtime, model_id):
        supported_tiers.append("priority")
    time.sleep(0.3)

    guardrail_compatible = _probe_guardrail_compatible(bedrock_runtime, model_id)
    time.sleep(0.3)

    return {
        "converse_supported": True,
        "tool_use": tool_use,
        "streaming_tool_use": streaming_tool_use,
        "extended_thinking": extended_thinking,
        "prompt_caching": None,  # Derived from LiteLLM cache pricing
        "supported_tiers": supported_tiers,
        "guardrail_compatible": guardrail_compatible,
    }


# ── Step 5: Fetch quality baselines from Artificial Analysis ────────

# Creator mapping for AA matching
AA_CREATOR_MAP = {
    "anthropic": "Anthropic",
    "amazon": "Amazon",
    "meta": "Meta",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "cohere": "Cohere",
    "ai21": "AI21 Labs",
    "google": "Google",
    "nvidia": "NVIDIA",
    "writer": "Writer",
    "openai": "OpenAI",
    "minimax": "MiniMax",
    "zai": "Z AI",
    "moonshotai": "Kimi",
    "moonshot": "Kimi",
    "qwen": "Alibaba",
}


def _get_aa_creator(model_id: str) -> str:
    """Get AA creator name from Bedrock model_id prefix."""
    prefix = model_id.split(".")[0].lower()
    return AA_CREATOR_MAP.get(prefix, "")


def _tokenize_name(name: str) -> list[str]:
    """Tokenize a model name into comparable parts."""
    s = name.lower()
    s = s.replace("-", " ").replace("_", " ")
    # Remove parenthetical annotations
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # Filter noise words
    noise = {'instruct', 'it', 'pt', 'bf16', 'vl', 'the', 'a', 'an', 'v1', 'v2'}
    return [t for t in s.split() if t not in noise]


def _compute_similarity(bedrock_tokens: list[str], aa_tokens: list[str]) -> float:
    """Compute similarity between two token lists using Jaccard + version bonuses."""
    if not bedrock_tokens or not aa_tokens:
        return 0.0

    b_set = set(bedrock_tokens)
    a_set = set(aa_tokens)
    overlap = b_set & a_set
    if not overlap:
        return 0.0

    union = b_set | a_set
    score = len(overlap) / len(union)

    # Bonus/penalty for version number matches
    version_pat = re.compile(r'^\d+(\.\d+)?$')
    b_ver = {t for t in b_set if version_pat.match(t)}
    a_ver = {t for t in a_set if version_pat.match(t)}
    if b_ver & a_ver:
        score += 0.15 * len(b_ver & a_ver)
    elif b_ver and a_ver and not (b_ver & a_ver):
        score -= 0.2

    # Bonus/penalty for size tokens (70b, 8b, etc.)
    size_pat = re.compile(r'^\d+b$')
    b_sz = {t for t in b_set if size_pat.match(t)}
    a_sz = {t for t in a_set if size_pat.match(t)}
    if b_sz & a_sz:
        score += 0.15 * len(b_sz & a_sz)
    elif b_sz and a_sz and not (b_sz & a_sz):
        score -= 0.15

    # Bonus for compound tokens (m2, k2.5, 8x7b, r1)
    compound_pat = re.compile(r'^[a-z]\d|^\d+x\d+')
    b_cmp = {t for t in b_set if compound_pat.match(t)}
    a_cmp = {t for t in a_set if compound_pat.match(t)}
    if b_cmp & a_cmp:
        score += 0.2 * len(b_cmp & a_cmp)
    elif b_cmp and a_cmp and not (b_cmp & a_cmp):
        score -= 0.15

    return score


def fetch_quality_baselines(api_key: str, cache_path: str = "") -> list[dict]:
    """Fetch AA Intelligence Index scores and pricing as structured data.

    Returns list of {name, score, creator, pricing} dicts.
    Pricing is in per-1M-token format from AA.
    Uses cache file if API is rate-limited or cache_path is provided.
    """

    # Try cache first if specified
    if cache_path:
        try:
            with open(cache_path) as f:
                data = json.load(f)
            logger.info(f"Step 5: Loaded AA data from cache: {cache_path}")
            results = []
            for m in data:
                score = (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
                pricing = m.get("pricing") or {}
                results.append({
                    "name": m["name"],
                    "score": score,
                    "creator": (m.get("model_creator") or {}).get("name", ""),
                    "pricing_input_per_1m": pricing.get("price_1m_input_tokens", 0),
                    "pricing_output_per_1m": pricing.get("price_1m_output_tokens", 0),
                })
            scored = [r for r in results if r["score"] is not None]
            logger.info(f"  {len(scored)} quality scores, {sum(1 for r in results if r['pricing_input_per_1m'] > 0)} with pricing from cache")
            return results
        except FileNotFoundError:
            logger.info("  Cache not found, fetching from API...")

    if not api_key:
        return []

    logger.info("Step 5: Fetching quality baselines from Artificial Analysis...")
    try:
        import requests
        resp = requests.get(
            "https://artificialanalysis.ai/api/v2/data/llms/models",
            headers={"x-api-key": api_key},
            timeout=30,
        )
        if resp.status_code == 429:
            logger.warning("  AA API rate limited (429). Use --aa-cache with a cached file.")
            return []
        resp.raise_for_status()
        data = resp.json().get("data", [])

        # Save to cache for future use
        default_cache = str(Path(__file__).parent / "_aa_models.json")
        with open(default_cache, "w") as f:
            json.dump(data, f)
        logger.info(f"  Cached AA data to {default_cache}")

        results = []
        for m in data:
            score = (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
            pricing = m.get("pricing") or {}
            results.append({
                "name": m["name"],
                "score": score,
                "creator": (m.get("model_creator") or {}).get("name", ""),
                "pricing_input_per_1m": pricing.get("price_1m_input_tokens", 0),
                "pricing_output_per_1m": pricing.get("price_1m_output_tokens", 0),
            })
        scored = [r for r in results if r["score"] is not None]
        logger.info(f"  Retrieved {len(scored)} quality scores, {sum(1 for r in results if r['pricing_input_per_1m'] > 0)} with pricing")
        return results
    except Exception as e:
        logger.warning(f"Failed to fetch AA scores: {e}")
        return []


def match_quality_baseline(bedrock_name: str, model_id: str, aa_models: list[dict]) -> tuple[float, dict]:
    """Find the best AA Intelligence Index match for a Bedrock model.

    Uses creator filtering + token-based similarity with preference
    for non-reasoning variants.

    Returns (quality_score, pricing_dict) where pricing_dict has
    input_per_1k and output_per_1k (converted from AA's per-1M format).
    """
    creator = _get_aa_creator(model_id)
    if not creator or not aa_models:
        return 0.0, {}

    # Filter by creator
    creator_models = [m for m in aa_models if m["creator"] == creator]
    if not creator_models:
        return 0.0, {}

    bedrock_tokens = _tokenize_name(bedrock_name)
    candidates = []

    for aa in creator_models:
        # Strip AA variant suffixes for matching
        aa_clean = re.sub(r'\s*\(.*?\)\s*$', '', aa["name"])
        aa_tokens = _tokenize_name(aa_clean)

        sim = _compute_similarity(bedrock_tokens, aa_tokens)
        if sim <= 0:
            continue

        # Variant preference scoring
        aa_lower = aa["name"].lower()
        variant_score = 0
        if "non-reasoning" in aa_lower and "high effort" in aa_lower:
            variant_score = 3
        elif "non-reasoning" in aa_lower:
            variant_score = 2
        elif "reasoning" not in aa_lower and "max" not in aa_lower and "adaptive" not in aa_lower:
            variant_score = 1
        # else: reasoning/max/adaptive = 0

        candidates.append((sim, variant_score, aa.get("score") or 0, aa))

    if not candidates:
        return 0.0, {}

    # Best match: highest similarity, then prefer non-reasoning, then highest score
    candidates.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    best = candidates[0]

    # Reject if similarity too low
    if best[0] < 0.15:
        return 0.0, {}

    aa_entry = best[3]
    quality = aa_entry.get("score") or 0.0

    # Convert pricing from per-1M to per-1K
    input_per_1m = aa_entry.get("pricing_input_per_1m", 0) or 0
    output_per_1m = aa_entry.get("pricing_output_per_1m", 0) or 0
    pricing = {}
    if input_per_1m > 0 or output_per_1m > 0:
        pricing = {
            "input_per_1k": round(input_per_1m / 1000, 6),
            "output_per_1k": round(output_per_1m / 1000, 6),
        }

    return quality, pricing


# ── Step 6: Build catalog ───────────────────────────────────────────

def build_catalog(
    models: list[dict],
    profiles: dict[str, list[str]],
    capabilities: dict[str, dict],
    aa_models: list[dict],
    litellm_data: dict[str, dict],
    existing_catalog: dict | None = None,
    region: str = "us-west-2",
) -> list[dict]:
    """Assemble the final catalog entries."""
    logger.info("Step 6: Building catalog...")

    # Load existing catalog for fields we can't auto-detect
    existing_by_id: dict[str, dict] = {}
    if existing_catalog:
        for m in existing_catalog.get("models", []):
            existing_by_id[m["model_id"]] = m
            # Also index by base model ID (without geo prefix)
            base = strip_geo_prefix(m["model_id"])
            if base != m["model_id"]:
                existing_by_id[base] = m

    catalog = []
    seen_base = set()

    for model_info in models:
        model_id = model_info["model_id"]
        base_id = strip_geo_prefix(model_id)

        # Skip if we've already processed this base model
        if base_id in seen_base:
            continue
        seen_base.add(base_id)

        # Models that use inference profiles keep their base ID
        # (the router's CRIS manager handles regional profile selection at invocation time)

        family = detect_family(model_id)
        display_name = model_info["model_name"]

        # Get CRIS profiles
        cris = profiles.get(base_id, [])
        # If no profiles found, use the model_id itself
        if not cris:
            cris = [model_id]

        # Get probed capabilities (try with and without geo prefix)
        caps = (
            capabilities.get(model_id)
            or capabilities.get(f"us.{model_id}")
            or capabilities.get(f"us.{base_id}")
            or {}
        )

        # Match quality baseline and pricing using token-based similarity
        quality_baseline, aa_pricing = match_quality_baseline(display_name, model_id, aa_models)

        # Use existing catalog values for fields we can't auto-detect
        existing = existing_by_id.get(model_id, {})

        # Build capabilities
        # Skip models that don't support the Converse API
        if caps.get("converse_supported") is False:
            logger.debug(f"  Skipping {model_id}: does not support Converse API")
            continue

        tool_use = caps.get("tool_use")
        if tool_use is None:
            tool_use = existing.get("capabilities", {}).get("tool_use", False)

        streaming_tool_use = caps.get("streaming_tool_use")
        if streaming_tool_use is None:
            streaming_tool_use = existing.get("capabilities", {}).get("streaming_tool_use", tool_use)

        extended_thinking = caps.get("extended_thinking")
        if extended_thinking is None:
            extended_thinking = existing.get("capabilities", {}).get("extended_thinking", False)

        # Derive prompt_caching from LiteLLM cache pricing or existing catalog
        prompt_caching = caps.get("prompt_caching")
        if prompt_caching is None:
            litellm_check = (
                litellm_data.get(model_id)
                or litellm_data.get(f"us.{base_id}")
                or litellm_data.get(f"bedrock/{model_id}")
                or litellm_data.get(base_id)
                or {}
            )
            if litellm_check.get("cache_read_input_token_cost") and litellm_check["cache_read_input_token_cost"] > 0:
                prompt_caching = True
            elif existing:
                prompt_caching = existing.get("capabilities", {}).get("prompt_caching", False)
            else:
                prompt_caching = False

        supported_tiers = caps.get("supported_tiers", ["standard"])
        if supported_tiers == ["standard"] and existing:
            supported_tiers = existing.get("supported_inference_tiers", ["standard"])

        # Token limits and pricing from LiteLLM (most reliable source)
        # Try multiple key formats: model_id, us.model_id, bedrock/model_id
        litellm_entry = (
            litellm_data.get(model_id)
            or litellm_data.get(f"us.{base_id}")
            or litellm_data.get(f"bedrock/{model_id}")
            or litellm_data.get(base_id)
            or {}
        )

        # Context window from LiteLLM > existing catalog > default
        if litellm_entry.get("max_input_tokens"):
            max_input = litellm_entry["max_input_tokens"]
        elif existing and existing.get("max_input_tokens", 0) > 0:
            max_input = existing["max_input_tokens"]
        else:
            max_input = 128000

        if litellm_entry.get("max_output_tokens"):
            max_output = litellm_entry["max_output_tokens"]
        elif existing and existing.get("max_output_tokens", 0) > 0:
            max_output = existing["max_output_tokens"]
        else:
            max_output = 4096

        # Pricing priority: existing catalog (has cache pricing) > LiteLLM > AA > zeros
        if existing and existing.get("pricing", {}).get("input_per_1k", 0) > 0:
            pricing = existing["pricing"]
        elif litellm_entry.get("input_cost_per_token"):
            pricing = {
                "input_per_1k": round(litellm_entry["input_cost_per_token"] * 1000, 6),
                "output_per_1k": round((litellm_entry.get("output_cost_per_token") or 0) * 1000, 6),
                "cache_read_per_1k": round((litellm_entry.get("cache_read_input_token_cost") or 0) * 1000, 6),
                "cache_write_per_1k": round((litellm_entry.get("cache_creation_input_token_cost") or 0) * 1000, 6),
            }
        elif aa_pricing:
            pricing = {
                "input_per_1k": aa_pricing["input_per_1k"],
                "output_per_1k": aa_pricing["output_per_1k"],
                "cache_read_per_1k": 0.0,
                "cache_write_per_1k": 0.0,
            }
        else:
            pricing = {
                "input_per_1k": 0.0,
                "output_per_1k": 0.0,
                "cache_read_per_1k": 0.0,
                "cache_write_per_1k": 0.0,
            }

        # Determine tier (needs pricing and capabilities computed first)
        if existing and "tier" in existing:
            tier = existing["tier"]
        else:
            tier = derive_tier(
                quality_baseline, model_id, display_name,
                price_in=pricing.get("input_per_1k", 0),
                caps={
                    "prompt_caching": prompt_caching,
                    "extended_thinking": extended_thinking,
                },
            )

        entry = {
            "model_id": model_id,
            "family": family,
            "tier": tier,
            "display_name": display_name,
            "capabilities": {
                "tool_use": tool_use,
                "vision": model_info["has_vision"],
                "streaming": model_info["streaming"],
                "streaming_tool_use": streaming_tool_use,
                "document_support": model_info["has_document"],
                "extended_thinking": extended_thinking,
                "prompt_caching": prompt_caching,
            },
            "max_input_tokens": max_input,
            "max_output_tokens": max_output,
            "pricing": pricing,
            "cris_profiles": cris,
            "supported_inference_tiers": supported_tiers,
            "guardrail_compatible": caps.get("guardrail_compatible", True),
            "quality_baseline": quality_baseline,
        }

        catalog.append(entry)

        # Also add global variant if CRIS profiles include it
        global_profiles = [p for p in cris if p.startswith("global.")]
        if global_profiles and not model_id.startswith("global."):
            global_id = f"global.{base_id}"
            global_entry = dict(entry)
            global_entry["model_id"] = global_id
            global_entry["display_name"] = f"{display_name} (Global)"
            global_entry["cris_profiles"] = global_profiles
            catalog.append(global_entry)

    logger.info(f"  Built catalog with {len(catalog)} entries")
    return catalog


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-refresh Bedrock model catalog")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--write", action="store_true", help="Write to models.json")
    parser.add_argument("--aa-key", default="", help="Artificial Analysis API key")
    parser.add_argument("--aa-cache", default="", help="Path to cached AA models JSON (skip API call)")
    parser.add_argument("--skip-probes", action="store_true", help="Skip capability probing (tool_use, thinking, guardrails)")
    parser.add_argument("--output", default="", help="Output file (default: scripts/_models.json)")
    args = parser.parse_args()

    t_start = time.time()

    session = boto3.Session(region_name=args.region)
    bedrock = session.client("bedrock", region_name=args.region)
    bedrock_runtime = session.client("bedrock-runtime", region_name=args.region)

    # Step 1: Discover models
    models = discover_models(bedrock, args.region)

    # Step 2: Discover CRIS profiles
    profiles = discover_profiles(bedrock)

    # Step 3: Fetch token limits from LiteLLM
    litellm_data = fetch_litellm_data()

    # Step 4: Probe capabilities
    capabilities: dict[str, dict] = {}
    if not args.skip_probes:

        logger.info("Step 4: Probing model capabilities (tool_use, streaming_tool_use, extended_thinking, guardrails, inference_tiers)...")
        # Use inference profile IDs (us.*) for models that need it,
        # otherwise use the raw model_id
        models_to_probe = []
        for m in models:
            mid = m["model_id"]
            inf_types = m.get("inference_types", [])
            if "INFERENCE_PROFILE" in inf_types and not mid.startswith(("us.", "eu.", "ap.", "global.")):
                models_to_probe.append(f"us.{mid}")
            else:
                models_to_probe.append(mid)

        # Probe in parallel (5 models at a time)
        def _probe_one(model_id: str) -> tuple[str, dict]:
            return model_id, probe_capabilities(bedrock_runtime, model_id, skip_probes=False)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_probe_one, mid): mid for mid in models_to_probe}
            done = 0
            for future in as_completed(futures):
                model_id, caps = future.result()
                capabilities[model_id] = caps
                done += 1
                if done % 10 == 0:
                    logger.info(f"  Probed {done}/{len(models_to_probe)} models...")

        logger.info(f"  Probed {len(capabilities)} models")
        # Report how many support Converse
        converse_ok = sum(1 for c in capabilities.values() if c.get("converse_supported") is not False)
        logger.info(f"  Converse API supported: {converse_ok}/{len(capabilities)}")
    else:
        logger.info("Step 4: Skipping capability probes (--skip-probes)")

    # Step 5: Quality baselines
    aa_models = fetch_quality_baselines(args.aa_key, cache_path=args.aa_cache)

    # Load existing catalog for fallback values
    existing_catalog = None
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH) as f:
            existing_catalog = json.load(f)

    # Step 6: Build catalog
    catalog = build_catalog(models, profiles, capabilities, aa_models, litellm_data, existing_catalog, region=args.region)

    # Output
    output_data = {"models": catalog}
    output_json = json.dumps(output_data, indent=2) + "\n"

    if args.write:
        out_path = Path(args.output) if args.output else Path(__file__).parent / "_models.json"
        with open(out_path, "w") as f:
            f.write(output_json)
        logger.info(f"Written {len(catalog)} models to {out_path}")
    else:
        # Print summary table
        print(f"\n{'Model ID':<55} {'Display Name':<30} {'Tier':<10} {'Quality':<8} {'Tools':<6} {'Think':<6}")
        print("=" * 120)
        for m in sorted(catalog, key=lambda x: -x.get("quality_baseline", 0)):
            mid = m["model_id"]
            if mid.startswith("global."):
                continue  # Skip global duplicates in display
            print(
                f"{mid:<55} "
                f"{m['display_name']:<30} "
                f"{m['tier']:<10} "
                f"{m.get('quality_baseline', 0):<8.1f} "
                f"{'✓' if m['capabilities']['tool_use'] else '✗':<6} "
                f"{'✓' if m['capabilities']['extended_thinking'] else '✗':<6}"
            )

        print(f"\nTotal: {len(catalog)} model entries")
        print(f"Use --write to save to {CATALOG_PATH}")

    elapsed = time.time() - t_start
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
