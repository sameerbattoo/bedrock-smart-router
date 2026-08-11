# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
from collections import defaultdict
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

# Suppress noisy urllib3 connection pool warnings and botocore credential logs
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

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
    "xai": "xai",
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
                price_in: float = 0, caps: dict | None = None,
                api_support: list[str] | None = None,
                price_out: float = 0, max_input: int = 0, max_output: int = 0) -> str:
    """Derive tier from multiple signals: name, size, price, quality, capabilities.

    Rules (in priority order):
    1. Reasoning: quality >= 50 OR name contains reasoning indicators
    1b. Responses-only frontier models (multi-signal scoring)
    2. Heavy: expensive (>= $4/M input) + full capabilities (cache + thinking)
    3. Micro: name says micro/nano OR (small model + cheap + low quality)
    4. Lite: name says lite/haiku/scout/mini OR small model (<=14B) + cheap
    5. Mid: name says pro/large/sonnet/maverick OR large model (>=70B) OR quality >= 15
    6. Default: lite
    """
    caps = caps or {}
    api_support = api_support or []
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

    # Rule 1b: Responses-only models — frontier models behind newest API.
    # These lack cache/thinking flags (Responses API handles state differently).
    # Use a composite score from pricing, quality, and capacity signals.
    if api_support == ["responses"]:
        score = 0
        if price_in >= 0.004:
            score += 3   # Very expensive input
        elif price_in >= 0.002:
            score += 2   # Expensive input
        if price_out >= 0.02:
            score += 2   # Very expensive output
        elif price_out >= 0.01:
            score += 1   # Expensive output
        if quality_baseline >= 38:
            score += 2   # High benchmark score
        elif quality_baseline >= 30:
            score += 1   # Moderate benchmark score
        if max_input >= 128000:
            score += 1   # Large context window
        if max_output >= 16384:
            score += 1   # Large output capacity

        if score >= 8:
            return "reasoning"
        if score >= 5:
            return "heavy"
        # Responses-only is at minimum mid (frontier API exclusivity)
        return "mid"

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
    """List inference profiles and map base model → profile IDs.

    Scans multiple regions to discover all CRIS profiles (us, eu, apac, global, etc.)
    and foundation models available per region.

    Returns: dict mapping base_model_id → list of regional availability entries:
        { "anthropic.claude-sonnet-4-6": [
            {"name": "us-east-1", "cris_profiles": ["global", "us"], "direct": true},
            {"name": "ap-south-1", "cris_profiles": ["apac"], "direct": false},
          ]
        }
    """
    logger.info("Step 2: Discovering inference profiles & regional availability...")

    SCAN_REGIONS = [
        "us-east-1", "us-east-2", "us-west-1", "us-west-2",
        "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
        "ap-south-1", "ap-southeast-1", "ap-southeast-2",
        "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
        "ca-central-1", "sa-east-1",
    ]

    # model_id -> { region -> { cris_profiles: set, direct: bool } }
    model_regions: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"cris_profiles": set(), "direct": False}))
    all_prefixes: set[str] = set()

    def _scan_region(region: str) -> tuple[str, list, list, str | None]:
        """Scan a single region for profiles and foundation models."""
        try:
            client = boto3.client("bedrock", region_name=region)
            # Inference profiles
            profiles = []
            kwargs = {"maxResults": 100, "typeEquals": "SYSTEM_DEFINED"}
            while True:
                resp = client.list_inference_profiles(**kwargs)
                for p in resp.get("inferenceProfileSummaries", []):
                    pid = p.get("inferenceProfileId", "")
                    prefix = pid.split(".")[0] if "." in pid else ""
                    base = ".".join(pid.split(".")[1:]) if "." in pid else pid
                    profiles.append({"prefix": prefix, "base_model": base})
                nt = resp.get("nextToken")
                if not nt:
                    break
                kwargs["nextToken"] = nt

            # Foundation models (direct access)
            fm_resp = client.list_foundation_models()
            direct_models = []
            for m in fm_resp.get("modelSummaries", []):
                if "ON_DEMAND" in m.get("inferenceTypesSupported", []):
                    direct_models.append(m.get("modelId", ""))

            return region, profiles, direct_models, None
        except Exception as e:
            return region, [], [], str(e)

    # Scan all regions in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_scan_region, r): r for r in SCAN_REGIONS}
        for future in as_completed(futures):
            region, profiles, direct_models, err = future.result()
            if err:
                logger.debug(f"  {region}: error - {err[:60]}")
                continue

            # Process inference profiles
            for p in profiles:
                base_model = p["base_model"]
                prefix = p["prefix"]
                all_prefixes.add(prefix)
                model_regions[base_model][region]["cris_profiles"].add(prefix)

            # Process direct-access models
            for model_id in direct_models:
                model_regions[model_id][region]["direct"] = True

            logger.debug(f"  {region}: {len(profiles)} profiles, {len(direct_models)} direct models")

    # Convert to output format
    result: dict[str, list[dict]] = {}
    for model_id, regions_data in model_regions.items():
        entries = []
        for region in sorted(regions_data.keys()):
            rdata = regions_data[region]
            entry: dict[str, Any] = {"name": region}
            if rdata["cris_profiles"]:
                entry["cris_profiles"] = sorted(rdata["cris_profiles"])
            if rdata["direct"]:
                entry["direct"] = True
            entries.append(entry)
        result[model_id] = entries

    logger.info(f"  Scanned {len(SCAN_REGIONS)} regions, found {len(result)} models")
    logger.info(f"  CRIS prefixes: {sorted(all_prefixes)}")
    models_with_cris = sum(1 for regions in result.values() if any("cris_profiles" in r for r in regions))
    models_direct_only = len(result) - models_with_cris
    logger.info(f"  Models with CRIS: {models_with_cris}, Direct-only: {models_direct_only}")

    return result


# ── Step 2b: Discover Mantle models ─────────────────────────────────

MANTLE_CACHE_PATH = Path(__file__).parent / "_mantle_models.json"


def discover_mantle_models(region: str) -> set[str]:
    """Discover models available on the bedrock-mantle endpoint.

    Scans multiple regions to get full availability. Returns a set of model IDs
    available via Chat Completions / Responses API, and also builds a
    model → regions mapping for the catalog.
    """
    logger.info("Step 2b: Discovering Mantle models (bedrock-mantle endpoint)...")

    if requests is None:
        logger.warning("  'requests' package not available, skipping Mantle discovery")
        return set()

    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    # Scan all regions where Mantle is available
    MANTLE_REGIONS = [
        "us-east-1", "us-east-2", "us-west-2",
        "eu-central-1", "eu-west-1", "eu-west-2", "eu-north-1", "eu-south-1",
        "ap-south-1", "ap-southeast-2", "ap-southeast-3", "ap-northeast-1",
        "sa-east-1",
    ]

    all_model_ids: set[str] = set()
    # model_id → set of regions
    model_regions: dict[str, set[str]] = {}

    def _scan_mantle_region(r: str) -> tuple[str, set[str], str | None]:
        try:
            sess = boto3.Session(region_name=r)
            creds = sess.get_credentials().get_frozen_credentials()
            url = f"https://bedrock-mantle.{r}.api.aws/v1/models"
            req = AWSRequest(method="GET", url=url, headers={"Content-Type": "application/json"})
            SigV4Auth(creds, "bedrock", r).add_auth(req)
            resp = requests.get(url, headers=dict(req.headers), timeout=10)
            if resp.status_code != 200:
                return r, set(), f"HTTP {resp.status_code}"
            data = resp.json()
            ids = set(m.get("id", "") for m in data.get("data", []) if m.get("id"))
            return r, ids, None
        except Exception as e:
            return r, set(), str(e)[:60]

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_scan_mantle_region, r): r for r in MANTLE_REGIONS}
        for future in as_completed(futures):
            r, ids, err = future.result()
            if err:
                logger.debug(f"  Mantle {r}: error - {err}")
                continue
            logger.debug(f"  Mantle {r}: {len(ids)} models")
            all_model_ids.update(ids)
            for mid in ids:
                model_regions.setdefault(mid, set()).add(r)

    # Cache for offline use
    MANTLE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {mid: sorted(regions) for mid, regions in model_regions.items()}
    with open(MANTLE_CACHE_PATH, "w") as f:
        json.dump(cache_data, f, indent=2)

    logger.info(f"  Found {len(all_model_ids)} models across {len(MANTLE_REGIONS)} Mantle regions")
    logger.info(f"  Saved to {MANTLE_CACHE_PATH.name}")

    # Store the regions mapping on the function for later use
    discover_mantle_models._model_regions = model_regions

    return all_model_ids


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
    """Test if model supports tool use (function calling) via Converse API.
    
    Sends a request with toolConfig. The model supports tool_use if it responds
    with a proper toolUse block OR the stopReason is "tool_use".
    
    Known false positives: Models that accept toolConfig but return tool calls
    as plain text (e.g., Gemma, Llama 3.3, Magistral). These don't support the
    Converse API tool_use conversation pattern (toolResult messages).
    
    Known false negatives: Models with reasoning/thinking that may hit maxTokens
    before emitting the toolUse block. We use maxTokens=500 and retry once on
    max_tokens stopReason to mitigate this.
    """
    # Known models that don't support Converse API tool_use pattern
    # (they accept toolConfig but return tool calls as text, not toolUse blocks)
    _KNOWN_NO_TOOL_USE = {
        "google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it",
    }
    base_id = model_id.split(".", 1)[-1] if "." in model_id and model_id.split(".")[0] in ("us", "eu", "ap", "global") else model_id
    if base_id in _KNOWN_NO_TOOL_USE or model_id in _KNOWN_NO_TOOL_USE:
        return False

    def _try_probe(max_tokens: int) -> bool | None:
        """Returns True/False if conclusive, None if inconclusive (max_tokens hit)."""
        try:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "What is 2+2? Use the calculator tool."}]}],
                toolConfig={
                    "tools": [{
                        "toolSpec": {
                            "name": "calculator",
                            "description": "Performs math calculations. You MUST use this tool for any math.",
                            "inputSchema": {"json": {
                                "type": "object",
                                "properties": {"expression": {"type": "string"}},
                                "required": ["expression"],
                            }},
                        }
                    }]
                },
                inferenceConfig={"maxTokens": max_tokens},
            )
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])
            has_tool_use = any(
                isinstance(block, dict) and "toolUse" in block
                for block in content
            )
            stop_reason = response.get("stopReason", "")
            # Model supports tools if it actually called one OR stop_reason is tool_use
            if has_tool_use or stop_reason == "tool_use":
                return True
            # If hit max_tokens, the model might support tools but ran out of budget
            # (common with reasoning models that emit thinking before tool call)
            if stop_reason == "max_tokens":
                return None  # Inconclusive
            # end_turn without tool call = doesn't support tool_use
            return False
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            msg = str(e).lower()
            if "throttl" in msg:
                return True
            if "tool" in msg or "not supported" in msg or "does not support" in msg:
                return False
            if "internal" in code.lower():
                return False
            return False
        except Exception:
            return False

    # First attempt with 500 tokens (enough for most models)
    result = _try_probe(500)
    if result is not None:
        return result
    # Retry with higher budget for reasoning models
    result = _try_probe(2000)
    if result is not None:
        return result
    # If still inconclusive after retry, assume it supports tools
    # (it accepted toolConfig without error, just ran out of tokens)
    return True


def _probe_streaming_tool_use(client: Any, model_id: str) -> bool:
    """Test if model supports streaming with tool use.
    
    This checks if the converse_stream API accepts toolConfig without error.
    Models that don't support tool_use at all should also fail this probe.
    """
    # If tool_use is known to be unsupported, streaming_tool_use is also unsupported
    _KNOWN_NO_TOOL_USE = {
        "google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it",
    }
    base_id = model_id.split(".", 1)[-1] if "." in model_id and model_id.split(".")[0] in ("us", "eu", "ap", "global") else model_id
    if base_id in _KNOWN_NO_TOOL_USE or model_id in _KNOWN_NO_TOOL_USE:
        return False

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
        # Model explicitly rejects the thinking field as unknown/extra
        if "extra inputs are not permitted" in msg or "extra_forbidden" in msg:
            return False
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


def _probe_priority_tier(client: Any, model_id: str) -> list[str]:
    """Probe if a model supports performanceConfig (optimized latency).
    
    Bedrock uses performanceConfig={"latency": "optimized"} for latency optimization.
    Models that don't support it return ValidationException.
    Returns ["standard", "optimized"] if supported, ["standard"] otherwise.
    Models that reject the parameter entirely (like Anthropic) get [].
    """
    # First test if the model accepts performanceConfig at all
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 10},
            performanceConfig={"latency": "optimized"},
        )
        return ["standard", "optimized"]
    except ClientError as e:
        msg = str(e).lower()
        if "throttl" in msg:
            return ["standard", "optimized"]  # Throttled = accepted the request
        # "not supported" for this model in this region = model doesn't support it
        if "not supported" in msg or "does not support" in msg:
            return ["standard"]
        # Generic validation error = parameter not recognized at all
        if "validation" in msg:
            return []
        return ["standard"]
    except Exception:
        return ["standard"]


def _probe_service_tiers(client: Any, model_id: str) -> list[str]:
    """Probe which service tiers (Standard/Flex/Priority) a model supports.

    Sends a converse request with serviceTier={"type": "flex"} and
    serviceTier={"type": "priority"} to check support. All models
    support "standard" by default.

    Returns a list like ["standard", "flex", "priority"].
    """
    supported = ["standard"]

    for tier_type in ["flex", "priority"]:
        try:
            client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
                inferenceConfig={"maxTokens": 5},
                serviceTier={"type": tier_type},
            )
            supported.append(tier_type)
        except ClientError as e:
            msg = str(e).lower()
            if "throttl" in msg:
                supported.append(tier_type)  # Throttled = model accepted the param
            # ValidationException about serviceTier = not supported, skip
        except Exception:
            pass

    return supported


def _probe_mantle_chat_completions(mantle_models: set[str], model_id: str, region: str) -> bool:
    """Test if model supports the Chat Completions API on bedrock-mantle.

    First checks if the model appears in the Mantle /v1/models list.
    If yes, sends a minimal Chat Completions request to confirm.
    """
    if not mantle_models:
        return False

    # Normalize model_id for Mantle lookup (strip version suffixes)
    base_id = strip_geo_prefix(model_id)
    candidates = [
        base_id,
        re.sub(r"-\d+:\d+$", "", base_id),
        re.sub(r"-v\d+:\d+$", "", base_id),
    ]
    # Find matching Mantle model ID
    mantle_id = None
    for candidate in candidates:
        if candidate in mantle_models:
            mantle_id = candidate
            break
    if not mantle_id:
        return False

    # Probe with a real request
    try:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        session = boto3.Session(region_name=region)
        credentials = session.get_credentials().get_frozen_credentials()

        url = f"https://bedrock-mantle.{region}.api.aws/v1/chat/completions"
        payload = json.dumps({
            "model": mantle_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        })
        request = AWSRequest(method="POST", url=url, data=payload, headers={"Content-Type": "application/json"})
        SigV4Auth(credentials, "bedrock", region).add_auth(request)

        resp = requests.post(url, data=payload, headers=dict(request.headers), timeout=15)
        if resp.status_code == 200:
            return True
        # 400 with "does not support" = model exists but doesn't support this API
        if resp.status_code == 400:
            err = resp.json().get("error", {}).get("message", "")
            if "does not support" in err:
                return False
            # Other 400 = validation issue, model likely supports it
            return True
        return False
    except Exception:
        return False


def _probe_mantle_responses(mantle_models: set[str], model_id: str, region: str) -> str | None:
    """Test if model supports the Responses API on bedrock-mantle.

    Sends a minimal Responses API request to confirm support.
    Tries both /v1/responses and /openai/v1/responses paths.

    Returns the working path (e.g., "/v1/responses") or None if not supported.
    """
    if not mantle_models:
        return None

    base_id = strip_geo_prefix(model_id)
    candidates = [
        base_id,
        re.sub(r"-\d+:\d+$", "", base_id),
        re.sub(r"-v\d+:\d+$", "", base_id),
    ]
    mantle_id = None
    for candidate in candidates:
        if candidate in mantle_models:
            mantle_id = candidate
            break
    if not mantle_id:
        return None

    try:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        session = boto3.Session(region_name=region)
        credentials = session.get_credentials().get_frozen_credentials()

        # Try both URL paths: /v1/responses (standard) and /openai/v1/responses (GPT-5.x)
        # GPT-5.x models use /openai/v1/responses, others use /v1/responses
        if "gpt-5" in mantle_id.lower():
            paths = ["/openai/v1/responses", "/v1/responses"]
        else:
            paths = ["/v1/responses", "/openai/v1/responses"]

        for path in paths:
            url = f"https://bedrock-mantle.{region}.api.aws{path}"
            payload = json.dumps({
                "model": mantle_id,
                "input": "hi",
                "store": False,
                "max_output_tokens": 16,  # Minimum required by /openai/v1/responses
            })
            request = AWSRequest(method="POST", url=url, data=payload, headers={"Content-Type": "application/json"})
            SigV4Auth(credentials, "bedrock", region).add_auth(request)

            resp = requests.post(url, data=payload, headers=dict(request.headers), timeout=15)
            if resp.status_code == 200:
                return path
            # Only a 200 confirms support. Any 400 (including validation errors)
            # is NOT a confirmation — the model may not support this API at all.
            # "does not support" is an explicit rejection; other 400s are ambiguous.
            if resp.status_code == 400:
                err = resp.json().get("error", {}).get("message", "")
                if "does not support" in err:
                    continue  # Explicit rejection — try next path
                # Other 400 errors (validation, missing params, etc.) are ambiguous.
                # Do NOT assume support — continue to next path.
                continue
        return None
    except Exception:
        return None


def _probe_responses_regions(model_id: str, regions: set[str]) -> set[str]:
    """Probe /v1/responses (or /openai/v1/responses) in each region to confirm availability.

    For Responses-only models, the /v1/models list may include regions where
    only Chat Completions is available. This function sends a minimal Responses
    request to each region and returns only the regions where it succeeds.

    Note: Some OpenAI models (GPT-5.4, GPT-5.5) use /openai/v1/responses
    instead of /v1/responses. We try both paths.

    Parameters
    ----------
    model_id : str
        The Mantle model ID (e.g., 'openai.gpt-5.4').
    regions : set[str]
        Candidate regions from /v1/models scan.

    Returns
    -------
    set[str]
        Regions where Responses API actually works for this model.
    """
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    confirmed: set[str] = set()

    # Determine which URL paths to try based on model ID
    # GPT-5.x models use /openai/v1/responses, others use /v1/responses
    if "gpt-5" in model_id.lower():
        paths = ["/openai/v1/responses", "/v1/responses"]
    else:
        paths = ["/v1/responses", "/openai/v1/responses"]

    def _probe_region(r: str) -> tuple[str, bool]:
        try:
            sess = boto3.Session(region_name=r)
            creds = sess.get_credentials().get_frozen_credentials()

            for path in paths:
                url = f"https://bedrock-mantle.{r}.api.aws{path}"
                payload = json.dumps({
                    "model": model_id,
                    "input": "hi",
                    "store": False,
                    "max_output_tokens": 16,  # Minimum required by /openai/v1/responses
                })
                req = AWSRequest(method="POST", url=url, data=payload,
                                headers={"Content-Type": "application/json"})
                SigV4Auth(creds, "bedrock", r).add_auth(req)
                resp = requests.post(url, data=payload, headers=dict(req.headers), timeout=15)
                if resp.status_code == 200:
                    return r, True
                if resp.status_code == 400:
                    err = resp.json().get("error", {}).get("message", "")
                    if "does not support" in err:
                        continue  # Try next path
                    # Other 400s are ambiguous — don't confirm support
                    continue
            return r, False
        except Exception:
            return r, False

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_probe_region, r): r for r in regions}
        for future in as_completed(futures):
            r, success = future.result()
            if success:
                confirmed.add(r)
            else:
                logger.debug(f"    {model_id}: Responses API NOT available in {r}")

    return confirmed


def _probe_guardrail_compatible(client: Any, model_id: str) -> bool:
    """Test if model supports Bedrock Guardrails.

    Sends a request with a fake guardrail ID. If the error is about the
    guardrail not being found (ResourceNotFoundException), the model supports
    guardrails. If it says guardrails are not supported, it doesn't.
    
    Known incompatible models that pass this probe incorrectly:
    - Google Gemma models (accept guardrailConfig but don't apply it)
    """
    # Known models that don't support guardrails despite passing the probe
    _KNOWN_INCOMPATIBLE = {"google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it"}
    base_id = model_id.split(".", 1)[-1] if "." in model_id and model_id.split(".")[0] in ("us", "eu", "ap", "global") else model_id
    if base_id in _KNOWN_INCOMPATIBLE or model_id in _KNOWN_INCOMPATIBLE:
        return False

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


def _probe_max_output_tokens(client: Any, model_id: str) -> int | None:
    """Probe the maximum output token limit enforced by the Converse API.

    Sends a request with an absurdly high maxTokens value (9,999,999).
    If the model has a server-side cap, Bedrock returns a ValidationException
    containing the actual limit: "exceeds the model limit of {N}".

    This probe ONLY works for Converse-compatible models. The Chat Completions
    and Responses APIs on bedrock-mantle do NOT enforce server-side maxTokens
    validation — they accept any value and let the model run until it stops.

    Returns:
        int: The enforced max output token limit.
        None: If the probe couldn't determine the limit (model accepted the
              value, errored without revealing the limit, or was throttled).

    Interpretation:
        - A returned integer is the authoritative max_output_tokens for the model.
        - None means fall back to LiteLLM / existing catalog / AWS docs values.
    """
    try:
        client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 9999999},
        )
        # Model accepted 9999999 without error — no server-side cap
        return None
    except ClientError as e:
        msg = str(e)
        # Look for the standard pattern: "exceeds the model limit of {N}"
        match = re.search(r"model limit of (\d+)", msg)
        if match:
            return int(match.group(1))
        # Throttled — can't determine
        if "throttl" in msg.lower():
            return None
        # Other validation errors that don't reveal the limit
        return None
    except Exception:
        return None


def probe_capabilities(bedrock_runtime: Any, model_id: str, skip_probes: bool = False,
                       mantle_models: set[str] | None = None, region: str = "us-west-2") -> dict[str, Any]:
    """Probe a model's capabilities via test API calls.

    Returns dict with: converse_supported, tool_use, streaming_tool_use,
    extended_thinking, prompt_caching, supported_tiers, api_support,
    max_output_tokens_probed.

    The max_output_tokens_probed field uses the Converse API's server-side
    maxTokens validation: sending an absurdly high value triggers an error
    that reveals the actual enforced limit. This only works for Converse-
    compatible models; Chat Completions and Responses APIs don't validate.
    """
    if skip_probes:
        return {
            "converse_supported": None,
            "tool_use": None,
            "streaming_tool_use": None,
            "extended_thinking": None,
            "prompt_caching": None,
            "supported_tiers": [],
            "guardrail_compatible": True,  # Safe default
            "api_support": None,  # Will be derived later if possible
            "max_output_tokens_probed": None,
        }

    # First check Converse API support
    converse_ok = _probe_converse_support(bedrock_runtime, model_id)

    # Build api_support list
    api_support = []
    if converse_ok:
        api_support.append("converse")

    # Probe Mantle APIs (Chat Completions + Responses)
    responses_path: str | None = None
    if mantle_models:
        if _probe_mantle_chat_completions(mantle_models, model_id, region):
            api_support.append("chat_completions")
            time.sleep(0.3)
        responses_path = _probe_mantle_responses(mantle_models, model_id, region)
        if responses_path:
            api_support.append("responses")
            time.sleep(0.3)

    if not converse_ok and not api_support:
        return {
            "converse_supported": False,
            "tool_use": False,
            "streaming_tool_use": False,
            "extended_thinking": False,
            "prompt_caching": None,
            "supported_tiers": [],
            "guardrail_compatible": False,
            "api_support": api_support or ["converse"],
        }

    # Converse-specific probes (only if Converse API is supported)
    tool_use = False
    streaming_tool_use = False
    extended_thinking = False
    supported_tiers: list[str] = []
    guardrail_compatible = True

    if converse_ok:
        time.sleep(0.3)
        tool_use = _probe_tool_use(bedrock_runtime, model_id)
        time.sleep(0.3)

        # Only probe streaming tools if tool_use is supported
        if tool_use:
            streaming_tool_use = _probe_streaming_tool_use(bedrock_runtime, model_id)
            time.sleep(0.3)

        extended_thinking = _probe_extended_thinking(bedrock_runtime, model_id)
        time.sleep(0.3)

        supported_tiers = _probe_priority_tier(bedrock_runtime, model_id)
        time.sleep(0.3)

        guardrail_compatible = _probe_guardrail_compatible(bedrock_runtime, model_id)
        time.sleep(0.3)

        service_tiers = _probe_service_tiers(bedrock_runtime, model_id)
        time.sleep(0.3)

        # Probe max output tokens (enforced server-side limit)
        max_output_probed = _probe_max_output_tokens(bedrock_runtime, model_id)
        time.sleep(0.3)

    return {
        "converse_supported": converse_ok,
        "tool_use": tool_use,
        "streaming_tool_use": streaming_tool_use,
        "extended_thinking": extended_thinking,
        "prompt_caching": None,  # Derived from LiteLLM cache pricing
        "supported_tiers": supported_tiers,
        "guardrail_compatible": guardrail_compatible,
        "api_support": api_support or ["converse"],
        "responses_path": responses_path,
        "supported_service_tiers": service_tiers if converse_ok else ["standard"],
        "max_output_tokens_probed": max_output_probed if converse_ok else None,
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
    "xai": "xAI",
}

# Some AA entries have creator=None but can be matched by name prefix
AA_NAME_PREFIX_FALLBACK = {
    "xai": "Grok",
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

    # Fallback: for providers where AA has creator=None (e.g., xAI/Grok),
    # match by name prefix instead
    if not creator_models:
        prefix = model_id.split(".")[0].lower()
        name_prefix = AA_NAME_PREFIX_FALLBACK.get(prefix)
        if name_prefix:
            creator_models = [
                m for m in aa_models
                if m.get("creator") is None and m["name"].startswith(name_prefix)
            ]

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

def _derive_api_support(base_id: str, model_id: str, caps: dict, mantle_models: set[str] | None) -> list[str]:
    """Derive api_support when probes were skipped, using Mantle model list.
    
    NOTE: Without probes, we can only confirm 'converse' (from ListFoundationModels)
    and mark models as *potentially* on Mantle. We do NOT assume chat_completions
    support without a real probe — the model may only support Messages or Responses API.
    """
    api_support = []

    # Converse: if we got here, the model passed the converse filter in discover_models
    if caps.get("converse_supported") is not False:
        api_support.append("converse")

    # We intentionally do NOT add chat_completions here without a probe.
    # Models on Mantle may support Messages API (Anthropic) or Responses API (OpenAI)
    # but not Chat Completions. Only the probe can confirm.

    return api_support or ["converse"]


def build_catalog(
    models: list[dict],
    profiles: dict[str, list[str]],
    capabilities: dict[str, dict],
    aa_models: list[dict],
    litellm_data: dict[str, dict],
    existing_catalog: dict | None = None,
    region: str = "us-west-2",
    mantle_models: set[str] | None = None,
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

        # Get regional availability (new format: per-region with cris_profiles + direct)
        region_entries = profiles.get(base_id, [])
        # Also check if model_id itself has entries (for direct-only models)
        if not region_entries:
            region_entries = profiles.get(model_id, [])

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

        supported_tiers = caps.get("supported_tiers", [])
        if not supported_tiers and existing:
            supported_tiers = existing.get("supported_latency_modes", [])

        # Token limits and pricing from LiteLLM (most reliable source)
        # Try multiple key formats: model_id, us.model_id, bedrock/model_id
        litellm_entry = (
            litellm_data.get(model_id)
            or litellm_data.get(f"us.{base_id}")
            or litellm_data.get(f"bedrock/{model_id}")
            or litellm_data.get(base_id)
            or {}
        )

        # Context window: LiteLLM > existing catalog > default
        # (No reliable probe exists for max_input_tokens)
        if litellm_entry.get("max_input_tokens"):
            max_input = litellm_entry["max_input_tokens"]
        elif existing and existing.get("max_input_tokens", 0) > 0:
            max_input = existing["max_input_tokens"]
        else:
            max_input = 128000

        # Max output tokens priority:
        #   1. Probed value (Converse API server-enforced limit — authoritative)
        #   2. LiteLLM (community-sourced, sometimes inaccurate)
        #   3. Existing catalog (previous known-good value)
        #   4. Default (4096)
        max_output_probed = caps.get("max_output_tokens_probed")
        if max_output_probed is not None:
            max_output = max_output_probed
        elif litellm_entry.get("max_output_tokens"):
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
        api_support = caps.get("api_support") or _derive_api_support(base_id, model_id, caps, mantle_models)
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
                api_support=api_support,
                price_out=pricing.get("output_per_1k", 0),
                max_input=max_input,
                max_output=max_output,
            )

        entry = {
            "model_id": model_id,
            "family": family,
            "regions": region_entries,
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
            "supported_latency_modes": supported_tiers,
            "guardrail_compatible": caps.get("guardrail_compatible", True),
            "quality_baseline": quality_baseline,
            "api_support": api_support,
        }

        # Add supported_service_tiers only if model supports flex/priority
        # (all models support standard by default, so we only store extras)
        service_tiers = caps.get("supported_service_tiers", ["standard"])
        if len(service_tiers) > 1:
            entry["supported_service_tiers"] = [t for t in service_tiers if t != "standard"]

        # Add responses_path if the model supports Responses API
        responses_path = caps.get("responses_path")
        if not responses_path and existing:
            responses_path = existing.get("responses_path")
        if responses_path:
            entry["responses_path"] = responses_path

        catalog.append(entry)

    logger.info(f"  Built catalog with {len(catalog)} entries")
    return catalog


# ── Step 6b: Add Mantle-only models ─────────────────────────────────

def add_mantle_only_models(
    catalog: list[dict],
    mantle_models: set[str],
    litellm_data: dict[str, dict],
    aa_models: list[dict],
    existing_catalog: dict | None = None,
    region: str = "us-west-2",
) -> list[dict]:
    """Add models that exist only on Mantle (not in bedrock-runtime) to the catalog.

    These models (e.g., GPT-5.4, GPT-5.5, Qwen3-Coder-Next) are only accessible
    via the bedrock-mantle endpoint and don't appear in ListFoundationModels.
    """
    if not mantle_models:
        return catalog

    # Find models already in catalog (normalize IDs for comparison)
    existing_ids = set()
    existing_by_base: dict[str, dict] = {}  # base_id → catalog entry
    # Provider prefix aliases (Mantle and Bedrock sometimes use different prefixes for the same provider)
    _PROVIDER_ALIASES = {
        "moonshotai": "moonshot",
        "moonshot": "moonshotai",
    }
    for m in catalog:
        mid = m["model_id"]
        existing_ids.add(mid)
        base = re.sub(r"-\d{8}-v\d+:\d+$", "", mid)  # Strip date-version like -20251001-v1:0
        base = re.sub(r"-v\d+:\d+$", "", base)  # Strip version like -v1:0
        base = re.sub(r"-\d+:\d+$", "", base)  # Strip version like -1:0
        existing_ids.add(base)
        existing_by_base[base] = m
        # Also index by aliased provider prefix
        prefix = base.split(".")[0] if "." in base else ""
        if prefix in _PROVIDER_ALIASES:
            aliased = _PROVIDER_ALIASES[prefix] + "." + ".".join(base.split(".")[1:])
            existing_by_base[aliased] = m
            existing_ids.add(aliased)

    # Find Mantle-only models
    mantle_only = [mid for mid in sorted(mantle_models) if mid not in existing_ids]

    if not mantle_only:
        logger.info("Step 6b: No Mantle-only models to add")
        return catalog

    logger.info(f"Step 6b: Processing {len(mantle_only)} Mantle-only models...")

    # Load existing catalog for fallback values
    existing_by_id: dict[str, dict] = {}
    if existing_catalog:
        for m in existing_catalog.get("models", []):
            existing_by_id[m["model_id"]] = m

    # Skip versioned duplicates (e.g., keep openai.gpt-5.4, skip openai.gpt-5.4-2026-03-05)
    # unless the base version isn't present
    base_versions: dict[str, str] = {}
    for mid in mantle_only:
        # Detect date-versioned models (e.g., openai.gpt-5.4-2026-03-05)
        date_match = re.search(r"-(\d{4}-\d{2}-\d{2})$", mid)
        if date_match:
            base = mid[:date_match.start()]
            if base in mantle_only or base in existing_ids:
                continue  # Skip, base version exists
        base_versions[mid] = mid

    for mantle_id in base_versions.values():
        # Check if this is actually a variant of an existing catalog model
        # e.g., "anthropic.claude-haiku-4-5" matches "anthropic.claude-haiku-4-5-20251001-v1:0"
        # Also handles: provider aliases (moonshotai↔moonshot) and suffix differences (-instruct)
        base_for_match = re.sub(r"-\d{8}(-v\d+)?$", "", mantle_id)
        # Try matching with and without -instruct suffix
        base_no_instruct = re.sub(r"-instruct$", "", base_for_match)
        existing_entry = (
            existing_by_base.get(mantle_id)
            or existing_by_base.get(base_for_match)
            or existing_by_base.get(base_no_instruct)
        )
        if existing_entry:
            # Model already in catalog — update its api_support with Mantle capability
            current_apis = existing_entry.get("api_support", ["converse"])
            if "chat_completions" not in current_apis:
                existing_entry["api_support"] = current_apis + ["chat_completions"]
                logger.info(f"    Updated {existing_entry['model_id']} → added chat_completions (matched from {mantle_id})")
            else:
                logger.debug(f"    Skipped {mantle_id} (already in catalog as {existing_entry['model_id']})")
            continue

        family = detect_family(mantle_id)

        # Determine api_support — these are Mantle-only
        api_support = ["chat_completions"]

        # OpenAI proprietary models (non-OSS) only support Responses API
        if "openai" in mantle_id.lower() and "oss" not in mantle_id.lower():
            api_support = ["responses"]
        elif "gpt-oss" in mantle_id.lower():
            api_support = ["chat_completions", "responses"]
        # Safeguard models are utility, not for general use
        elif "safeguard" in mantle_id.lower():
            continue

        # Derive display name from model_id
        # e.g., "openai.gpt-5.4" → "GPT-5.4", "qwen.qwen3-coder-next" → "Qwen3 Coder Next"
        raw_name = mantle_id.split(".", 1)[-1] if "." in mantle_id else mantle_id
        # Preserve version numbers (don't split on dots within the model name)
        # Convert hyphens to spaces for title casing, but keep version dots
        display_name = raw_name.replace("-", " ").replace("_", " ")
        # Title case each word but preserve uppercase acronyms like "GPT", "VL"
        words = display_name.split()
        display_name = " ".join(
            w.upper() if w.lower() in ("gpt", "vl", "glm", "oss") else w.title()
            for w in words
        )

        # Try to get quality/pricing from existing catalog or AA
        existing = existing_by_id.get(mantle_id, {})
        quality_baseline, aa_pricing = match_quality_baseline(display_name, mantle_id, aa_models)

        # LiteLLM pricing lookup
        litellm_entry = litellm_data.get(mantle_id) or litellm_data.get(f"bedrock/{mantle_id}") or litellm_data.get(f"bedrock_mantle/{mantle_id}") or {}
        # Try provider-specific prefixes (e.g., xai/grok-4.3 for xai.grok-4.3)
        if not litellm_entry.get("input_cost_per_token"):
            family_prefix = mantle_id.split(".", 1)[0] if "." in mantle_id else ""
            model_name = mantle_id.split(".", 1)[1] if "." in mantle_id else mantle_id
            if family_prefix:
                litellm_entry = litellm_data.get(f"{family_prefix}/{model_name}") or litellm_entry
        if litellm_entry.get("input_cost_per_token"):
            pricing = {
                "input_per_1k": round(litellm_entry["input_cost_per_token"] * 1000, 6),
                "output_per_1k": round((litellm_entry.get("output_cost_per_token") or 0) * 1000, 6),
                "cache_read_per_1k": 0.0,
                "cache_write_per_1k": 0.0,
            }
        elif aa_pricing:
            pricing = {"input_per_1k": aa_pricing["input_per_1k"], "output_per_1k": aa_pricing["output_per_1k"],
                       "cache_read_per_1k": 0.0, "cache_write_per_1k": 0.0}
        elif existing and existing.get("pricing", {}).get("input_per_1k", 0) > 0:
            pricing = existing["pricing"]
        else:
            pricing = {"input_per_1k": 0.0, "output_per_1k": 0.0, "cache_read_per_1k": 0.0, "cache_write_per_1k": 0.0}

        # Context window for Mantle-only models (no Converse probe available):
        # Existing catalog > LiteLLM (existing may have manual corrections from AWS docs)
        max_input = existing.get("max_input_tokens") or litellm_entry.get("max_input_tokens") or 128000
        max_output = existing.get("max_output_tokens") or litellm_entry.get("max_output_tokens") or 16384

        # Tier
        tier = existing.get("tier") or derive_tier(
            quality_baseline, mantle_id, display_name,
            price_in=pricing.get("input_per_1k", 0),
            api_support=api_support,
            price_out=pricing.get("output_per_1k", 0),
            max_input=max_input,
            max_output=max_output,
        )

        # Get region availability from Mantle scan
        mantle_region_map = getattr(discover_mantle_models, '_model_regions', {})
        model_mantle_regions = mantle_region_map.get(mantle_id, {region})

        # For Responses-only models, probe /v1/responses in each region to
        # confirm actual availability. The /v1/models list includes all models
        # known to Mantle but doesn't guarantee Responses API support per-region.
        if api_support == ["responses"] and len(model_mantle_regions) > 1:
            confirmed_regions = _probe_responses_regions(mantle_id, model_mantle_regions)
            if confirmed_regions:
                model_mantle_regions = confirmed_regions
                logger.info(f"    {mantle_id}: Responses API confirmed in {sorted(confirmed_regions)}")
            else:
                logger.warning(f"    {mantle_id}: Responses API not confirmed in any region, keeping /v1/models list")

        entry = {
            "model_id": mantle_id,
            "family": family,
            "regions": [{"name": r, "direct": True} for r in sorted(model_mantle_regions)],
            "tier": tier,
            "display_name": display_name,
            "capabilities": {
                "tool_use": True,  # Mantle models generally support tool_use via Chat Completions
                "vision": False,
                "streaming": True,
                "streaming_tool_use": True,
                "document_support": False,
                "extended_thinking": False,
                "prompt_caching": False,
            },
            "max_input_tokens": max_input,
            "max_output_tokens": max_output,
            "pricing": pricing,
            "supported_latency_modes": [],
            "guardrail_compatible": False,  # Mantle-only models don't use Bedrock guardrails
            "quality_baseline": quality_baseline,
            "api_support": api_support,
        }

        # Add responses_path for models that support Responses API
        if "responses" in api_support:
            # Determine path: GPT-5.x uses /openai/v1/responses, others use /v1/responses
            if "gpt-5" in mantle_id.lower():
                entry["responses_path"] = "/openai/v1/responses"
            else:
                entry["responses_path"] = "/v1/responses"
            # Override from existing catalog if available (previously probed)
            if existing and existing.get("responses_path"):
                entry["responses_path"] = existing["responses_path"]

        catalog.append(entry)
        logger.info(f"    Added: {mantle_id} (api_support={api_support}, tier={tier})")

    logger.info(f"  Catalog now has {len(catalog)} entries")
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

    # Step 2: Discover CRIS profiles (multi-region scan)
    profiles = discover_profiles(bedrock)

    # Step 2b: Discover Mantle models (bedrock-mantle endpoint)
    mantle_models = discover_mantle_models(args.region)

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
            return model_id, probe_capabilities(bedrock_runtime, model_id, skip_probes=False,
                                                mantle_models=mantle_models, region=args.region)

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
        # Report max_output_tokens probes
        output_probed = sum(1 for c in capabilities.values() if c.get("max_output_tokens_probed") is not None)
        logger.info(f"  Max output tokens probed: {output_probed}/{len(capabilities)} (Converse server-enforced limits)")
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
    catalog = build_catalog(models, profiles, capabilities, aa_models, litellm_data, existing_catalog, region=args.region, mantle_models=mantle_models)

    # Step 6b: Add Mantle-only models (not discovered via ListFoundationModels)
    catalog = add_mantle_only_models(catalog, mantle_models, litellm_data, aa_models, existing_catalog, region=args.region)

    # Step 6c: Remove utility/moderation models that shouldn't be routed to.
    # These are specialized models (content classification, guardrails) that don't
    # produce conversational responses and should never be selected by the router.
    UTILITY_MODEL_PATTERNS = ["safeguard", "rerank", "embed"]
    pre_filter_count = len(catalog)
    catalog = [
        m for m in catalog
        if not any(pat in m["model_id"].lower() for pat in UTILITY_MODEL_PATTERNS)
    ]
    removed = pre_filter_count - len(catalog)
    if removed:
        logger.info(f"Step 6c: Removed {removed} utility/moderation models (safeguard, rerank, embed)")

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
