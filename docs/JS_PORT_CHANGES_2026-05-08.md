# Bedrock Smart Router — Changes Report (May 8, 2026)

This document details all functionality implemented in the Python version on this date. The JS port needs to implement equivalent logic.

---

## Table of Contents

1. [Model Catalog Data Fixes](#1-model-catalog-data-fixes)
2. [Document Support Filtering](#2-document-support-filtering)
3. [Multimodal Payload Size → Complexity Boost](#3-multimodal-payload-size--complexity-boost)
4. [Boto Client Configuration Passthrough](#4-boto-client-configuration-passthrough)
5. [Retry Conflict Prevention](#5-retry-conflict-prevention)
6. [Model Discovery Script (`--discover`)](#6-model-discovery-script---discover)

---

## 1. Model Catalog Data Fixes

### Files Changed
- `bedrock_smart_router/data/models.json`

### What Was Wrong
The model catalog had incorrect capability flags and token limits for many models. These were audited against the official AWS Bedrock model card documentation.

### Changes Applied

#### `document_support` — was `false` for ALL models, corrected to `true` for:
- All Anthropic Claude models (Haiku 4.5, Sonnet 4.5, Sonnet 4.6, Opus 4.1, Opus 4.5, Opus 4.6, Opus 4.7 + all global variants)
- Amazon Nova Lite 1.0, Nova 2 Lite, Nova Pro 1.0 (+ global variants)
- Mistral Pixtral Large

Models correctly left as `document_support: false`:
- Amazon Nova Micro (text-only)
- All Meta Llama models (vision-only for Llama 4, text-only for 3.x)
- DeepSeek R1 (text-only)

#### `tool_use` — corrected for 3 models:
| Model | Was | Now | Reason |
|-------|-----|-----|--------|
| Nova Micro | `false` | `true` | AWS docs confirm "Client-side tool calling" supported |
| Nova Lite 1.0 | `false` | `true` | AWS docs confirm "Client-side tool calling" supported |
| Llama 3.1 8B | `false` | `true` | AWS docs confirm "Client-side tool calling" supported |

#### `extended_thinking` — corrected for 2 models:
| Model | Was | Now | Reason |
|-------|-----|-----|--------|
| Claude Haiku 4.5 | `false` | `true` | AWS docs say "Reasoning: Supported" |
| Claude Haiku 4.5 (Global) | `false` | `true` | Same model |

#### `max_output_tokens` — corrected for 8 models:
| Model | Was | Now | Source |
|-------|-----|-----|--------|
| Nova 2 Lite | 5,000 | 64,000 | AWS docs: "Max output tokens: 64K" |
| Nova 2 Lite (Global) | 5,000 | 64,000 | Same model |
| Claude Haiku 4.5 | 8,192 | 64,000 | AWS docs: "Max output tokens: 64K" |
| Claude Haiku 4.5 (Global) | 8,192 | 64,000 | Same model |
| Llama 3.1 8B | 2,048 | 4,096 | AWS docs: "Max output tokens: 4K" |
| Llama 4 Scout | 16,384 | 8,192 | AWS docs: "Max output tokens: 8K" |
| Llama 4 Maverick | 16,384 | 8,192 | AWS docs: "Max output tokens: 8K" |
| Mistral Pixtral Large | 8,192 | 16,384 | AWS docs: "Max output tokens: 16K" |

#### `max_input_tokens` — corrected for 1 model:
| Model | Was | Now | Source |
|-------|-----|-----|--------|
| Llama 4 Scout | 3,500,000 | 10,000,000 | AWS docs: "Context window: 10M tokens" |

### JS Port Action
Update your model catalog JSON with the same corrections. The JS catalog should mirror the Python `models.json` exactly.

---

## 2. Document Support Filtering

### Files Changed
- `bedrock_smart_router/request_analyzer.py`
- `bedrock_smart_router/models.py`
- `bedrock_smart_router/model_registry.py`
- `bedrock_smart_router/router.py`

### Problem
When a user sent a PDF via the Bedrock Converse API's `document` content block, the router could route it to a model that doesn't support documents (e.g., Llama, DeepSeek), causing a Bedrock API error. Vision (image) filtering already worked, but document filtering did not exist.

### Implementation

#### A. Request Analyzer — detect document content blocks

Added `_has_documents()` static method (mirrors existing `_has_images()`):

```python
@staticmethod
def _has_documents(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "document" in block:
                    return True
    return False
```

The `analyze()` method now calls this and sets `requires_document_support` on the result.

#### B. RequestAnalysis model — new field

Added to the `RequestAnalysis` dataclass:
```python
requires_document_support: bool = False
```

Positioned between `requires_vision` and `requires_tool_use`.

#### C. Model Registry — filter by document_support

Added `requires_document_support: bool = False` parameter to `eligible_models()`:

```python
if requires_document_support and not m.capabilities.document_support:
    continue
```

#### D. Router — pass the flag through

In `_resolve_model()`:
```python
candidates = self._registry.eligible_models(
    ...
    requires_document_support=analysis.requires_document_support,
    ...
)
```

In `_raise_no_models_error()`, added rejection reason:
```python
if analysis.requires_document_support and not m.capabilities.document_support:
    reasons.append("no document support")
```

Also added `requires_document_support` to the constraints dict for error reporting.

### JS Port Action
1. Add `_hasDocuments(messages)` detection in your request analyzer
2. Add `requiresDocumentSupport` field to your analysis result type
3. Add filtering in your `eligibleModels()` method
4. Pass it through in your routing pipeline
5. Include it in error rejection reasons

---

## 3. Multimodal Payload Size → Complexity Boost

### Files Changed
- `bedrock_smart_router/request_analyzer.py`

### Problem
A 50-page PDF with a one-word prompt ("summarize") was classified as "simple" because the complexity classifier only looked at text content. This caused it to route to the cheapest model, which may not handle large documents well.

### Implementation

#### A. Payload size measurement

Added `_multimodal_payload_bytes(messages)` static method:

```python
@staticmethod
def _multimodal_payload_bytes(messages: list[dict]) -> int:
    """Sum the byte size of all inline image and document payloads."""
    total = 0
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if "image" in block:
                source = block["image"].get("source", {})
                data = source.get("bytes")
                if isinstance(data, (bytes, bytearray)):
                    total += len(data)
            if "document" in block:
                source = block["document"].get("source", {})
                data = source.get("bytes")
                if isinstance(data, (bytes, bytearray)):
                    total += len(data)
    return total
```

Only measures inline `bytes` payloads. S3/URL references are skipped (can't measure what's not inline).

#### B. Composite score boost

After computing the weighted composite from 12 dimensions, a direct boost is applied based on payload size:

```python
payload_bytes = self._multimodal_payload_bytes(messages)
if payload_bytes > 0:
    if payload_bytes > 5_000_000:       # > 5MB
        composite += 0.30
    elif payload_bytes > 1_000_000:     # > 1MB
        composite += 0.20
    elif payload_bytes > 100_000:       # > 100KB
        composite += 0.10
    else:                               # < 100KB
        composite += 0.05
composite = max(0.0, min(1.0, composite))
```

#### C. Document analysis dimension boost

The `doc_score` dimension (index 7) also gets a +0.3 boost when any multimodal content is present:

```python
payload_bytes = self._multimodal_payload_bytes(messages)
if payload_bytes > 0:
    doc_score = min(1.0, doc_score + 0.3)
```

#### D. Token estimation for multimodal content

After estimating text tokens, additional tokens are added for multimodal payloads:

```python
if payload_bytes > 0:
    if has_documents:
        # ~3KB per page, ~1500 tokens per page
        est_pages = max(1, payload_bytes // 3000)
        est_input += int(est_pages * 1500)
    elif has_images:
        # ~750 tokens per image
        image_count = <count image blocks>
        est_input += image_count * 750
```

This ensures context window pre-validation works correctly for large payloads.

### Effect on Routing

| Payload | Complexity | Tier Selected |
|---------|-----------|---------------|
| Text only | simple (0.05) | micro/lite |
| Small image (<100KB) | simple (0.12) | lite |
| Large image (3MB) | moderate (0.28) | mid (Sonnet, Nova Pro) |
| Small PDF (80KB) | simple (0.13) | lite |
| Large PDF (5MB) | moderate (0.28) | mid |
| Huge PDF (10MB) | moderate (0.38) | mid/heavy |

### JS Port Action
1. Add `multimodalPayloadBytes(messages)` helper
2. Apply the same composite boost thresholds after weighted scoring
3. Boost the document_analysis dimension score by 0.3 when multimodal content is present
4. Add token estimation for images (~750/image) and documents (~1500 tokens per 3KB page)

---

## 4. Boto Client Configuration Passthrough

### Files Changed
- `bedrock_smart_router/config.py`
- `bedrock_smart_router/router.py`

### Problem
Users couldn't configure the underlying Bedrock client's timeouts, retries, or connection settings. The default 60-second `read_timeout` is insufficient for large document/image payloads.

### Implementation

#### A. RouterConfig — new field

Added to `RouterConfig`:
```python
boto_config: dict[str, Any] | None = None
```

Parsed in `from_dict()`:
```python
boto_config=data.get("boto_config"),
```

#### B. Router — two entry points

**Option A: Direct Config object (keyword arg on `create()`)**

```python
@classmethod
def create(cls, config=None, *, boto_session=None, boto_config=None, callbacks=None):
```

**Option B: Dict/YAML config**

```python
router = BedrockRouter.create({
    "boto_config": {
        "read_timeout": 300,
        "connect_timeout": 10,
        "retries": {"max_attempts": 3, "mode": "adaptive"},
    },
})
```

#### C. Resolution logic in `__init__`

```python
# Resolve: explicit param > config dict > None
resolved_boto_config = boto_config
if resolved_boto_config is None and config.boto_config:
    from botocore.config import Config as BotocoreConfig
    resolved_boto_config = BotocoreConfig(**config.boto_config)

# Pass to client
client_kwargs = {}
if resolved_boto_config is not None:
    client_kwargs["config"] = resolved_boto_config
self._bedrock = session.client("bedrock-runtime", **client_kwargs)
```

Option A (explicit kwarg) takes precedence over Option B (dict config).

### JS Port Action
In the JS/TS version, the equivalent is passing config to the AWS SDK v3 `BedrockRuntimeClient`:
```typescript
new BedrockRuntimeClient({
  region: "us-west-2",
  requestHandler: new NodeHttpHandler({
    connectionTimeout: 10000,
    socketTimeout: 300000,
  }),
  maxAttempts: 3,
  retryMode: "adaptive",
})
```

Implement a similar passthrough where users can provide SDK client config either as a direct object or via the config dict.

---

## 5. Retry Conflict Prevention

### Files Changed
- `bedrock_smart_router/router.py`

### Problem
The router has its own `RetryHandler` (3 retries with exponential backoff). If a user also configures retries in `boto_config`, a single throttled request could be retried up to `boto_retries × router_retries` times (e.g., 3 × 4 = 12 attempts).

### Implementation

After creating the `RetryHandler` with the user's config, check if `boto_config` includes retries. If so, replace the handler with a passthrough (max_retries=0):

```python
if resolved_boto_config is not None:
    has_user_retries = False
    if hasattr(resolved_boto_config, 'retries') and resolved_boto_config.retries:
        has_user_retries = True
    if config.boto_config and "retries" in config.boto_config:
        has_user_retries = True
    if has_user_retries:
        self._retry_handler = RetryHandler(RetryConfig(max_retries=0))
```

### Behavior Matrix

| Configuration | Who retries | Fallback to next model |
|---|---|---|
| No `boto_config` retries | Router's RetryHandler (3 retries + backoff) | Yes, after retries exhausted |
| `boto_config` with retries | boto3/SDK (internal) | Yes, on final failure |

The fallback chain (trying the next model in the chain) always works regardless of which layer handles retries.

### JS Port Action
Implement the same logic: if the user provides `maxAttempts` or retry config to the SDK client, disable your internal retry handler. The SDK will handle retries; your router just handles fallback to the next model.

---

## 6. Model Discovery Script (`--discover`)

### Files Changed
- `scripts/refresh_pricing.py`
- `README.md`

### Problem
Adding new models to the catalog was entirely manual. When AWS launches a new model, someone had to manually create the full JSON entry with all fields.

### Implementation

Added a `--discover` flag to `scripts/refresh_pricing.py` that:

1. Calls `bedrock:ListFoundationModels` to get all active models
2. Filters out embeddings, image generation, video, and LEGACY models
3. Compares against existing catalog entries (strips geo prefixes for matching)
4. Creates skeleton entries for missing models using family-based heuristics

#### Family Heuristics

| Family | tool_use | document_support | prompt_caching | streaming_tool_use | Default context |
|--------|----------|-----------------|----------------|-------------------|-----------------|
| anthropic | true | true | true | true | 200K in / 64K out |
| amazon | true | true | true | true | 300K in / 5K out |
| meta | true | false | false | false | 128K in / 8K out |
| mistral | true | false | false | true | 128K in / 8K out |
| deepseek | false | false | false | true | 128K in / 8K out |

#### Tier Inference from Model Name

Keywords in the model name determine the tier:
- `micro`, `mini`, `1b`, `3b`, `8b`, `nano` → `micro`
- `lite`, `haiku`, `small`, `scout`, `11b`, `12b` → `lite`
- `pro`, `sonnet`, `large`, `70b`, `maverick` → `mid`
- `premier`, `opus`, `405b` → `heavy`
- `r1`, `reasoning`, `think` → `reasoning`

#### What comes from the API vs heuristics

| Attribute | Source | Reliable? |
|-----------|--------|-----------|
| model_id | API | ✅ |
| family | API (`providerName`) | ✅ |
| display_name | API (`modelName`) | ✅ |
| vision | API (`"IMAGE" in inputModalities`) | ✅ |
| streaming | API (`responseStreamingSupported`) | ✅ |
| tool_use | Family heuristic | ⚠️ Needs review |
| document_support | Family heuristic | ⚠️ Needs review |
| extended_thinking | Family heuristic (default false) | ⚠️ Needs review |
| prompt_caching | Family heuristic | ⚠️ Needs review |
| max_input/output_tokens | Family default | ⚠️ Needs review |
| pricing | Pricing API (if available, else $0.00) | ⚠️ May be delayed |
| tier | Name heuristic | ⚠️ Needs review |

New entries are marked with `_needs_review: true` so maintainers know to verify them.

#### Usage

```bash
# Dry run — see what new models exist
python scripts/refresh_pricing.py --discover

# Add them to the catalog
python scripts/refresh_pricing.py --discover --fix

# Discover + create global entries
python scripts/refresh_pricing.py --discover --fix --regen-global

# Only discover from a specific provider
python scripts/refresh_pricing.py --discover --provider anthropic
```

### JS Port Action
Implement an equivalent discovery script/command that:
1. Calls `ListFoundationModels` via the AWS SDK
2. Compares against your model catalog
3. Generates skeleton entries with family-based defaults
4. Marks them for human review

---

## Summary of All Files Modified

| File | Changes |
|------|---------|
| `bedrock_smart_router/data/models.json` | Fixed `document_support`, `tool_use`, `extended_thinking`, `max_output_tokens`, `max_input_tokens` for 18+ models |
| `bedrock_smart_router/models.py` | Added `requires_document_support: bool` field to `RequestAnalysis` |
| `bedrock_smart_router/request_analyzer.py` | Added `_has_documents()`, `_multimodal_payload_bytes()`, composite boost logic, token estimation for multimodal |
| `bedrock_smart_router/model_registry.py` | Added `requires_document_support` parameter to `eligible_models()` |
| `bedrock_smart_router/router.py` | Pass `requires_document_support` to eligible_models, add rejection reason, add `boto_config` support, add retry conflict prevention |
| `bedrock_smart_router/config.py` | Added `boto_config: dict | None` field to `RouterConfig`, parse in `from_dict()` |
| `scripts/refresh_pricing.py` | Added `--discover` flag with full model discovery logic, family heuristics, tier inference |
| `README.md` | Added sections on multimodal routing, boto client config, retry conflict prevention, updated `--discover` docs |
