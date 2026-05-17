#!/usr/bin/env python3
"""Test CRIS profile selection with various exclude patterns.

Verifies that:
1. Excluding global.* → never invokes via global. prefix
2. Excluding us.* → never invokes via us. prefix (uses global. instead)
3. Excluding both global.* and us.* → uses direct model IDs only
4. No exclusions → uses preferred geography or global
5. No double-prefixing (global.global.*, us.us.*, etc.)
6. Models already with a prefix in the registry don't get double-prefixed
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bedrock_smart_router import BedrockRouter
from bedrock_smart_router.config import RoutingConfig

REGION = "us-west-2"

PROMPTS = [
    {"text": "Hello", "label": "simple"},
    {"text": "Explain the differences between Amazon S3 storage classes.", "label": "moderate"},
    {"text": "Design a distributed system for real-time fraud detection at scale with sub-100ms latency.", "label": "complex"},
]


def run_scenario(name, config, expected_prefixes_allowed, expected_prefixes_blocked):
    """Run a scenario and verify CRIS profile selection.
    
    Args:
        name: Scenario name
        config: Router config dict
        expected_prefixes_allowed: Prefixes that ARE allowed (e.g., ["us.", "global."])
        expected_prefixes_blocked: Prefixes that must NOT appear
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"  Config: excluded_models={config.get('excluded_models', [])}")
    print(f"  Allowed prefixes: {expected_prefixes_allowed or 'any'}")
    print(f"  Blocked prefixes: {expected_prefixes_blocked}")
    print(f"{'='*70}")
    
    router = BedrockRouter.create(config)
    print(f"  CRIS allow_global: {router._cris.config.allow_global}")
    
    passed = 0
    failed = 0
    
    for prompt in PROMPTS:
        try:
            resp = router.converse(
                messages=[{"role": "user", "content": [{"text": prompt["text"]}]}],
                routing=RoutingConfig(strategy="quality-optimized"),
                inferenceConfig={"maxTokens": 20},
            )
            d = resp.get("routing_decision")
            if not d:
                print(f"  [{prompt['label']}] ❌ No routing decision")
                failed += 1
                continue
            
            model = d.selected_model
            cris = d.cris_profile or model
            
            # Check for double-prefix bug
            for prefix in ["global.", "us.", "eu.", "ap."]:
                if cris.startswith(f"{prefix}{prefix}"):
                    print(f"  [{prompt['label']}] ❌ DOUBLE PREFIX: {cris}")
                    failed += 1
                    continue
            
            # Check blocked prefixes
            blocked = False
            for prefix in expected_prefixes_blocked:
                if cris.startswith(prefix):
                    print(f"  [{prompt['label']}] ❌ BLOCKED prefix used: {cris}")
                    failed += 1
                    blocked = True
                    break
            
            if not blocked:
                print(f"  [{prompt['label']}] ✅ model={model}, invoked_as={cris} ({d.latency_ms}ms)")
                passed += 1
                
        except Exception as e:
            err = str(e)[:200]
            # Check if error contains double-prefix
            if "global.global." in err or "us.us." in err:
                print(f"  [{prompt['label']}] ❌ DOUBLE PREFIX in error: {err}")
                failed += 1
            else:
                print(f"  [{prompt['label']}] ⚠️  Error (fallback exhausted): {err[:100]}")
                failed += 1
    
    print(f"\n  Result: {passed} passed, {failed} failed")
    return failed == 0


# ═══════════════════════════════════════════════════════════════════════
# Run all scenarios
# ═══════════════════════════════════════════════════════════════════════

results = []

# Scenario 1: Exclude global.* → should use us. or direct
results.append(run_scenario(
    "Exclude global.* (should use us. or direct)",
    {"region": REGION, "excluded_models": ["deepseek.*", "global.*"]},
    expected_prefixes_allowed=["us.", ""],  # us. or no prefix
    expected_prefixes_blocked=["global."],
))

# Scenario 2: Exclude us.* → should use global. or direct
results.append(run_scenario(
    "Exclude us.* (should use global. or direct)",
    {"region": REGION, "excluded_models": ["deepseek.*", "us.*"]},
    expected_prefixes_allowed=["global.", ""],  # global. or no prefix
    expected_prefixes_blocked=["us."],
))

# Scenario 3: Exclude both global.* and us.* → direct only
results.append(run_scenario(
    "Exclude both global.* and us.* (direct invocation only)",
    {"region": REGION, "excluded_models": ["deepseek.*", "global.*", "us.*"]},
    expected_prefixes_allowed=[""],  # no prefix only
    expected_prefixes_blocked=["global.", "us."],
))

# Scenario 4: No exclusions (default) → should work, prefer global
results.append(run_scenario(
    "No exclusions (default behavior)",
    {"region": REGION, "excluded_models": ["deepseek.*"]},
    expected_prefixes_allowed=["global.", "us.", ""],  # any
    expected_prefixes_blocked=[],  # nothing blocked
))

# Scenario 5: Exclude global.* with preferred_geography=us
results.append(run_scenario(
    "Exclude global.* + preferred_geography=us",
    {"region": REGION, "excluded_models": ["deepseek.*", "global.*"], "cris": {"preferred_geography": "us"}},
    expected_prefixes_allowed=["us.", ""],
    expected_prefixes_blocked=["global."],
))

# Scenario 6: Exclude specific model family + global
results.append(run_scenario(
    "Exclude anthropic.* and global.* (non-Anthropic models only)",
    {"region": REGION, "excluded_models": ["deepseek.*", "anthropic.*", "global.*"]},
    expected_prefixes_allowed=["us.", ""],
    expected_prefixes_blocked=["global.", "anthropic."],
))

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)
total = len(results)
passed = sum(results)
failed = total - passed
for i, (r, name) in enumerate(zip(results, [
    "Exclude global.*",
    "Exclude us.*", 
    "Exclude both global.* and us.*",
    "No exclusions",
    "Exclude global.* + preferred_geography=us",
    "Exclude anthropic.* and global.*",
])):
    print(f"  {'✅' if r else '❌'} Scenario {i+1}: {name}")

print(f"\n  {passed}/{total} scenarios passed")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
