#!/usr/bin/env python3
"""Refresh models.json pricing from the live AWS Pricing API.

Pulls on-demand Standard tier pricing for every model in our catalog,
compares it with what we have, and optionally auto-fixes mismatches.

Usage:
    # Dry run — show what's different, don't change anything
    python scripts/refresh_pricing.py

    # Apply fixes to models.json
    python scripts/refresh_pricing.py --fix

    # Check a specific provider only
    python scripts/refresh_pricing.py --provider amazon
    python scripts/refresh_pricing.py --provider anthropic

    # Use a different region
    python scripts/refresh_pricing.py --region us-east-1

    # Also check and fix service tier support
    python scripts/refresh_pricing.py --fix --check-tiers

    # Check for legacy models and remove them
    python scripts/refresh_pricing.py --check-legacy
    python scripts/refresh_pricing.py --fix --check-legacy

    # Verbose: show all AWS pricing rows (for debugging)
    python scripts/refresh_pricing.py --verbose

    # Also regenerate global.* entries (10% discount) after fixing
    python scripts/refresh_pricing.py --fix --regen-global

Requires: AWS credentials with pricing:GetProducts permission.
Note: The Pricing API endpoint is only in us-east-1 and ap-south-1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import boto3

CATALOG_PATH = Path(__file__).parent.parent / "bedrock_smart_router" / "data" / "models.json"

GLOBAL_DISCOUNT = 0.90  # Global CRIS profiles are ~10% cheaper

# ── Mapping: our display_name -> (pricing_api_provider, pricing_api_model) ──
#
# The AWS Pricing API uses its own naming for providers and models.
# Amazon Nova models have provider=None so we query by model name only.
# Newer Anthropic Claude 4.x models are NOT in the Pricing API yet.
#
MODEL_MAP: dict[str, tuple[str | None, str]] = {
    # Amazon Nova (provider=None in Pricing API)
    "Amazon Nova Micro 1.0":     (None, "Nova Micro"),
    "Amazon Nova Lite 1.0":      (None, "Nova Lite"),
    "Amazon Nova 2 Lite":        (None, "Nova 2.0 Lite"),
    "Amazon Nova Pro 1.0":       (None, "Nova Pro"),
    "Amazon Nova Premier 1.0":   (None, "Nova Premier"),
    # Anthropic — only legacy models are in the Pricing API
    # Claude 4.x models are NOT available via the API as of April 2026
    # "Claude Haiku 4.5":        ("Anthropic", "Claude Haiku 4.5"),  # NOT IN API
    # Meta Llama
    "Llama 3.1 8B Instruct":    ("Meta", "Llama 3.1 8B"),
    "Llama 3.1 70B Instruct":   ("Meta", "Llama 3.1 70B"),
    "Llama 3.2 1B Instruct":    ("Meta", "Llama 3.2 1B"),
    "Llama 3.2 3B Instruct":    ("Meta", "Llama 3.2 3B"),
    "Llama 3.2 11B Instruct":   ("Meta", "Llama 3.2 11B"),
    "Llama 3.2 90B Instruct":   ("Meta", "Llama 3.2 90B"),
    "Llama 3.3 70B Instruct":   ("Meta", "Llama 3.3 70B"),
    "Llama 4 Scout 17B Instruct":   ("Meta", "Llama 4 Scout 17B"),
    "Llama 4 Maverick 17B Instruct": ("Meta", "Llama 4 Maverick 17B"),
    # Mistral (provider="Mistral" for older models)
    "Mistral Small":             ("Mistral", "Mistral Small"),
    "Mistral Large 2":           ("Mistral", "Mistral Large 2407"),
    "Mistral Pixtral Large":     ("Mistral", "Pixtral Large 25.02"),
    # DeepSeek
    "DeepSeek R1":               ("DeepSeek", "R1"),
}


# ── Pricing API ─────────────────────────────────────────────────────

def fetch_model_pricing(
    client: Any,
    region: str,
    provider: str | None,
    model_name: str,
) -> list[dict]:
    """Fetch all pricing rows for a single model from the AWS Pricing API."""
    filters = [
        {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
        {"Type": "TERM_MATCH", "Field": "model", "Value": model_name},
    ]
    if provider:
        filters.append({"Type": "TERM_MATCH", "Field": "provider", "Value": provider})

    rows: list[dict] = []
    next_token: str | None = None

    while True:
        kwargs: dict = {"ServiceCode": "AmazonBedrock", "Filters": filters, "MaxResults": 100}
        if next_token:
            kwargs["NextToken"] = next_token
        resp = client.get_products(**kwargs)

        for price_json in resp["PriceList"]:
            data = json.loads(price_json)
            attrs = data.get("product", {}).get("attributes", {})
            terms = data.get("terms", {}).get("OnDemand", {})
            for term_val in terms.values():
                for dim_val in term_val.get("priceDimensions", {}).values():
                    rows.append({
                        "usagetype": attrs.get("usagetype", ""),
                        "inference_type": attrs.get("inferenceType", ""),
                        "service_tier": attrs.get("service_tier", ""),
                        "price": float(dim_val.get("pricePerUnit", {}).get("USD", "0")),
                        "unit": dim_val.get("unit", ""),
                    })

        next_token = resp.get("NextToken")
        if not next_token:
            break
    return rows


def extract_standard_pricing(rows: list[dict]) -> dict[str, float]:
    """Extract standard-tier on-demand pricing from API rows.

    Returns dict with keys: input_per_1k, output_per_1k,
    cache_read_per_1k, cache_write_per_1k.
    """
    pricing: dict[str, float] = {}
    for r in rows:
        usage = r["usagetype"].lower()
        inf = r["inference_type"].lower()

        # Skip batch, custom-model, and latency-optimized rows
        if "batch" in usage or "custom-model" in usage or "latency-optimized" in usage:
            continue

        # Skip non-standard tiers if tier field is populated
        tier = (r.get("service_tier") or "").lower()
        if tier and tier not in ("", "standard"):
            continue

        if "cache-read" in usage or "cache read" in inf:
            pricing["cache_read_per_1k"] = r["price"]
        elif "cache-write" in usage or "cache write" in inf:
            pricing["cache_write_per_1k"] = r["price"]
        elif "input" in inf:
            pricing["input_per_1k"] = r["price"]
        elif "output" in inf:
            pricing["output_per_1k"] = r["price"]

    return pricing


def extract_supported_tiers(rows: list[dict]) -> set[str]:
    """Extract which service tiers have pricing rows."""
    tiers: set[str] = set()
    for r in rows:
        usage = r["usagetype"].lower()
        if "batch" in usage or "custom-model" in usage or "latency-optimized" in usage:
            continue
        tier = (r.get("service_tier") or "").lower()
        if tier:
            tiers.add(tier)
        else:
            tiers.add("standard")
    return tiers


# ── Catalog helpers ─────────────────────────────────────────────────

def load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def save_catalog(data: dict) -> None:
    with open(CATALOG_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ── Legacy detection ────────────────────────────────────────────────

def fetch_legacy_model_ids(region: str) -> set[str]:
    """Query the Bedrock API for foundation models marked as LEGACY.

    Returns a set of base model IDs (e.g. ``meta.llama3-2-1b-instruct-v1:0``)
    that have ``modelLifecycle.status == "LEGACY"``.
    """
    bedrock = boto3.client("bedrock", region_name=region)
    legacy: set[str] = set()

    paginator_kwargs: dict[str, Any] = {}
    while True:
        resp = bedrock.list_foundation_models(**paginator_kwargs)
        for model in resp.get("modelSummaries", []):
            status = model.get("modelLifecycle", {}).get("status", "ACTIVE")
            if status == "LEGACY":
                legacy.add(model["modelId"])
        # ListFoundationModels doesn't paginate, but be safe
        if "nextToken" in resp:
            paginator_kwargs["nextToken"] = resp["nextToken"]
        else:
            break

    return legacy


def check_legacy_models(
    catalog: dict,
    region: str,
    fix: bool = False,
) -> int:
    """Check catalog models against the Bedrock LEGACY list.

    Returns the number of legacy models found.  When *fix* is True,
    removes them from the catalog dict in-place.
    """
    print(f"\n  {'─'*64}")
    print(f"  Checking for LEGACY models via Bedrock API (region: {region})...")

    legacy_ids = fetch_legacy_model_ids(region)
    if not legacy_ids:
        print(f"    No legacy models reported by Bedrock API")
        return 0

    print(f"    Bedrock reports {len(legacy_ids)} legacy foundation models")

    # Match catalog entries: strip the geography prefix (us.*, global.*)
    # to compare against the base model ID from the Bedrock API.
    legacy_found: list[str] = []
    for m in catalog["models"]:
        model_id = m["model_id"]
        # Strip geography prefix to get the base model ID
        base = model_id
        for prefix in ("us.", "global.", "eu.", "ap."):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        if base in legacy_ids:
            legacy_found.append(model_id)
            print(f"    ⚠ LEGACY: {model_id}  ({m['display_name']})")

    if not legacy_found:
        print(f"    ✓ No legacy models in catalog")
        return 0

    print(f"    Found {len(legacy_found)} legacy model(s) in catalog")

    if fix:
        legacy_set = set(legacy_found)
        before = len(catalog["models"])
        catalog["models"] = [m for m in catalog["models"] if m["model_id"] not in legacy_set]
        after = len(catalog["models"])
        print(f"    ✓ Removed {before - after} legacy models from catalog")

    return len(legacy_found)


# ── Main logic ──────────────────────────────────────────────────────

def refresh(
    region: str = "us-west-2",
    provider_filter: str | None = None,
    fix: bool = False,
    check_tiers: bool = False,
    check_legacy: bool = False,
    regen_global: bool = False,
    verbose: bool = False,
) -> int:
    catalog = load_catalog()
    client = boto3.client("pricing", region_name="us-east-1")

    by_display: dict[str, dict] = {}
    for m in catalog["models"]:
        by_display[m["display_name"]] = m

    mismatches = 0
    legacy_count = 0
    checked = 0
    skipped_no_api = 0

    print(f"\n  Refreshing pricing from AWS Pricing API")
    print(f"  Region: {region}")
    print(f"  Catalog: {CATALOG_PATH}")
    print(f"  Models in catalog: {len(catalog['models'])}")
    print()

    for display_name, (api_provider, api_model) in sorted(MODEL_MAP.items()):
        # Filter by provider if requested
        if provider_filter:
            cat_model = by_display.get(display_name)
            if cat_model and cat_model.get("family") != provider_filter.lower():
                continue

        cat_model = by_display.get(display_name)
        if not cat_model:
            continue

        # Skip global entries — they derive from regional
        if cat_model["model_id"].startswith("global."):
            continue

        checked += 1
        print(f"  {'─'*64}")
        print(f"  {display_name}  ({cat_model['model_id']})")

        rows = fetch_model_pricing(client, region, api_provider, api_model)
        if not rows:
            print(f"    ⚠ No data from Pricing API (model may not be in API yet)")
            skipped_no_api += 1
            continue

        if verbose:
            for r in rows:
                print(f"    RAW: {r['usagetype']:55s} {r['inference_type']:35s} ${r['price']:.10f}")

        aws_pricing = extract_standard_pricing(rows)
        our_pricing = cat_model["pricing"]
        all_ok = True

        for field in ["input_per_1k", "output_per_1k", "cache_read_per_1k", "cache_write_per_1k"]:
            ours = our_pricing.get(field, 0.0)
            theirs = aws_pricing.get(field)

            if theirs is None:
                print(f"    {field:25s}  ours=${ours:.10f}  (no AWS data)")
                continue

            if abs(ours - theirs) < 1e-12:
                print(f"    {field:25s}  ${ours:.10f}  ✓")
            else:
                diff_pct = ((theirs - ours) / theirs * 100) if theirs != 0 else 0
                print(f"    {field:25s}  ours=${ours:.10f}  AWS=${theirs:.10f}  ✗ ({diff_pct:+.1f}%)")
                mismatches += 1
                all_ok = False
                if fix:
                    cat_model["pricing"][field] = theirs

        if all_ok:
            print(f"    ✓ Pricing OK")

        # Check tiers
        if check_tiers:
            aws_tiers = extract_supported_tiers(rows)
            our_tiers = set(cat_model.get("supported_inference_tiers", []))
            if aws_tiers != our_tiers:
                print(f"    ⚠ Tiers: ours={sorted(our_tiers)}  AWS={sorted(aws_tiers)}")
                mismatches += 1
                if fix:
                    cat_model["supported_inference_tiers"] = sorted(aws_tiers)
            else:
                print(f"    ✓ Tiers OK: {sorted(our_tiers)}")

    # Check for legacy models if requested
    if check_legacy:
        legacy_count = check_legacy_models(catalog, region, fix=fix)
        mismatches += legacy_count

    # Regenerate global entries if requested
    if regen_global and fix:
        print(f"\n  {'─'*64}")
        print(f"  Regenerating global.* entries ({GLOBAL_DISCOUNT:.0%} of regional pricing)...")
        # Remove existing global entries
        catalog["models"] = [m for m in catalog["models"] if not m["model_id"].startswith("global.")]
        # Rebuild from regional entries that have global profiles
        new_globals = 0
        for m in list(catalog["models"]):
            global_profiles = [p for p in m.get("cris_profiles", []) if p.startswith("global.")]
            for gp in global_profiles:
                entry = json.loads(json.dumps(m))  # deep copy
                entry["model_id"] = gp
                entry["display_name"] = m["display_name"] + " (Global)"
                entry["cris_profiles"] = [gp]
                for key in ["input_per_1k", "output_per_1k", "cache_read_per_1k", "cache_write_per_1k"]:
                    val = entry["pricing"][key]
                    if val > 0:
                        entry["pricing"][key] = round(val * GLOBAL_DISCOUNT, 10)
                catalog["models"].append(entry)
                new_globals += 1
                print(f"    + {gp}")
        print(f"    Created {new_globals} global entries")

    # Not in API — list models we couldn't validate
    not_in_api = []
    for m in catalog["models"]:
        dn = m["display_name"]
        if dn not in MODEL_MAP and not dn.endswith("(Global)"):
            not_in_api.append(dn)

    # Summary
    print(f"\n  {'='*64}")
    print(f"  Checked:        {checked} models against AWS Pricing API")
    print(f"  Mismatches:     {mismatches}")
    print(f"  No API data:    {skipped_no_api}")
    if legacy_count:
        print(f"  Legacy models:  {legacy_count}")
    if not_in_api:
        print(f"  Not mapped:     {len(not_in_api)} (no Pricing API mapping)")
        for n in not_in_api:
            print(f"                    - {n}")
    if fix and mismatches > 0:
        save_catalog(catalog)
        print(f"\n  ✓ Applied fixes to {CATALOG_PATH}")
        print(f"    Total models in catalog: {len(catalog['models'])}")
        if legacy_count:
            print(f"    Legacy models removed: {legacy_count}")
    elif mismatches > 0:
        print(f"\n  Run with --fix to auto-update models.json")
    else:
        print(f"\n  ✓ All validated pricing matches!")
    print(f"  {'='*64}\n")

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh models.json pricing from the live AWS Pricing API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--region", default="us-west-2", help="AWS region (default: us-west-2)")
    parser.add_argument("--provider", help="Filter by family (amazon, anthropic, meta, mistral, deepseek)")
    parser.add_argument("--fix", action="store_true", help="Auto-update models.json with correct pricing")
    parser.add_argument("--check-tiers", action="store_true", help="Also validate supported_inference_tiers")
    parser.add_argument("--check-legacy", action="store_true", help="Check for LEGACY models via Bedrock API and remove them")
    parser.add_argument("--regen-global", action="store_true", help="Regenerate global.* entries after fixing")
    parser.add_argument("--verbose", action="store_true", help="Show raw pricing rows from AWS")
    args = parser.parse_args()

    mismatches = refresh(
        region=args.region,
        provider_filter=args.provider,
        fix=args.fix,
        check_tiers=args.check_tiers,
        check_legacy=args.check_legacy,
        regen_global=args.regen_global,
        verbose=args.verbose,
    )
    sys.exit(1 if mismatches > 0 else 0)


if __name__ == "__main__":
    main()
