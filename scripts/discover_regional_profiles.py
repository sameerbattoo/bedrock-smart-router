#!/usr/bin/env python3
"""Discover inference profiles and foundation models across all AWS Bedrock regions.

This script:
1. Loops through all Bedrock-supported regions
2. Calls ListInferenceProfiles to find CRIS profiles per region
3. Calls ListFoundationModels to find direct-access models per region
4. Merges both into a unified per-model regional availability map

Output: scripts/_regional_profiles.json

Usage:
    python scripts/discover_regional_profiles.py
"""

import json
import boto3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# All known Bedrock-supported regions (excluding those that don't support Bedrock)
BEDROCK_REGIONS = [
    # US
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    # Europe
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-north-1",
    # Asia Pacific
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    # Canada
    "ca-central-1",
    # South America
    "sa-east-1",
]


def get_inference_profiles(region: str) -> tuple[str, list, str | None]:
    """Get all system-defined inference profiles for a region."""
    try:
        client = boto3.client("bedrock", region_name=region)
        profiles = []
        kwargs = {"maxResults": 100, "typeEquals": "SYSTEM_DEFINED"}
        while True:
            resp = client.list_inference_profiles(**kwargs)
            for p in resp.get("inferenceProfileSummaries", []):
                pid = p.get("inferenceProfileId", "")
                prefix = pid.split(".")[0] if "." in pid else ""
                base = ".".join(pid.split(".")[1:]) if "." in pid else pid
                profiles.append({"profile_id": pid, "prefix": prefix, "base_model": base})
            nt = resp.get("nextToken")
            if not nt:
                break
            kwargs["nextToken"] = nt
        return region, profiles, None
    except Exception as e:
        return region, [], str(e)


def get_foundation_models(region: str) -> tuple[str, list, str | None]:
    """Get all foundation models available in a region (direct access)."""
    try:
        client = boto3.client("bedrock", region_name=region)
        resp = client.list_foundation_models()
        models = []
        for m in resp.get("modelSummaries", []):
            # Only include models that support ON_DEMAND inference
            if "ON_DEMAND" in m.get("inferenceTypesSupported", []):
                models.append({
                    "model_id": m.get("modelId", ""),
                    "model_name": m.get("modelName", ""),
                    "provider": m.get("providerName", ""),
                    "input_modalities": m.get("inputModalities", []),
                    "output_modalities": m.get("outputModalities", []),
                })
        return region, models, None
    except Exception as e:
        return region, [], str(e)


def discover_region(region: str) -> dict:
    """Discover both inference profiles and foundation models for a region."""
    _, profiles, prof_err = get_inference_profiles(region)
    _, models, model_err = get_foundation_models(region)
    return {
        "region": region,
        "profiles": profiles,
        "models": models,
        "profile_error": prof_err,
        "model_error": model_err,
    }


def main():
    print(f"Discovering profiles & models across {len(BEDROCK_REGIONS)} regions...")
    print("=" * 70)

    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(discover_region, r): r for r in BEDROCK_REGIONS}
        for future in as_completed(futures):
            data = future.result()
            region = data["region"]
            results[region] = data
            p_count = len(data["profiles"])
            m_count = len(data["models"])
            p_status = f"{p_count} profiles" if not data["profile_error"] else f"ERR: {data['profile_error'][:40]}"
            m_status = f"{m_count} models" if not data["model_error"] else f"ERR: {data['model_error'][:40]}"
            print(f"  {region:20s} | {p_status:25s} | {m_status}")

    # --- Aggregate per-model availability ---
    print(f"\n{'=' * 70}")
    print("Aggregating per-model availability...")

    # model_id -> { regions: { region: { cris_profiles: set, direct: bool } }, metadata }
    model_map = defaultdict(lambda: {
        "regions": defaultdict(lambda: {"cris_profiles": set(), "direct": False}),
        "provider": "",
        "model_name": "",
    })
    all_prefixes = set()

    for region, data in results.items():
        # From inference profiles
        for p in data["profiles"]:
            base_model = p["base_model"]
            prefix = p["prefix"]
            all_prefixes.add(prefix)
            model_map[base_model]["regions"][region]["cris_profiles"].add(prefix)

        # From foundation models (direct access)
        for m in data["models"]:
            model_id = m["model_id"]
            model_map[model_id]["regions"][region]["direct"] = True
            if not model_map[model_id]["provider"]:
                model_map[model_id]["provider"] = m["provider"]
                model_map[model_id]["model_name"] = m["model_name"]

    # --- Summary ---
    cris_models = {k: v for k, v in model_map.items() if any(r["cris_profiles"] for r in v["regions"].values())}
    direct_only_models = {k: v for k, v in model_map.items() if not any(r["cris_profiles"] for r in v["regions"].values()) and any(r["direct"] for r in v["regions"].values())}

    print(f"\nAll CRIS prefixes: {sorted(all_prefixes)}")
    print(f"Models with inference profiles: {len(cris_models)}")
    print(f"Models direct-only (no CRIS): {len(direct_only_models)}")
    print(f"Total unique models: {len(model_map)}")

    # Show APAC availability
    print(f"\n--- Models with APAC profiles ---")
    apac_models = [m for m, v in cris_models.items() if any("apac" in r["cris_profiles"] for r in v["regions"].values())]
    print(f"  Count: {len(apac_models)}")
    for m in sorted(apac_models):
        print(f"    {m}")

    # Show direct-only models
    print(f"\n--- Direct-only models (sample) ---")
    for m in sorted(direct_only_models.keys())[:15]:
        direct_regions = [r for r, d in direct_only_models[m]["regions"].items() if d["direct"]]
        provider = direct_only_models[m]["provider"]
        print(f"  {m} ({provider}): {len(direct_regions)} regions")

    # --- Write output ---
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regions_scanned": BEDROCK_REGIONS,
        "all_cris_prefixes": sorted(all_prefixes),
        "summary": {
            "total_models": len(model_map),
            "models_with_cris": len(cris_models),
            "models_direct_only": len(direct_only_models),
        },
        "models": {},
    }

    for model_id in sorted(model_map.keys()):
        data = model_map[model_id]
        regions_out = []
        for region in sorted(data["regions"].keys()):
            rdata = data["regions"][region]
            entry = {"name": region}
            if rdata["cris_profiles"]:
                entry["cris_profiles"] = sorted(rdata["cris_profiles"])
            if rdata["direct"]:
                entry["direct"] = True
            regions_out.append(entry)

        output["models"][model_id] = {
            "provider": data["provider"],
            "model_name": data["model_name"],
            "has_cris": any(r["cris_profiles"] for r in data["regions"].values()),
            "regions": regions_out,
        }

    out_path = "scripts/_regional_profiles.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Written to {out_path} ({len(output['models'])} models)")


if __name__ == "__main__":
    main()
