#!/usr/bin/env python3
"""
Download the BOLD dataset (Bias in Open-ended Language Generation).

Source: https://github.com/amazon-science/bold
Paper:  Dhamala et al., "BOLD: Dataset and Metrics for Measuring Biases in
        Open-Ended Language Generation", FAccT 2021.
License: Creative Commons Attribution Share Alike 4.0 International

23,679 text generation prompts across five domains: profession, gender,
race, religious ideologies, and political ideologies.

Downloads from the GitHub repository and saves as a single JSONL file.
"""
import json
import os
import sys
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

# Domain files in the BOLD GitHub repo
GITHUB_BASE = "https://raw.githubusercontent.com/amazon-science/bold/main/prompts"
DOMAIN_FILES = [
    "gender_prompt.json",
    "profession_prompt.json",
    "race_prompt.json",
    "religious_ideology_prompt.json",
    "political_ideology_prompt.json",
]


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    out_path = os.path.join(RAW_DIR, "bold_prompts.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"bold_prompts.jsonl already exists ({count} prompts), skipping.")
        return

    print("Downloading BOLD prompts from GitHub …")
    total = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for filename in DOMAIN_FILES:
            domain = filename.replace("_prompt.json", "")
            url = f"{GITHUB_BASE}/{filename}"
            print(f"  Fetching {filename} …")

            try:
                with urllib.request.urlopen(url) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                print(f"    ERROR downloading {url}: {e}")
                continue

            count = 0
            for subgroup, entities in data.items():
                for entity, prompts in entities.items():
                    for prompt_text in prompts:
                        prompt_text = prompt_text.strip()
                        if not prompt_text:
                            continue
                        out_f.write(json.dumps({
                            "domain": domain,
                            "subgroup": subgroup,
                            "entity": entity,
                            "prompt_text": prompt_text,
                        }, ensure_ascii=False) + "\n")
                        count += 1
            total += count
            print(f"    {domain}: {count} prompts")

    print(f"\n  Saved {total} total prompts to {out_path}")


if __name__ == "__main__":
    download()
