#!/usr/bin/env python3
"""
Download the T-REx sample dataset (10K documents, ~21 MB zip).

Source: https://hadyelsahar.github.io/t-rex/
Paper:  ElSahar et al., "T-REx: A Large Scale Alignment of Natural Language
        with Knowledge Base Triples", LREC 2018.

The sample is hosted on figshare. We download and extract the JSON files
into ./raw/ for the transform script to consume.
"""
import os
import sys
import zipfile
import urllib.request
import shutil

# figshare direct-download URL for the T-REx sample (10K docs, ~21 MB)
SAMPLE_URL = "https://ndownloader.figshare.com/articles/5151175/versions/1"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
ZIP_PATH = os.path.join(os.path.dirname(__file__), "trex_sample.zip")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print(f"Zip already exists at {ZIP_PATH}, skipping download.")
    else:
        print(f"Downloading T-REx sample from figshare …")
        urllib.request.urlretrieve(SAMPLE_URL, ZIP_PATH)
        print(f"Downloaded to {ZIP_PATH}")

    print("Extracting …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)

    # figshare wraps the actual data in a nested zip — extract any inner zips
    for fname in os.listdir(RAW_DIR):
        inner_path = os.path.join(RAW_DIR, fname)
        if fname.endswith(".zip") and zipfile.is_zipfile(inner_path):
            print(f"  Extracting inner archive: {fname}")
            with zipfile.ZipFile(inner_path, "r") as inner_zf:
                inner_zf.extractall(RAW_DIR)
            os.remove(inner_path)

    # Count JSON files (may be in subdirectories)
    json_count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".json"):
                json_count += 1
    print(f"Extracted {json_count} JSON file(s) into {RAW_DIR}/")


if __name__ == "__main__":
    download()
