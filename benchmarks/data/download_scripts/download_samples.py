#!/usr/bin/env python3
"""Download 50 samples from each of 3 multimodal datasets:
1. philschmid/ocr-invoice-data (invoice extraction)
2. lmms-lab/DocVQA (document VQA)
3. vis-nlp/ChartQA via HuggingFace (chart QA)

Saves images + ground truth JSON under ind_standard_datasets/

Requirements:
    pip install datasets Pillow
"""
import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Run: pip install datasets Pillow")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_SIZE = 50


def download_invoice_data():
    """Download 50 invoice samples from philschmid/ocr-invoice-data."""
    print("\n" + "=" * 60)
    print("1. Downloading philschmid/ocr-invoice-data (invoices)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "invoices")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ds = load_dataset("philschmid/ocr-invoice-data", split="train")
    print(f"  Dataset size: {len(ds)} samples")

    samples = []
    for i, item in enumerate(ds):
        if i >= SAMPLE_SIZE:
            break

        # Save image
        img = item.get("image")
        if img is None:
            continue

        img_filename = f"invoice_{i:03d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img.save(img_path)

        # Parse ground truth
        gt_json = item.get("json", "{}")
        if isinstance(gt_json, str):
            try:
                gt = json.loads(gt_json)
            except json.JSONDecodeError:
                gt = {"raw": gt_json}
        else:
            gt = gt_json

        samples.append({
            "id": f"inv_{i:03d}",
            "image_path": f"images/{img_filename}",
            "ground_truth": gt,
            "task": "invoice_extraction",
            "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
        })

    # Save metadata
    meta_path = os.path.join(out_dir, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {len(samples)} samples to {out_dir}")
    print(f"  Images: {img_dir}")
    print(f"  Metadata: {meta_path}")
    return len(samples)


def download_docvqa():
    """Download 50 DocVQA samples from lmms-lab/DocVQA."""
    print("\n" + "=" * 60)
    print("2. Downloading lmms-lab/DocVQA (document VQA)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "docvqa")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print(f"  Dataset size: {len(ds)} samples")

    samples = []
    for i, item in enumerate(ds):
        if i >= SAMPLE_SIZE:
            break

        # Save image
        img = item.get("image")
        if img is None:
            continue

        img_filename = f"doc_{i:03d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img.save(img_path)

        question = item.get("question", "")
        answers = item.get("answers", [])
        question_types = item.get("question_types", [])

        samples.append({
            "id": f"docvqa_{i:03d}",
            "image_path": f"images/{img_filename}",
            "question": question,
            "answers": answers,
            "question_types": question_types,
            "task": "document_vqa",
            "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
        })

    # Save metadata
    meta_path = os.path.join(out_dir, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {len(samples)} samples to {out_dir}")
    print(f"  Images: {img_dir}")
    print(f"  Metadata: {meta_path}")
    return len(samples)


def download_chartqa():
    """Download 50 ChartQA samples."""
    print("\n" + "=" * 60)
    print("3. Downloading ChartQA (chart question answering)")
    print("=" * 60)

    out_dir = os.path.join(BASE_DIR, "chartqa")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Try HuggingFace dataset
    try:
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
        print(f"  Dataset size: {len(ds)} samples")

        samples = []
        for i, item in enumerate(ds):
            if i >= SAMPLE_SIZE:
                break

            img = item.get("image")
            if img is None:
                continue

            img_filename = f"chart_{i:03d}.png"
            img_path = os.path.join(img_dir, img_filename)
            img.save(img_path)

            question = item.get("query", item.get("question", ""))
            answer = item.get("label", item.get("answer", ""))

            samples.append({
                "id": f"chart_{i:03d}",
                "image_path": f"images/{img_filename}",
                "question": question,
                "answer": answer if isinstance(answer, str) else str(answer),
                "task": "chart_qa",
                "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
            })

    except Exception as e:
        print(f"  HuggingFaceM4/ChartQA failed: {e}")
        print("  Trying alternative: ahmed-masry/ChartQA...")

        try:
            ds = load_dataset("ahmed-masry/ChartQA", split="test")
            print(f"  Dataset size: {len(ds)} samples")

            samples = []
            for i, item in enumerate(ds):
                if i >= SAMPLE_SIZE:
                    break

                img = item.get("image")
                if img is None:
                    continue

                img_filename = f"chart_{i:03d}.png"
                img_path = os.path.join(img_dir, img_filename)
                img.save(img_path)

                question = item.get("query", item.get("question", ""))
                answer = item.get("label", item.get("answer", [""]))
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""

                samples.append({
                    "id": f"chart_{i:03d}",
                    "image_path": f"images/{img_filename}",
                    "question": question,
                    "answer": str(answer),
                    "task": "chart_qa",
                    "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
                })

        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
            print("  Trying OCRBench as fallback (mixed document/chart tasks)...")

            ds = load_dataset("echo840/OCRBench", split="test")
            print(f"  Dataset size: {len(ds)} samples")

            samples = []
            for i, item in enumerate(ds):
                if i >= SAMPLE_SIZE:
                    break

                img = item.get("image")
                if img is None:
                    continue

                img_filename = f"chart_{i:03d}.png"
                img_path = os.path.join(img_dir, img_filename)
                img.save(img_path)

                question = item.get("question", "")
                answer = item.get("answer", "")

                samples.append({
                    "id": f"chart_{i:03d}",
                    "image_path": f"images/{img_filename}",
                    "question": question,
                    "answer": str(answer),
                    "task": "chart_qa",
                    "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
                })

    # Save metadata
    meta_path = os.path.join(out_dir, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Saved {len(samples)} samples to {out_dir}")
    print(f"  Images: {img_dir}")
    print(f"  Metadata: {meta_path}")
    return len(samples)


def main():
    print("Downloading multimodal benchmark samples")
    print(f"Target: {SAMPLE_SIZE} samples per dataset")
    print(f"Output: {BASE_DIR}")

    total = 0
    total += download_invoice_data()
    total += download_docvqa()
    total += download_chartqa()

    print("\n" + "=" * 60)
    print(f"DONE: {total} total samples downloaded")
    print("=" * 60)
    print(f"\nStructure:")
    print(f"  {BASE_DIR}/")
    print(f"    invoices/images/ + samples.json")
    print(f"    docvqa/images/ + samples.json")
    print(f"    chartqa/images/ + samples.json")


if __name__ == "__main__":
    main()
