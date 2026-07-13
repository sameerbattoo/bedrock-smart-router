# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Download 50 PDF samples from the DUDE dataset (Document Understanding).

DUDE contains multi-page business documents with QA annotations.
Source: https://huggingface.co/datasets/jordyvl/DUDE_loader

Requirements:
    pip install datasets pdf2image
"""
import json
import os
import sys
import requests

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Run: pip install datasets")
    sys.exit(1)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dude_pdfs")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

SAMPLE_SIZE = 50


def main():
    print("=" * 60)
    print("Downloading DUDE dataset (PDF Document Understanding)")
    print("=" * 60)

    # Try loading DUDE - it has PDFs with QA pairs
    try:
        ds = load_dataset("jordyvl/DUDE_loader", "Amazon_original", split="val")
        print(f"  Loaded DUDE: {len(ds)} samples")
        print(f"  Columns: {ds.column_names}")
    except Exception as e:
        print(f"  DUDE_loader failed: {e}")
        print("  Trying alternative: surgeai/GDP.pdf...")
        try:
            ds = load_dataset("surgeai/GDP.pdf", split="test")
            print(f"  Loaded GDP.pdf: {len(ds)} samples")
            print(f"  Columns: {ds.column_names}")
        except Exception as e2:
            print(f"  GDP.pdf also failed: {e2}")
            print("\n  Falling back to generating synthetic PDFs...")
            generate_synthetic_pdfs()
            return

    # Process samples
    samples = []
    saved_count = 0

    for i, item in enumerate(ds):
        if saved_count >= SAMPLE_SIZE:
            break

        # Try to get PDF URL or bytes
        pdf_url = item.get("pdf_url", item.get("document_url", item.get("url", "")))
        pdf_bytes = item.get("pdf", item.get("document", None))
        question = item.get("question", item.get("prompt", ""))
        answers = item.get("answers", item.get("answer", []))
        doc_id = item.get("docId", item.get("document_id", f"doc_{i:03d}"))

        if isinstance(answers, str):
            answers = [answers]
        elif isinstance(answers, list) and len(answers) > 0 and isinstance(answers[0], dict):
            answers = [a.get("answer", "") for a in answers]

        if not question:
            continue

        pdf_filename = f"dude_{saved_count:03d}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        # Save PDF
        if pdf_bytes is not None:
            if isinstance(pdf_bytes, bytes):
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)
            elif hasattr(pdf_bytes, "read"):
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes.read())
            else:
                continue
        elif pdf_url:
            try:
                resp = requests.get(pdf_url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 100:
                    with open(pdf_path, "wb") as f:
                        f.write(resp.content)
                else:
                    continue
            except Exception:
                continue
        else:
            # Try to get images and convert to PDF
            images = item.get("images", item.get("image", None))
            if images is not None:
                try:
                    if isinstance(images, list) and len(images) > 0:
                        images[0].save(pdf_path, "PDF", save_all=True, append_images=images[1:])
                    elif hasattr(images, "save"):
                        images.save(pdf_path, "PDF")
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue

        samples.append({
            "id": f"dude_{saved_count:03d}",
            "pdf_path": f"pdfs/{pdf_filename}",
            "question": question,
            "answers": answers,
            "doc_id": str(doc_id),
            "task": "document_pdf_qa",
            "difficulty": "simple" if saved_count < 17 else ("medium" if saved_count < 34 else "complex"),
        })
        saved_count += 1

        if saved_count % 10 == 0:
            print(f"  Downloaded {saved_count}/{SAMPLE_SIZE}...")

    # Save metadata
    meta_path = os.path.join(BASE_DIR, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\n  Saved {len(samples)} PDF samples")
    print(f"  PDFs: {PDF_DIR}")
    print(f"  Metadata: {meta_path}")

    if samples:
        print(f"\n  Sample Q: {samples[0]['question']}")
        print(f"  Sample A: {samples[0]['answers']}")


def generate_synthetic_pdfs():
    """Fallback: generate synthetic PDFs with known content for testing."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("  reportlab not installed. Run: pip install reportlab")
        print("  Generating minimal PDFs with fpdf2 or raw bytes...")
        generate_minimal_pdfs()
        return

    print("  Generating 50 synthetic PDFs with reportlab...")

    invoice_data = [
        {"inv_no": f"INV-2024-{1000+i}", "date": f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}",
         "company": f"Company {chr(65+i%26)}", "total": round(100 + i * 47.5, 2)}
        for i in range(50)
    ]

    samples = []
    for i, inv in enumerate(invoice_data):
        pdf_filename = f"dude_{i:03d}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, f"INVOICE {inv['inv_no']}")
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, f"Date: {inv['date']}")
        c.drawString(72, 700, f"From: {inv['company']}")
        c.drawString(72, 680, f"Total: ${inv['total']}")
        c.drawString(72, 640, "Items:")
        c.drawString(90, 620, f"1. Service A - ${inv['total'] * 0.6:.2f}")
        c.drawString(90, 600, f"2. Service B - ${inv['total'] * 0.4:.2f}")
        c.save()

        questions = [
            (f"What is the invoice number?", [inv["inv_no"]]),
            (f"What is the total amount?", [f"${inv['total']}"]),
            (f"What company issued this invoice?", [inv["company"]]),
        ]
        q, a = questions[i % 3]

        samples.append({
            "id": f"dude_{i:03d}",
            "pdf_path": f"pdfs/{pdf_filename}",
            "question": q,
            "answers": a,
            "task": "document_pdf_qa",
            "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
        })

    meta_path = os.path.join(BASE_DIR, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Generated {len(samples)} synthetic PDF samples")
    print(f"  PDFs: {PDF_DIR}")
    print(f"  Metadata: {meta_path}")


def generate_minimal_pdfs():
    """Generate minimal PDFs without reportlab using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("  Installing fpdf2...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "fpdf2"], capture_output=True)
        from fpdf import FPDF

    print("  Generating 50 synthetic PDFs with fpdf2...")

    samples = []
    for i in range(50):
        inv_no = f"INV-2024-{1000+i}"
        date = f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}"
        company = f"Company {chr(65+i%26)} Corp"
        total = round(100 + i * 47.5, 2)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"INVOICE {inv_no}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Date: {date}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"From: {company}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Bill To: Customer {i+1}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, "", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Line Items:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        item1 = round(total * 0.6, 2)
        item2 = round(total * 0.4, 2)
        pdf.cell(0, 7, f"  1. Consulting Services - ${item1}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"  2. Software License - ${item2}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, "", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"TOTAL: ${total}", new_x="LMARGIN", new_y="NEXT")

        pdf_filename = f"dude_{i:03d}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        pdf.output(pdf_path)

        questions = [
            (f"What is the invoice number?", [inv_no]),
            (f"What is the total amount?", [f"${total}"]),
            (f"What company issued this invoice?", [company]),
            (f"What is the date of this invoice?", [date]),
            (f"How many line items are there?", ["2"]),
        ]
        q, a = questions[i % 5]

        samples.append({
            "id": f"dude_{i:03d}",
            "pdf_path": f"pdfs/{pdf_filename}",
            "question": q,
            "answers": a,
            "task": "document_pdf_qa",
            "difficulty": "simple" if i < 17 else ("medium" if i < 34 else "complex"),
        })

    meta_path = os.path.join(BASE_DIR, "samples.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"  Generated {len(samples)} synthetic PDF samples")
    print(f"  PDFs: {PDF_DIR}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
