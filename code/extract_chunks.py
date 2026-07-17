import os
import json
import re
from typing import List, Dict

import fitz
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "datasets")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "temp")
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "pdf_chunks.json")


CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving Chinese and English content.
    """
    text = text.replace("\x00", " ")

    # Normalize spaces and tabs, but keep line breaks first.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove too many blank lines.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip each line.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def split_text_by_char(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple character-based chunking.
    Works reasonably well for Chinese because Chinese text does not always use spaces.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def list_pdf_files(folder: str) -> List[str]:
    pdf_files = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(folder, filename))

    pdf_files.sort()
    return pdf_files


def extract_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    source_file = os.path.basename(pdf_path)
    source_stem = os.path.splitext(source_file)[0]

    chunks_data = []

    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        page_number = page_idx + 1
        page = doc[page_idx]

        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        page_chunks = split_text_by_char(text)

        for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"{source_stem}_p{page_number}_c{chunk_idx}"

            chunks_data.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "page": page_number,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                }
            )

    doc.close()
    return chunks_data


def main():
    print("Project root:", PROJECT_ROOT)
    print("PDF folder:", PDF_FOLDER)
    print("Output file:", OUTPUT_FILE)

    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_files = list_pdf_files(PDF_FOLDER)

    print(f"\nFound {len(pdf_files)} PDF files.")

    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        pdf_chunks = extract_chunks_from_pdf(pdf_path)
        all_chunks.extend(pdf_chunks)

        print(
            f"{os.path.basename(pdf_path)} -> {len(pdf_chunks)} chunks"
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")

    if all_chunks:
        print("\nSample chunk:")
        print("=" * 80)
        print(json.dumps(all_chunks[0], ensure_ascii=False, indent=2)[:2000])
        print("=" * 80)


if __name__ == "__main__":
    main()