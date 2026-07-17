import os
import fitz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "datasets")


def list_pdfs(folder):
    pdfs = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdfs.append(os.path.join(folder, filename))

    return pdfs


def read_first_page(pdf_path):
    doc = fitz.open(pdf_path)

    if len(doc) == 0:
        doc.close()
        return ""

    page = doc[0]
    text = page.get_text("text")

    doc.close()
    return text


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("PDF folder:", PDF_FOLDER)

    pdfs = list_pdfs(PDF_FOLDER)

    print(f"\nFound {len(pdfs)} PDF files:")

    for pdf in pdfs:
        print("-", os.path.basename(pdf))

    print("\nTesting first PDF...\n")

    if not pdfs:
        print("No PDF files found.")
    else:
        first_pdf = pdfs[0]
        text = read_first_page(first_pdf)

        print("File:", os.path.basename(first_pdf))
        print("First page text preview:")
        print("=" * 80)
        print(text[:1500])
        print("=" * 80)