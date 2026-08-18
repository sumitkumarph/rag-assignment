from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""

    reader = PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def extract_all_pdfs(pdf_directory: Path):
    """Extract text from every PDF in the directory."""

    documents = []

    pdf_files = list(pdf_directory.glob("*.pdf"))

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        text = extract_text_from_pdf(pdf_file)

        documents.append({
            "filename": pdf_file.name,
            "text": text
        })

    return documents