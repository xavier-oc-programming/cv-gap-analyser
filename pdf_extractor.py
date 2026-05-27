"""
pdf_extractor.py — PDF text extraction using PyMuPDF (fitz).

No classes. Single public function: extract_text_from_bytes().
"""
from config import MAX_PDF_SIZE_MB, MAX_TEXT_LENGTH


def extract_text_from_bytes(pdf_bytes: bytes) -> dict:
    """
    Extract text from PDF bytes using PyMuPDF.

    Args:
        pdf_bytes: raw PDF file content

    Returns:
        {
          "text": str,
          "page_count": int,
          "word_count": int,
          "truncated": bool,
          "extraction_method": "pymupdf"
        }

    Raises:
        ValueError if content is not a valid PDF.
        ValueError if file exceeds MAX_PDF_SIZE_MB.
    """
    import fitz  # PyMuPDF

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(
            f"PDF exceeds maximum size of {MAX_PDF_SIZE_MB} MB "
            f"(uploaded: {size_mb:.1f} MB)."
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ValueError("Uploaded file is not a valid PDF.")

    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    full_text = "\n".join(pages).strip()
    original_length = len(full_text)
    full_text = full_text[:MAX_TEXT_LENGTH]
    truncated = len(full_text) < original_length

    return {
        "text": full_text,
        "page_count": len(pages),
        "word_count": len(full_text.split()),
        "truncated": truncated,
        "extraction_method": "pymupdf",
    }
