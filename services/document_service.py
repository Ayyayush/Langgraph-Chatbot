# =========================================================
# DOCUMENT SERVICE (PDF -> text -> chunks)
# =========================================================
# Handles everything about turning an uploaded PDF into small
# text chunks that are ready to be embedded and stored in
# ChromaDB. Kept deliberately simple (no external chunking
# libraries) so it's easy to explain line by line.

from pypdf import PdfReader


def extract_text_from_pdf(file_obj) -> str:
    """
    Extract all text from an uploaded PDF (a Streamlit
    UploadedFile behaves like a file object, so PdfReader can
    read it directly).

    Returns "" if the PDF has no extractable text (e.g. a
    scanned/image-only PDF) instead of raising.
    """
    try:
        reader = PdfReader(file_obj)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")

    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n".join(pages_text).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    A simple sliding window is enough for a student project -
    it avoids pulling in a separate text-splitter library while
    still giving the retriever reasonably sized, overlapping
    pieces of context.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, keeping overlap

    return chunks
