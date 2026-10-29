"""PDF loading and chunking. Every chunk keeps its page number for citations."""
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PyPdfError


@dataclass(frozen=True)
class Chunk:
    id: int
    page: int
    text: str


class DocumentError(ValueError):
    """The upload could not be read as a usable PDF."""


def extract_pages(data: bytes, max_pages: int = 300) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("Encrypted PDFs are not supported.")
        if len(reader.pages) > max_pages:
            raise DocumentError(f"PDF has more than {max_pages} pages.")
        pages = [(p.extract_text() or "") for p in reader.pages]
    except DocumentError:
        raise
    except (PyPdfError, ValueError, OSError) as exc:
        raise DocumentError("Could not read this file as a PDF.") from exc
    if not any(p.strip() for p in pages):
        raise DocumentError("No text found. Scanned PDFs need OCR first.")
    return pages


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_pages(pages: list[str], chunk_chars: int = 1200) -> list[Chunk]:
    """Split each page on blank lines / sentences into chunks of about chunk_chars."""
    chunks: list[Chunk] = []
    for page_no, raw in enumerate(pages, start=1):
        text = _clean(raw)
        if not text:
            continue
        pieces = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.;:])\s*\n", text) if p.strip()]
        buf = ""
        for piece in pieces:
            piece = piece.replace("\n", " ")
            if buf and len(buf) + len(piece) + 1 > chunk_chars:
                chunks.append(Chunk(len(chunks) + 1, page_no, buf))
                buf = ""
            while len(piece) > chunk_chars:                # very long paragraph
                cut = piece.rfind(" ", 0, chunk_chars)
                cut = cut if cut > chunk_chars // 2 else chunk_chars
                chunks.append(Chunk(len(chunks) + 1, page_no, piece[:cut].strip()))
                piece = piece[cut:].strip()
            buf = f"{buf} {piece}".strip()
        if buf:
            chunks.append(Chunk(len(chunks) + 1, page_no, buf))
    return chunks
