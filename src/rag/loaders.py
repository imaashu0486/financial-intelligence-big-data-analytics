"""
Document loaders for the RAG pipeline.

Supports loading text from TXT and PDF files, preserving document metadata
(name, source path, page number) for downstream chunking and retrieval.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.logger import logger


@dataclass
class DocumentPage:
    """A single page (or logical unit) of text extracted from a document."""

    document_id: str
    document_name: str
    source_path: str
    page_number: int
    text: str

    def __post_init__(self) -> None:
        """Strip whitespace from text on creation."""
        self.text = self.text.strip()


def _generate_document_id(source_path: str) -> str:
    """Generate a deterministic document ID from the file path."""
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]


def load_txt(path: Path) -> List[DocumentPage]:
    """
    Load a plain-text document.

    TXT files are treated as a single page (page_number=1).

    Parameters
    ----------
    path : Path
        Path to the .txt file.

    Returns
    -------
    List[DocumentPage]
        A list containing one DocumentPage (or empty if file is empty).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {path}")

    doc_id = _generate_document_id(str(path.resolve()))
    doc_name = path.name

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1 for files with non-UTF8 encoding
        try:
            text = path.read_text(encoding="latin-1")
            logger.warning(f"File {doc_name} loaded with latin-1 fallback encoding.")
        except Exception as e:
            logger.error(f"Could not read {doc_name}: {e}")
            return []

    text = text.strip()
    if not text:
        logger.warning(f"Empty document: {doc_name}")
        return []

    page = DocumentPage(
        document_id=doc_id,
        document_name=doc_name,
        source_path=str(path.resolve()),
        page_number=1,
        text=text,
    )
    return [page]


def load_pdf(path: Path) -> List[DocumentPage]:
    """
    Load a PDF document, extracting text per page.

    Requires PyPDF2. If PyPDF2 is not installed, raises ImportError
    with a clear message.

    Parameters
    ----------
    path : Path
        Path to the .pdf file.

    Returns
    -------
    List[DocumentPage]
        One DocumentPage per non-empty page.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If PyPDF2 is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF loading. "
            "Install it with: pip install PyPDF2>=3.0.0"
        )

    doc_id = _generate_document_id(str(path.resolve()))
    doc_name = path.name
    pages: List[DocumentPage] = []

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.error(f"Could not read PDF {doc_name}: {e}")
        return []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Could not extract text from page {i + 1} of {doc_name}: {e}")
            text = ""

        text = text.strip()
        if not text:
            continue

        pages.append(
            DocumentPage(
                document_id=doc_id,
                document_name=doc_name,
                source_path=str(path.resolve()),
                page_number=i + 1,
                text=text,
            )
        )

    if not pages:
        logger.warning(f"No text extracted from PDF: {doc_name}")

    return pages


def load_directory(
    dir_path: Path,
    extensions: Optional[List[str]] = None,
) -> List[DocumentPage]:
    """
    Load all supported documents from a directory.

    Parameters
    ----------
    dir_path : Path
        Directory containing documents.
    extensions : list of str, optional
        File extensions to load. Default: [".txt", ".pdf"].

    Returns
    -------
    List[DocumentPage]
        All pages from all successfully loaded documents.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    if extensions is None:
        extensions = [".txt", ".pdf"]

    # Map extensions to loader functions
    loaders = {
        ".txt": load_txt,
        ".pdf": load_pdf,
    }

    all_pages: List[DocumentPage] = []
    files = sorted(dir_path.iterdir())

    for file_path in files:
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in extensions:
            continue

        loader = loaders.get(ext)
        if loader is None:
            logger.warning(f"No loader for extension {ext}: {file_path.name}")
            continue

        try:
            pages = loader(file_path)
            if pages:
                logger.info(
                    f"Loaded {file_path.name}: {len(pages)} page(s), "
                    f"{sum(len(p.text) for p in pages)} chars"
                )
                all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(
        f"Directory loading complete: {len(all_pages)} pages "
        f"from {dir_path}"
    )
    return all_pages
