"""
Document loader — stub for PDF/text document ingestion (Phase 7).

For Phase 1 this simply lists available documents in the configured directory.
Full text extraction via PyPDF / OCR will be added in Phase 7.

Usage:
    from src.ingestion.document_loader import list_documents
    docs = list_documents("data/documents")
"""

from pathlib import Path
from typing import List, Union

from src.utils.logger import logger

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def list_documents(directory: Union[str, Path]) -> List[Path]:
    """
    List supported document files in the given directory.

    Parameters
    ----------
    directory : str | Path
        Path to the documents folder.

    Returns
    -------
    list[Path]
        Sorted list of document file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Documents directory does not exist: {directory}")
        return []

    docs = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    logger.info(f"Found {len(docs)} documents in {directory}")
    for doc in docs:
        logger.debug(f"  📄 {doc.name} ({doc.stat().st_size / 1024:.1f} KB)")

    return docs
