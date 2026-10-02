from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm

from .config import (
    OCR_LANGUAGE,
    SHOW_PROGRESS,
    USE_OCR,
)


# =========================================================
# Data structures
# =========================================================


@dataclass
class PageContent:
    """
    Parsed content from one PDF page.

    Attributes
    ----------
    page_number:
        1-based page number from the PDF.

    text:
        Cleaned article/body text intended for RAG chunking.

    raw_text:
        Complete Markdown returned by PyMuPDF4LLM before
        layout-based cleanup.

    captions:
        Figure/table captions stored separately from body text.

    metadata:
        Page-level metadata returned by PyMuPDF4LLM.

    toc_items:
        Table-of-contents / heading information associated
        with this page.

    page_boxes:
        Layout boxes returned by PyMuPDF4LLM. These describe
        regions such as text, captions, pictures, headers,
        footers, etc.
    """

    page_number: int

    text: str

    raw_text: str

    captions: list[str]

    metadata: dict

    toc_items: list

    page_boxes: list


@dataclass
class ParsedDocument:
    """
    Parsed representation of one research PDF.
    """

    path: Path

    filename: str

    sha256: str

    page_count: int

    title: str | None

    author: str | None

    pages: list[PageContent]


# =========================================================
# File utilities
# =========================================================


def calculate_sha256(
    path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA-256 hash of a file.

    The hash will later allow the indexer to determine whether
    a PDF has changed and therefore needs to be reprocessed.
    """

    path = Path(path)

    hasher = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            block = file.read(block_size)

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


# =========================================================
# PDF discovery
# =========================================================


def discover_pdfs(
    root: Path,
) -> list[Path]:
    """
    Recursively discover PDF files below a directory.

    Parameters
    ----------
    root:
        Root research-library directory.

    Returns
    -------
    list[Path]
        Sorted list of discovered PDFs.
    """

    root = Path(root)

    if not root.exists():

        raise FileNotFoundError(
            f"Document directory does not exist:\n{root}"
        )

    if not root.is_dir():

        raise NotADirectoryError(
            f"Document root is not a directory:\n{root}"
        )

    pdfs = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
        )
    ]

    pdfs.sort(
        key=lambda path: str(path).lower()
    )

    return pdfs


# =========================================================
# Layout helpers
# =========================================================


def _extract_box_text(
    raw_text: str,
    box: dict,
) -> str:
    """
    Recover the Markdown text belonging to one layout box.

    PyMuPDF4LLM layout boxes may contain a ``pos`` field with
    character offsets into the complete page Markdown.

    If position data are unavailable or invalid, an empty
    string is returned.
    """

    position = box.get("pos")

    if not position:
        return ""

    if not isinstance(
        position,
        (list, tuple),
    ):
        return ""

    if len(position) != 2:
        return ""

    start, stop = position

    try:

        start = int(start)

        stop = int(stop)

    except (TypeError, ValueError):

        return ""

    if start < 0:
        return ""

    if stop <= start:
        return ""

    if start >= len(raw_text):
        return ""

    stop = min(
        stop,
        len(raw_text),
    )

    return raw_text[
        start:stop
    ].strip()


def _join_body_segments(
    segments: list[str],
) -> str:
    """
    Join article-text layout segments into readable Markdown.

    This function keeps paragraph separation while repairing
    obvious words that were split by a column or layout boundary.

    Example
    -------
    ``phenom-`` + ``enon``

    becomes

    ``phenomenon``

    This repair is intentionally conservative. It is performed
    only when the previous segment ends in a hyphen and the next
    segment begins with a lowercase character.
    """

    cleaned = [
        segment.strip()
        for segment in segments
        if segment
        and segment.strip()
    ]

    if not cleaned:
        return ""

    result = cleaned[0]

    for segment in cleaned[1:]:

        previous = result.rstrip()

        following = segment.lstrip()

        if (
            previous.endswith("-")
            and following
            and following[0].islower()
        ):

            result = (
                previous[:-1]
                + following
            )

        else:

            result = (
                previous
                + "\n\n"
                + following
            )

    return result.strip()


def clean_layout_text(
    chunk: dict,
) -> tuple[str, list[str]]:
    """
    Separate main article text from visual material.

    The raw page Markdown returned by PyMuPDF4LLM can include
    text associated with figures, diagrams, captions, page
    headers, and page footers.

    For RAG, we want the main article body to remain coherent,
    while figure/table captions should still be preserved for
    later indexing.

    Parameters
    ----------
    chunk:
        One page-chunk dictionary returned by
        PyMuPDF4LLM ``to_markdown(page_chunks=True)``.

    Returns
    -------
    tuple[str, list[str]]
        ``(body_text, captions)``
    """

    raw_text = str(
        chunk.get("text") or ""
    )

    page_boxes = list(
        chunk.get("page_boxes") or []
    )

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------
    #
    # Some PDFs or PyMuPDF4LLM versions may not provide
    # page_boxes. In that situation it is safer to preserve
    # all extracted text than to lose document content.

    if not page_boxes:

        return (
            raw_text.strip(),
            [],
        )


    body_segments: list[str] = []

    captions: list[str] = []


    # -----------------------------------------------------
    # Preserve reading order
    # -----------------------------------------------------

    page_boxes = sorted(
        page_boxes,
        key=lambda box: box.get(
            "index",
            0,
        ),
    )


    for box in page_boxes:

        box_class = str(
            box.get("class") or ""
        ).strip().lower()


        segment = _extract_box_text(
            raw_text,
            box,
        )


        if not segment:
            continue


        # -------------------------------------------------
        # Ignore visual / repetitive layout elements
        # -------------------------------------------------

        if box_class in {
            "picture",
            "page-header",
            "page-footer",
        }:

            continue


        # -------------------------------------------------
        # Preserve figure / table captions separately
        # -------------------------------------------------

        if box_class == "caption":

            captions.append(
                segment
            )

            continue


        # -------------------------------------------------
        # Keep all remaining text classes
        # -------------------------------------------------
        #
        # These may include:
        #
        # text
        # title
        # section-header
        # formula
        # table
        # etc.
        #
        # We intentionally do not aggressively remove them
        # because research-paper structure is valuable to RAG.

        body_segments.append(
            segment
        )


    body_text = _join_body_segments(
        body_segments
    )


    # -----------------------------------------------------
    # Safety fallback
    # -----------------------------------------------------
    #
    # If layout classification unexpectedly discarded almost
    # the entire page, preserve the original text instead.

    if (
        len(body_text) < 50
        and len(raw_text.strip()) > 200
    ):

        body_text = raw_text.strip()


    return (
        body_text,
        captions,
    )


# =========================================================
# PDF parsing
# =========================================================


def parse_pdf(
    path: Path,
) -> ParsedDocument:
    """
    Parse one research PDF using PyMuPDF4LLM.

    The document is returned as page-level records containing:

    - cleaned body text
    - original raw Markdown
    - figure/table captions
    - metadata
    - table-of-contents information
    - layout boxes
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"PDF does not exist:\n{path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Path is not a file:\n{path}"
        )

    if path.suffix.lower() != ".pdf":

        raise ValueError(
            f"Expected a PDF file, received:\n{path}"
        )


    print()

    print(
        f"Parsing: {path.name}"
    )


    # =====================================================
    # Extract page-level Markdown
    # =====================================================

    page_chunks = pymupdf4llm.to_markdown(
        str(path),

        # Return one dictionary for every PDF page.
        page_chunks=True,

        # OCR is available when required.
        use_ocr=USE_OCR,

        ocr_language=OCR_LANGUAGE,

        show_progress=SHOW_PROGRESS,

        # We only need textual information during this stage.
        write_images=False,

        embed_images=False,

        # Remove repetitive journal page headers and footers.
        header=False,

        footer=False,

        # Do not inject text from figure / picture regions
        # into the main article reading order.
        force_text=False,
    )


    if not isinstance(
        page_chunks,
        list,
    ):

        raise RuntimeError(
            "Expected PyMuPDF4LLM to return page chunks "
            "as a list."
        )


    pages: list[PageContent] = []


    document_title: str | None = None

    document_author: str | None = None


    # =====================================================
    # Convert each page into our internal representation
    # =====================================================

    for fallback_page_number, chunk in enumerate(
        page_chunks,
        start=1,
    ):

        metadata = dict(
            chunk.get("metadata") or {}
        )


        # -------------------------------------------------
        # Page number
        # -------------------------------------------------

        page_number = metadata.get(
            "page_number",
            fallback_page_number,
        )

        try:

            page_number = int(
                page_number
            )

        except (TypeError, ValueError):

            page_number = (
                fallback_page_number
            )


        # -------------------------------------------------
        # Document-level metadata
        # -------------------------------------------------

        if document_title is None:

            title = metadata.get(
                "title"
            )

            if title:

                title = str(
                    title
                ).strip()

                if title:

                    document_title = title


        if document_author is None:

            author = metadata.get(
                "author"
            )

            if author:

                author = str(
                    author
                ).strip()

                if author:

                    document_author = author


        # -------------------------------------------------
        # Raw extraction
        # -------------------------------------------------

        raw_text = str(
            chunk.get("text") or ""
        ).strip()


        # -------------------------------------------------
        # Layout-aware cleanup
        # -------------------------------------------------

        text, captions = (
            clean_layout_text(
                chunk
            )
        )


        # -------------------------------------------------
        # TOC items
        # -------------------------------------------------

        toc_items = list(
            chunk.get("toc_items") or []
        )


        # -------------------------------------------------
        # Layout boxes
        # -------------------------------------------------

        page_boxes = list(
            chunk.get("page_boxes") or []
        )


        # -------------------------------------------------
        # Store page
        # -------------------------------------------------

        pages.append(

            PageContent(
                page_number=page_number,

                text=text,

                raw_text=raw_text,

                captions=captions,

                metadata=metadata,

                toc_items=toc_items,

                page_boxes=page_boxes,
            )

        )


    # =====================================================
    # Build document object
    # =====================================================

    return ParsedDocument(

        path=path,

        filename=path.name,

        sha256=calculate_sha256(
            path
        ),

        page_count=len(
            pages
        ),

        title=document_title,

        author=document_author,

        pages=pages,
    )


# =========================================================
# Markdown export
# =========================================================


def save_parsed_markdown(
    document: ParsedDocument,
    output_path: Path,
) -> None:
    """
    Save cleaned extracted text as human-readable Markdown.

    Main article text is written first.

    Figure/table captions are placed at the end of their
    corresponding page so they remain searchable without
    interrupting the surrounding article prose.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    parts: list[str] = []


    # =====================================================
    # File-level information
    # =====================================================

    parts.append(
        f"<!-- Source: {document.filename} -->"
    )

    parts.append(
        f"<!-- SHA256: {document.sha256} -->"
    )

    parts.append("")


    if document.title:

        parts.append(
            f"# {document.title}"
        )

    else:

        parts.append(
            f"# {document.filename}"
        )


    parts.append("")


    if document.author:

        parts.append(
            f"**Author:** {document.author}"
        )

        parts.append("")


    # =====================================================
    # Individual pages
    # =====================================================

    for page in document.pages:

        parts.append(
            "\n---\n"
        )


        parts.append(
            f"<!-- PDF_PAGE={page.page_number} -->"
        )


        parts.append("")


        # -------------------------------------------------
        # Main article text
        # -------------------------------------------------

        if page.text:

            parts.append(
                page.text
            )

            parts.append("")


        # -------------------------------------------------
        # Captions
        # -------------------------------------------------

        if page.captions:

            parts.append(
                "### Figure / Table Captions"
            )

            parts.append("")


            for caption in page.captions:

                parts.append(
                    caption
                )

                parts.append("")


    combined = "\n".join(
        parts
    )


    output_path.write_text(
        combined,
        encoding="utf-8",
    )


# =========================================================
# Raw Markdown export
# =========================================================


def save_raw_markdown(
    document: ParsedDocument,
    output_path: Path,
) -> None:
    """
    Save the uncleaned PyMuPDF4LLM extraction.

    This is useful for debugging differences between the raw
    parser output and the cleaned RAG representation.

    It is not intended to be used directly for embeddings.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    parts: list[str] = []


    parts.append(
        f"<!-- Source: {document.filename} -->"
    )

    parts.append(
        f"<!-- SHA256: {document.sha256} -->"
    )

    parts.append("")


    for page in document.pages:

        parts.append(
            "\n---\n"
        )

        parts.append(
            f"<!-- PDF_PAGE={page.page_number} -->"
        )

        parts.append("")

        parts.append(
            page.raw_text
        )

        parts.append("")


    combined = "\n".join(
        parts
    )


    output_path.write_text(
        combined,
        encoding="utf-8",
    )


# =========================================================
# Diagnostics
# =========================================================


def document_statistics(
    document: ParsedDocument,
) -> dict:
    """
    Return basic extraction statistics.

    These values are useful during development for quickly
    identifying pages that extracted poorly.
    """

    total_characters = sum(
        len(page.text)
        for page in document.pages
    )


    total_raw_characters = sum(
        len(page.raw_text)
        for page in document.pages
    )


    empty_pages = sum(
        1
        for page in document.pages
        if not page.text.strip()
    )


    pages_with_toc = sum(
        1
        for page in document.pages
        if page.toc_items
    )


    pages_with_layout = sum(
        1
        for page in document.pages
        if page.page_boxes
    )


    caption_count = sum(
        len(page.captions)
        for page in document.pages
    )


    return {

        "filename":
            document.filename,

        "page_count":
            document.page_count,

        "total_characters":
            total_characters,

        "total_raw_characters":
            total_raw_characters,

        "empty_pages":
            empty_pages,

        "pages_with_toc":
            pages_with_toc,

        "pages_with_layout":
            pages_with_layout,

        "caption_count":
            caption_count,

        "sha256":
            document.sha256,
    }