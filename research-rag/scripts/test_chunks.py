from __future__ import annotations

import argparse
import json

from dataclasses import asdict
from pathlib import Path

from research_rag.chunker import (
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    chunk_document,
    chunk_statistics,
)

from research_rag.config import (
    DOCUMENT_ROOT,
    PROCESSED_DIR,
)

from research_rag.parser import (
    discover_pdfs,
    parse_pdf,
)


# =========================================================
# CLI
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect research-aware RAG chunks "
            "before generating embeddings."
        )
    )


    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help=(
            "Optional PDF path. If omitted, the first PDF "
            "in DOCUMENT_ROOT will be used."
        ),
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Maximum number of chunks to print. "
            "Default: 20."
        ),
    )


    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Print every chunk."
        ),
    )


    return parser.parse_args()


# =========================================================
# Helpers
# =========================================================


def separator():

    print()

    print(
        "=" * 88
    )

    print()


def short_separator():

    print()

    print(
        "-" * 88
    )

    print()


def save_jsonl(
    chunks,
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            record = asdict(
                chunk
            )

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )


def save_diagnostic_markdown(
    chunks,
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    parts: list[str] = []


    parts.append(
        "# Research RAG Chunk Diagnostic"
    )

    parts.append("")


    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        parts.append(
            "---"
        )

        parts.append("")

        parts.append(
            f"## Chunk {index}"
        )

        parts.append("")

        parts.append(
            f"- **Chunk ID:** `{chunk.chunk_id}`"
        )

        parts.append(
            f"- **File:** `{chunk.filename}`"
        )

        if (
            chunk.page_start
            == chunk.page_end
        ):

            page_display = (
                str(
                    chunk.page_start
                )
            )

        else:

            page_display = (
                f"{chunk.page_start}"
                f"–{chunk.page_end}"
            )


        parts.append(
            f"- **Page(s):** {page_display}"
        )

        parts.append(
            f"- **Section:** {chunk.section}"
        )

        parts.append(
            f"- **Type:** `{chunk.chunk_type}`"
        )

        parts.append(
            f"- **Approx. tokens:** {chunk.token_count}"
        )

        parts.append("")

        parts.append(
            chunk.text
        )

        parts.append("")


    output_path.write_text(
        "\n".join(
            parts
        ),
        encoding="utf-8",
    )


# =========================================================
# Main
# =========================================================


def main():

    args = parse_arguments()


    separator()

    print(
        "Research RAG — Chunking test"
    )

    print()

    print(
        f"Chunk target:   {CHUNK_TARGET}"
    )

    print(
        f"Chunk minimum:  {CHUNK_MIN}"
    )

    print(
        f"Chunk maximum:  {CHUNK_MAX}"
    )

    print(
        f"Overlap target: {CHUNK_OVERLAP}"
    )


    # -----------------------------------------------------
    # Select PDF
    # -----------------------------------------------------

    if args.pdf:

        pdf_path = Path(
            args.pdf
        ).expanduser()

    else:

        pdfs = discover_pdfs(
            DOCUMENT_ROOT
        )


        if not pdfs:

            raise RuntimeError(
                "No PDF files found in:\n"
                f"{DOCUMENT_ROOT}"
            )


        pdf_path = (
            pdfs[0]
        )


    print()

    print(
        f"PDF:\n{pdf_path}"
    )


    # -----------------------------------------------------
    # Parse PDF
    # -----------------------------------------------------

    document = parse_pdf(
        pdf_path
    )


    # -----------------------------------------------------
    # Chunk document
    # -----------------------------------------------------

    chunks = chunk_document(
        document
    )


    stats = chunk_statistics(
        chunks
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    separator()

    print(
        "CHUNK SUMMARY"
    )

    print()

    print(
        f"Chunks:          {stats['chunk_count']}"
    )

    print(
        f"Total tokens:    {stats['total_tokens']:,}"
    )

    print(
        f"Smallest chunk:  {stats['min_tokens']}"
    )

    print(
        f"Largest chunk:   {stats['max_tokens']}"
    )

    print(
        f"Mean chunk:      {stats['mean_tokens']:.1f}"
    )

    print(
        f"Below minimum:   {stats['below_min']}"
    )

    print(
        f"Above maximum:   {stats['above_max']}"
    )


    print()

    print(
        "Chunk types:"
    )


    for chunk_type, count in sorted(
        stats["types"].items()
    ):

        print(
            f"  {chunk_type:<20} {count}"
        )


    # -----------------------------------------------------
    # Print chunks
    # -----------------------------------------------------

    if args.all:

        display_chunks = (
            chunks
        )

    else:

        display_chunks = (
            chunks[
                :max(
                    args.limit,
                    0,
                )
            ]
        )


    for index, chunk in enumerate(
        display_chunks,
        start=1,
    ):

        separator()

        print(
            f"CHUNK {index}"
        )

        print()

        print(
            f"ID:       {chunk.chunk_id}"
        )

        print(
            f"Type:     {chunk.chunk_type}"
        )

        print(
            f"Section:  {chunk.section}"
        )


        if (
            chunk.page_start
            == chunk.page_end
        ):

            page_display = (
                str(
                    chunk.page_start
                )
            )

        else:

            page_display = (
                f"{chunk.page_start}"
                f"-{chunk.page_end}"
            )


        print(
            f"Page(s):  {page_display}"
        )

        print(
            f"Tokens:   {chunk.token_count}"
        )


        short_separator()

        print(
            chunk.text
        )


    # -----------------------------------------------------
    # Save diagnostics
    # -----------------------------------------------------

    jsonl_path = (
        PROCESSED_DIR
        / f"{pdf_path.stem}_chunks.jsonl"
    )


    markdown_path = (
        PROCESSED_DIR
        / f"{pdf_path.stem}_chunks.md"
    )


    save_jsonl(
        chunks,
        jsonl_path,
    )


    save_diagnostic_markdown(
        chunks,
        markdown_path,
    )


    separator()

    print(
        "Saved:"
    )

    print()

    print(
        f"JSONL:\n{jsonl_path}"
    )

    print()

    print(
        f"Diagnostic Markdown:\n{markdown_path}"
    )

    print()

    print(
        "Inspect the Markdown before generating embeddings."
    )

    separator()


if __name__ == "__main__":
    main()