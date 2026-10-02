from __future__ import annotations

import argparse

from pathlib import Path

from research_rag.config import (
    DOCUMENT_ROOT,
    PROCESSED_DIR,
)

from research_rag.parser import (
    discover_pdfs,
    document_statistics,
    parse_pdf,
    save_parsed_markdown,
)


# =========================================================
# Command-line arguments
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Test research PDF extraction using "
            "PyMuPDF4LLM."
        )
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help=(
            "Optional path to a specific PDF. "
            "If omitted, the first PDF in DOCUMENT_ROOT "
            "will be used."
        ),
    )

    return parser.parse_args()


# =========================================================
# Display helpers
# =========================================================


def print_separator():

    print()
    print("=" * 80)
    print()


def print_page_preview(
    page,
    preview_characters: int = 1500,
):

    print_separator()

    print(
        f"PAGE {page.page_number}"
    )

    print()


    if page.toc_items:

        print("TOC entries:")

        for item in page.toc_items:

            print(
                f"  {item}"
            )

        print()


    preview = page.text[
        :preview_characters
    ]


    print(preview)


    if len(page.text) > preview_characters:

        print()
        print(
            "... [preview truncated]"
        )


# =========================================================
# Main
# =========================================================


def main():

    args = parse_arguments()


    print_separator()

    print(
        "Research RAG — PDF extraction test"
    )

    print()

    print(
        f"Document root:\n{DOCUMENT_ROOT}"
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

        print()

        print(
            f"PDFs discovered: {len(pdfs)}"
        )


        if not pdfs:

            raise RuntimeError(
                "\nNo PDF files were found.\n\n"
                "Place at least one research paper in:\n"
                f"{DOCUMENT_ROOT}"
            )


        print()

        print(
            "First PDFs:"
        )

        for pdf in pdfs[:10]:

            try:

                relative = pdf.relative_to(
                    DOCUMENT_ROOT
                )

            except ValueError:

                relative = pdf


            print(
                f"  - {relative}"
            )


        pdf_path = pdfs[0]


    # -----------------------------------------------------
    # Parse
    # -----------------------------------------------------

    document = parse_pdf(
        pdf_path
    )


    stats = document_statistics(
        document
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print_separator()

    print(
        "EXTRACTION SUMMARY"
    )

    print()


    print(
        f"File:             {document.filename}"
    )

    print(
        f"Title:            {document.title or '(not found)'}"
    )

    print(
        f"Author:           {document.author or '(not found)'}"
    )

    print(
        f"Pages:            {stats['page_count']}"
    )

    print(
        f"Characters:       {stats['total_characters']:,}"
    )

    print(
        f"Empty pages:      {stats['empty_pages']}"
    )

    print(
        f"Pages with TOC:   {stats['pages_with_toc']}"
    )

    print(
        f"SHA-256:          {stats['sha256']}"
    )


    # -----------------------------------------------------
    # Preview first few pages
    # -----------------------------------------------------

    pages_to_preview = min(
        3,
        len(document.pages),
    )


    for page in document.pages[
        :pages_to_preview
    ]:

        print_page_preview(
            page
        )


    # -----------------------------------------------------
    # Save extracted Markdown
    # -----------------------------------------------------

    output_path = (
        PROCESSED_DIR
        / f"{pdf_path.stem}.md"
    )


    save_parsed_markdown(
        document,
        output_path,
    )


    print_separator()

    print(
        "Saved extracted Markdown:"
    )

    print(
        output_path
    )

    print()

    print(
        "Open this file in VS Code and compare it "
        "with the original PDF."
    )

    print_separator()


if __name__ == "__main__":
    main()