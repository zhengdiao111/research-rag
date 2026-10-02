from __future__ import annotations

import argparse

from research_rag.config import (
    EMBED_MODEL,
    SEMANTIC_TOP_K,
)

from research_rag.database import (
    semantic_search,
)

from research_rag.embeddings import (
    OllamaEmbedder,
)


# =========================================================
# Command-line arguments
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Semantic search over locally indexed "
            "research PDFs and books."
        )
    )


    parser.add_argument(
        "query",
        nargs="+",
        help=(
            "Natural-language search query."
        ),
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=SEMANTIC_TOP_K,
        help=(
            "Number of results to return. "
            f"Default: {SEMANTIC_TOP_K}"
        ),
    )


    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1500,
        help=(
            "Maximum characters to display "
            "for each result."
        ),
    )


    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Print the complete retrieved chunk "
            "instead of a preview."
        ),
    )


    return parser.parse_args()


# =========================================================
# Display helpers
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


def format_pages(
    page_start: int,
    page_end: int,
) -> str:

    if page_start == page_end:

        return str(
            page_start
        )

    return (
        f"{page_start}-"
        f"{page_end}"
    )


# =========================================================
# Main
# =========================================================


def main():

    args = parse_arguments()


    # -----------------------------------------------------
    # Prepare query
    # -----------------------------------------------------

    query = " ".join(
        args.query
    ).strip()


    if not query:

        raise ValueError(
            "Query cannot be empty."
        )


    if args.top_k <= 0:

        raise ValueError(
            "--top-k must be greater than 0."
        )


    if args.preview_chars <= 0:

        raise ValueError(
            "--preview-chars must be greater than 0."
        )


    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    separator()

    print(
        "Research RAG — Semantic Search"
    )

    print()

    print(
        "Query:"
    )

    print(
        query
    )

    print()

    print(
        f"Embedding model: {EMBED_MODEL}"
    )

    print(
        f"Top K:           {args.top_k}"
    )


    # -----------------------------------------------------
    # Embed query
    # -----------------------------------------------------

    print()

    print(
        "Embedding query..."
    )


    embedder = (
        OllamaEmbedder()
    )


    # IMPORTANT:
    #
    # This uses Nomic's search_query prefix internally.
    #
    # Indexed document chunks use search_document.
    #
    # Keeping these two task prefixes distinct improves
    # retrieval performance for nomic-embed-text.

    query_vector = (
        embedder.embed_query(
            query
        )
    )


    print(
        f"Query embedding dimension: "
        f"{len(query_vector)}"
    )


    # -----------------------------------------------------
    # Search LanceDB
    # -----------------------------------------------------

    print()

    print(
        "Searching local research index..."
    )


    results = semantic_search(

        query_vector=(
            query_vector
        ),

        top_k=(
            args.top_k
        ),
    )


    # -----------------------------------------------------
    # No results
    # -----------------------------------------------------

    if not results:

        separator()

        print(
            "No results found."
        )

        print()

        return


    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    separator()

    print(
        f"RESULTS — {len(results)} retrieved"
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        separator()

        print(
            f"RESULT {rank}"
        )

        print()


        # ---------------------------------------------
        # Distance
        # ---------------------------------------------

        distance = result.get(
            "_distance"
        )


        if distance is not None:

            try:

                distance_display = (
                    f"{float(distance):.6f}"
                )

            except (
                TypeError,
                ValueError,
            ):

                distance_display = (
                    str(distance)
                )


            print(
                f"Distance:    "
                f"{distance_display}"
            )


        # ---------------------------------------------
        # Document
        # ---------------------------------------------

        filename = str(
            result.get(
                "filename",
                "",
            )
        )


        title = str(
            result.get(
                "title",
                "",
            )
        ).strip()


        author = str(
            result.get(
                "author",
                "",
            )
        ).strip()


        print(
            f"File:        "
            f"{filename}"
        )


        if title:

            print(
                f"Title:       "
                f"{title}"
            )


        if author:

            print(
                f"Author:      "
                f"{author}"
            )


        # ---------------------------------------------
        # Location
        # ---------------------------------------------

        page_start = int(
            result.get(
                "page_start",
                0,
            )
        )


        page_end = int(
            result.get(
                "page_end",
                page_start,
            )
        )


        page_display = (
            format_pages(
                page_start,
                page_end,
            )
        )


        print(
            f"Page(s):     "
            f"{page_display}"
        )


        print(
            f"Section:     "
            f"{result.get('section', '')}"
        )


        print(
            f"Chunk type:  "
            f"{result.get('chunk_type', '')}"
        )


        print(
            f"Tokens:      "
            f"{result.get('token_count', '')}"
        )


        print(
            f"Chunk ID:    "
            f"{result.get('chunk_id', '')}"
        )


        # ---------------------------------------------
        # Text
        # ---------------------------------------------

        short_separator()


        text = str(
            result.get(
                "text",
                "",
            )
        ).strip()


        if args.full:

            display_text = (
                text
            )

        else:

            display_text = (
                text[
                    :args.preview_chars
                ]
            )


            if (
                len(text)
                > args.preview_chars
            ):

                display_text += (
                    "\n\n"
                    "... [truncated]"
                )


        print(
            display_text
        )


    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------

    separator()

    print(
        "Search complete."
    )

    print()

    print(
        "Lower distance generally means "
        "greater semantic similarity."
    )

    print()


if __name__ == "__main__":
    main()