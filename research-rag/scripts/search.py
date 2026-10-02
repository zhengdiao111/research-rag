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
# Arguments
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Semantic search over locally indexed "
            "research PDFs."
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
            "Number of chunks to return."
        ),
    )


    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1500,
        help=(
            "Maximum characters displayed "
            "for each chunk."
        ),
    )


    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Print complete chunk text."
        ),
    )


    return parser.parse_args()


# =========================================================
# Helpers
# =========================================================


def separator():

    print()
    print("=" * 88)
    print()


# =========================================================
# Main
# =========================================================


def main():

    args = parse_arguments()


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


    print()

    print(
        "Research RAG — Semantic search"
    )

    print()

    print(
        f"Query:\n{query}"
    )

    print()

    print(
        f"Embedding model: "
        f"{EMBED_MODEL}"
    )


    # -----------------------------------------------------
    # Query embedding
    # -----------------------------------------------------

    embedder = (
        OllamaEmbedder()
    )


    query_vector = (
        embedder.embed_text(
            query
        )
    )


    # -----------------------------------------------------
    # Vector search
    # -----------------------------------------------------

    results = semantic_search(
        query_vector=query_vector,

        top_k=args.top_k,
    )


    if not results:

        print()

        print(
            "No results found."
        )

        return


    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    for rank, result in enumerate(
        results,
        start=1,
    ):

        separator()

        print(
            f"RESULT {rank}"
        )

        print()


        distance = result.get(
            "_distance"
        )


        if distance is not None:

            print(
                f"Distance:  "
                f"{float(distance):.6f}"
            )


        print(
            f"File:      "
            f"{result['filename']}"
        )


        page_start = int(
            result[
                "page_start"
            ]
        )

        page_end = int(
            result[
                "page_end"
            ]
        )


        if page_start == page_end:

            page_display = (
                str(
                    page_start
                )
            )

        else:

            page_display = (
                f"{page_start}-"
                f"{page_end}"
            )


        print(
            f"Page(s):   "
            f"{page_display}"
        )

        print(
            f"Section:   "
            f"{result['section']}"
        )

        print(
            f"Type:      "
            f"{result['chunk_type']}"
        )

        print(
            f"Tokens:    "
            f"{result['token_count']}"
        )


        print()

        print(
            "-" * 88
        )

        print()


        text = str(
            result["text"]
        )


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


    separator()


if __name__ == "__main__":
    main()