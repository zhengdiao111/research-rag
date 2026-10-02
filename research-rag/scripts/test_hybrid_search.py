from __future__ import annotations

import argparse

from research_rag.embeddings import (
    OllamaEmbedder,
)

from research_rag.retrieval import (
    hybrid_search,
    lexical_search,
    vector_search,
)


# =========================================================
# Arguments
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Compare vector, lexical, and hybrid "
            "retrieval."
        )
    )


    parser.add_argument(
        "query",
        nargs="+",
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )


    return parser.parse_args()


# =========================================================
# Display
# =========================================================


def display_results(
    label: str,
    results: list[dict],
):

    print()

    print(
        "=" * 88
    )

    print(
        label
    )

    print(
        "=" * 88
    )

    print()


    if not results:

        print(
            "No results."
        )

        return


    for rank, result in enumerate(
        results,
        start=1,
    ):

        filename = (
            result.get(
                "filename",
                ""
            )
        )

        page_start = (
            result.get(
                "page_start",
                ""
            )
        )

        page_end = (
            result.get(
                "page_end",
                ""
            )
        )

        section = (
            result.get(
                "section",
                ""
            )
        )


        if (
            page_start
            == page_end
        ):

            pages = (
                str(
                    page_start
                )
            )

        else:

            pages = (
                f"{page_start}-"
                f"{page_end}"
            )


        print(
            f"{rank:>2}. "
            f"{filename}"
        )

        print(
            f"    Page(s): {pages}"
        )

        print(
            f"    Section: {section}"
        )


        distance = (
            result.get(
                "_distance"
            )
        )


        score = (
            result.get(
                "_score"
            )
        )


        relevance = (
            result.get(
                "_relevance_score"
            )
        )


        if distance is not None:

            print(
                f"    Vector distance: "
                f"{distance}"
            )


        if score is not None:

            print(
                f"    FTS score: "
                f"{score}"
            )


        if relevance is not None:

            print(
                f"    Hybrid relevance: "
                f"{relevance}"
            )


        print()


# =========================================================
# Main
# =========================================================


def main():

    args = parse_arguments()


    query = " ".join(
        args.query
    ).strip()


    print()

    print(
        f"Query: {query}"
    )


    embedder = (
        OllamaEmbedder()
    )


    query_vector = (
        embedder.embed_query(
            query
        )
    )


    vector_results = (
        vector_search(
            query_vector=(
                query_vector
            ),
            top_k=(
                args.top_k
            ),
        )
    )


    lexical_results = (
        lexical_search(
            query_text=(
                query
            ),
            top_k=(
                args.top_k
            ),
        )
    )


    hybrid_results = (
        hybrid_search(
            query_text=(
                query
            ),
            query_vector=(
                query_vector
            ),
            top_k=(
                args.top_k
            ),
        )
    )


    display_results(
        "VECTOR SEARCH",
        vector_results,
    )


    display_results(
        "LEXICAL / BM25 SEARCH",
        lexical_results,
    )


    display_results(
        "HYBRID SEARCH",
        hybrid_results,
    )


if __name__ == "__main__":

    main()