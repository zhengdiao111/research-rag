from __future__ import annotations

import argparse

from research_rag.config import (
    RETRIEVAL_CANDIDATE_MULTIPLIER,
)

from research_rag.embeddings import (
    OllamaEmbedder,
)

from research_rag.retrieval import (
    hybrid_search,
    refine_hybrid_results,
)


# =========================================================
# Arguments
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Compare raw hybrid retrieval with "
            "research-aware refined retrieval."
        )
    )


    parser.add_argument(
        "query",
        nargs="+",
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )


    return parser.parse_args()


# =========================================================
# Display
# =========================================================


def display_results(
    title: str,
    results: list[dict],
):

    print()

    print(
        "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )

    print()


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


        chunk_type = (
            result.get(
                "chunk_type",
                ""
            )
        )


        if (
            page_start
            == page_end
        ):

            pages = str(
                page_start
            )

        else:

            pages = (
                f"{page_start}-"
                f"{page_end}"
            )


        print(
            f"{rank:>2}. {filename}"
        )

        print(
            f"    Page(s): "
            f"{pages}"
        )

        print(
            f"    Section: "
            f"{section}"
        )

        print(
            f"    Type:    "
            f"{chunk_type}"
        )


        relevance = (
            result.get(
                "_relevance_score"
            )
        )


        if relevance is not None:

            print(
                f"    RRF relevance: "
                f"{relevance}"
            )


        base = (
            result.get(
                "_base_relevance_score"
            )
        )


        if base is not None:

            print(
                f"    Base score:    "
                f"{base}"
            )


        weight = (
            result.get(
                "_chunk_type_weight"
            )
        )


        if weight is not None:

            print(
                f"    Type weight:   "
                f"{weight}"
            )


        refined = (
            result.get(
                "_refined_score"
            )
        )


        if refined is not None:

            print(
                f"    Refined score: "
                f"{refined}"
            )


        caption_like = (
            result.get(
                "_caption_like"
            )
        )


        if caption_like is not None:

            print(
                f"    Caption-like:  "
                f"{caption_like}"
            )


        print()


# =========================================================
# Main
# =========================================================


def main():

    args = (
        parse_arguments()
    )


    query = " ".join(
        args.query
    ).strip()


    if not query:

        raise ValueError(
            "Query cannot be empty."
        )


    print()

    print(
        f"Query: {query}"
    )


    print(
        f"Final top-k: "
        f"{args.top_k}"
    )


    candidate_k = (
        args.top_k
        * RETRIEVAL_CANDIDATE_MULTIPLIER
    )


    print(
        f"Candidate pool: "
        f"{candidate_k}"
    )


    embedder = (
        OllamaEmbedder()
    )


    query_vector = (
        embedder.embed_query(
            query
        )
    )


    # =====================================================
    # Retrieve a larger raw hybrid candidate pool
    # =====================================================

    candidates = (
        hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=candidate_k,
        )
    )


    # =====================================================
    # Raw RRF top-k
    # =====================================================

    raw_top_k = (
        candidates[
            :args.top_k
        ]
    )


    # =====================================================
    # Research-aware refinement
    # =====================================================

    refined = (
        refine_hybrid_results(
            results=candidates,
            top_k=args.top_k,
        )
    )


    # =====================================================
    # Display comparison
    # =====================================================

    display_results(
        "RAW HYBRID / RRF TOP RESULTS",
        raw_top_k,
    )


    display_results(
        "REFINED RESEARCH-AWARE RESULTS",
        refined,
    )


if __name__ == "__main__":

    main()