from __future__ import annotations

import argparse

from research_rag.config import (
    LLM_MODEL,
    RAG_EVIDENCE_BUDGET,
    RAG_FINAL_K,
    RAG_RETRIEVAL_K,
)

from research_rag.rag import (
    answer_question,
    format_pages,
)


# =========================================================
# Arguments
# =========================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Ask grounded questions over the local "
            "research library."
        )
    )


    parser.add_argument(
        "question",
        nargs="+",
    )


    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=RAG_RETRIEVAL_K,
    )


    parser.add_argument(
        "--final-k",
        type=int,
        default=RAG_FINAL_K,
    )


    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=RAG_EVIDENCE_BUDGET,
    )


    parser.add_argument(
        "--show-evidence",
        action="store_true",
    )


    parser.add_argument(
        "--show-raw-answer",
        action="store_true",
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


def show_evidence(
    evidence,
):

    separator()

    print(
        "EVIDENCE"
    )


    for item in evidence:

        separator()


        pages = (
            format_pages(
                item.page_start,
                item.page_end,
            )
        )


        print(
            f"[{item.source_id}]"
        )

        print(
            f"File:     {item.filename}"
        )

        print(
            f"Page(s):  {pages}"
        )

        print(
            f"Section:  {item.section}"
        )

        print(
            f"Type:     {item.chunk_type}"
        )

        print(
            f"Tokens:   {item.token_count}"
        )


        if item.distance is not None:

            print(
                f"Distance: "
                f"{item.distance:.6f}"
            )


        print()

        print(
            item.text
        )


# =========================================================
# Main
# =========================================================


def main():

    args = parse_arguments()


    question = " ".join(
        args.question
    ).strip()


    separator()

    print(
        "Research RAG — Grounded Question Answering"
    )

    print()

    print(
        f"Model: {LLM_MODEL}"
    )

    print()

    print(
        "Question:"
    )

    print()

    print(
        question
    )

    print()

    print(
        "Retrieving evidence..."
    )


    result = (
        answer_question(

            question=question,

            retrieval_k=(
                args.retrieval_k
            ),

            final_k=(
                args.final_k
            ),

            evidence_budget=(
                args.evidence_budget
            ),
        )
    )


    print()

    print(
        f"Retrieved candidates: "
        f"{result.retrieved_count}"
    )

    print(
        f"Evidence chunks used: "
        f"{len(result.evidence)}"
    )

    print(
        f"Evidence tokens:      "
        f"{result.evidence_tokens:,}"
    )


    # =====================================================
    # Sufficiency diagnostics
    # =====================================================

    separator()

    print(
        "EVIDENCE SUFFICIENCY CHECK"
    )

    print()


    check = (
        result.sufficiency_check
    )


    print(
        f"Sufficient: {check.sufficient}"
    )


    if (
        check.llm_sufficient
        is not None
    ):

        print(
            f"LLM check:  "
            f"{check.llm_sufficient}"
        )


    print()

    print(
        f"Reason: {check.reason}"
    )


    if check.query_terms:

        print()

        print(
            "Query terms:"
        )

        print(
            ", ".join(
                check.query_terms
            )
        )


    if check.matched_terms:

        print()

        print(
            "Matched terms:"
        )

        print(
            ", ".join(
                check.matched_terms
            )
        )


    if check.anchor_terms:

        print()

        print(
            "Anchor terms:"
        )

        print(
            ", ".join(
                check.anchor_terms
            )
        )


    if args.show_evidence:

        show_evidence(
            result.evidence
        )


    # =====================================================
    # Answer
    # =====================================================

    separator()

    print(
        "ANSWER"
    )

    print()

    print(
        result.answer
    )


    # =====================================================
    # Abstention information
    # =====================================================

    if result.abstained:

        separator()

        print(
            "RAG ABSTAINED"
        )

        print()

        print(
            f"Reason: "
            f"{result.abstention_reason}"
        )


    # =====================================================
    # Citation diagnostics
    # =====================================================

    if not result.abstained:

        separator()

        print(
            "CITATION CHECK"
        )

        print()


        citation = (
            result.citation_check
        )


        print(
            f"Has citations: "
            f"{citation.has_citations}"
        )

        print(
            f"Valid:         "
            f"{citation.valid}"
        )

        print(
            f"Repair used:   "
            f"{result.citation_repaired}"
        )


        if (
            citation.cited_source_ids
        ):

            print(
                "Cited sources: "
                + ", ".join(
                    citation.cited_source_ids
                )
            )


        if (
            citation.unknown_source_ids
        ):

            print(
                "Unknown IDs: "
                + ", ".join(
                    citation.unknown_source_ids
                )
            )


    # =====================================================
    # Raw answer
    # =====================================================

    if (
        args.show_raw_answer
        and result.raw_answer
    ):

        separator()

        print(
            "RAW MODEL ANSWER"
        )

        print()

        print(
            result.raw_answer
        )


    # =====================================================
    # Source map
    # =====================================================

    if result.evidence:

        separator()

        print(
            "SOURCE MAP"
        )

        print()


        for item in result.evidence:

            pages = (
                format_pages(
                    item.page_start,
                    item.page_end,
                )
            )


            print(
                f"[{item.source_id}] "
                f"{item.filename}, "
                f"p. {pages} — "
                f"{item.section}"
            )


    separator()


if __name__ == "__main__":

    main()