from __future__ import annotations

import argparse

from research_rag.config import (
    LLM_MODEL,
    RAG_EVIDENCE_BUDGET,
    RAG_FINAL_K,
    RAG_RETRIEVAL_K,
    RETRIEVAL_MODE,
)

from research_rag.rag import (
    answer_question,
    format_pages,
)


# =========================================================
# Command-line arguments
# =========================================================


def parse_arguments():
    """
    Parse command-line arguments for grounded research RAG.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Ask grounded questions over the local "
            "research PDF library."
        )
    )


    parser.add_argument(
        "question",
        nargs="+",
        help=(
            "Question to answer from the local "
            "research library."
        ),
    )


    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=RAG_RETRIEVAL_K,
        help=(
            "Number of candidate chunks retained after "
            "retrieval refinement."
        ),
    )


    parser.add_argument(
        "--final-k",
        type=int,
        default=RAG_FINAL_K,
        help=(
            "Maximum number of evidence chunks sent "
            "to the LLM."
        ),
    )


    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=RAG_EVIDENCE_BUDGET,
        help=(
            "Approximate maximum number of evidence "
            "tokens supplied to the LLM."
        ),
    )


    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help=(
            "Print the complete evidence packet supplied "
            "to the LLM."
        ),
    )


    parser.add_argument(
        "--show-raw-answer",
        action="store_true",
        help=(
            "Print the raw model answer before deterministic "
            "filename/page citation rendering."
        ),
    )


    parser.add_argument(
        "--show-all-claims",
        action="store_true",
        help=(
            "Print all claim-support verification results, "
            "including supported claims."
        ),
    )


    return parser.parse_args()


# =========================================================
# Display helpers
# =========================================================


def separator():
    """
    Print a standard section separator.
    """

    print()

    print(
        "=" * 88
    )

    print()


# =========================================================
# Header
# =========================================================


def display_basic_header(
    question: str,
):
    """
    Display model and retrieval configuration.
    """

    separator()

    print(
        "Research RAG — Grounded Question Answering"
    )

    print()

    print(
        f"Model:     {LLM_MODEL}"
    )

    print(
        f"Retrieval: {RETRIEVAL_MODE}"
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


# =========================================================
# Retrieval summary
# =========================================================


def display_retrieval_summary(
    result,
):
    """
    Display retrieval and evidence-budget statistics.
    """

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


# =========================================================
# Evidence sufficiency
# =========================================================


def display_sufficiency_check(
    result,
):
    """
    Display deterministic and LLM evidence-sufficiency
    diagnostics.
    """

    separator()

    print(
        "EVIDENCE SUFFICIENCY CHECK"
    )

    print()


    check = (
        result.sufficiency_check
    )


    print(
        f"Sufficient: "
        f"{check.sufficient}"
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


    if check.matched_anchor_terms:

        print()

        print(
            "Matched anchor terms:"
        )

        print(
            ", ".join(
                check.matched_anchor_terms
            )
        )


# =========================================================
# Generation diagnostics
# =========================================================


def display_generation_diagnostics(
    result,
):
    """
    Display LLM completion metadata.

    This indicates whether:
    - generation ended normally
    - the concise retry was used
    - how many output tokens were generated
    """

    separator()

    print(
        "GENERATION DIAGNOSTICS"
    )

    print()


    print(
        f"Retry used:    "
        f"{result.generation_retried}"
    )


    if (
        result.generation_done_reason
        is not None
    ):

        print(
            f"Done reason:   "
            f"{result.generation_done_reason}"
        )

    else:

        print(
            "Done reason:   unavailable"
        )


    if (
        result.generation_eval_count
        is not None
    ):

        print(
            f"Output tokens: "
            f"{result.generation_eval_count}"
        )

    else:

        print(
            "Output tokens: unavailable"
        )


# =========================================================
# Evidence display
# =========================================================


def display_evidence(
    evidence,
):
    """
    Print the complete evidence packet selected for the LLM.

    Primarily useful for retrieval/debugging work.
    """

    separator()

    print(
        "EVIDENCE SENT TO LLM"
    )


    if not evidence:

        print()

        print(
            "No evidence selected."
        )

        return


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


        if (
            item.distance
            is not None
        ):

            print(
                f"Distance: "
                f"{item.distance}"
            )


        print()

        print(
            item.text
        )


# =========================================================
# Answer
# =========================================================


def display_answer(
    result,
):
    """
    Display the final user-facing answer.
    """

    separator()

    print(
        "ANSWER"
    )

    print()

    print(
        result.answer
    )


# =========================================================
# Abstention
# =========================================================


def display_abstention(
    result,
):
    """
    Display why the RAG pipeline refused to return a normal
    scientific answer.
    """

    if not result.abstained:

        return


    separator()

    print(
        "RAG ABSTAINED"
    )

    print()

    print(
        f"Reason: "
        f"{result.abstention_reason}"
    )


# =========================================================
# Citation validation
# =========================================================


def display_citation_check(
    result,
):
    """
    Display citation-ID validation diagnostics.
    """

    citation = (
        result.citation_check
    )


    # -----------------------------------------------------
    # If the query was rejected before answer generation,
    # citation diagnostics are not meaningful.
    # -----------------------------------------------------

    if (
        result.abstained
        and not result.raw_answer
    ):

        return


    separator()

    print(
        "CITATION CHECK"
    )

    print()


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
            "Unknown IDs:   "
            + ", ".join(
                citation.unknown_source_ids
            )
        )


# =========================================================
# Claim-to-citation support verification
# =========================================================


def display_claim_support_check(
    result,
    show_all_claims: bool = False,
):
    """
    Display claim-level evidence-support diagnostics.

    By default, only problematic claims are shown.

    Use --show-all-claims to display every verified claim.
    """

    check = (
        result.claim_support_check
    )


    if check is None:

        return


    separator()

    print(
        "CLAIM-TO-CITATION SUPPORT CHECK"
    )

    print()


    print(
        f"Valid:           "
        f"{check.valid}"
    )

    print(
        f"Pruning attempted: "
        f"{result.claim_support_pruning_attempted}"
    )

    print(
        f"Pruning accepted:  "
        f"{result.claim_support_pruned}"
    )

    print(
        f"Repair attempted:  "
        f"{result.claim_support_repair_attempted}"
    )

    print(
        f"Repair accepted:   "
        f"{result.claim_support_repaired}"
    )

    print(
        f"Supported:       "
        f"{check.supported_count}"
    )

    print(
        f"Partial:         "
        f"{check.partial_count}"
    )

    print(
        f"Unsupported:     "
        f"{check.unsupported_count}"
    )

    print(
        f"Uncited claims:  "
        f"{len(check.uncited_claims)}"
    )


    if (
        check.missing_claim_ids
    ):

        print(
            f"Missing results: "
            f"{len(check.missing_claim_ids)}"
        )


    if (
        check.unknown_claim_ids
    ):

        print(
            f"Unknown IDs:     "
            f"{len(check.unknown_claim_ids)}"
        )


    if (
        check.verifier_done_reason
        is not None
    ):

        print(
            f"Verifier end:    "
            f"{check.verifier_done_reason}"
        )


    if (
        check.verifier_eval_count
        is not None
    ):

        print(
            f"Verifier tokens: "
            f"{check.verifier_eval_count}"
        )


    print()

    print(
        f"Reason: "
        f"{check.reason}"
    )


    # =====================================================
    # Determine which claim results to display
    # =====================================================

    if show_all_claims:

        items_to_show = (
            check.items
        )

    else:

        items_to_show = [

            item

            for item in check.items

            if (
                item.verdict
                != "SUPPORTED"
            )
        ]


    # =====================================================
    # Claim results
    # =====================================================

    if items_to_show:

        print()

        if show_all_claims:

            print(
                "Claim results:"
            )

        else:

            print(
                "Problem claims:"
            )


        for item in items_to_show:

            print()

            print(
                f"{item.claim_id} — "
                f"{item.verdict}"
            )

            print(
                f"Claim:"
            )

            print(
                item.claim_text
            )

            print()

            print(
                "Sources:"
            )

            print(
                ", ".join(
                    item.source_ids
                )
            )

            print()

            print(
                "Verifier reason:"
            )

            print(
                item.reason
            )


    # =====================================================
    # Uncited substantive claims
    # =====================================================

    if check.uncited_claims:

        print()

        print(
            "Uncited substantive claims:"
        )


        for index, claim in enumerate(
            check.uncited_claims,
            start=1,
        ):

            print()

            print(
                f"{index}. {claim}"
            )


    # =====================================================
    # Missing verifier results
    # =====================================================

    if check.missing_claim_ids:

        print()

        print(
            "Missing verifier claim IDs:"
        )

        print(
            ", ".join(
                check.missing_claim_ids
            )
        )


    # =====================================================
    # Unknown verifier IDs
    # =====================================================

    if check.unknown_claim_ids:

        print()

        print(
            "Unexpected verifier claim IDs:"
        )

        print(
            ", ".join(
                check.unknown_claim_ids
            )
        )


# =========================================================
# Raw model answer
# =========================================================


def display_raw_answer(
    result,
):
    """
    Display the raw source-ID answer before filename/page
    citation rendering.
    """

    if not result.raw_answer:

        return


    separator()

    print(
        "RAW MODEL ANSWER"
    )

    print()

    print(
        result.raw_answer
    )


# =========================================================
# Source map
# =========================================================


def display_source_map(
    evidence,
):
    """
    Display deterministic source-ID → PDF metadata mapping.
    """

    if not evidence:

        return


    separator()

    print(
        "SOURCE MAP"
    )

    print()


    for item in evidence:

        pages = (
            format_pages(
                item.page_start,
                item.page_end,
            )
        )


        if (
            item.page_start
            == item.page_end
        ):

            page_label = (
                f"p. {pages}"
            )

        else:

            page_label = (
                f"pp. {pages}"
            )


        section = (
            item.section.strip()

            if item.section

            else "Unknown section"
        )


        print(
            f"[{item.source_id}] "
            f"{item.filename}, "
            f"{page_label} — "
            f"{section}"
        )


# =========================================================
# Main
# =========================================================


def main():
    """
    Run one grounded RAG question from the command line.
    """

    args = (
        parse_arguments()
    )


    question = " ".join(
        args.question
    ).strip()


    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    # =====================================================
    # Header
    # =====================================================

    display_basic_header(
        question
    )


    # =====================================================
    # Run RAG pipeline
    # =====================================================

    result = (
        answer_question(

            question=(
                question
            ),

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


    # =====================================================
    # Retrieval diagnostics
    # =====================================================

    display_retrieval_summary(
        result
    )


    # =====================================================
    # Evidence sufficiency
    # =====================================================

    display_sufficiency_check(
        result
    )


    # =====================================================
    # Generation diagnostics
    #
    # Only display if generation actually occurred.
    # =====================================================

    generation_occurred = (

        result.generation_done_reason
        is not None

        or result.generation_eval_count
        is not None

        or bool(
            result.raw_answer
        )
    )


    if generation_occurred:

        display_generation_diagnostics(
            result
        )


    # =====================================================
    # Optional evidence dump
    # =====================================================

    if args.show_evidence:

        display_evidence(
            result.evidence
        )


    # =====================================================
    # Final answer
    # =====================================================

    display_answer(
        result
    )


    # =====================================================
    # Abstention diagnostics
    # =====================================================

    display_abstention(
        result
    )


    # =====================================================
    # Citation-ID validation
    # =====================================================

    display_citation_check(
        result
    )


    # =====================================================
    # Claim-to-citation support verification
    # =====================================================

    display_claim_support_check(

        result,

        show_all_claims=(
            args.show_all_claims
        ),
    )


    # =====================================================
    # Optional raw answer
    # =====================================================

    if args.show_raw_answer:

        display_raw_answer(
            result
        )


    # =====================================================
    # Source map
    # =====================================================

    display_source_map(
        result.evidence
    )


    separator()


if __name__ == "__main__":

    main()