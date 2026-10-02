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
            "Number of candidate chunks retrieved "
            "before evidence selection."
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
            "citation rendering."
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


def display_generation_diagnostics(
    result,
):
    """
    Display LLM completion metadata.

    This makes it easy to see whether a concise retry was
    required and whether Ollama finished normally.
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


def display_evidence(
    evidence,
):
    """
    Print the complete selected evidence packet.

    This is primarily useful for debugging retrieval and
    evidence selection.
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


        if item.distance is not None:

            print(
                f"Distance: "
                f"{item.distance}"
            )


        print()

        print(
            item.text
        )


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


def display_abstention(
    result,
):
    """
    Display why the RAG pipeline refused to produce a
    normal answer.
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


def display_citation_check(
    result,
):
    """
    Display citation-validation diagnostics.

    Citation diagnostics remain useful even when the system
    abstains because of a citation failure.
    """

    citation = (
        result.citation_check
    )


    # -----------------------------------------------------
    # For a normal evidence-insufficiency abstention there
    # was intentionally no generated citation-bearing
    # answer, so skip this section.
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


def display_raw_answer(
    result,
):
    """
    Display the raw source-ID answer before citation
    rendering.
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


def display_source_map(
    evidence,
):
    """
    Display the deterministic mapping between source IDs and
    PDF metadata.
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
    # Only display this when generation actually occurred.
    # If the lexical gate rejected the query before Qwen was
    # called, the metadata will be absent.
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
    # Citation validation
    # =====================================================

    display_citation_check(
        result
    )


    # =====================================================
    # Optional raw model answer
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