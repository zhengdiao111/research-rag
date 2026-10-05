from __future__ import annotations

import re

from dataclasses import dataclass

from .chunker import (
    count_tokens,
)

from .config import (
    CLAIM_SUPPORT_REPAIR,
    CLAIM_SUPPORT_VERIFY,
    RAG_CITATION_REPAIR,
    RAG_CITATION_VERIFY,
    RAG_EVIDENCE_BUDGET,
    RAG_FINAL_K,
    RAG_RETRIEVAL_K,
    RAG_SUFFICIENCY_CHECK,
)

from .retrieval import (
    retrieve,
)

from .embeddings import (
    OllamaEmbedder,
)

from .llm import (
    OllamaChatClient,
)

from .claim_verifier import (
    ClaimSupportCheck,
    prune_answer_to_supported_claims,
    repair_claim_support,
    verify_claim_support,
)


# =========================================================
# Data structures
# =========================================================


@dataclass
class EvidenceItem:

    source_id: str

    chunk_id: str

    filename: str

    title: str

    author: str

    page_start: int

    page_end: int

    section: str

    chunk_type: str

    text: str

    token_count: int

    distance: float | None


@dataclass
class CitationCheck:

    cited_source_ids: list[str]

    unknown_source_ids: list[str]

    has_citations: bool

    valid: bool


@dataclass
class SufficiencyCheck:

    sufficient: bool

    reason: str

    query_terms: list[str]

    matched_terms: list[str]

    anchor_terms: list[str]

    matched_anchor_terms: list[str]

    llm_sufficient: bool | None


@dataclass
class RAGResult:

    question: str

    answer: str

    raw_answer: str

    evidence: list[EvidenceItem]

    retrieved_count: int

    evidence_tokens: int

    citation_check: CitationCheck

    sufficiency_check: SufficiencyCheck

    abstained: bool

    abstention_reason: str | None

    citation_repaired: bool

    generation_retried: bool

    generation_done_reason: str | None

    generation_eval_count: int | None

    claim_support_check: ClaimSupportCheck | None = None

    claim_support_pruning_attempted: bool = False
    
    claim_support_pruned: bool = False

    claim_support_repair_attempted: bool = False

    claim_support_repaired: bool = False


# =========================================================
# Stop words
# =========================================================


_STOP_WORDS = {

    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",

    "are",
    "was",
    "were",
    "will",
    "would",
    "could",
    "should",

    "the",
    "and",
    "for",
    "from",
    "with",
    "without",
    "into",
    "onto",
    "about",

    "this",
    "that",
    "these",
    "those",

    "their",
    "there",
    "they",
    "them",

    "have",
    "has",
    "had",

    "does",
    "did",
    "doing",

    "can",
    "may",
    "might",

    "its",
    "our",
    "your",

    "use",
    "used",
    "using",

    "based",
    "between",
    "through",
    "within",

    "explain",
    "describe",
    "discuss",
}


# =========================================================
# Text helpers
# =========================================================


_WORD_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_-]*"
)


def tokenize_words(
    text: str,
) -> list[str]:

    return [
        match.group(0).lower()

        for match in _WORD_PATTERN.finditer(
            text
        )
    ]


def extract_query_terms(
    question: str,
) -> list[str]:
    """
    Extract useful lexical terms from a question.
    """

    terms: list[str] = []


    for token in tokenize_words(
        question
    ):

        if token in _STOP_WORDS:

            continue


        if len(token) < 4:

            continue


        if token not in terms:

            terms.append(
                token
            )


    return terms


def extract_anchor_terms(
    terms: list[str],
) -> list[str]:
    """
    Identify distinctive query terms.

    Examples:

        pembrolizumab
        SMARCA2
        BRM014
        reaction-diffusion
        rs123456
    """

    anchors: list[str] = []


    for term in terms:

        contains_digit = any(
            char.isdigit()
            for char in term
        )


        contains_hyphen = (
            "-"
            in term
        )


        unusually_long = (
            len(term)
            >= 12
        )


        if (
            contains_digit
            or contains_hyphen
            or unusually_long
        ):

            anchors.append(
                term
            )


    return anchors


# =========================================================
# Output completeness checks
# =========================================================


def has_incomplete_source_citation(
    text: str,
) -> bool:
    """
    Detect output that ends while writing a source citation.

    Examples:

        ... result [S
        ... result [S2
        ... result [S12
    """

    stripped = (
        text.rstrip()
    )


    return bool(
        re.search(
            r"\[S\d*$",
            stripped,
        )
    )


def generation_finished_cleanly(
    llm: OllamaChatClient,
    text: str,
) -> bool:
    """
    Require:

    1. Ollama reports normal completion when done_reason
       metadata is available.

    2. The answer does not end inside an unfinished
       [S<number>] citation.

    Some Ollama versions/builds may omit done_reason.
    When that happens, structural output checks are used.
    """

    citation_complete = (
        not has_incomplete_source_citation(
            text
        )
    )


    done_reason = (
        llm.last_done_reason
    )


    if done_reason is None:

        return (
            citation_complete
        )


    return (
        done_reason
        == "stop"

        and citation_complete
    )


# =========================================================
# Paragraph overlap cleanup
# =========================================================


def normalize_paragraph(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip().lower()


def remove_repeated_paragraphs(
    text: str,
    seen_paragraphs: set[str],
) -> str:
    """
    Remove paragraphs repeated because neighboring chunks
    intentionally overlap.
    """

    paragraphs = [

        paragraph.strip()

        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )

        if paragraph.strip()
    ]


    kept: list[str] = []


    for paragraph in paragraphs:

        normalized = (
            normalize_paragraph(
                paragraph
            )
        )


        track = (
            len(normalized)
            >= 120
        )


        if (
            track
            and normalized
            in seen_paragraphs
        ):

            continue


        kept.append(
            paragraph
        )


        if track:

            seen_paragraphs.add(
                normalized
            )


    return "\n\n".join(
        kept
    ).strip()


# =========================================================
# Evidence construction
# =========================================================


def result_to_evidence(
    result: dict,
    source_id: str,
    text: str,
) -> EvidenceItem:

    distance = (
        result.get(
            "_distance"
        )
    )


    if distance is not None:

        distance = float(
            distance
        )


    return EvidenceItem(

        source_id=source_id,

        chunk_id=str(
            result.get(
                "chunk_id",
                "",
            )
        ),

        filename=str(
            result.get(
                "filename",
                "",
            )
        ),

        title=str(
            result.get(
                "title",
                "",
            )
            or ""
        ),

        author=str(
            result.get(
                "author",
                "",
            )
            or ""
        ),

        page_start=int(
            result.get(
                "page_start",
                0,
            )
        ),

        page_end=int(
            result.get(
                "page_end",
                0,
            )
        ),

        section=str(
            result.get(
                "section",
                "",
            )
        ),

        chunk_type=str(
            result.get(
                "chunk_type",
                "",
            )
        ),

        text=text,

        token_count=(
            count_tokens(
                text
            )
        ),

        distance=distance,
    )


def select_evidence(
    results: list[dict],
    final_k: int = RAG_FINAL_K,
    token_budget: int = RAG_EVIDENCE_BUDGET,
) -> list[EvidenceItem]:
    """
    Select evidence after:

    - removing duplicate chunk IDs
    - removing repeated overlap paragraphs
    - applying the evidence token budget
    - applying the final-k limit
    """

    selected: list[
        EvidenceItem
    ] = []


    seen_chunk_ids: set[
        str
    ] = set()


    seen_paragraphs: set[
        str
    ] = set()


    used_tokens = 0


    for result in results:

        chunk_id = str(
            result.get(
                "chunk_id",
                "",
            )
        )


        if not chunk_id:

            continue


        if chunk_id in seen_chunk_ids:

            continue


        seen_chunk_ids.add(
            chunk_id
        )


        original_text = str(
            result.get(
                "text",
                "",
            )
        ).strip()


        if not original_text:

            continue


        cleaned_text = (
            remove_repeated_paragraphs(
                original_text,
                seen_paragraphs,
            )
        )


        if not cleaned_text:

            continue


        tokens = (
            count_tokens(
                cleaned_text
            )
        )


        if tokens < 20:

            continue


        if (
            used_tokens
            + tokens
            > token_budget
        ):

            continue


        source_id = (
            f"S{len(selected) + 1}"
        )


        item = (
            result_to_evidence(
                result=result,
                source_id=source_id,
                text=cleaned_text,
            )
        )


        selected.append(
            item
        )


        used_tokens += (
            item.token_count
        )


        if len(
            selected
        ) >= final_k:

            break


    return selected


# =========================================================
# Page formatting
# =========================================================


def format_pages(
    page_start: int,
    page_end: int,
) -> str:

    if (
        page_start
        == page_end
    ):

        return str(
            page_start
        )


    return (
        f"{page_start}-{page_end}"
    )


# =========================================================
# Evidence formatting
# =========================================================


def build_evidence_context(
    evidence: list[EvidenceItem],
) -> str:

    blocks: list[
        str
    ] = []


    for item in evidence:

        pages = (
            format_pages(
                item.page_start,
                item.page_end,
            )
        )


        lines = [

            f"[{item.source_id}]",

            f"File: {item.filename}",

            f"Page(s): {pages}",

            f"Section: {item.section}",

            f"Type: {item.chunk_type}",
        ]


        if item.title:

            lines.append(
                f"Title: {item.title}"
            )


        if item.author:

            lines.append(
                f"Author: {item.author}"
            )


        block = (
            "\n".join(
                lines
            )

            + "\n\n"

            + item.text
        )


        blocks.append(
            block
        )


    return (
        "\n\n"
        + "=" * 72
        + "\n\n"
    ).join(
        blocks
    )


# =========================================================
# Evidence sufficiency — lexical gate
# =========================================================


def lexical_sufficiency_check(
    question: str,
    evidence: list[EvidenceItem],
) -> SufficiencyCheck:
    """
    Deterministic relevance gate.

    This catches obvious retrieval failures before the LLM
    can attempt to answer them.

    Example:

        Question mentions "pembrolizumab"

    but:

        "pembrolizumab"

    occurs nowhere in the selected evidence.
    """

    query_terms = (
        extract_query_terms(
            question
        )
    )


    anchor_terms = (
        extract_anchor_terms(
            query_terms
        )
    )


    evidence_text = " ".join(
        item.text
        for item in evidence
    )


    evidence_words = set(
        tokenize_words(
            evidence_text
        )
    )


    matched_terms = [

        term

        for term in query_terms

        if term in evidence_words
    ]


    matched_anchor_terms = [

        term

        for term in anchor_terms

        if term in evidence_words
    ]


    # -----------------------------------------------------
    # Distinctive entity absent
    # -----------------------------------------------------

    if (
        anchor_terms
        and not matched_anchor_terms
    ):

        return SufficiencyCheck(

            sufficient=False,

            reason=(
                "Distinctive query term(s) were absent "
                "from all retrieved evidence: "
                + ", ".join(
                    anchor_terms
                )
            ),

            query_terms=query_terms,

            matched_terms=matched_terms,

            anchor_terms=anchor_terms,

            matched_anchor_terms=(
                matched_anchor_terms
            ),

            llm_sufficient=None,
        )


    # -----------------------------------------------------
    # Zero meaningful lexical overlap
    # -----------------------------------------------------

    if (
        query_terms
        and not matched_terms
    ):

        return SufficiencyCheck(

            sufficient=False,

            reason=(
                "No meaningful query terms occurred "
                "in the retrieved evidence."
            ),

            query_terms=query_terms,

            matched_terms=matched_terms,

            anchor_terms=anchor_terms,

            matched_anchor_terms=(
                matched_anchor_terms
            ),

            llm_sufficient=None,
        )


    return SufficiencyCheck(

        sufficient=True,

        reason=(
            "Lexical evidence gate passed."
        ),

        query_terms=query_terms,

        matched_terms=matched_terms,

        anchor_terms=anchor_terms,

        matched_anchor_terms=(
            matched_anchor_terms
        ),

        llm_sufficient=None,
    )


# =========================================================
# Evidence sufficiency — LLM gate
# =========================================================


SUFFICIENCY_SYSTEM_PROMPT = """
You are an evidence sufficiency classifier.

You are NOT answering the scientific question.

Your only job is to determine whether the supplied evidence
contains enough relevant information to attempt an answer.

Use only the supplied evidence.

Return exactly two lines:

SUFFICIENT: YES

or

SUFFICIENT: NO

and then:

REASON: <brief explanation>

Choose NO if:
- the evidence is about a different subject,
- the key entity or concept is missing,
- only tangential material is present,
- the requested detail cannot be supported.

Do not answer the original question.
""".strip()


def llm_sufficiency_check(
    question: str,
    evidence: list[EvidenceItem],
    llm: OllamaChatClient,
) -> tuple[
    bool,
    str,
]:
    """
    Ask the local LLM whether the retrieved evidence is
    sufficient to attempt the answer.

    Parsing failure is treated as insufficient evidence.
    """

    context = (
        build_evidence_context(
            evidence
        )
    )


    prompt = (
        "QUESTION\n"
        "========\n\n"
        f"{question}\n\n"

        "EVIDENCE\n"
        "========\n\n"
        f"{context}"
    )


    response = (
        llm.chat(
            [
                {
                    "role":
                        "system",

                    "content":
                        SUFFICIENCY_SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ]
        )
    )


    match = re.search(
        r"SUFFICIENT\s*:\s*(YES|NO)",
        response,
        flags=re.IGNORECASE,
    )


    reason_match = re.search(
        r"REASON\s*:\s*(.+)",
        response,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )


    if reason_match:

        reason = (
            reason_match
            .group(1)
            .strip()
        )

    else:

        reason = (
            response.strip()
        )


    if not match:

        return (
            False,
            (
                "Could not reliably parse the evidence "
                "sufficiency classifier."
            ),
        )


    sufficient = (
        match.group(1)
        .upper()
        == "YES"
    )


    return (
        sufficient,
        reason,
    )


# =========================================================
# Main answer prompts
# =========================================================


SYSTEM_PROMPT = """
You are a research assistant answering questions from a
local scientific document library.

Follow these rules strictly:

1. Use only the evidence supplied in the user message.
2. Do not rely on outside knowledge.
3. Do not invent facts, mechanisms, citations, authors,
   titles, page numbers, or source IDs.
4. Every substantive factual claim must have an inline
   source citation such as [S1] or [S2].
5. Multiple sources may be cited as [S1][S3].
6. Cite only source IDs supplied in the evidence.
7. Never use author-year citations such as (Smith, 2024).
8. Never cite filenames directly.
9. If the evidence does not support a claim, do not make it.
10. If the evidence is insufficient, state:
    "The available evidence is insufficient to answer this
    question."

Do not create a bibliography or Sources section.

Write a focused scientific synthesis appropriate for a
researcher.

Unless the user explicitly requests an extensive review:

- answer directly,
- use 2-5 short paragraphs,
- stay below about 500 words,
- prioritize the most relevant evidence,
- do not summarize every retrieved source,
- do not repeat mechanisms or conclusions,
- stop once the question has been answered.
""".strip()


CONCISE_RETRY_SYSTEM_PROMPT = """
You are a research assistant answering questions from a
local scientific document library.

Your previous attempt was too long and was truncated.

Generate a NEW, shorter answer from scratch.

Strict rules:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Every substantive factual claim must have an inline
   citation in the exact form [S1], [S2], etc.
4. Use only source IDs provided in the evidence.
5. Never use author-year citations.
6. Never write filenames or page numbers yourself.
7. Do not include a bibliography or Sources section.
8. Do not discuss every retrieved source.
9. Prioritize only the evidence necessary to answer the
   question.
10. Maximum length: 350 words.
11. Use at most 4 short paragraphs.
12. Finish with a complete sentence and then stop.

If the evidence is insufficient, write exactly:

The available evidence is insufficient to answer this question.

Return only the answer.
""".strip()


def build_user_prompt(
    question: str,
    evidence: list[EvidenceItem],
) -> str:

    return (
        "QUESTION\n"
        "========\n\n"
        f"{question}\n\n"

        "EVIDENCE\n"
        "========\n\n"
        f"{build_evidence_context(evidence)}\n\n"

        "ANSWER INSTRUCTIONS\n"
        "===================\n\n"
        "Answer only from the evidence above. "
        "Every substantive claim must include one or more "
        "citations in the exact form [S1], [S2], etc."
    )


# =========================================================
# Grounded generation with automatic concise retry
# =========================================================


def generate_grounded_answer(
    question: str,
    evidence: list[EvidenceItem],
    llm: OllamaChatClient,
) -> tuple[
    str,
    bool,
    str | None,
    int | None,
]:
    """
    Generate the grounded answer.

    If the first generation ends because Ollama reached the
    output-length limit, retry once using a much stricter
    concise prompt.

    Returns:

        answer
        retry_used
        done_reason
        eval_count
    """

    user_prompt = (
        build_user_prompt(
            question,
            evidence,
        )
    )


    # =====================================================
    # First attempt
    # =====================================================

    first_answer = (
        llm.chat(
            [
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ]
        )
    )


    first_done_reason = (
        llm.last_done_reason
    )


    first_eval_count = (
        llm.last_eval_count
    )


    if generation_finished_cleanly(
        llm,
        first_answer,
    ):

        return (
            first_answer,
            False,
            first_done_reason,
            first_eval_count,
        )


    # =====================================================
    # Retry only when generation hit the length limit
    # =====================================================

    if (
        first_done_reason
        != "length"
    ):

        return (
            first_answer,
            False,
            first_done_reason,
            first_eval_count,
        )


    # =====================================================
    # Concise retry
    # =====================================================

    retry_answer = (
        llm.chat(
            [
                {
                    "role":
                        "system",

                    "content":
                        CONCISE_RETRY_SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ]
        )
    )


    retry_done_reason = (
        llm.last_done_reason
    )


    retry_eval_count = (
        llm.last_eval_count
    )


    return (
        retry_answer,
        True,
        retry_done_reason,
        retry_eval_count,
    )


# =========================================================
# Citation validation
# =========================================================


_SOURCE_CITATION_PATTERN = re.compile(
    r"\[S(\d+)\]"
)


def verify_citations(
    answer: str,
    evidence: list[EvidenceItem],
) -> CitationCheck:

    valid_ids = {
        item.source_id
        for item in evidence
    }


    matches = (
        _SOURCE_CITATION_PATTERN.findall(
            answer
        )
    )


    cited_ids: list[
        str
    ] = []


    for number in matches:

        source_id = (
            f"S{number}"
        )


        if source_id not in cited_ids:

            cited_ids.append(
                source_id
            )


    unknown = [

        source_id

        for source_id in cited_ids

        if source_id not in valid_ids
    ]


    has_citations = bool(
        cited_ids
    )


    valid = (
        has_citations
        and not unknown
    )


    return CitationCheck(

        cited_source_ids=(
            cited_ids
        ),

        unknown_source_ids=(
            unknown
        ),

        has_citations=(
            has_citations
        ),

        valid=(
            valid
        ),
    )


# =========================================================
# Citation repair
# =========================================================


CITATION_REPAIR_SYSTEM_PROMPT = """
Rewrite the supplied draft so that it obeys strict RAG
citation rules.

Rules:

1. Use only the supplied evidence.
2. Preserve only claims supported by that evidence.
3. Every substantive factual claim must contain at least
   one citation in the form [S1], [S2], etc.
4. Use only source IDs that exist in the evidence.
5. Do not use author-year citations.
6. Do not write filenames or page numbers yourself.
7. Do not add outside knowledge.
8. Be concise. Do not exceed 350 words.
9. If the evidence is insufficient, write exactly:

The available evidence is insufficient to answer this question.

Return only the revised answer.
""".strip()


def repair_citations(
    question: str,
    draft_answer: str,
    evidence: list[EvidenceItem],
    llm: OllamaChatClient,
) -> str:

    prompt = (
        "QUESTION\n"
        "========\n\n"
        f"{question}\n\n"

        "EVIDENCE\n"
        "========\n\n"
        f"{build_evidence_context(evidence)}\n\n"

        "DRAFT ANSWER\n"
        "============\n\n"
        f"{draft_answer}"
    )


    return (
        llm.chat(
            [
                {
                    "role":
                        "system",

                    "content":
                        CITATION_REPAIR_SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ]
        )
    )


# =========================================================
# Deterministic citation rendering
# =========================================================


def citation_display(
    item: EvidenceItem,
) -> str:

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


    return (
        f"[{item.filename}, "
        f"{page_label}]"
    )


def render_citations(
    answer: str,
    evidence: list[EvidenceItem],
) -> str:

    evidence_map = {

        item.source_id:
            item

        for item in evidence
    }


    def replace_match(
        match: re.Match,
    ) -> str:

        source_id = (
            f"S{match.group(1)}"
        )


        item = (
            evidence_map.get(
                source_id
            )
        )


        if item is None:

            return (
                match.group(0)
            )


        return (
            citation_display(
                item
            )
        )


    return (
        _SOURCE_CITATION_PATTERN.sub(
            replace_match,
            answer,
        )
    )


# =========================================================
# Abstention helper
# =========================================================


def make_abstention_result(
    question: str,
    evidence: list[EvidenceItem],
    retrieved_count: int,
    sufficiency_check: SufficiencyCheck,
    reason: str,
    generation_done_reason: str | None = None,
    generation_eval_count: int | None = None,
    generation_retried: bool = False,
    raw_answer: str = "",
) -> RAGResult:

    return RAGResult(

        question=question,

        answer=(
            "The available evidence is insufficient "
            "to answer this question."
        ),

        raw_answer=(
            raw_answer
        ),

        evidence=evidence,

        retrieved_count=(
            retrieved_count
        ),

        evidence_tokens=sum(
            item.token_count
            for item in evidence
        ),

        citation_check=(
            CitationCheck(
                cited_source_ids=[],
                unknown_source_ids=[],
                has_citations=False,
                valid=True,
            )
        ),

        sufficiency_check=(
            sufficiency_check
        ),

        abstained=True,

        abstention_reason=(
            reason
        ),

        citation_repaired=False,

        generation_retried=(
            generation_retried
        ),

        generation_done_reason=(
            generation_done_reason
        ),

        generation_eval_count=(
            generation_eval_count
        ),
    )


# =========================================================
# Main RAG pipeline
# =========================================================


def answer_question(
    question: str,
    retrieval_k: int = RAG_RETRIEVAL_K,
    final_k: int = RAG_FINAL_K,
    evidence_budget: int = RAG_EVIDENCE_BUDGET,
) -> RAGResult:

    question = (
        str(
            question
        )
        .strip()
    )


    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    # =====================================================
    # Query embedding
    # =====================================================

    embedder = (
        OllamaEmbedder()
    )


    query_vector = (
        embedder.embed_query(
            question
        )
    )


    # =====================================================
    # Retrieval
    # =====================================================

    retrieved = (
        retrieve(
            question=(
                question
            ),
            query_vector=(
                query_vector
            ),
            top_k=(
                retrieval_k
            ),
        )
    )


    if not retrieved:

        empty_check = (
            SufficiencyCheck(
                sufficient=False,
                reason=(
                    "No evidence was retrieved."
                ),
                query_terms=[],
                matched_terms=[],
                anchor_terms=[],
                matched_anchor_terms=[],
                llm_sufficient=None,
            )
        )


        return (
            make_abstention_result(

                question=question,

                evidence=[],

                retrieved_count=0,

                sufficiency_check=(
                    empty_check
                ),

                reason=(
                    "No evidence was retrieved."
                ),
            )
        )


    # =====================================================
    # Evidence selection
    # =====================================================

    evidence = (
        select_evidence(

            results=(
                retrieved
            ),

            final_k=(
                final_k
            ),

            token_budget=(
                evidence_budget
            ),
        )
    )


    if not evidence:

        empty_check = (
            SufficiencyCheck(
                sufficient=False,
                reason=(
                    "No usable evidence remained."
                ),
                query_terms=[],
                matched_terms=[],
                anchor_terms=[],
                matched_anchor_terms=[],
                llm_sufficient=None,
            )
        )


        return (
            make_abstention_result(

                question=question,

                evidence=[],

                retrieved_count=(
                    len(
                        retrieved
                    )
                ),

                sufficiency_check=(
                    empty_check
                ),

                reason=(
                    "No usable evidence remained "
                    "after filtering."
                ),
            )
        )


    # =====================================================
    # Lexical sufficiency gate
    # =====================================================

    sufficiency = (
        lexical_sufficiency_check(
            question,
            evidence,
        )
    )


    if (
        RAG_SUFFICIENCY_CHECK
        and not sufficiency.sufficient
    ):

        return (
            make_abstention_result(

                question=question,

                evidence=evidence,

                retrieved_count=(
                    len(
                        retrieved
                    )
                ),

                sufficiency_check=(
                    sufficiency
                ),

                reason=(
                    sufficiency.reason
                ),
            )
        )


    # =====================================================
    # Initialize LLM
    # =====================================================

    llm = (
        OllamaChatClient()
    )


    # =====================================================
    # LLM sufficiency gate
    # =====================================================

    if RAG_SUFFICIENCY_CHECK:

        (
            llm_sufficient,
            llm_reason,
        ) = (
            llm_sufficiency_check(

                question=question,

                evidence=evidence,

                llm=llm,
            )
        )


        sufficiency.llm_sufficient = (
            llm_sufficient
        )


        if not llm_sufficient:

            sufficiency.sufficient = False

            sufficiency.reason = (
                llm_reason
            )


            return (
                make_abstention_result(

                    question=question,

                    evidence=evidence,

                    retrieved_count=(
                        len(
                            retrieved
                        )
                    ),

                    sufficiency_check=(
                        sufficiency
                    ),

                    reason=(
                        llm_reason
                    ),

                    generation_done_reason=(
                        llm.last_done_reason
                    ),

                    generation_eval_count=(
                        llm.last_eval_count
                    ),

                    generation_retried=False,
                )
            )


    # =====================================================
    # Generate grounded answer
    #
    # The first attempt uses the normal concise prompt.
    #
    # If that attempt reaches Ollama's length limit,
    # generate_grounded_answer() retries once with a much
    # stricter 350-word prompt.
    # =====================================================

    (
        raw_answer,
        generation_retried,
        generation_done_reason,
        generation_eval_count,
    ) = (
        generate_grounded_answer(

            question=question,

            evidence=evidence,

            llm=llm,
        )
    )


    # =====================================================
    # Generation completeness check
    # =====================================================

    if not generation_finished_cleanly(
        llm,
        raw_answer,
    ):

        return RAGResult(

            question=question,

            answer=(
                "A complete answer could not be "
                "generated within the configured "
                "output limit."
            ),

            raw_answer=(
                raw_answer
            ),

            evidence=evidence,

            retrieved_count=(
                len(
                    retrieved
                )
            ),

            evidence_tokens=sum(
                item.token_count
                for item in evidence
            ),

            citation_check=(
                CitationCheck(
                    cited_source_ids=[],
                    unknown_source_ids=[],
                    has_citations=False,
                    valid=False,
                )
            ),

            sufficiency_check=(
                sufficiency
            ),

            abstained=True,

            abstention_reason=(
                "Generation did not finish cleanly. "
                f"done_reason="
                f"{generation_done_reason!r}"
            ),

            citation_repaired=False,

            generation_retried=(
                generation_retried
            ),

            generation_done_reason=(
                generation_done_reason
            ),

            generation_eval_count=(
                generation_eval_count
            ),
        )


    # =====================================================
    # Citation validation
    # =====================================================

    citation_check = (
        verify_citations(
            raw_answer,
            evidence,
        )
    )


    citation_repaired = False


    # =====================================================
    # Citation repair
    # =====================================================

    if (
        RAG_CITATION_VERIFY
        and not citation_check.valid
        and RAG_CITATION_REPAIR
    ):

        repaired_answer = (
            repair_citations(

                question=question,

                draft_answer=(
                    raw_answer
                ),

                evidence=evidence,

                llm=llm,
            )
        )


        repair_done_reason = (
            llm.last_done_reason
        )


        repair_eval_count = (
            llm.last_eval_count
        )


        # ---------------------------------------------
        # Only accept a completed repair.
        # ---------------------------------------------

        if generation_finished_cleanly(
            llm,
            repaired_answer,
        ):

            repaired_check = (
                verify_citations(
                    repaired_answer,
                    evidence,
                )
            )


            if repaired_check.valid:

                raw_answer = (
                    repaired_answer
                )

                citation_check = (
                    repaired_check
                )

                citation_repaired = (
                    True
                )

                generation_done_reason = (
                    repair_done_reason
                )

                generation_eval_count = (
                    repair_eval_count
                )


    # =====================================================
    # Fail closed if citations remain invalid
    # =====================================================

    if (
        RAG_CITATION_VERIFY
        and not citation_check.valid
    ):

        return RAGResult(

            question=question,

            answer=(
                "A citation-valid answer could not be "
                "generated from the retrieved evidence."
            ),

            raw_answer=(
                raw_answer
            ),

            evidence=evidence,

            retrieved_count=(
                len(
                    retrieved
                )
            ),

            evidence_tokens=sum(
                item.token_count
                for item in evidence
            ),

            citation_check=(
                citation_check
            ),

            sufficiency_check=(
                sufficiency
            ),

            abstained=True,

            abstention_reason=(
                "Citation validation failed."
            ),

            citation_repaired=(
                citation_repaired
            ),

            generation_retried=(
                generation_retried
            ),

            generation_done_reason=(
                generation_done_reason
            ),

            generation_eval_count=(
                generation_eval_count
            ),
        )

    # =====================================================
    # Claim-to-citation support verification
    # =====================================================

    claim_support_check = None

    claim_support_pruning_attempted = False

    claim_support_pruned = False

    claim_support_repair_attempted = False

    claim_support_repaired = False


    if CLAIM_SUPPORT_VERIFY:

        # =================================================
        # First verification of generated answer
        # =================================================

        claim_support_check = (
            verify_claim_support(

                answer=(
                    raw_answer
                ),

                evidence=(
                    evidence
                ),

                llm=(
                    llm
                ),
            )
        )


        # =================================================
        # Stage 1 — deterministic pruning
        #
        # Remove:
        #   PARTIAL claims
        #   UNSUPPORTED claims
        #   uncited prose
        #
        # No LLM generation occurs here.
        # =================================================

        if (
            not claim_support_check.valid
        ):

            claim_support_pruning_attempted = (
                True
            )


            pruned_answer = (
                prune_answer_to_supported_claims(

                    answer=(
                        raw_answer
                    ),

                    support_check=(
                        claim_support_check
                    ),
                )
            )


            # ---------------------------------------------
            # Only evaluate pruning if something usable
            # remains and the answer actually changed.
            # ---------------------------------------------

            if (
                pruned_answer

                and (
                    pruned_answer.strip()
                    != raw_answer.strip()
                )
            ):

                pruned_citation_check = (
                    verify_citations(

                        pruned_answer,

                        evidence,
                    )
                )


                if (
                    pruned_citation_check.valid
                ):

                    pruned_support_check = (
                        verify_claim_support(

                            answer=(
                                pruned_answer
                            ),

                            evidence=(
                                evidence
                            ),

                            llm=(
                                llm
                            ),
                        )
                    )


                    # -------------------------------------
                    # Use the pruned candidate as the new
                    # working answer even if it still needs
                    # LLM repair.
                    #
                    # This means the repair model starts
                    # from a shorter, safer draft.
                    # -------------------------------------

                    raw_answer = (
                        pruned_answer
                    )

                    citation_check = (
                        pruned_citation_check
                    )

                    claim_support_check = (
                        pruned_support_check
                    )


                    if (
                        pruned_support_check.valid
                    ):

                        claim_support_pruned = (
                            True
                        )


        # =================================================
        # Stage 2 — LLM repair fallback
        #
        # Only run if deterministic pruning did not produce
        # a completely supported answer.
        # =================================================

        if (
            claim_support_check is not None

            and not claim_support_check.valid

            and CLAIM_SUPPORT_REPAIR
        ):

            claim_support_repair_attempted = (
                True
            )


            support_repaired_answer = (
                repair_claim_support(

                    question=(
                        question
                    ),

                    draft_answer=(
                        raw_answer
                    ),

                    evidence=(
                        evidence
                    ),

                    support_check=(
                        claim_support_check
                    ),

                    llm=(
                        llm
                    ),
                )
            )


            support_repair_done_reason = (
                llm.last_done_reason
            )


            support_repair_eval_count = (
                llm.last_eval_count
            )


            # ---------------------------------------------
            # Only consider a completed repair.
            # ---------------------------------------------

            if (
                generation_finished_cleanly(
                    llm,
                    support_repaired_answer,
                )
            ):

                support_repair_citations = (
                    verify_citations(

                        support_repaired_answer,

                        evidence,
                    )
                )


                # -----------------------------------------
                # Repaired answer must still obey citation
                # syntax before semantic verification.
                # -----------------------------------------

                if (
                    support_repair_citations.valid
                ):

                    repaired_support_check = (
                        verify_claim_support(

                            answer=(
                                support_repaired_answer
                            ),

                            evidence=(
                                evidence
                            ),

                            llm=(
                                llm
                            ),
                        )
                    )


                    # -------------------------------------
                    # Keep the repaired candidate and its
                    # corresponding diagnostics together.
                    # -------------------------------------

                    raw_answer = (
                        support_repaired_answer
                    )

                    citation_check = (
                        support_repair_citations
                    )

                    claim_support_check = (
                        repaired_support_check
                    )


                    if (
                        repaired_support_check.valid
                    ):

                        claim_support_repaired = (
                            True
                        )


                        generation_done_reason = (
                            support_repair_done_reason
                        )


                        generation_eval_count = (
                            support_repair_eval_count
                        )


    # =====================================================
    # Fail closed if claim support remains invalid
    # =====================================================

    if (
        CLAIM_SUPPORT_VERIFY

        and claim_support_check
        is not None

        and not claim_support_check.valid
    ):

        return RAGResult(

            question=question,

            answer=(
                "The retrieved evidence appears sufficient, "
                "but the generated answer did not pass "
                "claim-to-citation verification."
            ),

            raw_answer=(
                raw_answer
            ),

            evidence=evidence,

            retrieved_count=(
                len(
                    retrieved
                )
            ),

            evidence_tokens=sum(
                item.token_count
                for item in evidence
            ),

            citation_check=(
                citation_check
            ),

            sufficiency_check=(
                sufficiency
            ),

            abstained=True,

            abstention_reason=(
                "Claim-to-citation support "
                "verification failed: "
                f"{claim_support_check.reason}"
            ),

            citation_repaired=(
                citation_repaired
            ),

            generation_retried=(
                generation_retried
            ),

            generation_done_reason=(
                generation_done_reason
            ),

            generation_eval_count=(
                generation_eval_count
            ),

            claim_support_check=(
                claim_support_check
            ),

            claim_support_pruning_attempted=(
                claim_support_pruning_attempted
            ),

            claim_support_pruned=(
                claim_support_pruned
            ),

            claim_support_repair_attempted=(
                claim_support_repair_attempted
            ),

            claim_support_repaired=(
                claim_support_repaired
            ),
        )

    # =====================================================
    # Render verified citations
    # =====================================================

    rendered_answer = (
        render_citations(
            raw_answer,
            evidence,
        )
    )


    # =====================================================
    # Final structural safety check
    # =====================================================

    if has_incomplete_source_citation(
        rendered_answer
    ):

        return RAGResult(

            question=question,

            answer=(
                "A complete citation-safe answer could "
                "not be generated."
            ),

            raw_answer=(
                raw_answer
            ),

            evidence=evidence,

            retrieved_count=(
                len(
                    retrieved
                )
            ),

            evidence_tokens=sum(
                item.token_count
                for item in evidence
            ),

            citation_check=(
                CitationCheck(
                    cited_source_ids=[],
                    unknown_source_ids=[],
                    has_citations=False,
                    valid=False,
                )
            ),

            sufficiency_check=(
                sufficiency
            ),

            abstained=True,

            abstention_reason=(
                "Final answer contained an incomplete "
                "citation marker."
            ),

            citation_repaired=(
                citation_repaired
            ),

            generation_retried=(
                generation_retried
            ),

            generation_done_reason=(
                generation_done_reason
            ),

            generation_eval_count=(
                generation_eval_count
            ),

            claim_support_check=(
                claim_support_check
            ),

            claim_support_pruning_attempted=(
                claim_support_pruning_attempted
            ),

            claim_support_pruned=(
                claim_support_pruned
            ),

            claim_support_repair_attempted=(
                claim_support_repair_attempted
            ),

            claim_support_repaired=(
                claim_support_repaired
            ),
        )


    # =====================================================
    # Success
    # =====================================================

    return RAGResult(

        question=question,

        answer=(
            rendered_answer
        ),

        raw_answer=(
            raw_answer
        ),

        evidence=evidence,

        retrieved_count=(
            len(
                retrieved
            )
        ),

        evidence_tokens=sum(
            item.token_count
            for item in evidence
        ),

        citation_check=(
            citation_check
        ),

        sufficiency_check=(
            sufficiency
        ),

        abstained=False,

        abstention_reason=None,

        citation_repaired=(
            citation_repaired
        ),

        generation_retried=(
            generation_retried
        ),

        generation_done_reason=(
            generation_done_reason
        ),

        generation_eval_count=(
            generation_eval_count
        ),

        claim_support_check=(
            claim_support_check
        ),

        claim_support_pruning_attempted=(
            claim_support_pruning_attempted
        ),

        claim_support_pruned=(
            claim_support_pruned
        ),

        claim_support_repair_attempted=(
            claim_support_repair_attempted
        ),

        claim_support_repaired=(
            claim_support_repaired
        ),
    )