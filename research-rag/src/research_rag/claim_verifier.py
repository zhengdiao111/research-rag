from __future__ import annotations

import json
import re

from dataclasses import dataclass

from .config import (
    CLAIM_SUPPORT_MAX_CLAIMS,
    CLAIM_SUPPORT_MAX_OUTPUT_TOKENS,
)

from .llm import (
    OllamaChatClient,
)


# =========================================================
# Verification batching
# =========================================================

# Small batches reduce claim/evidence cross-talk in a
# relatively small local language model.
CLAIM_SUPPORT_BATCH_SIZE = 2


# =========================================================
# Data structures
# =========================================================


@dataclass
class ClaimUnit:
    """
    One citation-bearing claim extracted from the answer.
    """

    claim_id: str

    text: str

    source_ids: list[str]


@dataclass
class ClaimSupportItem:
    """
    Verification result for one claim.
    """

    claim_id: str

    claim_text: str

    source_ids: list[str]

    verdict: str

    reason: str


@dataclass
class ClaimSupportCheck:
    """
    Aggregate claim-to-citation verification result.
    """

    valid: bool

    items: list[ClaimSupportItem]

    supported_count: int

    partial_count: int

    unsupported_count: int

    uncited_claims: list[str]

    missing_claim_ids: list[str]

    unknown_claim_ids: list[str]

    reason: str

    verifier_done_reason: str | None

    verifier_eval_count: int | None


# =========================================================
# Patterns
# =========================================================


_SOURCE_PATTERN = re.compile(
    r"\[S(\d+)\]"
)


_WORD_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_-]*"
)


_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
    r"(?=(?:[*_`\"“‘(\[]*[A-Z0-9]))"
)


# =========================================================
# Citation helpers
# =========================================================


def remove_source_citations(
    text: str,
) -> str:
    """
    Remove internal [S#] citation markers.
    """

    cleaned = (
        _SOURCE_PATTERN.sub(
            "",
            text,
        )
    )


    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )


    return (
        cleaned
        .strip()
        .strip("-•")
        .strip()
    )


def source_ids_from_text(
    text: str,
) -> list[str]:
    """
    Extract source IDs in first-occurrence order.

    Example:

        [S2][S4][S2]

    becomes:

        ["S2", "S4"]
    """

    source_ids: list[str] = []


    for number in (
        _SOURCE_PATTERN.findall(
            text
        )
    ):

        source_id = (
            f"S{number}"
        )


        if (
            source_id
            not in source_ids
        ):

            source_ids.append(
                source_id
            )


    return source_ids


# =========================================================
# Claim extraction
# =========================================================


def is_potentially_substantive(
    text: str,
) -> bool:
    """
    Determine whether uncited prose is substantial enough
    that it should normally have a citation.
    """

    cleaned = (
        remove_source_citations(
            text
        )
    )


    words = (
        _WORD_PATTERN.findall(
            cleaned
        )
    )


    return (
        len(words)
        >= 8
    )


def split_answer_into_segments(
    answer: str,
) -> list[str]:
    """
    Split an answer into sentence-like units while retaining
    inline [S#] citations.
    """

    segments: list[str] = []


    paragraphs = re.split(
        r"\n\s*\n",
        answer,
    )


    for paragraph in paragraphs:

        paragraph = (
            paragraph.strip()
        )


        if not paragraph:

            continue


        lines = [

            line.strip()

            for line in paragraph.splitlines()

            if line.strip()
        ]


        for line in lines:

            line = re.sub(
                r"^\s*[-*•]\s+",
                "",
                line,
            )


            pieces = (
                _SENTENCE_SPLIT_PATTERN.split(
                    line
                )
            )


            for piece in pieces:

                piece = (
                    piece.strip()
                )


                if piece:

                    segments.append(
                        piece
                    )


    return segments


def extract_claim_units(
    answer: str,
) -> tuple[
    list[ClaimUnit],
    list[str],
]:
    """
    Extract citation-bearing claims and substantive uncited
    claims.
    """

    claims: list[
        ClaimUnit
    ] = []


    uncited_claims: list[
        str
    ] = []


    for segment in (
        split_answer_into_segments(
            answer
        )
    ):

        source_ids = (
            source_ids_from_text(
                segment
            )
        )


        claim_text = (
            remove_source_citations(
                segment
            )
        )


        if not claim_text:

            continue


        # -------------------------------------------------
        # Citation-bearing claim
        # -------------------------------------------------

        if source_ids:

            claims.append(
                ClaimUnit(

                    claim_id=(
                        f"C{len(claims) + 1}"
                    ),

                    text=(
                        claim_text
                    ),

                    source_ids=(
                        source_ids
                    ),
                )
            )

            continue


        # -------------------------------------------------
        # Substantive uncited claim
        # -------------------------------------------------

        if (
            is_potentially_substantive(
                segment
            )
        ):

            uncited_claims.append(
                claim_text
            )


    return (
        claims,
        uncited_claims,
    )


# =========================================================
# Deterministic answer pruning
# =========================================================


def prune_answer_to_supported_claims(
    answer: str,
    support_check: ClaimSupportCheck,
) -> str:
    """
    Deterministically remove anything that was not verified
    as SUPPORTED.

    Keeps:
        citation-bearing sentences whose claim verdict was
        SUPPORTED.

    Removes:
        PARTIAL claims
        UNSUPPORTED claims
        substantive uncited sentences
        other uncited prose

    Also performs a very conservative cleanup when pruning
    removes a sentence that previously supplied context for
    the next retained sentence.

    No LLM is involved.
    """

    supported_ids = {

        item.claim_id

        for item in support_check.items

        if (
            item.verdict
            == "SUPPORTED"
        )
    }


    if not supported_ids:

        return ""


    # =====================================================
    # Conservative opening cleanup
    # =====================================================

    def clean_orphaned_opening(
        text: str,
    ) -> str:
        """
        Clean a sentence opening only when earlier material
        from the same paragraph was removed.

        These transformations alter discourse structure,
        not scientific content.
        """

        cleaned = (
            text.strip()
        )


        # -------------------------------------------------
        # Remove transition words whose contrast/addition
        # may have depended on a deleted sentence.
        # -------------------------------------------------

        transition_patterns = [
            r"^Conversely,\s+",
            r"^Additionally,\s+",
            r"^Furthermore,\s+",
            r"^Moreover,\s+",
            r"^However,\s+",
            r"^Therefore,\s+",
            r"^Thus,\s+",
            r"^Consequently,\s+",
            r"^In contrast,\s+",
            r"^By contrast,\s+",
        ]


        for pattern in transition_patterns:

            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )


        # -------------------------------------------------
        # Conservative anaphora cleanup.
        #
        # We intentionally handle only common noun phrases
        # where replacing This/These with The changes the
        # discourse reference but does not introduce a new
        # scientific claim.
        # -------------------------------------------------

        demonstrative_patterns = [

            (
                r"^This low Reynolds number regime\b",
                "The low Reynolds number regime",
            ),

            (
                r"^This high Reynolds number regime\b",
                "The high Reynolds number regime",
            ),

            (
                r"^This approximation\b",
                "The approximation",
            ),

            (
                r"^This approach\b",
                "The approach",
            ),

            (
                r"^This process\b",
                "The process",
            ),

            (
                r"^This framework\b",
                "The framework",
            ),

            (
                r"^This theory\b",
                "The theory",
            ),

            (
                r"^This model\b",
                "The model",
            ),

            (
                r"^This mechanism\b",
                "The mechanism",
            ),

            (
                r"^This analysis\b",
                "The analysis",
            ),

            (
                r"^This result\b",
                "The result",
            ),

            (
                r"^These analyses\b",
                "The analyses",
            ),

            (
                r"^These results\b",
                "The results",
            ),

            (
                r"^These equations\b",
                "The equations",
            ),

            (
                r"^These parameters\b",
                "The parameters",
            ),

            (
                r"^These observations\b",
                "The observations",
            ),

            (
                r"^These deformations\b",
                "The deformations",
            ),
        ]


        for (
            pattern,
            replacement,
        ) in demonstrative_patterns:

            updated = re.sub(
                pattern,
                replacement,
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )


            if updated != cleaned:

                cleaned = updated

                break


        # -------------------------------------------------
        # Restore capitalization if removing a transition
        # exposed a lowercase first character.
        # -------------------------------------------------

        if (
            cleaned
            and cleaned[0].islower()
        ):

            cleaned = (
                cleaned[0].upper()
                + cleaned[1:]
            )


        return cleaned


    # =====================================================
    # Reconstruct answer
    # =====================================================

    kept_paragraphs: list[str] = []


    claim_counter = 0


    paragraphs = re.split(
        r"\n\s*\n",
        answer,
    )


    for paragraph in paragraphs:

        paragraph = (
            paragraph.strip()
        )


        if not paragraph:

            continue


        # -------------------------------------------------
        # Collect the original sentence-like units first.
        # -------------------------------------------------

        original_segments: list[str] = []


        lines = [

            line.strip()

            for line in paragraph.splitlines()

            if line.strip()
        ]


        for line in lines:

            line = re.sub(
                r"^\s*[-*•]\s+",
                "",
                line,
            )


            pieces = (
                _SENTENCE_SPLIT_PATTERN.split(
                    line
                )
            )


            for piece in pieces:

                piece = (
                    piece.strip()
                )


                if piece:

                    original_segments.append(
                        piece
                    )


        # -------------------------------------------------
        # Decide which segments survive verification.
        # -------------------------------------------------

        kept_segments: list[str] = []


        removed_before_first_kept = False

        first_kept_found = False


        for piece in original_segments:

            claim_text = (
                remove_source_citations(
                    piece
                )
            )


            if not claim_text:

                continue


            source_ids = (
                source_ids_from_text(
                    piece
                )
            )


            # ---------------------------------------------
            # Uncited prose is removed.
            # ---------------------------------------------

            if not source_ids:

                if not first_kept_found:

                    removed_before_first_kept = (
                        True
                    )

                continue


            # ---------------------------------------------
            # Citation-bearing sentences correspond to the
            # claim IDs produced by extract_claim_units().
            # ---------------------------------------------

            claim_counter += 1


            claim_id = (
                f"C{claim_counter}"
            )


            if (
                claim_id
                not in supported_ids
            ):

                if not first_kept_found:

                    removed_before_first_kept = (
                        True
                    )

                continue


            # ---------------------------------------------
            # If earlier material in this paragraph was
            # deleted, clean only the first retained
            # sentence's discourse opening.
            # ---------------------------------------------

            if (
                not first_kept_found
                and removed_before_first_kept
            ):

                piece = (
                    clean_orphaned_opening(
                        piece
                    )
                )


            kept_segments.append(
                piece
            )


            first_kept_found = True


        if kept_segments:

            kept_paragraphs.append(
                " ".join(
                    kept_segments
                )
            )


    return (
        "\n\n".join(
            kept_paragraphs
        )
        .strip()
    )


# =========================================================
# Evidence helpers
# =========================================================


def build_evidence_map(
    evidence,
) -> dict:
    """
    Build source-ID -> EvidenceItem mapping.
    """

    return {

        item.source_id:
            item

        for item in evidence
    }


# =========================================================
# Structured-output schema
# =========================================================


def build_claim_support_schema(
    claims: list[ClaimUnit],
) -> dict:
    """
    Require exactly one verifier result for every claim in
    the current batch.
    """

    properties: dict = {}

    required: list[str] = []


    for claim in claims:

        claim_id = (
            claim.claim_id
        )


        properties[
            claim_id
        ] = {

            "type":
                "object",

            "properties": {

                "verdict": {

                    "type":
                        "string",

                    "enum": [
                        "SUPPORTED",
                        "PARTIAL",
                        "UNSUPPORTED",
                    ],
                },

                "reason": {

                    "type":
                        "string",

                    "minLength":
                        1,

                    "maxLength":
                        500,
                },
            },

            "required": [
                "verdict",
                "reason",
            ],

            "additionalProperties":
                False,
        }


        required.append(
            claim_id
        )


    return {

        "type":
            "object",

        "properties":
            properties,

        "required":
            required,

        "additionalProperties":
            False,
    }


# =========================================================
# Verifier system prompt
# =========================================================


CLAIM_SUPPORT_SYSTEM_PROMPT = """
You are a strict scientific claim-to-citation verifier.

You are NOT answering the user's original question.

Evaluate every supplied claim independently.

For each claim, determine whether the ENTIRE claim is
supported by ONLY the evidence explicitly attached to that
claim.

Do NOT:
- skip claims,
- combine claims,
- use evidence attached to another claim,
- use outside knowledge,
- fill gaps using general scientific knowledge,
- assume a citation is correct merely because it exists.

IMPORTANT CLAUSE-LEVEL RULE:

A claim may contain multiple factual clauses.

Every substantive clause must be supported by the cited
evidence.

If the main statement is supported but an additional clause,
interpretation, causal statement, example, implication,
generalization, adjective, or qualification is not clearly
supported, the verdict must be PARTIAL rather than
SUPPORTED.

Examples:

Evidence supports:
    "X occurs."

Claim says:
    "X occurs and causes Y."

Verdict:
    PARTIAL

unless the evidence also supports X causing Y.


Evidence supports:
    "Removing active forcing reduces the equations to
    Newtonian hydrodynamics."

Claim adds:
    "thereby bridging the gap between active and passive
    matter."

Verdict:
    PARTIAL

unless that interpretation is also explicitly supported.


Evidence supports:
    "A parameter can be positive or negative."

Claim adds:
    "therefore allowing complex collective behavior."

Verdict:
    PARTIAL

unless that consequence is also supported.


VERDICTS:

SUPPORTED

Every substantive part of the claim is directly supported
by the cited evidence, is a conservative paraphrase, or is a
direct and unavoidable logical consequence.

PARTIAL

The central idea is supported, but one or more substantive
details, clauses, interpretations, implications, causal
statements, examples, or generalizations are not clearly
established.

UNSUPPORTED

The cited evidence does not establish the central claim,
addresses a different fact, contradicts the claim, or the
claim requires information not contained in the cited
evidence.

Be especially strict about:
- causal relationships,
- quantitative values,
- mathematical relationships,
- signs and directions,
- named mechanisms,
- biological entities,
- topological charge notation,
- conservation laws,
- clinical recommendations,
- claims of experimental validation,
- broad generalizations,
- interpretations extending beyond the source text.

For the reason field:
- discuss the SAME claim being evaluated,
- discuss ONLY its attached evidence,
- identify unsupported clauses when returning PARTIAL or
  UNSUPPORTED,
- keep the explanation concise.

Return a verdict for EVERY required claim ID.
""".strip()


# =========================================================
# Verification prompt construction
# =========================================================


def build_claim_verification_prompt(
    claims: list[ClaimUnit],
    evidence,
) -> str:
    """
    Construct one small verifier batch.

    Each claim receives only the evidence cited by that
    claim.
    """

    evidence_map = (
        build_evidence_map(
            evidence
        )
    )


    blocks: list[str] = []


    for claim in claims:

        lines: list[str] = [

            (
                f"CLAIM "
                f"{claim.claim_id}"
            ),

            "",

            "Claim text:",

            claim.text,

            "",

            (
                "Evidence allowed for "
                f"{claim.claim_id}:"
            ),
        ]


        for source_id in (
            claim.source_ids
        ):

            item = (
                evidence_map.get(
                    source_id
                )
            )


            if item is None:

                lines.extend(
                    [
                        "",
                        f"[{source_id}]",
                        "SOURCE NOT FOUND",
                    ]
                )

                continue


            if (
                item.page_start
                == item.page_end
            ):

                page_text = (
                    str(
                        item.page_start
                    )
                )

            else:

                page_text = (
                    f"{item.page_start}-"
                    f"{item.page_end}"
                )


            lines.extend(
                [
                    "",
                    f"[{source_id}]",
                    (
                        f"File: "
                        f"{item.filename}"
                    ),
                    (
                        f"Page(s): "
                        f"{page_text}"
                    ),
                    (
                        f"Section: "
                        f"{item.section}"
                    ),
                    "",
                    item.text,
                ]
            )


        blocks.append(
            "\n".join(
                lines
            )
        )


    block_separator = (
        "\n\n"
        + "=" * 72
        + "\n\n"
    )


    claim_ids = ", ".join(
        claim.claim_id
        for claim in claims
    )


    return (
        "Evaluate exactly these claim IDs:\n\n"
        f"{claim_ids}\n\n"
        "Evaluate each claim independently.\n\n"
        "For each claim, use ONLY the evidence printed "
        "directly beneath that claim.\n\n"
        "Do not transfer evidence or reasoning from one "
        "claim to another.\n\n"
        + block_separator.join(
            blocks
        )
    )


# =========================================================
# Batch-failure helper
# =========================================================


def build_batch_failure_check(
    *,
    reason: str,
    all_claims: list[ClaimUnit],
    completed_items: list[ClaimSupportItem],
    returned_ids: set[str],
    uncited_claims: list[str],
    unknown_claim_ids: list[str],
    verifier_done_reason: str | None,
    verifier_eval_count: int | None,
) -> ClaimSupportCheck:
    """
    Build a fail-closed verification result if one verifier
    batch fails.
    """

    supported_count = sum(

        item.verdict
        == "SUPPORTED"

        for item in completed_items
    )


    partial_count = sum(

        item.verdict
        == "PARTIAL"

        for item in completed_items
    )


    unsupported_count = sum(

        item.verdict
        == "UNSUPPORTED"

        for item in completed_items
    )


    missing_claim_ids = [

        claim.claim_id

        for claim in all_claims

        if (
            claim.claim_id
            not in returned_ids
        )
    ]


    return ClaimSupportCheck(

        valid=False,

        items=(
            completed_items
        ),

        supported_count=(
            supported_count
        ),

        partial_count=(
            partial_count
        ),

        unsupported_count=(
            unsupported_count
        ),

        uncited_claims=(
            uncited_claims
        ),

        missing_claim_ids=(
            missing_claim_ids
        ),

        unknown_claim_ids=(
            unknown_claim_ids
        ),

        reason=(
            reason
        ),

        verifier_done_reason=(
            verifier_done_reason
        ),

        verifier_eval_count=(
            verifier_eval_count
        ),
    )


# =========================================================
# Main claim verification
# =========================================================


def verify_claim_support(
    answer: str,
    evidence,
    llm: OllamaChatClient,
) -> ClaimSupportCheck:
    """
    Verify every citation-bearing claim against only the
    evidence cited by that claim.

    Claims are processed in small batches.

    Verification fails closed when:
    - claims are missing,
    - unknown IDs are returned,
    - substantive uncited claims remain,
    - PARTIAL claims remain,
    - UNSUPPORTED claims remain,
    - verifier output is malformed or incomplete.
    """

    (
        claims,
        uncited_claims,
    ) = (
        extract_claim_units(
            answer
        )
    )


    # =====================================================
    # No cited claims
    # =====================================================

    if not claims:

        return ClaimSupportCheck(

            valid=False,

            items=[],

            supported_count=0,

            partial_count=0,

            unsupported_count=0,

            uncited_claims=(
                uncited_claims
            ),

            missing_claim_ids=[],

            unknown_claim_ids=[],

            reason=(
                "No citation-bearing claims were "
                "available for support verification."
            ),

            verifier_done_reason=None,

            verifier_eval_count=None,
        )


    # =====================================================
    # Claim-count safety limit
    # =====================================================

    if (
        len(claims)
        > CLAIM_SUPPORT_MAX_CLAIMS
    ):

        return ClaimSupportCheck(

            valid=False,

            items=[],

            supported_count=0,

            partial_count=0,

            unsupported_count=0,

            uncited_claims=(
                uncited_claims
            ),

            missing_claim_ids=[

                claim.claim_id

                for claim in claims
            ],

            unknown_claim_ids=[],

            reason=(
                "The answer contained more claims than "
                "the configured verification limit: "
                f"{len(claims)} > "
                f"{CLAIM_SUPPORT_MAX_CLAIMS}."
            ),

            verifier_done_reason=None,

            verifier_eval_count=None,
        )


    # =====================================================
    # Aggregate batch results
    # =====================================================

    all_items: list[
        ClaimSupportItem
    ] = []


    returned_ids: set[
        str
    ] = set()


    unknown_claim_ids: list[
        str
    ] = []


    total_eval_count = 0

    saw_eval_count = False


    final_done_reason: str | None = None


    # =====================================================
    # Verify small batches
    # =====================================================

    for batch_start in range(
        0,
        len(claims),
        CLAIM_SUPPORT_BATCH_SIZE,
    ):

        batch = claims[
            batch_start:
            batch_start
            + CLAIM_SUPPORT_BATCH_SIZE
        ]


        prompt = (
            build_claim_verification_prompt(
                claims=batch,
                evidence=evidence,
            )
        )


        schema = (
            build_claim_support_schema(
                batch
            )
        )


        raw = (
            llm.chat(

                [
                    {
                        "role":
                            "system",

                        "content":
                            CLAIM_SUPPORT_SYSTEM_PROMPT,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            prompt,
                    },
                ],

                response_format=(
                    schema
                ),

                temperature=0.0,

                max_output_tokens=(
                    CLAIM_SUPPORT_MAX_OUTPUT_TOKENS
                ),
            )
        )


        batch_done_reason = (
            llm.last_done_reason
        )


        batch_eval_count = (
            llm.last_eval_count
        )


        final_done_reason = (
            batch_done_reason
        )


        if (
            batch_eval_count
            is not None
        ):

            total_eval_count += (
                batch_eval_count
            )

            saw_eval_count = True


        # =================================================
        # Batch must finish cleanly
        # =================================================

        if (
            batch_done_reason
            is not None

            and batch_done_reason
            != "stop"
        ):

            return (
                build_batch_failure_check(

                    reason=(
                        "Claim verifier batch did not "
                        "finish cleanly. "
                        f"done_reason="
                        f"{batch_done_reason!r}"
                    ),

                    all_claims=claims,

                    completed_items=(
                        all_items
                    ),

                    returned_ids=(
                        returned_ids
                    ),

                    uncited_claims=(
                        uncited_claims
                    ),

                    unknown_claim_ids=(
                        unknown_claim_ids
                    ),

                    verifier_done_reason=(
                        batch_done_reason
                    ),

                    verifier_eval_count=(
                        total_eval_count
                        if saw_eval_count
                        else None
                    ),
                )
            )


        # =================================================
        # Parse JSON
        # =================================================

        try:

            data = (
                json.loads(
                    raw
                )
            )


        except json.JSONDecodeError:

            return (
                build_batch_failure_check(

                    reason=(
                        "Claim verifier returned "
                        "invalid JSON."
                    ),

                    all_claims=claims,

                    completed_items=(
                        all_items
                    ),

                    returned_ids=(
                        returned_ids
                    ),

                    uncited_claims=(
                        uncited_claims
                    ),

                    unknown_claim_ids=(
                        unknown_claim_ids
                    ),

                    verifier_done_reason=(
                        batch_done_reason
                    ),

                    verifier_eval_count=(
                        total_eval_count
                        if saw_eval_count
                        else None
                    ),
                )
            )


        if not isinstance(
            data,
            dict,
        ):

            return (
                build_batch_failure_check(

                    reason=(
                        "Claim verifier JSON was not "
                        "an object."
                    ),

                    all_claims=claims,

                    completed_items=(
                        all_items
                    ),

                    returned_ids=(
                        returned_ids
                    ),

                    uncited_claims=(
                        uncited_claims
                    ),

                    unknown_claim_ids=(
                        unknown_claim_ids
                    ),

                    verifier_done_reason=(
                        batch_done_reason
                    ),

                    verifier_eval_count=(
                        total_eval_count
                        if saw_eval_count
                        else None
                    ),
                )
            )


        batch_map = {

            claim.claim_id:
                claim

            for claim in batch
        }


        batch_ids = set(
            batch_map
        )


        # =================================================
        # Detect unexpected IDs
        # =================================================

        for returned_id in (
            data.keys()
        ):

            returned_id = (
                str(
                    returned_id
                )
            )


            if (
                returned_id
                not in batch_ids

                and returned_id
                not in unknown_claim_ids
            ):

                unknown_claim_ids.append(
                    returned_id
                )


        # =================================================
        # Parse expected results
        # =================================================

        for claim in batch:

            claim_id = (
                claim.claim_id
            )


            entry = (
                data.get(
                    claim_id
                )
            )


            if not isinstance(
                entry,
                dict,
            ):

                continue


            verdict = (
                str(
                    entry.get(
                        "verdict",
                        "",
                    )
                )
                .strip()
                .upper()
            )


            reason = (
                str(
                    entry.get(
                        "reason",
                        "",
                    )
                )
                .strip()
            )


            if verdict not in {
                "SUPPORTED",
                "PARTIAL",
                "UNSUPPORTED",
            }:

                verdict = (
                    "UNSUPPORTED"
                )


                reason = (
                    "Verifier returned an invalid verdict. "
                    + reason
                ).strip()


            if not reason:

                reason = (
                    "Verifier did not provide an "
                    "explanation."
                )


            returned_ids.add(
                claim_id
            )


            all_items.append(
                ClaimSupportItem(

                    claim_id=(
                        claim_id
                    ),

                    claim_text=(
                        claim.text
                    ),

                    source_ids=(
                        claim.source_ids
                    ),

                    verdict=(
                        verdict
                    ),

                    reason=(
                        reason
                    ),
                )
            )


    # =====================================================
    # Final completeness
    # =====================================================

    missing_claim_ids = [

        claim.claim_id

        for claim in claims

        if (
            claim.claim_id
            not in returned_ids
        )
    ]


    supported_count = sum(

        item.verdict
        == "SUPPORTED"

        for item in all_items
    )


    partial_count = sum(

        item.verdict
        == "PARTIAL"

        for item in all_items
    )


    unsupported_count = sum(

        item.verdict
        == "UNSUPPORTED"

        for item in all_items
    )


    valid = (

        not missing_claim_ids

        and not unknown_claim_ids

        and partial_count
        == 0

        and unsupported_count
        == 0

        and not uncited_claims

        and supported_count
        == len(claims)
    )


    # =====================================================
    # Human-readable reason
    # =====================================================

    if valid:

        reason = (
            "All citation-bearing claims were supported "
            "by their cited evidence."
        )


    else:

        reason_parts: list[str] = []


        if partial_count:

            reason_parts.append(
                f"{partial_count} partially supported"
            )


        if unsupported_count:

            reason_parts.append(
                f"{unsupported_count} unsupported"
            )


        if uncited_claims:

            reason_parts.append(
                f"{len(uncited_claims)} uncited "
                "substantive claim(s)"
            )


        if missing_claim_ids:

            reason_parts.append(
                f"{len(missing_claim_ids)} verifier "
                "result(s) missing"
            )


        if unknown_claim_ids:

            reason_parts.append(
                f"{len(unknown_claim_ids)} unknown "
                "claim ID(s)"
            )


        if not reason_parts:

            reason_parts.append(
                "not every extracted claim was "
                "confirmed supported"
            )


        reason = (
            "Claim-support verification failed: "
            + ", ".join(
                reason_parts
            )
            + "."
        )


    return ClaimSupportCheck(

        valid=(
            valid
        ),

        items=(
            all_items
        ),

        supported_count=(
            supported_count
        ),

        partial_count=(
            partial_count
        ),

        unsupported_count=(
            unsupported_count
        ),

        uncited_claims=(
            uncited_claims
        ),

        missing_claim_ids=(
            missing_claim_ids
        ),

        unknown_claim_ids=(
            unknown_claim_ids
        ),

        reason=(
            reason
        ),

        verifier_done_reason=(
            final_done_reason
        ),

        verifier_eval_count=(
            total_eval_count
            if saw_eval_count
            else None
        ),
    )


# =========================================================
# LLM repair prompt
# =========================================================


CLAIM_SUPPORT_REPAIR_SYSTEM_PROMPT = """
You are repairing a scientific retrieval-augmented
generation answer.

The draft still contains one or more claims that could not
be deterministically retained as fully supported.

Rewrite the answer using ONLY the supplied evidence.

STRICT RULES:

1. Remove every unsupported factual detail.

2. Narrow or weaken claims when the evidence supports only
   a more limited statement.

3. If a sentence contains multiple factual clauses, every
   clause must be supported.

4. Prefer one main factual claim per sentence.

5. EVERY substantive factual sentence MUST contain at least
   one [S#] citation.

6. Put citations in the same sentence as the claim they
   support.

Example:

    The continuity equation conserves animal number [S3].

7. Use citations only in the exact forms:

    [S1]
    [S2]
    [S1][S3]

8. Cite only source IDs contained in the supplied evidence.

9. Do not use outside knowledge.

10. Do not invent mechanisms, interpretations, causal
    relationships, implications, or examples.

11. Correct malformed quantities, mathematical notation,
    signs, topological charges, biological names, or other
    terminology only when the supplied evidence clearly
    supports the correction.

12. Do not write filenames or page numbers.

13. Do not include a bibliography, references section, or
    Sources section.

14. Prefer a shorter answer containing strongly supported
    claims over a longer answer with weak claims.

15. Use approximately 2-4 short paragraphs and normally
    remain under 350 words.

Before returning the answer, inspect every substantive
sentence:

    Does it contain at least one [S#] citation?

If not, either add the correct citation or remove the
sentence.

Return only the revised answer.
""".strip()


# =========================================================
# Repair feedback
# =========================================================


def build_repair_feedback(
    check: ClaimSupportCheck,
) -> str:
    """
    Convert verifier failures to repair feedback.
    """

    lines: list[str] = []


    for item in (
        check.items
    ):

        if (
            item.verdict
            == "SUPPORTED"
        ):

            continue


        lines.extend(
            [
                (
                    f"{item.claim_id}: "
                    f"{item.verdict}"
                ),

                (
                    f"Claim: "
                    f"{item.claim_text}"
                ),

                (
                    f"Cited sources: "
                    f"{', '.join(item.source_ids)}"
                ),

                (
                    f"Reason: "
                    f"{item.reason}"
                ),

                "",
            ]
        )


    for index, claim in enumerate(
        check.uncited_claims,
        start=1,
    ):

        lines.extend(
            [
                (
                    f"UNCITED-{index}: "
                    "substantive sentence has no citation"
                ),

                (
                    f"Claim: "
                    f"{claim}"
                ),

                "",
            ]
        )


    if check.missing_claim_ids:

        lines.extend(
            [
                (
                    "MISSING VERIFIER RESULTS: "
                    + ", ".join(
                        check.missing_claim_ids
                    )
                ),

                "",
            ]
        )


    if check.unknown_claim_ids:

        lines.extend(
            [
                (
                    "UNKNOWN VERIFIER IDS: "
                    + ", ".join(
                        check.unknown_claim_ids
                    )
                ),

                "",
            ]
        )


    if not lines:

        lines.append(
            check.reason
        )


    return (
        "\n".join(
            lines
        )
        .strip()
    )


# =========================================================
# Evidence context for LLM repair
# =========================================================


def build_full_evidence_context(
    evidence,
) -> str:
    """
    Render all selected evidence using internal source IDs.
    """

    blocks: list[str] = []


    for item in evidence:

        if (
            item.page_start
            == item.page_end
        ):

            pages = (
                str(
                    item.page_start
                )
            )

        else:

            pages = (
                f"{item.page_start}-"
                f"{item.page_end}"
            )


        blocks.append(
            (
                f"[{item.source_id}]\n"
                f"File: {item.filename}\n"
                f"Page(s): {pages}\n"
                f"Section: {item.section}\n\n"
                f"{item.text}"
            )
        )


    block_separator = (
        "\n\n"
        + "=" * 72
        + "\n\n"
    )


    return (
        block_separator.join(
            blocks
        )
    )


# =========================================================
# LLM repair
# =========================================================


def repair_claim_support(
    question: str,
    draft_answer: str,
    evidence,
    support_check: ClaimSupportCheck,
    llm: OllamaChatClient,
) -> str:
    """
    LLM fallback repair.

    This should be used only after deterministic pruning
    fails to produce an acceptable answer.
    """

    prompt = (
        "QUESTION\n"
        "========\n\n"
        f"{question}\n\n"

        "DRAFT ANSWER\n"
        "============\n\n"
        f"{draft_answer}\n\n"

        "CLAIM-SUPPORT PROBLEMS\n"
        "======================\n\n"
        f"{build_repair_feedback(support_check)}\n\n"

        "AVAILABLE EVIDENCE\n"
        "==================\n\n"
        f"{build_full_evidence_context(evidence)}"
    )


    repaired_answer = (
        llm.chat(

            [
                {
                    "role":
                        "system",

                    "content":
                        CLAIM_SUPPORT_REPAIR_SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            temperature=0.1,
        )
    )


    return (
        repaired_answer
        .strip()
    )