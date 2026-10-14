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


def build_pruned_support_check(
    support_check: ClaimSupportCheck,
) -> ClaimSupportCheck:
    """
    Build a deterministic support result after pruning.

    prune_answer_to_supported_claims() retains only claims
    that were already classified as SUPPORTED.

    Therefore, once citation syntax has been revalidated,
    there is no need to ask the LLM to semantically verify
    those same claims a second time.

    Claim IDs are renumbered to match the pruned answer.
    """

    retained_items: list[
        ClaimSupportItem
    ] = []

    for item in support_check.items:

        if (
            item.verdict
            != "SUPPORTED"
        ):
            continue

        new_claim_id = (
            f"C{len(retained_items) + 1}"
        )

        retained_items.append(
            ClaimSupportItem(
                claim_id=(
                    new_claim_id
                ),
                claim_text=(
                    item.claim_text
                ),
                source_ids=list(
                    item.source_ids
                ),
                verdict=(
                    "SUPPORTED"
                ),
                reason=(
                    "Retained from the original "
                    "claim-support verification after "
                    "deterministic pruning."
                ),
            )
        )

    valid = bool(
        retained_items
    )

    return ClaimSupportCheck(
        valid=valid,
        items=retained_items,
        supported_count=(
            len(
                retained_items
            )
        ),
        partial_count=0,
        unsupported_count=0,
        uncited_claims=[],
        missing_claim_ids=[],
        unknown_claim_ids=[],
        reason=(
            "Deterministic pruning retained only claims "
            "previously verified as SUPPORTED."
            if valid
            else
            "No previously supported claims remained "
            "after deterministic pruning."
        ),
        verifier_done_reason=(
            support_check.verifier_done_reason
        ),
        verifier_eval_count=(
            support_check.verifier_eval_count
        ),
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

Your primary goal is SOURCE FIDELITY, not literal wording
identity.

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
- infer missing comparison groups,
- infer missing mechanisms,
- transfer properties between nearby variables,
- assume a citation is correct merely because it exists,
- reject a claim merely because another source reports a
  different result.


SEMANTIC-EQUIVALENCE RULE:

Scientific claims do NOT need to copy the source
word-for-word.

A conservative paraphrase should be marked SUPPORTED when it
preserves:

- the same subject,
- the same factual relationship,
- the same direction,
- the same comparison,
- the same condition,
- the same quantitative meaning,
- the same level of uncertainty,
- the same analytical interpretation.

For example:

Evidence:

    "The probability of event X increased."

Claim:

    "Event X occurred more often."

This may be SUPPORTED when the context clearly describes the
frequency or occurrence of the same event and the claim does
not invent a new rate, duration, denominator, or numerical
quantity.

Do NOT reject a claim solely because synonyms or ordinary
scientific paraphrases are used.

However, semantic equivalence must not change the underlying
scientific variable.


IMPORTANT CLAIM-LEVEL RULE:

A claim may contain multiple factual clauses.

Every substantive clause must be supported.

If the central statement is supported but an additional
clause, interpretation, causal statement, comparison,
example, implication, generalization, qualification,
condition, mechanism, or substantive detail is not clearly
supported, the verdict must be PARTIAL rather than
SUPPORTED.


ATTRIBUTE AND VALUE PROVENANCE:

Every property, number, direction, duration, frequency,
probability, rate, spacing, count, or other measurement must
remain attached to the exact entity or variable that the
evidence assigns it to.

Nearby facts are not interchangeable.

For example:

Evidence states:

    "Objects within a group are separated by 4 seconds."

and separately:

    "Groups are separated by 100 seconds."

Claim states:

    "Groups have a spacing of 4 seconds."

Verdict:

    PARTIAL or UNSUPPORTED.

The value 4 seconds belongs to a different variable.


Another example:

Evidence states:

    "The mutation increases the frequency of inactive
    periods."

Claim states:

    "The mutation increases the duration of inactive
    periods."

Verdict:

    UNSUPPORTED.

Frequency and duration are different variables.


Another example:

Evidence states:

    "Factor X has a residence time of several minutes."

and:

    "Events occur every few seconds."

Claim states:

    "Factor X has a residence time of a few seconds."

Verdict:

    UNSUPPORTED.

A nearby time value may not be reassigned to another
measurement.


Distinguish carefully between:

- frequency,
- probability,
- rate,
- duration,
- interval,
- spacing,
- count,
- amplitude,
- residence time,
- time between events,
- time between groups,
- within-group spacing.


COMPARATOR AND REFERENCE-GROUP FIDELITY:

The comparison group, reference group, baseline, condition,
and population are substantive parts of the claim.

A claim must preserve them accurately.

Evidence:

    "Group A had a higher value than the reference group."

Claim:

    "Group A had a higher value than Group B."

Verdict:

    PARTIAL

unless Group B is explicitly established as the reference.


Do not assume membership in a reference category merely
because an entity appears nearby in the same passage.


ANALYSIS-CONTEXT FIDELITY:

Preserve distinctions between:

- presence versus absence,
- categorical comparisons,
- continuous relationships,
- correlations,
- associations,
- regressions,
- perturbations,
- temporal relationships,
- predictions,
- mechanistic interpretations,
- direct comparisons,
- causal effects.

A continuous correlation does not by itself establish a
specific categorical comparison.


CONTEXT FIDELITY:

Preserve scientifically meaningful context such as:

- system,
- population,
- experimental condition,
- treatment,
- measurement,
- time scale,
- analysis method,
- model assumptions,
- observational versus experimental evidence,
- computational versus empirical evidence.


UNCERTAINTY FIDELITY:

Preserve qualifications such as:

- may,
- might,
- could,
- suggests,
- associated with,
- correlated with,
- consistent with,
- approximately,
- no significant difference,
- under the tested conditions.

Do not strengthen certainty.


CAUSAL-LANGUAGE FIDELITY:

Words such as:

- causes,
- drives,
- promotes,
- enables,
- supports,
- determines,
- produces,
- leads to,
- results in,
- is responsible for

may imply causal, functional, or mechanistic relationships.

If the evidence reports only an association, correlation,
prediction, regression result, or statistical relationship,
do not treat stronger causal wording as equivalent.


TRANSITIVE-INFERENCE RULE:

Do not construct a new relationship by chaining individually
supported facts.

Evidence:

    A depends on B.

and:

    B interacts with C.

does NOT establish:

    A depends on C.


GENERALIZATION FIDELITY:

Do not generalize a result beyond the systems, populations,
conditions, or examples actually represented by the cited
evidence.

Evidence:

    "System A and System B display X."

Claim:

    "X is a general feature of this entire class of systems."

Verdict:

    PARTIAL or UNSUPPORTED

unless the evidence explicitly supports that broader
generalization.


Terms such as:

- generally,
- universally,
- broadly,
- all,
- characteristic of,
- general feature,

require evidence of the broader scope they assert.


EXPLICIT CROSS-REFERENCE RULE:

A source may establish that two systems share a property
using language such as:

    "as seen with X"
    "similar to X"
    "as observed in X"
    "also observed in X"
    "likewise"

When the source clearly attributes the SAME property to both
systems, that shared property may be treated as explicitly
supported.

Do not extend the cross-reference to unrelated nearby
properties, measurements, mechanisms, or values.


SOURCE-DISAGREEMENT RULE:

Do NOT mark a claim unsupported merely because another
paper, experiment, condition, model, population, or analysis
reports a different result.

Judge each claim against its attached evidence.

Scientific disagreement is allowed.


SAME-SOURCE CONTEXT RULE:

Different analyses within the same source may yield
different findings.

Do not force them to agree.

Preserve the analysis, comparison, condition, and variable
associated with each finding.


VERDICT CONSISTENCY RULE:

Your verdict MUST agree with your own reason.

If your reason identifies ANY substantive unsupported:

- clause,
- value,
- variable,
- comparator,
- mechanism,
- interpretation,
- condition,
- generalization,
- increase in certainty,

you MUST NOT return SUPPORTED.

Use PARTIAL when the central claim remains supported.

Use UNSUPPORTED when the central claim itself is not
established.


VERDICTS:

SUPPORTED

Every substantive part of the claim is directly supported,
is a conservative semantic paraphrase, or is a direct and
unavoidable logical consequence.

The claim preserves:

- meaning,
- subject,
- variable,
- value,
- scope,
- comparison,
- condition,
- analytical context,
- uncertainty,
- direction,
- quantitative meaning,
- strength of evidence.


PARTIAL

The central idea is supported, but one or more substantive
details are not clearly established.


UNSUPPORTED

The cited evidence does not establish the central claim,
addresses a different fact, contradicts the claim, assigns
the relevant value to a different variable, or requires
information absent from the cited evidence.


Be especially strict about:

- quantitative values,
- which variable a value belongs to,
- frequency versus duration,
- probability versus rate,
- within-group versus between-group spacing,
- residence time versus event interval,
- causal relationships,
- mechanistic relationships,
- comparison groups,
- reference groups,
- experimental conditions,
- temporal relationships,
- statistical significance,
- uncertainty,
- categorical versus continuous analyses,
- broad generalizations,
- interpretations beyond the source.


For the reason field:

- discuss the SAME claim,
- discuss ONLY its attached evidence,
- identify the exact unsupported component when returning
  PARTIAL or UNSUPPORTED,
- accept conservative semantic paraphrases,
- do not demand identical wording,
- do not criticize a claim because another source disagrees,
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

The draft contains one or more claims that could not be
retained as fully supported.

Your goal is to produce a shorter, fully supported answer
while preserving as much source-supported meaning as
possible.

Rewrite the answer using ONLY the supplied evidence.

SOURCE FIDELITY IS MORE IMPORTANT THAN COMPLETENESS.


STRICT RULES:

1. Remove every unsupported factual detail.

2. Narrow or weaken claims when the evidence supports only a
   more limited statement.

3. If a sentence contains multiple factual clauses, every
   clause must be supported.

4. Prefer one main factual claim per sentence.

5. EVERY substantive factual sentence MUST contain at least
   one [S#] citation.

6. Put citations in the same sentence as the claim they
   support.

7. Cite only source IDs contained in the supplied evidence.

8. Do not use outside knowledge.


SEMANTIC PARAPHRASING

9. Conservative semantic paraphrasing is allowed.

   Do not require source wording to be copied literally.

   Preserve:

   - subject,
   - relationship,
   - direction,
   - scope,
   - uncertainty,
   - quantitative meaning,
   - analytical context.


10. For example, when the source clearly describes increased
    probability or frequency of the SAME event:

        "the probability of X increased"

    may be repaired as:

        "X occurred more often"

    provided no new rate, duration, denominator, or
    numerical meaning is introduced.


ATTRIBUTE AND VALUE PROVENANCE

11. Keep every property and quantitative value attached to
    the exact variable or entity to which the evidence
    assigns it.

12. Never transfer a value merely because it appears nearby
    in the same evidence passage.

13. Preserve distinctions among:

    - frequency,
    - probability,
    - rate,
    - duration,
    - spacing,
    - interval,
    - count,
    - amplitude,
    - residence time,
    - within-group spacing,
    - between-group timing.


14. If the evidence says:

        X lasts 20 minutes.

    and:

        Y occurs every 4 seconds.

    do not assign 4 seconds to X.


15. If the evidence says a perturbation changes frequency but
    not duration, do not rewrite the result as a duration
    effect.


COMPARISONS

16. Preserve the exact comparison or reference group.

17. Do not introduce, substitute, or infer a comparator.

18. When the evidence does not identify a comparison group,
    baseline, or population, omit it rather than guessing.


ANALYTICAL CONTEXT

19. Preserve the original type of analysis.

Distinguish:

    presence versus absence,
    categorical versus continuous analysis,
    correlation,
    association,
    regression,
    perturbation,
    prediction,
    direct comparison,
    mechanistic interpretation,
    causal effect.


CAUSAL LANGUAGE

20. Do not strengthen cautious statistical language.

Prefer:

    X correlated with Y.
    X was associated with Y.
    X predicted Y.

over:

    X drives Y.
    X promotes Y.
    X supports Y.
    X determines Y.
    X causes Y.

unless the stronger relationship is explicitly supported.


21. Do not create a new dependency, mechanism, causal
    relationship, comparison, or function by combining
    separate facts.


UNCERTAINTY

22. Preserve qualifications such as:

    may,
    might,
    could,
    suggests,
    approximately,
    consistent with,
    no significant difference,
    under the tested conditions.


GENERALIZATION

23. Do not broaden findings from one or a few systems into a
    universal or class-wide conclusion unless explicitly
    supported.

Do not convert:

    "A and B show X"

into:

    "X is a general feature of these systems"

without evidence for that broader statement.


CROSS-REFERENCES

24. Explicit language such as:

    "as seen with X"
    "similar to X"
    "also observed in X"

may establish the SAME referenced property for both cases.

Do not transfer unrelated nearby properties.


SOURCE DISAGREEMENT

25. Do not force different sources, experiments, models,
    conditions, populations, or analyses to agree.

26. Different analyses within one source may legitimately
    differ.


REPAIR STRATEGY

27. When a claim fails verification, prefer deleting only the
    unsupported portion rather than replacing the whole
    sentence.

28. If a narrower supported statement remains, retain it.

29. Prefer a shorter answer containing strong claims over a
    longer answer containing weak claims.

30. Do not write filenames or page numbers.

31. Do not include a bibliography or Sources section.

32. Use approximately 2-4 short paragraphs and normally stay
    below 350 words.


Before returning the answer, inspect every substantive
sentence and ask:

    A. Does it contain a citation?

    B. Does the cited evidence support the ENTIRE sentence?

    C. Does every value belong to the exact variable named?

    D. Did I confuse frequency, probability, duration, rate,
       spacing, interval, or residence time?

    E. Did I invent or substitute a comparison group?

    F. Did I strengthen association/correlation into causal
       or mechanistic language?

    G. Did I combine separate facts into a new relationship?

    H. Did I remove an important uncertainty?

    I. Did I generalize beyond the represented systems or
       population?

    J. Did I force distinct sources or analyses to agree?

    K. Did I overextend an explicit cross-reference?

If A is no, add the correct citation or remove the sentence.

If B through I or K indicate a fidelity problem, narrow the
statement to exactly what the evidence supports or remove the
unsupported portion.

If J is yes, restore the source-specific distinction.

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