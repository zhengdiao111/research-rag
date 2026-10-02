from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass, replace
from typing import Iterable

import tiktoken

from .config import (
    ALLOW_CROSS_PAGE_CHUNKS,
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    TOKENIZER_ENCODING,
)

from .parser import ParsedDocument


# =========================================================
# Tokenizer
# =========================================================


_ENCODER = tiktoken.get_encoding(
    TOKENIZER_ENCODING
)


def count_tokens(
    text: str,
) -> int:
    """
    Count approximate tokens used for chunk sizing.

    This tokenizer is used only to keep chunk sizes
    consistent. The eventual embedding model may use
    a different tokenizer internally.
    """

    if not text:
        return 0

    return len(
        _ENCODER.encode(
            text,
            disallowed_special=(),
        )
    )


def decode_tokens(
    tokens: list[int],
) -> str:
    """
    Decode tokenizer IDs back into text.
    """

    return _ENCODER.decode(
        tokens
    )


# =========================================================
# Internal data structures
# =========================================================


@dataclass
class TextBlock:
    """
    Intermediate paragraph / heading representation.
    """

    text: str

    page_start: int

    page_end: int

    section: str

    chunk_type: str

    is_heading: bool = False


@dataclass
class Segment:
    """
    A paragraph-sized piece ready for chunk assembly.
    """

    text: str

    page_start: int

    page_end: int

    section: str

    chunk_type: str


@dataclass
class ResearchChunk:
    """
    Final research-RAG chunk.

    Attributes
    ----------
    chunk_id:
        Stable deterministic ID for this particular chunk.

    document_id:
        Stable identifier derived from the source PDF hash.

    filename:
        Original PDF filename.

    page_start / page_end:
        Physical PDF page range represented by the chunk.

    section:
        Current article section or heading.

    chunk_type:
        Semantic category such as body, methods,
        references, caption, metadata, etc.

    text:
        Text that will eventually be embedded.

    token_count:
        Approximate token count used by the chunker.

    source_sha256:
        Full source PDF hash.
    """

    chunk_id: str

    document_id: str

    filename: str

    page_start: int

    page_end: int

    section: str

    chunk_type: str

    text: str

    token_count: int

    source_sha256: str


# =========================================================
# Markdown helpers
# =========================================================


HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})\s+(.+?)\s*$"
)


def clean_heading(
    text: str,
) -> str:
    """
    Remove common Markdown formatting from a heading.
    """

    text = text.strip()

    text = re.sub(
        r"^#{1,6}\s*",
        "",
        text,
    )

    text = re.sub(
        r"</?[^>]+>",
        "",
        text,
    )

    text = text.replace(
        "**",
        "",
    )

    text = text.replace(
        "__",
        "",
    )

    text = text.replace(
        "*",
        "",
    )

    text = text.replace(
        "_",
        "",
    )

    return text.strip()


# =========================================================
# Basic text normalization
# =========================================================


def normalize_text(
    text: str,
) -> str:
    """
    Perform conservative cleanup suitable for scientific text.

    Deliberately avoids spelling correction because automatic
    correction could damage:

    - gene names
    - protein names
    - abbreviations
    - chemical names
    - equations
    - identifiers
    """

    if not text:
        return ""

    # Unicode replacement character from imperfect PDF text.
    text = text.replace(
        "\ufffd",
        "",
    )

    # Soft hyphen.
    text = text.replace(
        "\u00ad",
        "",
    )

    # Non-breaking space.
    text = text.replace(
        "\u00a0",
        " ",
    )

    # Normalize Windows line endings.
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Collapse repeated spaces and tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Avoid excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# Section classification
# =========================================================


def classify_heading(
    heading: str,
) -> str:
    """
    Classify common scientific-paper section headings.
    """

    normalized = heading.lower().strip()

    normalized = re.sub(
        r"[^a-z0-9\s&/-]",
        "",
        normalized,
    )

    # References
    if any(
        phrase in normalized
        for phrase in (
            "references",
            "bibliography",
            "literature cited",
        )
    ):
        return "references"

    # Acknowledgments
    if any(
        phrase in normalized
        for phrase in (
            "acknowledgment",
            "acknowledgement",
            "acknowledgments",
            "acknowledgements",
        )
    ):
        return "acknowledgments"

    # Abstract / summary
    if normalized in {
        "abstract",
        "summary",
    }:
        return "abstract"

    # Introduction
    if any(
        phrase in normalized
        for phrase in (
            "introduction",
            "background",
        )
    ):
        return "introduction"

    # Methods
    if any(
        phrase in normalized
        for phrase in (
            "materials and methods",
            "methods",
            "methodology",
            "experimental procedures",
            "experimental methods",
        )
    ):
        return "methods"

    # Results
    if normalized.startswith(
        "result"
    ):
        return "results"

    # Discussion
    if normalized.startswith(
        "discussion"
    ):
        return "discussion"

    # Conclusions
    if any(
        phrase in normalized
        for phrase in (
            "conclusion",
            "conclusions",
            "concluding remarks",
        )
    ):
        return "conclusion"

    # Supplementary material
    if any(
        phrase in normalized
        for phrase in (
            "supplementary",
            "supporting information",
        )
    ):
        return "supplementary"

    return "body"


# =========================================================
# Metadata detection
# =========================================================


_METADATA_SIGNALS = (
    "view the article online",
    "permissions",
    "terms of service",
    "issn",
    "copyright ©",
    "copyright (c)",
    "all rights reserved",
    "reprints and permissions",
    "downloaded from http",
)


def looks_like_metadata(
    text: str,
) -> bool:
    """
    Identify obvious publisher boilerplate.
    """

    lowered = text.lower()

    return any(
        signal in lowered
        for signal in _METADATA_SIGNALS
    )


def page_is_mostly_metadata(
    paragraphs: list[str],
) -> bool:
    """
    Detect publisher-only pages such as the final Science
    permissions / DOI page.

    Requires multiple metadata signals so normal article pages
    are not accidentally removed.
    """

    if not paragraphs:
        return False

    hits = sum(
        1
        for text in paragraphs
        if looks_like_metadata(text)
    )

    return (
        hits >= 2
        and hits >= max(
            2,
            len(paragraphs) // 2,
        )
    )


# =========================================================
# Page parsing
# =========================================================


def split_markdown_units(
    text: str,
) -> list[tuple[str, str]]:
    """
    Split page Markdown into heading and paragraph units.

    Returns
    -------
    list of:
        ("heading", text)
        ("paragraph", text)
    """

    text = normalize_text(
        text
    )

    if not text:
        return []

    units: list[
        tuple[str, str]
    ] = []

    paragraph_lines: list[str] = []


    def flush_paragraph() -> None:

        if not paragraph_lines:
            return

        paragraph = " ".join(
            line.strip()
            for line in paragraph_lines
            if line.strip()
        )

        paragraph = normalize_text(
            paragraph
        )

        if paragraph:

            units.append(
                (
                    "paragraph",
                    paragraph,
                )
            )

        paragraph_lines.clear()


    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:

            flush_paragraph()

            continue


        match = HEADING_PATTERN.match(
            stripped
        )

        if match:

            flush_paragraph()

            heading = clean_heading(
                stripped
            )

            if heading:

                units.append(
                    (
                        "heading",
                        heading,
                    )
                )

            continue


        paragraph_lines.append(
            stripped
        )


    flush_paragraph()

    return units


# =========================================================
# Document -> blocks
# =========================================================


def document_to_blocks(
    document: ParsedDocument,
) -> list[TextBlock]:
    """
    Convert parsed pages into semantic text blocks.
    """

    blocks: list[TextBlock] = []

    current_section = (
        document.title
        or "Main text"
    )

    current_type = "body"


    for page in document.pages:

        units = split_markdown_units(
            page.text
        )


        paragraphs = [
            text
            for kind, text in units
            if kind == "paragraph"
        ]


        mostly_metadata = (
            page_is_mostly_metadata(
                paragraphs
            )
        )


        for kind, unit_text in units:

            if kind == "heading":

                current_section = (
                    clean_heading(
                        unit_text
                    )
                    or current_section
                )

                current_type = (
                    classify_heading(
                        current_section
                    )
                )


                blocks.append(
                    TextBlock(
                        text=current_section,

                        page_start=(
                            page.page_number
                        ),

                        page_end=(
                            page.page_number
                        ),

                        section=(
                            current_section
                        ),

                        chunk_type=(
                            current_type
                        ),

                        is_heading=True,
                    )
                )

                continue


            block_type = (
                current_type
            )


            if mostly_metadata:

                block_type = "metadata"

            elif looks_like_metadata(
                unit_text
            ):

                block_type = "metadata"


            blocks.append(
                TextBlock(
                    text=unit_text,

                    page_start=(
                        page.page_number
                    ),

                    page_end=(
                        page.page_number
                    ),

                    section=(
                        current_section
                    ),

                    chunk_type=(
                        block_type
                    ),

                    is_heading=False,
                )
            )


        # ---------------------------------------------
        # Captions preserved separately by parser.py
        # ---------------------------------------------

        for caption in page.captions:

            caption = normalize_text(
                caption
            )

            if not caption:
                continue

            blocks.append(
                TextBlock(
                    text=caption,

                    page_start=(
                        page.page_number
                    ),

                    page_end=(
                        page.page_number
                    ),

                    section=(
                        "Figure / Table Caption"
                    ),

                    chunk_type="caption",

                    is_heading=False,
                )
            )


    return blocks


# =========================================================
# Sidebar / interrupted-word repair
# =========================================================


def starts_with_lowercase(
    text: str,
) -> bool:
    """
    Return True when the first alphabetic character is lowercase.
    """

    for char in text:

        if char.isalpha():

            return char.islower()

    return False


def repair_interrupted_layout(
    blocks: list[TextBlock],
) -> list[TextBlock]:
    """
    Repair one common multi-column PDF extraction artifact.

    Example layout:

        ... observed phenom-

        ## Defects program tissue shapes

        [short sidebar paragraph]

        enon is driven by mechanical forces.

    The heading/sidebar is preserved as a separate ``sidebar``
    block while:

        phenom- + enon

    becomes:

        phenomenon

    This repair is intentionally conservative. It requires:

    - an unfinished hyphenated word
    - immediately followed by a heading
    - followed by a short paragraph
    - followed by lowercase continuation text
    - all on the same PDF page
    """

    blocks = list(
        blocks
    )

    i = 0


    while i <= len(blocks) - 4:

        first = blocks[i]

        heading = blocks[i + 1]

        sidebar_body = blocks[i + 2]

        continuation = blocks[i + 3]


        same_page = (
            first.page_end
            == heading.page_start
            == sidebar_body.page_start
            == continuation.page_start
        )


        pattern_matches = (

            not first.is_heading

            and first.text.rstrip().endswith(
                "-"
            )

            and heading.is_heading

            and not sidebar_body.is_heading

            and not continuation.is_heading

            and same_page

            and starts_with_lowercase(
                continuation.text
            )

            and count_tokens(
                sidebar_body.text
            ) <= 250
        )


        if not pattern_matches:

            i += 1

            continue


        original_section = (
            first.section
        )

        original_type = (
            first.chunk_type
        )

        sidebar_section = (
            heading.section
        )


        # ---------------------------------------------
        # Join interrupted word
        # ---------------------------------------------

        left = first.text.rstrip()

        right = continuation.text.lstrip()


        joined_text = (
            left[:-1]
            + right
        )


        merged = replace(
            first,

            text=joined_text,

            page_end=(
                continuation.page_end
            ),
        )


        # ---------------------------------------------
        # Preserve sidebar as its own retrieval object
        # ---------------------------------------------

        sidebar = TextBlock(

            text=(
                f"{heading.text}\n\n"
                f"{sidebar_body.text}"
            ),

            page_start=(
                heading.page_start
            ),

            page_end=(
                sidebar_body.page_end
            ),

            section=(
                sidebar_section
            ),

            chunk_type="sidebar",

            is_heading=False,
        )


        # ---------------------------------------------
        # The false sidebar heading may have changed the
        # section assigned to following article paragraphs.
        #
        # Restore their previous section until the next
        # real heading.
        # ---------------------------------------------

        j = i + 4

        while j < len(blocks):

            block = blocks[j]

            if block.is_heading:
                break

            if (
                block.page_start
                != first.page_start
            ):
                break

            if (
                block.section
                == sidebar_section
            ):

                blocks[j] = replace(
                    block,

                    section=(
                        original_section
                    ),

                    chunk_type=(
                        original_type
                    ),
                )

            j += 1


        blocks[
            i:i + 4
        ] = [
            merged,
            sidebar,
        ]


        i += 2


    return blocks


# =========================================================
# Remove heading markers
# =========================================================


def remove_heading_blocks(
    blocks: Iterable[TextBlock],
) -> list[TextBlock]:
    """
    Headings are stored as metadata on their paragraphs rather
    than embedded as standalone chunks.
    """

    return [
        block
        for block in blocks
        if not block.is_heading
    ]


# =========================================================
# Long paragraph splitting
# =========================================================


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9(\[])"
)


def split_long_block(
    block: TextBlock,
) -> list[Segment]:
    """
    Split one very long paragraph.

    Sentence boundaries are preferred.

    Token-level slicing is used only as the final fallback.
    """

    text = block.text.strip()

    token_count = count_tokens(
        text
    )


    if token_count <= CHUNK_MAX:

        return [
            Segment(
                text=text,

                page_start=(
                    block.page_start
                ),

                page_end=(
                    block.page_end
                ),

                section=(
                    block.section
                ),

                chunk_type=(
                    block.chunk_type
                ),
            )
        ]


    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(
            text
        )
        if sentence.strip()
    ]


    # If sentence parsing did not help, fall back to
    # token slicing.

    if len(sentences) <= 1:

        tokens = _ENCODER.encode(
            text,
            disallowed_special=(),
        )

        pieces: list[
            Segment
        ] = []


        start = 0


        while start < len(tokens):

            stop = min(
                start + CHUNK_TARGET,
                len(tokens),
            )

            piece = decode_tokens(
                tokens[start:stop]
            ).strip()


            if piece:

                pieces.append(
                    Segment(
                        text=piece,

                        page_start=(
                            block.page_start
                        ),

                        page_end=(
                            block.page_end
                        ),

                        section=(
                            block.section
                        ),

                        chunk_type=(
                            block.chunk_type
                        ),
                    )
                )


            if stop >= len(tokens):
                break


            start = max(
                stop - CHUNK_OVERLAP,
                start + 1,
            )


        return pieces


    pieces: list[Segment] = []

    current: list[str] = []

    current_tokens = 0


    for sentence in sentences:

        sentence_tokens = (
            count_tokens(
                sentence
            )
        )


        if (
            current
            and (
                current_tokens
                + sentence_tokens
                > CHUNK_TARGET
            )
        ):

            piece = " ".join(
                current
            ).strip()


            pieces.append(
                Segment(
                    text=piece,

                    page_start=(
                        block.page_start
                    ),

                    page_end=(
                        block.page_end
                    ),

                    section=(
                        block.section
                    ),

                    chunk_type=(
                        block.chunk_type
                    ),
                )
            )


            current = []

            current_tokens = 0


        # A single enormous sentence still needs token slicing.

        if (
            not current
            and sentence_tokens
            > CHUNK_MAX
        ):

            temporary = TextBlock(
                text=sentence,

                page_start=(
                    block.page_start
                ),

                page_end=(
                    block.page_end
                ),

                section=(
                    block.section
                ),

                chunk_type=(
                    block.chunk_type
                ),
            )


            sentence_token_ids = (
                _ENCODER.encode(
                    temporary.text,
                    disallowed_special=(),
                )
            )


            start = 0


            while start < len(
                sentence_token_ids
            ):

                stop = min(
                    start + CHUNK_TARGET,
                    len(sentence_token_ids),
                )


                piece = decode_tokens(
                    sentence_token_ids[
                        start:stop
                    ]
                ).strip()


                if piece:

                    pieces.append(
                        Segment(
                            text=piece,

                            page_start=(
                                block.page_start
                            ),

                            page_end=(
                                block.page_end
                            ),

                            section=(
                                block.section
                            ),

                            chunk_type=(
                                block.chunk_type
                            ),
                        )
                    )


                if stop >= len(
                    sentence_token_ids
                ):
                    break


                start = max(
                    stop - CHUNK_OVERLAP,
                    start + 1,
                )


            continue


        current.append(
            sentence
        )

        current_tokens += (
            sentence_tokens
        )


    if current:

        piece = " ".join(
            current
        ).strip()


        pieces.append(
            Segment(
                text=piece,

                page_start=(
                    block.page_start
                ),

                page_end=(
                    block.page_end
                ),

                section=(
                    block.section
                ),

                chunk_type=(
                    block.chunk_type
                ),
            )
        )


    return pieces


# =========================================================
# Blocks -> segments
# =========================================================


def blocks_to_segments(
    blocks: list[TextBlock],
) -> list[Segment]:
    """
    Convert semantic blocks into chunk-sized segments.
    """

    segments: list[
        Segment
    ] = []


    for block in blocks:

        if not block.text.strip():
            continue

        segments.extend(
            split_long_block(
                block
            )
        )


    return segments


# =========================================================
# Overlap helper
# =========================================================


def overlap_tail(
    segments: list[Segment],
    token_budget: int,
) -> list[Segment]:
    """
    Preserve approximately ``token_budget`` tokens from the end
    of the previous chunk.

    Prefers whole paragraphs. If the final paragraph alone is
    too large, only its token tail is used.
    """

    if (
        not segments
        or token_budget <= 0
    ):
        return []


    selected: list[
        Segment
    ] = []

    total = 0


    for segment in reversed(
        segments
    ):

        tokens = count_tokens(
            segment.text
        )


        if (
            total + tokens
            <= token_budget
        ):

            selected.append(
                segment
            )

            total += tokens

            continue


        remaining = (
            token_budget - total
        )


        if remaining > 0:

            encoded = (
                _ENCODER.encode(
                    segment.text,
                    disallowed_special=(),
                )
            )


            tail_text = decode_tokens(
                encoded[-remaining:]
            ).strip()


            if tail_text:

                selected.append(
                    Segment(
                        text=tail_text,

                        page_start=(
                            segment.page_start
                        ),

                        page_end=(
                            segment.page_end
                        ),

                        section=(
                            segment.section
                        ),

                        chunk_type=(
                            segment.chunk_type
                        ),
                    )
                )


        break


    selected.reverse()

    return selected


# =========================================================
# Chunk assembly
# =========================================================


def can_combine(
    current: list[Segment],
    candidate: Segment,
) -> bool:
    """
    Decide whether a candidate may belong to the current chunk.
    """

    if not current:
        return True


    first = current[0]


    # Never mix semantically different section types.
    if (
        candidate.chunk_type
        != first.chunk_type
    ):
        return False


    # Keep named sections distinct.
    if (
        candidate.section
        != first.section
    ):
        return False


    # For now, avoid combining across physical pages because
    # multi-column publisher layouts can occasionally have
    # imperfect reading order.

    if not ALLOW_CROSS_PAGE_CHUNKS:

        if (
            candidate.page_start
            != current[-1].page_end
        ):
            return False


    return True


def make_chunk_text(
    segments: list[Segment],
) -> str:
    """
    Join paragraph-sized segments while retaining paragraphs.
    """

    return "\n\n".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()


def assemble_segment_chunks(
    segments: list[Segment],
) -> list[list[Segment]]:
    """
    Assemble paragraph-sized segments into final chunk groups.
    """

    groups: list[
        list[Segment]
    ] = []

    current: list[
        Segment
    ] = []


    def flush_current() -> None:

        nonlocal current

        if not current:
            return

        groups.append(
            current
        )

        current = []


    for segment in segments:

        if not current:

            current = [
                segment
            ]

            continue


        if not can_combine(
            current,
            segment,
        ):

            previous = current

            flush_current()


            # Overlap is only appropriate when the next segment
            # belongs to the same section/type and, unless
            # explicitly allowed, the same physical page.

            overlap_allowed = (

                previous

                and previous[-1].section
                == segment.section

                and previous[-1].chunk_type
                == segment.chunk_type

                and (
                    ALLOW_CROSS_PAGE_CHUNKS

                    or (
                        previous[-1].page_end
                        == segment.page_start
                    )
                )
            )


            if overlap_allowed:

                current = overlap_tail(
                    previous,
                    CHUNK_OVERLAP,
                )


            current.append(
                segment
            )

            continue


        candidate = (
            current
            + [segment]
        )


        candidate_tokens = (
            count_tokens(
                make_chunk_text(
                    candidate
                )
            )
        )


        current_tokens = (
            count_tokens(
                make_chunk_text(
                    current
                )
            )
        )


        # Preferred case: stay below target.

        if (
            candidate_tokens
            <= CHUNK_TARGET
        ):

            current.append(
                segment
            )

            continue


        # Avoid producing a tiny previous chunk when adding the
        # new segment still stays below the maximum size.

        if (
            current_tokens
            < CHUNK_MIN

            and candidate_tokens
            <= CHUNK_MAX
        ):

            current.append(
                segment
            )

            continue


        # Otherwise close the current chunk.

        previous = current

        flush_current()


        overlap = overlap_tail(
            previous,
            CHUNK_OVERLAP,
        )


        current = overlap


        # Protect against overlap itself causing a large chunk.

        candidate_with_overlap = (
            current
            + [segment]
        )


        if (
            count_tokens(
                make_chunk_text(
                    candidate_with_overlap
                )
            )
            > CHUNK_MAX
        ):

            current = []


        current.append(
            segment
        )


    flush_current()

    return groups


# =========================================================
# Chunk IDs
# =========================================================


def make_chunk_id(
    document_sha256: str,
    chunk_index: int,
    text: str,
    page_start: int,
    page_end: int,
    section: str,
    chunk_type: str,
) -> str:
    """
    Generate deterministic chunk IDs.
    """

    payload = "|".join(
        [
            document_sha256,
            str(chunk_index),
            str(page_start),
            str(page_end),
            section,
            chunk_type,
            text,
        ]
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()

    return digest[:24]


# =========================================================
# Public chunking API
# =========================================================


def chunk_document(
    document: ParsedDocument,
) -> list[ResearchChunk]:
    """
    Convert one ParsedDocument into research-aware chunks.
    """

    blocks = document_to_blocks(
        document
    )


    blocks = repair_interrupted_layout(
        blocks
    )


    blocks = remove_heading_blocks(
        blocks
    )


    segments = blocks_to_segments(
        blocks
    )


    grouped = (
        assemble_segment_chunks(
            segments
        )
    )


    document_id = (
        document.sha256[:24]
    )


    chunks: list[
        ResearchChunk
    ] = []


    for index, group in enumerate(
        grouped
    ):

        if not group:
            continue


        text = make_chunk_text(
            group
        )


        if not text:
            continue


        page_start = min(
            segment.page_start
            for segment in group
        )


        page_end = max(
            segment.page_end
            for segment in group
        )


        section = (
            group[0].section
        )


        chunk_type = (
            group[0].chunk_type
        )


        token_count = (
            count_tokens(
                text
            )
        )


        chunk_id = make_chunk_id(
            document_sha256=(
                document.sha256
            ),

            chunk_index=index,

            text=text,

            page_start=page_start,

            page_end=page_end,

            section=section,

            chunk_type=chunk_type,
        )


        chunks.append(
            ResearchChunk(
                chunk_id=(
                    chunk_id
                ),

                document_id=(
                    document_id
                ),

                filename=(
                    document.filename
                ),

                page_start=(
                    page_start
                ),

                page_end=(
                    page_end
                ),

                section=(
                    section
                ),

                chunk_type=(
                    chunk_type
                ),

                text=text,

                token_count=(
                    token_count
                ),

                source_sha256=(
                    document.sha256
                ),
            )
        )


    return chunks


# =========================================================
# Diagnostics
# =========================================================


def chunk_statistics(
    chunks: list[ResearchChunk],
) -> dict:
    """
    Return useful development diagnostics.
    """

    if not chunks:

        return {
            "chunk_count": 0,
            "total_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "mean_tokens": 0.0,
            "below_min": 0,
            "above_max": 0,
            "types": {},
        }


    token_counts = [
        chunk.token_count
        for chunk in chunks
    ]


    types: dict[
        str,
        int,
    ] = {}


    for chunk in chunks:

        types[
            chunk.chunk_type
        ] = (
            types.get(
                chunk.chunk_type,
                0,
            )
            + 1
        )


    return {

        "chunk_count":
            len(chunks),

        "total_tokens":
            sum(token_counts),

        "min_tokens":
            min(token_counts),

        "max_tokens":
            max(token_counts),

        "mean_tokens":
            (
                sum(token_counts)
                / len(token_counts)
            ),

        "below_min":
            sum(
                1
                for value in token_counts
                if value < CHUNK_MIN
            ),

        "above_max":
            sum(
                1
                for value in token_counts
                if value > CHUNK_MAX
            ),

        "types":
            types,
    }