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
    Approximate token count used only for chunk sizing.

    The embedding model uses its own tokenizer, so this
    should not be treated as the model's true token count.
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
    Decode chunking-token IDs back to text.
    """

    return _ENCODER.decode(
        tokens
    )


# =========================================================
# Data structures
# =========================================================


@dataclass
class MarkdownUnit:
    """
    One parsed Markdown unit.

    kind:
        "heading" or "paragraph"

    heading_level:
        1-6 for Markdown headings.
        None for paragraphs.
    """

    kind: str

    text: str

    heading_level: int | None = None


@dataclass
class HeadingContext:
    """
    Active heading in the document hierarchy.
    """

    level: int

    section: str

    chunk_type: str


@dataclass
class TextBlock:
    """
    Intermediate semantic block.
    """

    text: str

    page_start: int

    page_end: int

    section: str

    chunk_type: str

    is_heading: bool = False

    heading_level: int | None = None


@dataclass
class Segment:
    """
    Paragraph-sized unit ready for chunk assembly.
    """

    text: str

    page_start: int

    page_end: int

    section: str

    chunk_type: str


@dataclass
class ResearchChunk:
    """
    Final RAG chunk.
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
    Remove Markdown formatting from a heading.
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

    text = text.replace(
        "~~",
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
    Conservative cleanup for scientific text.

    Does not perform spelling correction because that could
    damage gene names, protein names, chemical names,
    identifiers, mathematical notation, etc.
    """

    if not text:
        return ""

    # Unicode replacement character.
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

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Repeated spaces / tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# Heading classification
# =========================================================


def normalize_heading_for_classification(
    heading: str,
) -> str:
    """
    Normalize heading text before classification.
    """

    normalized = (
        heading
        .lower()
        .strip()
    )

    normalized = re.sub(
        r"[^a-z0-9\s&:/-]",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def classify_heading(
    heading: str,
) -> str:
    """
    Directly classify a heading.

    Returning "body" means the heading itself does not
    explicitly identify a special section. Hierarchical
    inheritance is handled separately.
    """

    normalized = (
        normalize_heading_for_classification(
            heading
        )
    )


    # =====================================================
    # References / bibliography
    # =====================================================

    if normalized in {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature cited",
    }:

        return "references"


    if normalized.startswith(
        "references and"
    ):

        return "references"


    # =====================================================
    # Further reading
    # =====================================================

    if normalized in {
        "further reading",
        "further readings",
        "recommended reading",
        "recommended readings",
        "suggested reading",
        "suggested readings",
        "additional reading",
        "additional readings",
        "further reading and listening",
    }:

        return "further_reading"


    # =====================================================
    # Acknowledgments
    # =====================================================

    if normalized in {
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
    }:

        return "acknowledgments"


    # =====================================================
    # Abstract / summary
    # =====================================================

    if normalized in {
        "abstract",
        "summary",
    }:

        return "abstract"


    # =====================================================
    # Introduction
    # =====================================================

    if normalized in {
        "introduction",
        "background",
    }:

        return "introduction"


    # =====================================================
    # Methods
    # =====================================================

    if normalized in {
        "methods",
        "methodology",
        "materials and methods",
        "methods and materials",
        "experimental methods",
        "experimental procedures",
        "experimental section",
    }:

        return "methods"


    # =====================================================
    # Results
    # =====================================================

    if (
        normalized == "results"
        or normalized.startswith(
            "results "
        )
    ):

        return "results"


    # =====================================================
    # Discussion
    # =====================================================

    if (
        normalized == "discussion"
        or normalized.startswith(
            "discussion "
        )
    ):

        return "discussion"


    # =====================================================
    # Conclusions
    # =====================================================

    if normalized in {
        "conclusion",
        "conclusions",
        "concluding remarks",
        "concluding discussion",
    }:

        return "conclusion"


    # =====================================================
    # Supplementary / appendix
    # =====================================================

    if any(
        phrase in normalized
        for phrase in (
            "supplementary material",
            "supplemental material",
            "supplementary information",
            "supporting information",
        )
    ):

        return "supplementary"


    if (
        normalized == "appendix"
        or normalized == "appendices"
        or normalized.startswith(
            "appendix "
        )
    ):

        return "supplementary"


    # =====================================================
    # Book front matter
    # =====================================================

    if normalized in {
        "contents",
        "table of contents",
        "preface",
        "foreword",
        "prologue",
        "introduction to the book",
        "about the author",
        "about the authors",
        "contributors",
        "list of contributors",
        "dedication",
        "title page",
        "copyright page",
    }:

        return "frontmatter"


    # =====================================================
    # Index
    # =====================================================

    if normalized in {
        "index",
        "subject index",
        "author index",
        "name index",
        "general index",
    }:

        return "index"


    # =====================================================
    # Notes / endnotes
    # =====================================================

    if normalized in {
        "notes",
        "endnotes",
        "chapter notes",
        "notes to chapters",
    }:

        return "notes"


    # =====================================================
    # Glossary
    # =====================================================

    if normalized in {
        "glossary",
        "glossary of terms",
    }:

        return "glossary"


    # =====================================================
    # Ordinary chapter headings
    # =====================================================
    #
    # Chapters remain normal semantic body content.

    if re.match(
        r"^chapter\b",
        normalized,
    ):

        return "body"


    # =====================================================
    # Default
    # =====================================================

    return "body"


# =========================================================
# Hierarchical section inheritance
# =========================================================


INHERITABLE_PARENT_TYPES = {
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "supplementary",
    "references",
    "acknowledgments",
    "index",
    "notes",
    "glossary",
    "further_reading",
}


def resolve_heading_type(
    heading: str,
    heading_level: int,
    heading_stack: list[HeadingContext],
) -> str:
    """
    Determine the semantic type for a heading.

    Example
    -------

    # Index
        -> index

    ## bacteria
        direct classification = body
        parent = index
        final type = index

    This also improves normal papers:

    ## Results
        -> results

    ### Transcription increases after stimulation
        direct classification = body
        parent = results
        final type = results
    """

    direct_type = (
        classify_heading(
            heading
        )
    )


    # An explicitly recognized heading always wins.

    if direct_type != "body":

        return direct_type


    # Search the active ancestors from nearest to farthest.

    for parent in reversed(
        heading_stack
    ):

        if parent.level >= heading_level:
            continue


        if (
            parent.chunk_type
            in INHERITABLE_PARENT_TYPES
        ):

            return (
                parent.chunk_type
            )


    return "body"


# =========================================================
# Metadata / publisher boilerplate
# =========================================================


_METADATA_SIGNALS = (
    "view the article online",
    "permissions",
    "terms of service",
    "issn",
    "isbn",
    "copyright ©",
    "copyright (c)",
    "all rights reserved",
    "reprints and permissions",
    "downloaded from http",
    "library of congress control number",
    "printed in china",
    "printed in the united states",
)


def looks_like_metadata(
    text: str,
) -> bool:
    """
    Identify obvious publisher boilerplate.
    """

    lowered = (
        text.lower()
    )

    return any(
        signal in lowered
        for signal in _METADATA_SIGNALS
    )


def page_is_mostly_metadata(
    paragraphs: list[str],
) -> bool:
    """
    Detect publisher-only pages.

    Multiple metadata signals are required to reduce false
    positives.
    """

    if not paragraphs:
        return False


    hits = sum(
        1
        for text in paragraphs
        if looks_like_metadata(
            text
        )
    )


    return (
        hits >= 2
        and hits >= max(
            2,
            len(paragraphs) // 2,
        )
    )


# =========================================================
# Markdown page parsing
# =========================================================


def split_markdown_units(
    text: str,
) -> list[MarkdownUnit]:
    """
    Split Markdown into headings and paragraphs while
    preserving heading level.
    """

    text = normalize_text(
        text
    )


    if not text:

        return []


    units: list[
        MarkdownUnit
    ] = []


    paragraph_lines: list[
        str
    ] = []


    def flush_paragraph() -> None:

        if not paragraph_lines:
            return


        paragraph = " ".join(
            line.strip()
            for line in paragraph_lines
            if line.strip()
        )


        paragraph = (
            normalize_text(
                paragraph
            )
        )


        if paragraph:

            units.append(
                MarkdownUnit(
                    kind="paragraph",
                    text=paragraph,
                    heading_level=None,
                )
            )


        paragraph_lines.clear()


    for line in text.splitlines():

        stripped = (
            line.strip()
        )


        if not stripped:

            flush_paragraph()

            continue


        match = (
            HEADING_PATTERN.match(
                stripped
            )
        )


        if match:

            flush_paragraph()


            markdown_marks = (
                match.group(1)
            )


            heading_level = len(
                markdown_marks
            )


            heading = (
                clean_heading(
                    match.group(2)
                )
            )


            if heading:

                units.append(
                    MarkdownUnit(
                        kind="heading",
                        text=heading,
                        heading_level=(
                            heading_level
                        ),
                    )
                )


            continue


        paragraph_lines.append(
            stripped
        )


    flush_paragraph()


    return units


# =========================================================
# Heading stack helpers
# =========================================================


def update_heading_stack(
    heading_stack: list[HeadingContext],
    heading: str,
    level: int,
    chunk_type: str,
) -> None:
    """
    Update active Markdown heading hierarchy.

    Any existing heading at the same or deeper level is
    closed when a new heading appears.
    """

    while (
        heading_stack
        and heading_stack[-1].level
        >= level
    ):

        heading_stack.pop()


    heading_stack.append(
        HeadingContext(
            level=level,
            section=heading,
            chunk_type=chunk_type,
        )
    )


# =========================================================
# Document -> blocks
# =========================================================


def document_to_blocks(
    document: ParsedDocument,
) -> list[TextBlock]:
    """
    Convert parsed PDF pages into hierarchical semantic
    text blocks.
    """

    blocks: list[
        TextBlock
    ] = []


    current_section = (
        document.title
        or "Main text"
    )


    current_type = (
        "body"
    )


    heading_stack: list[
        HeadingContext
    ] = []


    for page in document.pages:

        units = (
            split_markdown_units(
                page.text
            )
        )


        paragraphs = [
            unit.text

            for unit in units

            if (
                unit.kind
                == "paragraph"
            )
        ]


        mostly_metadata = (
            page_is_mostly_metadata(
                paragraphs
            )
        )


        for unit in units:

            # =================================================
            # Heading
            # =================================================

            if unit.kind == "heading":

                heading_level = (
                    unit.heading_level
                    or 1
                )


                # ---------------------------------------------
                # Remove same-level and deeper headings BEFORE
                # determining inheritance.
                # ---------------------------------------------

                while (
                    heading_stack
                    and heading_stack[-1].level
                    >= heading_level
                ):

                    heading_stack.pop()


                heading_type = (
                    resolve_heading_type(
                        heading=(
                            unit.text
                        ),

                        heading_level=(
                            heading_level
                        ),

                        heading_stack=(
                            heading_stack
                        ),
                    )
                )


                current_section = (
                    unit.text
                )


                current_type = (
                    heading_type
                )


                heading_stack.append(
                    HeadingContext(
                        level=(
                            heading_level
                        ),

                        section=(
                            current_section
                        ),

                        chunk_type=(
                            current_type
                        ),
                    )
                )


                blocks.append(
                    TextBlock(
                        text=(
                            current_section
                        ),

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

                        heading_level=(
                            heading_level
                        ),
                    )
                )


                continue


            # =================================================
            # Paragraph
            # =================================================

            block_type = (
                current_type
            )


            if mostly_metadata:

                block_type = (
                    "metadata"
                )


            elif looks_like_metadata(
                unit.text
            ):

                block_type = (
                    "metadata"
                )


            blocks.append(
                TextBlock(
                    text=(
                        unit.text
                    ),

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

                    heading_level=None,
                )
            )


        # =================================================
        # Captions
        # =================================================

        for caption in page.captions:

            caption = (
                normalize_text(
                    caption
                )
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

                    chunk_type=(
                        "caption"
                    ),

                    is_heading=False,

                    heading_level=None,
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
    Return True when the first alphabetic character is
    lowercase.
    """

    for char in text:

        if char.isalpha():

            return (
                char.islower()
            )


    return False


def repair_interrupted_layout(
    blocks: list[TextBlock],
) -> list[TextBlock]:
    """
    Repair the multi-column interruption pattern previously
    observed in Science PDFs.

    Example:

        observed phenom-

        ## Defects program tissue shapes

        [sidebar]

        enon is driven...

    becomes:

        observed phenomenon is driven...

    while preserving the sidebar separately.
    """

    blocks = list(
        blocks
    )


    i = 0


    while i <= len(
        blocks
    ) - 4:

        first = (
            blocks[i]
        )

        heading = (
            blocks[i + 1]
        )

        sidebar_body = (
            blocks[i + 2]
        )

        continuation = (
            blocks[i + 3]
        )


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


        # -------------------------------------------------
        # Repair interrupted word
        # -------------------------------------------------

        left = (
            first.text.rstrip()
        )

        right = (
            continuation.text.lstrip()
        )


        joined_text = (
            left[:-1]
            + right
        )


        merged = replace(
            first,

            text=(
                joined_text
            ),

            page_end=(
                continuation.page_end
            ),
        )


        # -------------------------------------------------
        # Preserve sidebar
        # -------------------------------------------------

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

            chunk_type=(
                "sidebar"
            ),

            is_heading=False,

            heading_level=None,
        )


        # -------------------------------------------------
        # Restore original section to immediately following
        # prose if the false sidebar heading changed it.
        # -------------------------------------------------

        j = (
            i + 4
        )


        while j < len(
            blocks
        ):

            block = (
                blocks[j]
            )


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
# Remove standalone headings
# =========================================================


def remove_heading_blocks(
    blocks: Iterable[TextBlock],
) -> list[TextBlock]:
    """
    Heading information is retained as metadata, so headings
    do not need independent chunks.
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
    Split a long paragraph into smaller semantic segments.

    Preference:
        sentence boundaries
        then token-level fallback
    """

    text = (
        block.text.strip()
    )


    token_count = (
        count_tokens(
            text
        )
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

        for sentence
        in _SENTENCE_SPLIT.split(
            text
        )

        if sentence.strip()
    ]


    # =====================================================
    # Token-level fallback
    # =====================================================

    if len(
        sentences
    ) <= 1:

        tokens = (
            _ENCODER.encode(
                text,
                disallowed_special=(),
            )
        )


        pieces: list[
            Segment
        ] = []


        start = 0


        while start < len(
            tokens
        ):

            stop = min(
                start
                + CHUNK_TARGET,

                len(tokens),
            )


            piece = (
                decode_tokens(
                    tokens[
                        start:stop
                    ]
                )
                .strip()
            )


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
                tokens
            ):

                break


            start = max(
                stop
                - CHUNK_OVERLAP,

                start + 1,
            )


        return pieces


    # =====================================================
    # Sentence-aware split
    # =====================================================

    pieces: list[
        Segment
    ] = []


    current: list[
        str
    ] = []


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

            piece = (
                " ".join(
                    current
                )
                .strip()
            )


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


        # -------------------------------------------------
        # Single huge sentence
        # -------------------------------------------------

        if (
            not current

            and sentence_tokens
            > CHUNK_MAX
        ):

            encoded = (
                _ENCODER.encode(
                    sentence,
                    disallowed_special=(),
                )
            )


            start = 0


            while start < len(
                encoded
            ):

                stop = min(
                    start
                    + CHUNK_TARGET,

                    len(
                        encoded
                    ),
                )


                piece = (
                    decode_tokens(
                        encoded[
                            start:stop
                        ]
                    )
                    .strip()
                )


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
                    encoded
                ):

                    break


                start = max(
                    stop
                    - CHUNK_OVERLAP,

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

        pieces.append(
            Segment(
                text=(
                    " ".join(
                        current
                    )
                    .strip()
                ),

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
# Overlap
# =========================================================


def overlap_tail(
    segments: list[Segment],
    token_budget: int,
) -> list[Segment]:

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

        token_count = (
            count_tokens(
                segment.text
            )
        )


        if (
            total + token_count
            <= token_budget
        ):

            selected.append(
                segment
            )

            total += (
                token_count
            )

            continue


        remaining = (
            token_budget
            - total
        )


        if remaining > 0:

            encoded = (
                _ENCODER.encode(
                    segment.text,
                    disallowed_special=(),
                )
            )


            tail_text = (
                decode_tokens(
                    encoded[
                        -remaining:
                    ]
                )
                .strip()
            )


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

    if not current:
        return True


    first = (
        current[0]
    )


    if (
        candidate.chunk_type
        != first.chunk_type
    ):

        return False


    if (
        candidate.section
        != first.section
    ):

        return False


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

    return "\n\n".join(
        segment.text.strip()

        for segment in segments

        if segment.text.strip()
    ).strip()


def assemble_segment_chunks(
    segments: list[Segment],
) -> list[list[Segment]]:

    groups: list[
        list[Segment]
    ] = []


    current: list[
        Segment
    ] = []


    def flush_current():

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

            previous = (
                current
            )


            flush_current()


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

                current = (
                    overlap_tail(
                        previous,
                        CHUNK_OVERLAP,
                    )
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


        if (
            candidate_tokens
            <= CHUNK_TARGET
        ):

            current.append(
                segment
            )

            continue


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


        previous = (
            current
        )


        flush_current()


        current = (
            overlap_tail(
                previous,
                CHUNK_OVERLAP,
            )
        )


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

    payload = "|".join(
        [
            document_sha256,
            str(
                chunk_index
            ),
            str(
                page_start
            ),
            str(
                page_end
            ),
            section,
            chunk_type,
            text,
        ]
    )


    digest = (
        hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


    return digest[:24]


# =========================================================
# Public API
# =========================================================


def chunk_document(
    document: ParsedDocument,
) -> list[ResearchChunk]:
    """
    Convert one parsed research document into final RAG
    chunks.
    """

    blocks = (
        document_to_blocks(
            document
        )
    )


    blocks = (
        repair_interrupted_layout(
            blocks
        )
    )


    blocks = (
        remove_heading_blocks(
            blocks
        )
    )


    segments = (
        blocks_to_segments(
            blocks
        )
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


        text = (
            make_chunk_text(
                group
            )
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


        chunk_id = (
            make_chunk_id(
                document_sha256=(
                    document.sha256
                ),

                chunk_index=(
                    index
                ),

                text=(
                    text
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
            )
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

                text=(
                    text
                ),

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
            len(
                chunks
            ),

        "total_tokens":
            sum(
                token_counts
            ),

        "min_tokens":
            min(
                token_counts
            ),

        "max_tokens":
            max(
                token_counts
            ),

        "mean_tokens":
            (
                sum(
                    token_counts
                )
                / len(
                    token_counts
                )
            ),

        "below_min":
            sum(
                1
                for value
                in token_counts
                if value < CHUNK_MIN
            ),

        "above_max":
            sum(
                1
                for value
                in token_counts
                if value > CHUNK_MAX
            ),

        "types":
            types,
    }