from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunker import (
    ResearchChunk,
    chunk_document,
)

from .config import (
    DOCUMENT_ROOT,
    EMBED_MODEL,
    SEARCHABLE_CHUNK_TYPES,
)

from .database import (
    replace_semantic_table,
)

from .embeddings import (
    OllamaEmbedder,
)

from .index_manifest import (
    build_current_index_manifest,
    write_index_manifest,
)

from .parser import (
    discover_pdfs,
    parse_pdf,
)


# =========================================================
# Summary
# =========================================================


@dataclass
class IndexSummary:

    pdf_count: int

    total_chunks: int

    searchable_chunks: int

    skipped_chunks: int

    embedding_dimension: int

    table_rows: int


# =========================================================
# Searchability
# =========================================================


def is_searchable_chunk(
    chunk: ResearchChunk,
) -> bool:
    """
    Return whether a chunk should enter the semantic index.
    """

    return (
        chunk.chunk_type
        in SEARCHABLE_CHUNK_TYPES
    )


# =========================================================
# LanceDB record construction
# =========================================================


def make_record(
    document,
    chunk: ResearchChunk,
    vector: list[float],
) -> dict:
    """
    Convert one embedded chunk into a LanceDB row.
    """

    return {

        "chunk_id":
            chunk.chunk_id,

        "document_id":
            chunk.document_id,

        "filename":
            chunk.filename,

        "source_path":
            str(
                document.path
            ),

        "source_sha256":
            chunk.source_sha256,

        "title":
            document.title or "",

        "author":
            document.author or "",

        "page_start":
            int(
                chunk.page_start
            ),

        "page_end":
            int(
                chunk.page_end
            ),

        "section":
            chunk.section,

        "chunk_type":
            chunk.chunk_type,

        "text":
            chunk.text,

        "token_count":
            int(
                chunk.token_count
            ),

        "embedding_model":
            EMBED_MODEL,

        "vector":
            vector,
    }


# =========================================================
# Build index
# =========================================================


def build_semantic_index(
    document_root: Path = DOCUMENT_ROOT,
) -> IndexSummary:
    """
    Parse, chunk, embed, and rebuild the complete research
    semantic index.

    Safety behavior
    ---------------

    The existing LanceDB table is not overwritten until:

    - every PDF has been parsed,
    - every searchable chunk has been produced,
    - every embedding has been generated,
    - all replacement records exist in memory,
    - the embedding dimension has been validated.

    Therefore, parsing or embedding failures leave the
    existing semantic table untouched.

    After the replacement table is successfully created,
    the current index manifest is written. A manifest write
    failure will cause future incremental syncs to fail
    closed rather than silently accepting an unknown index.
    """

    document_root = (
        Path(
            document_root
        )
        .expanduser()
        .resolve()
    )


    pdfs = sorted(
        discover_pdfs(
            document_root
        ),
        key=lambda path: (
            str(path)
            .casefold()
        ),
    )


    if not pdfs:

        raise RuntimeError(
            "No PDFs found in:\n"
            f"{document_root}"
        )


    print()
    print("=" * 80)

    print(
        "Building research semantic index"
    )

    print("=" * 80)

    print()

    print(
        f"PDFs found: {len(pdfs)}"
    )

    print(
        "Searchable chunk types:"
    )


    for chunk_type in sorted(
        SEARCHABLE_CHUNK_TYPES
    ):

        print(
            f"  - {chunk_type}"
        )


    embedder = (
        OllamaEmbedder()
    )


    records: list[
        dict
    ] = []


    total_chunks = 0

    searchable_chunks = 0

    skipped_chunks = 0


    # =====================================================
    # Process every document before touching LanceDB
    # =====================================================

    for document_index, pdf_path in enumerate(
        pdfs,
        start=1,
    ):

        print()
        print("=" * 80)

        print(
            f"Document "
            f"{document_index}/{len(pdfs)}"
        )

        print(
            pdf_path.name
        )

        print("=" * 80)


        # -------------------------------------------------
        # Parse
        # -------------------------------------------------

        document = (
            parse_pdf(
                pdf_path
            )
        )


        # -------------------------------------------------
        # Chunk
        # -------------------------------------------------

        chunks = (
            chunk_document(
                document
            )
        )


        total_chunks += (
            len(
                chunks
            )
        )


        searchable = [

            chunk

            for chunk in chunks

            if (
                is_searchable_chunk(
                    chunk
                )
            )
        ]


        skipped = (
            len(
                chunks
            )
            - len(
                searchable
            )
        )


        searchable_chunks += (
            len(
                searchable
            )
        )

        skipped_chunks += (
            skipped
        )


        print()

        print(
            f"Total chunks:      "
            f"{len(chunks)}"
        )

        print(
            f"Semantic chunks:   "
            f"{len(searchable)}"
        )

        print(
            f"Excluded chunks:   "
            f"{skipped}"
        )


        if not searchable:

            print(
                "No searchable chunks "
                "for this document."
            )

            continue


        # -------------------------------------------------
        # Embed
        # -------------------------------------------------

        texts = [

            chunk.text

            for chunk in searchable
        ]


        vectors = (
            embedder.embed_document_batches(
                texts
            )
        )


        if (
            len(
                vectors
            )
            != len(
                searchable
            )
        ):

            raise RuntimeError(
                "Internal embedding count mismatch."
            )


        # -------------------------------------------------
        # Build database records
        # -------------------------------------------------

        for chunk, vector in zip(
            searchable,
            vectors,
            strict=True,
        ):

            records.append(
                make_record(
                    document=(
                        document
                    ),
                    chunk=(
                        chunk
                    ),
                    vector=(
                        vector
                    ),
                )
            )


    # =====================================================
    # Validate complete replacement BEFORE database write
    # =====================================================

    if not records:

        raise RuntimeError(
            "No searchable chunks were generated."
        )


    dimension = (
        embedder.dimension
    )


    if dimension is None:

        raise RuntimeError(
            "Could not determine embedding dimension."
        )


    if dimension <= 0:

        raise RuntimeError(
            "Embedding dimension must be greater than 0."
        )


    for record_index, record in enumerate(
        records,
        start=1,
    ):

        vector = (
            record.get(
                "vector"
            )
        )


        if (
            vector is None
            or len(
                vector
            )
            != dimension
        ):

            raise RuntimeError(
                "Embedding dimension mismatch in "
                f"record {record_index}."
            )


    # =====================================================
    # Replace LanceDB table
    # =====================================================

    print()
    print("=" * 80)

    print(
        "Writing replacement LanceDB table..."
    )

    print("=" * 80)


    replace_semantic_table(
        records
    )


    # =====================================================
    # Persist reproducibility manifest
    # =====================================================

    current_manifest = (
        build_current_index_manifest()
    )


    write_index_manifest(
        current_manifest
    )


    return IndexSummary(

        pdf_count=(
            len(
                pdfs
            )
        ),

        total_chunks=(
            total_chunks
        ),

        searchable_chunks=(
            searchable_chunks
        ),

        skipped_chunks=(
            skipped_chunks
        ),

        embedding_dimension=(
            dimension
        ),

        table_rows=(
            len(
                records
            )
        ),
    )
