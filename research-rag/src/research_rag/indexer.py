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
    Parse, chunk, embed, and index the complete research
    library.

    Milestone 3 rebuilds the semantic table every run.
    """

    document_root = Path(
        document_root
    )


    pdfs = discover_pdfs(
        document_root
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
    # Process documents
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

        document = parse_pdf(
            pdf_path
        )


        # -------------------------------------------------
        # Chunk
        # -------------------------------------------------

        chunks = chunk_document(
            document
        )


        total_chunks += len(
            chunks
        )


        searchable = [
            chunk
            for chunk in chunks
            if is_searchable_chunk(
                chunk
            )
        ]


        skipped = (
            len(chunks)
            - len(searchable)
        )


        searchable_chunks += len(
            searchable
        )

        skipped_chunks += skipped


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


        if len(
            vectors
        ) != len(
            searchable
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
                    document=document,
                    chunk=chunk,
                    vector=vector,
                )
            )


    # =====================================================
    # Save LanceDB table
    # =====================================================

    if not records:

        raise RuntimeError(
            "No searchable chunks were generated."
        )


    print()
    print("=" * 80)

    print(
        "Writing LanceDB table..."
    )

    print("=" * 80)


    replace_semantic_table(
        records
    )


    dimension = (
        embedder.dimension
    )


    if dimension is None:

        raise RuntimeError(
            "Could not determine embedding dimension."
        )


    return IndexSummary(

        pdf_count=len(
            pdfs
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

        table_rows=len(
            records
        ),
    )