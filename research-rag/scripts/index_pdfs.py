from research_rag.config import (
    EMBED_MODEL,
    LANCEDB_PATH,
    SEMANTIC_TABLE_NAME,
)

from research_rag.indexer import (
    build_semantic_index,
)


def main():

    print()
    print("=" * 80)

    print(
        "Research RAG — Semantic indexing"
    )

    print("=" * 80)

    print()

    print(
        f"Embedding model: "
        f"{EMBED_MODEL}"
    )

    print(
        f"LanceDB path:    "
        f"{LANCEDB_PATH}"
    )

    print(
        f"Table:           "
        f"{SEMANTIC_TABLE_NAME}"
    )


    summary = (
        build_semantic_index()
    )


    print()
    print("=" * 80)

    print(
        "INDEX COMPLETE"
    )

    print("=" * 80)

    print()

    print(
        f"PDFs processed:       "
        f"{summary.pdf_count}"
    )

    print(
        f"Chunks generated:     "
        f"{summary.total_chunks}"
    )

    print(
        f"Semantic chunks:      "
        f"{summary.searchable_chunks}"
    )

    print(
        f"Excluded chunks:      "
        f"{summary.skipped_chunks}"
    )

    print(
        f"Embedding dimension:  "
        f"{summary.embedding_dimension}"
    )

    print(
        f"LanceDB rows:         "
        f"{summary.table_rows}"
    )

    print()

    print(
        "Semantic search is ready."
    )

    print()


if __name__ == "__main__":
    main()