from __future__ import annotations

from collections.abc import Sequence

import lancedb
import pandas as pd

from .config import (
    LANCEDB_PATH,
    SEMANTIC_TABLE_NAME,
)


# =========================================================
# Connection
# =========================================================


def connect_database():
    """
    Connect to the local LanceDB database.
    """

    LANCEDB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )


    return lancedb.connect(
        str(
            LANCEDB_PATH
        )
    )


# =========================================================
# Table creation
# =========================================================


def replace_semantic_table(
    records: list[dict],
):
    """
    Replace the semantic chunk table.

    Milestone 3 deliberately rebuilds the table each time.

    Incremental indexing will be added later.
    """

    if not records:

        raise ValueError(
            "Cannot create semantic table "
            "with zero records."
        )


    db = connect_database()


    table = db.create_table(
        SEMANTIC_TABLE_NAME,

        data=records,

        mode="overwrite",
    )


    return table


# =========================================================
# Table access
# =========================================================


def open_semantic_table():
    """
    Open the semantic chunk table.
    """

    db = connect_database()


    try:

        table = db.open_table(
            SEMANTIC_TABLE_NAME
        )

    except Exception as exc:

        raise RuntimeError(
            "\nSemantic index does not exist yet.\n\n"
            "Run:\n"
            "uv run python scripts\\index_pdfs.py"
        ) from exc


    return table


# =========================================================
# Vector search
# =========================================================


def semantic_search(
    query_vector: Sequence[float],
    top_k: int,
) -> list[dict]:
    """
    Search the semantic chunk table.

    Lower _distance values indicate closer vectors.
    """

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    table = (
        open_semantic_table()
    )


    results = (

        table
        .search(
            list(
                query_vector
            )
        )

        .select(
            [
                "chunk_id",
                "document_id",
                "filename",
                "source_path",
                "source_sha256",
                "title",
                "author",
                "page_start",
                "page_end",
                "section",
                "chunk_type",
                "text",
                "token_count",
                "embedding_model",
                "_distance",
            ]
        )

        .limit(
            top_k
        )

        .to_pandas()
    )


    if results.empty:

        return []


    return results.to_dict(
        orient="records"
    )