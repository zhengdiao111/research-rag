from __future__ import annotations

from typing import Any

import numpy as np

from lancedb.rerankers import (
    RRFReranker,
)

from .config import (
    FTS_COLUMN,
    HYBRID_RRF_K,
    RETRIEVAL_MODE,
    VECTOR_COLUMN,
)

from .database import (
    open_semantic_table,
    semantic_search,
)


# =========================================================
# Helpers
# =========================================================


def dataframe_to_records(
    dataframe,
) -> list[dict[str, Any]]:
    """
    Convert LanceDB/Pandas output into the list[dict]
    representation used throughout the RAG pipeline.
    """

    if dataframe is None:

        return []


    if dataframe.empty:

        return []


    records = (
        dataframe.to_dict(
            orient="records"
        )
    )


    # -----------------------------------------------------
    # The raw vector is large and no longer needed once
    # retrieval is complete.
    # -----------------------------------------------------

    for record in records:

        record.pop(
            VECTOR_COLUMN,
            None,
        )


    return records


def normalize_query_vector(
    query_vector,
):
    """
    Convert the embedding into a format LanceDB accepts
    reliably for vector/hybrid search.

    Accepts:

        list
        tuple
        numpy.ndarray
        objects supporting .tolist()
    """

    if isinstance(
        query_vector,
        np.ndarray,
    ):

        return (
            query_vector
            .astype(
                np.float32
            )
            .tolist()
        )


    if hasattr(
        query_vector,
        "tolist",
    ):

        query_vector = (
            query_vector.tolist()
        )


    if isinstance(
        query_vector,
        tuple,
    ):

        query_vector = list(
            query_vector
        )


    if not isinstance(
        query_vector,
        list,
    ):

        raise TypeError(
            "query_vector must be a list, tuple, "
            "NumPy array, or object supporting .tolist()."
        )


    return [
        float(value)
        for value in query_vector
    ]


# =========================================================
# Vector retrieval
# =========================================================


def vector_search(
    query_vector,
    top_k: int,
) -> list[dict]:
    """
    Existing semantic vector retrieval.

    This wraps the Milestone 3 semantic_search() function
    so the rest of the application can switch cleanly
    between vector and hybrid retrieval.
    """

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    return (
        semantic_search(
            query_vector=(
                query_vector
            ),
            top_k=(
                top_k
            ),
        )
    )


# =========================================================
# Lexical / BM25 retrieval
# =========================================================


def lexical_search(
    query_text: str,
    top_k: int,
) -> list[dict]:
    """
    Search the LanceDB FTS index using BM25.
    """

    query_text = (
        str(
            query_text
        )
        .strip()
    )


    if not query_text:

        return []


    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    table = (
        open_semantic_table()
    )


    try:

        query = (
            table.search(
                query_text,
                query_type="fts",
                fts_columns=FTS_COLUMN,
            )
        )


        dataframe = (
            query
            .limit(
                top_k
            )
            .to_pandas()
        )


    except Exception as exc:

        raise RuntimeError(
            "\nFull-text search failed.\n\n"
            "The FTS index may not exist or may be "
            "incompatible with the current table.\n\n"
            "Run:\n\n"
            "uv run python "
            "scripts\\create_fts_index.py\n\n"
            f"Original error:\n{exc}"
        ) from exc


    return (
        dataframe_to_records(
            dataframe
        )
    )


# =========================================================
# Hybrid retrieval
# =========================================================


def hybrid_search(
    query_text: str,
    query_vector,
    top_k: int,
) -> list[dict]:
    """
    Hybrid retrieval using:

        vector similarity
        +
        BM25 full-text search
        +
        Reciprocal Rank Fusion

    IMPORTANT:

    In the current synchronous LanceDB API, the vector and
    text queries are supplied separately:

        table.search(query_type="hybrid")
            .vector(...)
            .text(...)

    rather than passing a tuple to table.search().
    """

    query_text = (
        str(
            query_text
        )
        .strip()
    )


    if not query_text:

        return []


    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    query_vector = (
        normalize_query_vector(
            query_vector
        )
    )


    table = (
        open_semantic_table()
    )


    # -----------------------------------------------------
    # Reciprocal Rank Fusion
    #
    # return_score="all" is useful during development
    # because LanceDB retains the individual vector / FTS
    # scoring information as well as the fused relevance
    # score.
    # -----------------------------------------------------

    reranker = (
        RRFReranker(
            K=HYBRID_RRF_K,
            return_score="all",
        )
    )


    try:

        query = (
            table.search(
                query_type="hybrid",
                vector_column_name=(
                    VECTOR_COLUMN
                ),
                fts_columns=(
                    FTS_COLUMN
                ),
            )
            .vector(
                query_vector
            )
            .text(
                query_text
            )
            .rerank(
                reranker
            )
        )


        dataframe = (
            query
            .limit(
                top_k
            )
            .to_pandas()
        )


    except Exception as exc:

        raise RuntimeError(
            "\nHybrid search failed.\n\n"
            "FTS index creation succeeded, so this "
            "usually indicates a hybrid-query API or "
            "vector-format issue.\n\n"
            f"Query text:\n{query_text}\n\n"
            f"Vector dimension:\n"
            f"{len(query_vector)}\n\n"
            f"Original error:\n{exc}"
        ) from exc


    return (
        dataframe_to_records(
            dataframe
        )
    )


# =========================================================
# Unified retrieval entry point
# =========================================================


def retrieve(
    question: str,
    query_vector,
    top_k: int,
    mode: str = RETRIEVAL_MODE,
) -> list[dict]:
    """
    Unified retrieval API.

    Supported modes:

        vector
        hybrid
    """

    mode = (
        str(
            mode
        )
        .strip()
        .lower()
    )


    if mode == "vector":

        return (
            vector_search(
                query_vector=(
                    query_vector
                ),
                top_k=(
                    top_k
                ),
            )
        )


    if mode == "hybrid":

        return (
            hybrid_search(
                query_text=(
                    question
                ),
                query_vector=(
                    query_vector
                ),
                top_k=(
                    top_k
                ),
            )
        )


    raise ValueError(
        f"Unknown retrieval mode: {mode}"
    )