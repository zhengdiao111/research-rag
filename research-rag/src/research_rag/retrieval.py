from __future__ import annotations

import math

from typing import Any

import numpy as np

from lancedb.rerankers import (
    RRFReranker,
)

from .config import (
    FTS_COLUMN,
    HYBRID_RRF_K,
    RERANK_CAPTION_WEIGHT,
    RERANK_ENABLED,
    RERANK_MAX_CAPTIONS,
    RERANK_METHODS_WEIGHT,
    RERANK_SUPPLEMENTARY_WEIGHT,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
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
    # The full embedding vector is large and is no longer
    # needed after retrieval.
    # -----------------------------------------------------

    for record in records:

        record.pop(
            VECTOR_COLUMN,
            None,
        )


    return records


def normalize_query_vector(
    query_vector,
) -> list[float]:
    """
    Convert an embedding into a format accepted reliably
    by LanceDB.

    Supports:

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


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Convert a value to float while treating None and NaN
    as the supplied default.
    """

    if value is None:

        return default


    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


    if math.isnan(
        result
    ):

        return default


    return result


# =========================================================
# Chunk-type classification
# =========================================================


def normalize_chunk_type(
    result: dict,
) -> str:

    return (
        str(
            result.get(
                "chunk_type",
                "",
            )
            or ""
        )
        .strip()
        .lower()
    )


def normalize_section(
    result: dict,
) -> str:

    return (
        str(
            result.get(
                "section",
                "",
            )
            or ""
        )
        .strip()
        .lower()
    )


def is_caption_like(
    result: dict,
) -> bool:
    """
    Detect caption-like chunks.

    We check both the explicit chunk_type and the section
    heading because some parsed PDFs may represent a figure
    or table using a generic chunk type but retain the
    figure/table label in the section metadata.
    """

    chunk_type = (
        normalize_chunk_type(
            result
        )
    )


    section = (
        normalize_section(
            result
        )
    )


    if chunk_type == "caption":

        return True


    caption_prefixes = (
        "figure ",
        "figure:",
        "fig. ",
        "fig ",
        "table ",
        "table:",
    )


    return section.startswith(
        caption_prefixes
    )


# =========================================================
# Research-aware weighting
# =========================================================


def chunk_type_weight(
    result: dict,
) -> float:
    """
    Apply a conservative weight based on research-document
    chunk type.

    Important:

    These are deliberately mild adjustments.

    Retrieval relevance remains the dominant ranking signal.
    """

    chunk_type = (
        normalize_chunk_type(
            result
        )
    )


    # -----------------------------------------------------
    # Figure/table captions are useful but should generally
    # not displace explanatory prose when equally relevant.
    # -----------------------------------------------------

    if is_caption_like(
        result
    ):

        return (
            RERANK_CAPTION_WEIGHT
        )


    # -----------------------------------------------------
    # Methods are still valuable, especially for questions
    # asking how something was done.

    # The penalty is intentionally tiny.
    # -----------------------------------------------------

    if chunk_type == "methods":

        return (
            RERANK_METHODS_WEIGHT
        )


    # -----------------------------------------------------
    # Supplementary material is useful but receives a small
    # preference penalty relative to primary narrative text.
    # -----------------------------------------------------

    if chunk_type == "supplementary":

        return (
            RERANK_SUPPLEMENTARY_WEIGHT
        )


    # -----------------------------------------------------
    # Main explanatory scientific text remains neutral.
    # -----------------------------------------------------

    return 1.0


# =========================================================
# Vector retrieval
# =========================================================


def vector_search(
    query_vector,
    top_k: int,
) -> list[dict]:
    """
    Existing semantic vector retrieval.
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

        dataframe = (
            table.search(
                query_text,
                query_type="fts",
                fts_columns=(
                    FTS_COLUMN
                ),
            )
            .limit(
                top_k
            )
            .to_pandas()
        )


    except Exception as exc:

        raise RuntimeError(
            "\nFull-text search failed.\n\n"
            "Make sure the LanceDB FTS index exists.\n\n"
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
# Raw hybrid retrieval
# =========================================================


def hybrid_search(
    query_text: str,
    query_vector,
    top_k: int,
) -> list[dict]:
    """
    Native LanceDB hybrid retrieval:

        vector similarity
        +
        BM25 full-text retrieval
        +
        Reciprocal Rank Fusion

    This returns the raw RRF-ranked result set.

    Research-specific weighting happens later.
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


    reranker = (
        RRFReranker(
            K=(
                HYBRID_RRF_K
            ),
            return_score="all",
        )
    )


    try:

        dataframe = (
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
            .limit(
                top_k
            )
            .to_pandas()
        )


    except Exception as exc:

        raise RuntimeError(
            "\nHybrid search failed.\n\n"
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
# Research-aware post-ranking
# =========================================================


def score_candidate(
    result: dict,
    fallback_rank: int,
) -> dict:
    """
    Attach research-aware scoring diagnostics to one hybrid
    candidate.

    LanceDB's RRF relevance score remains the base score.

    The only adjustment at this stage is a conservative
    multiplier based on chunk type.
    """

    fallback_score = (
        1.0
        / (
            HYBRID_RRF_K
            + fallback_rank
        )
    )


    base_score = (
        safe_float(
            result.get(
                "_relevance_score"
            ),
            default=(
                fallback_score
            ),
        )
    )


    weight = (
        chunk_type_weight(
            result
        )
    )


    refined_score = (
        base_score
        * weight
    )


    scored = dict(
        result
    )


    scored[
        "_base_relevance_score"
    ] = (
        base_score
    )


    scored[
        "_chunk_type_weight"
    ] = (
        weight
    )


    scored[
        "_refined_score"
    ] = (
        refined_score
    )


    scored[
        "_caption_like"
    ] = (
        is_caption_like(
            result
        )
    )


    return scored


def refine_hybrid_results(
    results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Research-aware post-ranking.

    Strategy:

    1. Preserve LanceDB RRF relevance as the primary signal.
    2. Apply a small chunk-type multiplier.
    3. Sort by the adjusted score.
    4. Limit caption-like chunks in the first pass.
    5. If there are not enough non-caption candidates,
       restore skipped captions so recall is not lost.

    This prevents figure captions from dominating evidence
    while still allowing highly relevant captions through.
    """

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    if not results:

        return []


    # -----------------------------------------------------
    # Score all candidates.
    # -----------------------------------------------------

    scored = [

        score_candidate(
            result=result,
            fallback_rank=rank,
        )

        for rank, result in enumerate(
            results,
            start=1,
        )
    ]


    # -----------------------------------------------------
    # Sort by refined score.

    # Base RRF score serves as the secondary key.
    # -----------------------------------------------------

    scored.sort(
        key=lambda item: (
            safe_float(
                item.get(
                    "_refined_score"
                )
            ),
            safe_float(
                item.get(
                    "_base_relevance_score"
                )
            ),
        ),
        reverse=True,
    )


    selected: list[
        dict
    ] = []


    deferred_captions: list[
        dict
    ] = []


    caption_count = 0


    # -----------------------------------------------------
    # First pass:
    #
    # choose highest ranked items while respecting the
    # caption limit.
    # -----------------------------------------------------

    for result in scored:

        if len(
            selected
        ) >= top_k:

            break


        caption_like = bool(
            result.get(
                "_caption_like",
                False,
            )
        )


        if caption_like:

            if (
                caption_count
                >= RERANK_MAX_CAPTIONS
            ):

                deferred_captions.append(
                    result
                )

                continue


            caption_count += 1


        selected.append(
            result
        )


    # -----------------------------------------------------
    # Second pass:
    #
    # If fewer than top_k survived, restore the best
    # deferred captions.

    # This prevents the caption cap from reducing recall.
    # -----------------------------------------------------

    if len(
        selected
    ) < top_k:

        for result in deferred_captions:

            if len(
                selected
            ) >= top_k:

                break


            selected.append(
                result
            )


    # -----------------------------------------------------
    # Final sort keeps output deterministic after deferred
    # candidates are restored.
    # -----------------------------------------------------

    selected.sort(
        key=lambda item: (
            safe_float(
                item.get(
                    "_refined_score"
                )
            ),
            safe_float(
                item.get(
                    "_base_relevance_score"
                )
            ),
        ),
        reverse=True,
    )


    return selected[
        :top_k
    ]


# =========================================================
# Refined hybrid retrieval
# =========================================================


def refined_hybrid_search(
    query_text: str,
    query_vector,
    top_k: int,
) -> list[dict]:
    """
    Production Milestone 5B retrieval.

    Instead of asking LanceDB for exactly top_k results,
    retrieve a larger candidate pool and refine it locally.
    """

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than 0."
        )


    candidate_k = max(
        top_k,
        (
            top_k
            * RETRIEVAL_CANDIDATE_MULTIPLIER
        ),
    )


    candidates = (
        hybrid_search(
            query_text=(
                query_text
            ),
            query_vector=(
                query_vector
            ),
            top_k=(
                candidate_k
            ),
        )
    )


    if not RERANK_ENABLED:

        return candidates[
            :top_k
        ]


    return (
        refine_hybrid_results(
            results=(
                candidates
            ),
            top_k=(
                top_k
            ),
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
    Unified retrieval API used by rag.py.

    vector
        Semantic vector retrieval only.

    hybrid
        Vector + BM25 + RRF + research-aware refinement.
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
            refined_hybrid_search(
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