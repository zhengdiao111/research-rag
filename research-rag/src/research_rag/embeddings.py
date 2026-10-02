from __future__ import annotations

import re

from collections.abc import Sequence

import httpx
import numpy as np

from .config import (
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    EMBED_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)


# =========================================================
# Nomic retrieval prefixes
# =========================================================


DOCUMENT_PREFIX = "search_document: "

QUERY_PREFIX = "search_query: "


# =========================================================
# Exceptions
# =========================================================


class OllamaEmbeddingError(
    RuntimeError
):
    """
    General Ollama embedding failure.
    """


class EmbeddingInputTooLongError(
    OllamaEmbeddingError
):
    """
    Raised when one input exceeds the embedding model's
    native context window.
    """


# =========================================================
# Vector utilities
# =========================================================


def l2_normalize(
    vector: Sequence[float],
) -> list[float]:
    """
    L2-normalize one embedding vector.
    """

    array = np.asarray(
        vector,
        dtype=np.float32,
    )

    if array.ndim != 1:

        raise ValueError(
            "Embedding vector must be one-dimensional."
        )

    norm = float(
        np.linalg.norm(
            array
        )
    )

    if norm == 0.0:

        raise OllamaEmbeddingError(
            "Received a zero-length embedding vector."
        )

    return (
        array / norm
    ).tolist()


def weighted_pool_vectors(
    vectors: Sequence[
        Sequence[float]
    ],
    weights: Sequence[float],
) -> list[float]:
    """
    Combine several sub-chunk embeddings into one vector.

    This is used only as a fallback when a single original
    chunk exceeds the embedding model's context window.

    Longer sub-chunks contribute proportionally more weight.
    """

    if not vectors:

        raise ValueError(
            "Cannot pool zero vectors."
        )

    if len(vectors) != len(
        weights
    ):

        raise ValueError(
            "Vector count must match weight count."
        )

    matrix = np.asarray(
        vectors,
        dtype=np.float32,
    )

    weight_array = np.asarray(
        weights,
        dtype=np.float32,
    )

    if np.any(
        weight_array <= 0
    ):

        raise ValueError(
            "Pooling weights must be positive."
        )

    pooled = np.average(
        matrix,
        axis=0,
        weights=weight_array,
    )

    return l2_normalize(
        pooled
    )


# =========================================================
# Text splitting
# =========================================================


def split_text_near_middle(
    text: str,
) -> tuple[str, str]:
    """
    Split text into two approximately equal pieces.

    Preferred boundaries:

    1. paragraph boundary
    2. sentence boundary
    3. whitespace
    4. hard character midpoint

    This function is used only when Ollama explicitly tells
    us the input exceeds the model context.
    """

    text = text.strip()

    if len(text) < 2:

        raise EmbeddingInputTooLongError(
            "Embedding input is too long but cannot "
            "be split further."
        )

    middle = (
        len(text) // 2
    )

    lower = int(
        len(text) * 0.20
    )

    upper = int(
        len(text) * 0.80
    )


    patterns = [

        # Paragraphs
        r"\n\s*\n",

        # Sentences
        r"(?<=[.!?])\s+",

        # Any whitespace
        r"\s+",
    ]


    for pattern in patterns:

        candidates: list[int] = []

        for match in re.finditer(
            pattern,
            text,
        ):

            position = (
                match.end()
            )

            if (
                lower
                <= position
                <= upper
            ):

                candidates.append(
                    position
                )


        if not candidates:

            continue


        split_position = min(
            candidates,
            key=lambda value: abs(
                value - middle
            ),
        )


        left = text[
            :split_position
        ].strip()

        right = text[
            split_position:
        ].strip()


        if left and right:

            return (
                left,
                right,
            )


    # -----------------------------------------------------
    # Final fallback
    # -----------------------------------------------------

    left = text[
        :middle
    ].strip()

    right = text[
        middle:
    ].strip()


    if not left or not right:

        raise EmbeddingInputTooLongError(
            "Could not split oversized embedding input."
        )


    return (
        left,
        right,
    )


# =========================================================
# Ollama embedding client
# =========================================================


class OllamaEmbedder:
    """
    Local embedding client using Ollama's /api/embed endpoint.

    Supports:

    - batch embedding
    - Nomic task prefixes
    - context-overflow detection
    - recursive safe splitting
    - weighted pooling of oversized chunks
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = EMBED_MODEL,
        timeout_seconds: float = EMBED_TIMEOUT_SECONDS,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.model = model

        self.timeout_seconds = (
            timeout_seconds
        )

        self._dimension: (
            int | None
        ) = None


    # =====================================================
    # Properties
    # =====================================================

    @property
    def dimension(
        self,
    ) -> int | None:

        return self._dimension


    # =====================================================
    # Ollama request
    # =====================================================

    def _request_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        url = (
            f"{self.base_url}"
            "/api/embed"
        )


        payload = {

            "model":
                self.model,

            "input":
                texts,

            # Never silently remove document text.
            "truncate":
                False,
        }


        try:

            response = httpx.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()


        except httpx.ConnectError as exc:

            raise OllamaEmbeddingError(
                "\nCould not connect to Ollama.\n\n"
                f"Expected Ollama at:\n"
                f"{self.base_url}\n\n"
                "Make sure Ollama is running."
            ) from exc


        except httpx.HTTPStatusError as exc:

            body = (
                exc.response.text
            )

            body_lower = (
                body.lower()
            )


            if (
                exc.response.status_code
                == 400

                and (
                    "input length exceeds"
                    in body_lower
                )
            ):

                raise EmbeddingInputTooLongError(
                    body
                ) from exc


            raise OllamaEmbeddingError(
                "\nOllama embedding request failed.\n\n"
                f"Status: "
                f"{exc.response.status_code}\n\n"
                f"Response:\n"
                f"{body}"
            ) from exc


        except httpx.HTTPError as exc:

            raise OllamaEmbeddingError(
                f"Ollama HTTP error:\n{exc}"
            ) from exc


        try:

            data = (
                response.json()
            )

        except ValueError as exc:

            raise OllamaEmbeddingError(
                "Ollama returned invalid JSON."
            ) from exc


        vectors = data.get(
            "embeddings"
        )


        if not isinstance(
            vectors,
            list,
        ):

            raise OllamaEmbeddingError(
                "Ollama response did not contain "
                "an 'embeddings' list."
            )


        if len(vectors) != len(
            texts
        ):

            raise OllamaEmbeddingError(
                "Embedding count does not match input count.\n"
                f"Inputs: {len(texts)}\n"
                f"Vectors: {len(vectors)}"
            )


        return vectors


    # =====================================================
    # Raw embedding
    # =====================================================

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Embed already-prepared strings.

        This is the low-level method.

        For RAG indexing, prefer:

            embed_document_batches()

        For search queries, prefer:

            embed_query()
        """

        cleaned = [
            str(text).strip()
            for text in texts
        ]


        if not cleaned:

            return []


        for index, text in enumerate(
            cleaned
        ):

            if not text:

                raise ValueError(
                    "Cannot embed empty text. "
                    f"Input index: {index}"
                )


        raw_vectors = (
            self._request_embeddings(
                cleaned
            )
        )


        vectors: list[
            list[float]
        ] = []


        for raw_vector in raw_vectors:

            normalized = (
                l2_normalize(
                    raw_vector
                )
            )


            dimension = len(
                normalized
            )


            if self._dimension is None:

                self._dimension = (
                    dimension
                )


            elif (
                dimension
                != self._dimension
            ):

                raise OllamaEmbeddingError(
                    "Embedding dimension changed "
                    "within the same model session.\n"
                    f"Expected: "
                    f"{self._dimension}\n"
                    f"Received: "
                    f"{dimension}"
                )


            vectors.append(
                normalized
            )


        return vectors


    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Embed one already-prepared input string.
        """

        return self.embed_texts(
            [text]
        )[0]


    # =====================================================
    # Safe overflow handling
    # =====================================================

    def _embed_one_safe(
        self,
        text: str,
        prefix: str,
        depth: int = 0,
        max_depth: int = 12,
    ) -> list[float]:
        """
        Embed one logical chunk.

        If it exceeds the model context:

        - split it in two
        - embed both pieces recursively
        - combine them into one normalized vector

        This preserves one LanceDB row per original RAG chunk.
        """

        text = text.strip()


        if not text:

            raise ValueError(
                "Cannot embed empty text."
            )


        prepared = (
            prefix
            + text
        )


        try:

            return self.embed_text(
                prepared
            )


        except EmbeddingInputTooLongError:

            if depth >= max_depth:

                raise EmbeddingInputTooLongError(
                    "Exceeded maximum recursive split depth "
                    "while embedding an oversized chunk."
                )


            left, right = (
                split_text_near_middle(
                    text
                )
            )


            if depth == 0:

                print(
                    "  Oversized chunk detected."
                )

                print(
                    f"  Characters: {len(text):,}"
                )

                print(
                    "  Splitting safely for embedding..."
                )


            left_vector = (
                self._embed_one_safe(
                    left,
                    prefix=prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )


            right_vector = (
                self._embed_one_safe(
                    right,
                    prefix=prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )


            return weighted_pool_vectors(

                vectors=[
                    left_vector,
                    right_vector,
                ],

                weights=[
                    max(
                        len(left),
                        1,
                    ),

                    max(
                        len(right),
                        1,
                    ),
                ],
            )


    def _embed_batch_safe(
        self,
        texts: list[str],
        prefix: str,
        absolute_start: int,
    ) -> list[list[float]]:
        """
        Embed a batch.

        Fast path:
            send the entire batch.

        Fallback:
            if Ollama reports that at least one input is too
            long, isolate the inputs and safely embed them
            one-by-one.
        """

        prepared = [
            prefix + text.strip()
            for text in texts
        ]


        try:

            return self.embed_texts(
                prepared
            )


        except EmbeddingInputTooLongError:

            print(
                "Batch contains at least one input "
                "that exceeds the embedding context."
            )

            print(
                "Retrying this batch individually..."
            )


            vectors: list[
                list[float]
            ] = []


            for local_index, text in enumerate(
                texts
            ):

                chunk_number = (
                    absolute_start
                    + local_index
                    + 1
                )


                try:

                    vector = (
                        self.embed_text(
                            prefix
                            + text.strip()
                        )
                    )


                except EmbeddingInputTooLongError:

                    print()

                    print(
                        f"Chunk {chunk_number} "
                        f"is over the model context."
                    )


                    vector = (
                        self._embed_one_safe(
                            text,
                            prefix=prefix,
                        )
                    )


                vectors.append(
                    vector
                )


            return vectors


    # =====================================================
    # RAG document embeddings
    # =====================================================

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Embed document passages using Nomic's required
        search_document prefix.
        """

        texts = list(
            texts
        )


        return self._embed_batch_safe(

            texts=texts,

            prefix=DOCUMENT_PREFIX,

            absolute_start=0,
        )


    def embed_document_batches(
        self,
        texts: Sequence[str],
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> list[list[float]]:
        """
        Embed document chunks in batches.

        Handles individual context-overflow chunks
        automatically.
        """

        texts = list(
            texts
        )


        if not texts:

            return []


        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than 0."
            )


        all_vectors: list[
            list[float]
        ] = []


        total = len(
            texts
        )


        for start in range(
            0,
            total,
            batch_size,
        ):

            stop = min(
                start + batch_size,
                total,
            )


            print(
                f"Embedding chunks "
                f"{start + 1}-{stop} "
                f"of {total}..."
            )


            batch = texts[
                start:stop
            ]


            vectors = (
                self._embed_batch_safe(

                    texts=batch,

                    prefix=DOCUMENT_PREFIX,

                    absolute_start=start,
                )
            )


            all_vectors.extend(
                vectors
            )


        return all_vectors


    # =====================================================
    # RAG query embeddings
    # =====================================================

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Embed a retrieval query using Nomic's required
        search_query prefix.
        """

        query = str(
            query
        ).strip()


        if not query:

            raise ValueError(
                "Query cannot be empty."
            )


        return self._embed_one_safe(

            text=query,

            prefix=QUERY_PREFIX,
        )


    # =====================================================
    # Backwards-compatible alias
    # =====================================================

    def embed_batches(
        self,
        texts: Sequence[str],
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> list[list[float]]:
        """
        Backwards-compatible alias.

        Existing indexing code calling embed_batches()
        will now automatically use document retrieval
        embeddings.
        """

        return self.embed_document_batches(
            texts=texts,
            batch_size=batch_size,
        )