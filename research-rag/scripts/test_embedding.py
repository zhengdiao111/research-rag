from __future__ import annotations

import numpy as np

from research_rag.config import (
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)

from research_rag.embeddings import (
    OllamaEmbedder,
)


def main():

    print()
    print("=" * 80)

    print(
        "Research RAG — Embedding test"
    )

    print("=" * 80)

    print()

    print(
        f"Ollama: {OLLAMA_BASE_URL}"
    )

    print(
        f"Model:  {EMBED_MODEL}"
    )

    print()


    samples = [

        (
            "Topological defects can control "
            "the shape of cellular tissues."
        ),

        (
            "Cells can organize into "
            "three-dimensional structures."
        ),

        (
            "Chocolate cake is made with "
            "flour, sugar, and cocoa."
        ),
    ]


    embedder = (
        OllamaEmbedder()
    )


    vectors = (
        embedder.embed_texts(
            samples
        )
    )


    print(
        f"Embeddings returned: "
        f"{len(vectors)}"
    )

    print(
        f"Embedding dimension: "
        f"{embedder.dimension}"
    )

    print()


    for index, vector in enumerate(
        vectors,
        start=1,
    ):

        norm = float(
            np.linalg.norm(
                np.asarray(
                    vector,
                    dtype=np.float32,
                )
            )
        )


        print(
            f"Vector {index}:"
        )

        print(
            f"  shape = "
            f"({len(vector)},)"
        )

        print(
            f"  L2 norm = "
            f"{norm:.6f}"
        )


    # -----------------------------------------------------
    # Simple sanity check
    # -----------------------------------------------------

    first = np.asarray(
        vectors[0],
        dtype=np.float32,
    )

    second = np.asarray(
        vectors[1],
        dtype=np.float32,
    )

    third = np.asarray(
        vectors[2],
        dtype=np.float32,
    )


    biological_similarity = float(
        np.dot(
            first,
            second,
        )
    )


    unrelated_similarity = float(
        np.dot(
            first,
            third,
        )
    )


    print()

    print(
        "Similarity sanity check:"
    )

    print()

    print(
        "  biological pair: "
        f"{biological_similarity:.4f}"
    )

    print(
        "  unrelated pair:  "
        f"{unrelated_similarity:.4f}"
    )

    print()


    if (
        biological_similarity
        > unrelated_similarity
    ):

        print(
            "PASS: related biological text "
            "is more similar."
        )

    else:

        print(
            "WARNING: semantic similarity "
            "did not behave as expected."
        )


    print()
    print("=" * 80)


if __name__ == "__main__":
    main()