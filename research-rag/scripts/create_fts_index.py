from __future__ import annotations

from lancedb.index import FTS

from research_rag.config import (
    FTS_COLUMN,
)

from research_rag.database import (
    open_semantic_table,
)


def main():

    print()

    print(
        "=" * 80
    )

    print(
        "Research RAG — Create FTS Index"
    )

    print(
        "=" * 80
    )

    print()


    table = (
        open_semantic_table()
    )


    print(
        f"Creating FTS index on column: "
        f"{FTS_COLUMN}"
    )

    print()


    config = FTS(

        base_tokenizer="simple",

        language="English",

        lower_case=True,

        stem=True,

        remove_stop_words=True,

        ascii_folding=True,

        with_position=False,
    )


    table.create_index(
        FTS_COLUMN,
        config=config,
        replace=True,
    )


    print(
        "FTS index created."
    )

    print()

    print(
        "Available indices:"
    )


    for index in table.list_indices():

        print(
            f"  {index}"
        )


    print()

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()