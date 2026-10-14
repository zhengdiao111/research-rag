from __future__ import annotations

import argparse

from pathlib import Path


from research_rag.config import (
    DOCUMENT_ROOT,
)

from research_rag.incremental_indexer import (
    sync_semantic_index,
)


# =========================================================
# CLI
# =========================================================


def parse_arguments():
    """
    Parse incremental-indexing options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Incrementally synchronize the local research "
            "PDF library with LanceDB."
        )
    )


    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            DOCUMENT_ROOT
        ),
        help=(
            "Research PDF root directory."
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what would change without parsing, "
            "embedding, or modifying LanceDB."
        ),
    )


    parser.add_argument(
        "--sync-deletions",
        action="store_true",
        help=(
            "Remove LanceDB rows for PDFs that no longer "
            "exist under the document root."
        ),
    )


    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help=(
            "Skip LanceDB post-write optimization."
        ),
    )


    return (
        parser.parse_args()
    )


# =========================================================
# Display helpers
# =========================================================


def separator():

    print()

    print(
        "=" * 88
    )

    print()


def print_file_group(
    title: str,
    paths: list[str],
):
    """
    Print one classification group.
    """

    print(
        f"{title}: "
        f"{len(paths)}"
    )


    for path in paths:

        print(
            f"  - {path}"
        )


# =========================================================
# Main
# =========================================================


def main():

    args = (
        parse_arguments()
    )


    separator()

    print(
        "Research RAG — Incremental Index Sync"
    )

    print()

    print(
        f"Root: "
        f"{args.root.resolve()}"
    )

    print(
        f"Dry run: "
        f"{args.dry_run}"
    )

    print(
        f"Sync deletions: "
        f"{args.sync_deletions}"
    )

    print(
        f"Optimize after write: "
        f"{not args.no_optimize}"
    )


    print()

    print(
        "Scanning PDF library..."
    )


    summary = (
        sync_semantic_index(

            document_root=(
                args.root
            ),

            dry_run=(
                args.dry_run
            ),

            sync_deletions=(
                args.sync_deletions
            ),

            optimize_after_write=(
                not args.no_optimize
            ),
        )
    )


    separator()

    print(
        "SYNC PLAN / RESULT"
    )

    print()

    print(
        f"PDFs scanned: "
        f"{summary.scanned_pdfs}"
    )

    print()


    print_file_group(
        "NEW",
        summary.new_files,
    )

    print()


    print_file_group(
        "CHANGED",
        summary.changed_files,
    )

    print()


    print(
        f"UNCHANGED: "
        f"{len(summary.unchanged_files)}"
    )

    print()


    print_file_group(
        "DELETED FROM DISK",
        summary.deleted_detected,
    )


    if summary.dry_run:

        separator()

        print(
            "DRY RUN — no database changes were made."
        )

        separator()

        return


    separator()

    print(
        "WRITE SUMMARY"
    )

    print()

    print_file_group(
        "Indexed / replaced",
        summary.indexed_files,
    )

    print()

    print_file_group(
        "Deleted from LanceDB",
        summary.deleted_removed,
    )

    print()

    print(
        f"Chunks generated:          "
        f"{summary.chunks_generated}"
    )

    print(
        f"Searchable chunks written: "
        f"{summary.searchable_chunks_written}"
    )

    print(
        f"LanceDB optimized:         "
        f"{summary.optimized}"
    )


    if summary.optimize_error:

        print()

        print(
            "Optimization warning:"
        )

        print(
            summary.optimize_error
        )


    if (
        summary.deleted_detected
        and not args.sync_deletions
    ):

        print()

        print(
            "NOTE:"
        )

        print(
            "Deleted PDFs were detected but their indexed "
            "rows were intentionally preserved."
        )

        print(
            "Use --sync-deletions when you intentionally "
            "want stale document rows removed."
        )


    separator()


if __name__ == "__main__":

    main()