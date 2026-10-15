from __future__ import annotations

import argparse

from pathlib import Path


from research_rag.config import (
    DOCUMENT_ROOT,
    INDEX_MANIFEST_PATH,
)

from research_rag.database import (
    open_semantic_table,
)

from research_rag.incremental_indexer import (
    sync_semantic_index,
)

from research_rag.index_manifest import (
    build_current_index_manifest,
)

from research_rag.indexer import (
    build_semantic_index,
)

from research_rag.parser import (
    discover_pdfs,
)


# =========================================================
# CLI
# =========================================================


def parse_arguments():
    """
    Parse index synchronization / rebuild options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Synchronize or fully rebuild the local "
            "research-PDF LanceDB index."
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
            "During normal incremental sync, remove "
            "LanceDB rows for PDFs that no longer exist "
            "under the document root."
        ),
    )


    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help=(
            "Skip LanceDB post-write optimization."
        ),
    )


    mode_group = (
        parser.add_mutually_exclusive_group()
    )


    mode_group.add_argument(
        "--adopt-current-index",
        action="store_true",
        help=(
            "Initialize a missing index manifest using "
            "the current pipeline configuration without "
            "rebuilding the existing index. Use this only "
            "for the one-time migration of an index known "
            "to have been built with the current settings."
        ),
    )


    mode_group.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Fully rebuild the semantic index from all "
            "PDFs using the current parser, chunker, "
            "filters, and embedding model. This mode "
            "intentionally bypasses stale-manifest "
            "blocking because rebuilding is how a stale "
            "index is repaired."
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
# Optimization helper
# =========================================================


def optimize_current_table() -> tuple[
    bool,
    str | None,
]:
    """
    Optimize the active semantic table.

    Optimization failure does not invalidate an otherwise
    successful rebuild or incremental write.
    """

    try:

        table = (
            open_semantic_table()
        )

        table.optimize()

        return (
            True,
            None,
        )


    except Exception as exc:

        return (
            False,
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


# =========================================================
# Rebuild mode
# =========================================================


def run_rebuild(
    args,
    current_manifest,
) -> None:
    """
    Execute or preview a complete semantic-index rebuild.
    """

    root = (
        args.root
        .expanduser()
        .resolve()
    )


    separator()

    print(
        "FULL REBUILD MODE"
    )

    print()

    print(
        "This rebuild intentionally bypasses the current "
        "manifest compatibility gate."
    )

    print()

    print(
        "The existing LanceDB table will not be overwritten "
        "until all PDFs have been parsed, chunked, embedded, "
        "and validated successfully."
    )

    print()


    if args.sync_deletions:

        print(
            "NOTE:"
        )

        print(
            "--sync-deletions is unnecessary during a full "
            "rebuild. The replacement table contains only "
            "PDFs currently present under the document root."
        )

        print()


    # =====================================================
    # Rebuild dry run
    # =====================================================

    if args.dry_run:

        pdfs = sorted(
            discover_pdfs(
                root
            ),
            key=lambda path: (
                str(path)
                .casefold()
            ),
        )


        print(
            f"PDFs that would be rebuilt: "
            f"{len(pdfs)}"
        )

        print(
            f"Target fingerprint: "
            f"{current_manifest.fingerprint}"
        )

        print(
            f"Manifest path: "
            f"{INDEX_MANIFEST_PATH}"
        )


        separator()

        print(
            "DRY RUN — no PDFs were parsed, no embeddings "
            "were generated, and no database or manifest "
            "changes were made."
        )

        separator()

        return


    # =====================================================
    # Build complete replacement
    # =====================================================

    summary = (
        build_semantic_index(
            document_root=(
                root
            )
        )
    )


    # =====================================================
    # Optional optimization
    # =====================================================

    optimized = False

    optimize_error: str | None = None


    if not args.no_optimize:

        (
            optimized,
            optimize_error,
        ) = (
            optimize_current_table()
        )


    # =====================================================
    # Rebuild summary
    # =====================================================

    separator()

    print(
        "FULL REBUILD COMPLETE"
    )

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

    print(
        f"LanceDB optimized:    "
        f"{optimized}"
    )

    print()

    print(
        f"Manifest fingerprint: "
        f"{current_manifest.fingerprint}"
    )

    print(
        f"Manifest path:        "
        f"{INDEX_MANIFEST_PATH}"
    )


    if optimize_error:

        print()

        print(
            "Optimization warning:"
        )

        print(
            optimize_error
        )


    separator()


# =========================================================
# Incremental mode
# =========================================================


def run_incremental(
    args,
) -> None:
    """
    Execute the existing incremental synchronization path.
    """

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

            adopt_missing_manifest=(
                args.adopt_current_index
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


# =========================================================
# Main
# =========================================================


def main():

    args = (
        parse_arguments()
    )


    current_manifest = (
        build_current_index_manifest()
    )


    separator()

    print(
        "Research RAG — Index Management"
    )

    print()

    print(
        f"Root: "
        f"{args.root.expanduser().resolve()}"
    )

    print(
        f"Dry run: "
        f"{args.dry_run}"
    )

    print(
        f"Rebuild: "
        f"{args.rebuild}"
    )

    print(
        f"Sync deletions: "
        f"{args.sync_deletions}"
    )

    print(
        f"Optimize after write: "
        f"{not args.no_optimize}"
    )

    print(
        f"Adopt current index: "
        f"{args.adopt_current_index}"
    )

    print(
        f"Pipeline fingerprint: "
        f"{current_manifest.fingerprint[:16]}..."
    )


    if args.rebuild:

        run_rebuild(
            args=(
                args
            ),
            current_manifest=(
                current_manifest
            ),
        )

        return


    run_incremental(
        args
    )


if __name__ == "__main__":

    main()
