from __future__ import annotations



import hashlib



from dataclasses import dataclass, field

from pathlib import Path





from .chunker import (

    chunk_document,

)



from .config import (

    DOCUMENT_ROOT,

)



from .database import (

    open_semantic_table,

)



from .embeddings import (

    OllamaEmbedder,

)



from .indexer import (

    is_searchable_chunk,

    make_record,

)


from .index_manifest import (

    ensure_index_manifest_compatible,

)



from .parser import (

    discover_pdfs,

    parse_pdf,

)





# =========================================================

# Data structures

# =========================================================





@dataclass

class IndexedDocumentState:

    """

    Existing document state reconstructed from LanceDB.

    """



    normalized_path: str



    source_paths: set[str] = field(

        default_factory=set

    )



    sha256s: set[str] = field(

        default_factory=set

    )



    row_count: int = 0





@dataclass

class CurrentDocumentState:

    """

    Current filesystem state for one PDF.

    """



    path: Path



    normalized_path: str



    sha256: str





@dataclass

class SyncPlan:

    """

    Classification of the filesystem relative to the

    existing LanceDB index.

    """



    new_files: list[CurrentDocumentState]



    changed_files: list[CurrentDocumentState]



    unchanged_files: list[CurrentDocumentState]



    deleted_files: list[IndexedDocumentState]





@dataclass

class SyncSummary:

    """

    User-facing summary of one indexing sync.

    """



    scanned_pdfs: int



    new_files: list[str]



    changed_files: list[str]



    unchanged_files: list[str]



    deleted_detected: list[str]



    indexed_files: list[str]



    deleted_removed: list[str]



    chunks_generated: int



    searchable_chunks_written: int



    optimized: bool



    optimize_error: str | None



    dry_run: bool





# =========================================================

# Path normalization

# =========================================================





def normalize_path(

    path: str | Path,

) -> str:

    """

    Normalize a path for comparison.



    Windows paths are case-insensitive, so casefold() is

    used to avoid treating:



        C:\\Research\\Paper.pdf



    and:



        c:\\research\\paper.pdf



    as different documents.

    """



    resolved = (

        Path(path)

        .expanduser()

        .resolve(strict=False)

    )





    return (

        str(resolved)

        .casefold()

    )





# =========================================================

# SHA-256

# =========================================================





def file_sha256(

    path: Path,

    block_size: int = 1024 * 1024,

) -> str:

    """

    Compute a streaming SHA-256 hash without parsing the PDF.



    This is intentionally much cheaper than running the

    complete PDF extraction/chunking pipeline.

    """



    digest = (

        hashlib.sha256()

    )





    with path.open(

        "rb"

    ) as handle:



        while True:



            block = (

                handle.read(

                    block_size

                )

            )





            if not block:



                break





            digest.update(

                block

            )





    return (

        digest.hexdigest()

    )





# =========================================================

# Existing LanceDB state

# =========================================================





def load_indexed_document_state(

    table,

) -> dict[str, IndexedDocumentState]:

    """

    Reconstruct document-level state from the existing

    chunk-level LanceDB table.



    Only source_path and source_sha256 are read.



    The vector column is deliberately not materialized.

    """



    dataset = (

        table.to_lance()

    )





    arrow_table = (

        dataset.to_table(

            columns=[

                "source_path",

                "source_sha256",

            ]

        )

    )





    states: dict[

        str,

        IndexedDocumentState,

    ] = {}





    for row in (

        arrow_table.to_pylist()

    ):



        raw_path = (

            row.get(

                "source_path"

            )

        )



        raw_sha256 = (

            row.get(

                "source_sha256"

            )

        )





        if not raw_path:



            continue





        source_path = (

            str(

                raw_path

            )

        )





        normalized = (

            normalize_path(

                source_path

            )

        )





        state = (

            states.get(

                normalized

            )

        )





        if state is None:



            state = (

                IndexedDocumentState(

                    normalized_path=(

                        normalized

                    ),

                )

            )



            states[

                normalized

            ] = state





        state.source_paths.add(

            source_path

        )





        if raw_sha256:



            state.sha256s.add(

                str(

                    raw_sha256

                )

                .strip()

                .lower()

            )





        state.row_count += 1





    return states





# =========================================================

# Current filesystem state

# =========================================================





def scan_current_library(

    document_root: str | Path = DOCUMENT_ROOT,

) -> dict[str, CurrentDocumentState]:

    """

    Discover every PDF and calculate its SHA-256.



    PDFs are NOT parsed here.

    """



    root = (

        Path(

            document_root

        )

        .expanduser()

        .resolve()

    )





    if not root.exists():



        raise FileNotFoundError(

            "Research document root does not exist:\n"

            f"{root}"

        )





    if not root.is_dir():



        raise NotADirectoryError(

            "Research document root is not a directory:\n"

            f"{root}"

        )





    pdf_paths = sorted(

        discover_pdfs(

            root

        ),

        key=lambda path: (

            str(path)

            .casefold()

        ),

    )





    current: dict[

        str,

        CurrentDocumentState,

    ] = {}





    for path in pdf_paths:



        path = (

            Path(path)

            .expanduser()

            .resolve()

        )





        normalized = (

            normalize_path(

                path

            )

        )





        current[

            normalized

        ] = (

            CurrentDocumentState(



                path=path,



                normalized_path=(

                    normalized

                ),



                sha256=(

                    file_sha256(

                        path

                    )

                ),

            )

        )





    return current





# =========================================================

# Sync planning

# =========================================================





def build_sync_plan(

    table,

    document_root: str | Path = DOCUMENT_ROOT,

) -> SyncPlan:

    """

    Compare filesystem state against indexed state.



    Classifications:



        NEW

            path not currently in LanceDB



        CHANGED

            path exists but file hash changed



        UNCHANGED

            path exists and hash is identical



        DELETED

            LanceDB path no longer exists under the

            document root

    """



    indexed = (

        load_indexed_document_state(

            table

        )

    )





    current = (

        scan_current_library(

            document_root

        )

    )





    new_files: list[

        CurrentDocumentState

    ] = []





    changed_files: list[

        CurrentDocumentState

    ] = []





    unchanged_files: list[

        CurrentDocumentState

    ] = []





    deleted_files: list[

        IndexedDocumentState

    ] = []





    # =====================================================

    # Current files

    # =====================================================



    for (

        normalized,

        current_state,

    ) in current.items():



        indexed_state = (

            indexed.get(

                normalized

            )

        )





        if indexed_state is None:



            new_files.append(

                current_state

            )



            continue





        # -------------------------------------------------

        # One path should normally have exactly one hash.

        #

        # Multiple historical hashes indicate an

        # inconsistent state and should force replacement.

        # -------------------------------------------------



        if (

            indexed_state.sha256s

            == {

                current_state.sha256

            }

        ):



            unchanged_files.append(

                current_state

            )





        else:



            changed_files.append(

                current_state

            )





    # =====================================================

    # Indexed paths missing from filesystem

    # =====================================================



    for (

        normalized,

        indexed_state,

    ) in indexed.items():



        if (

            normalized

            not in current

        ):



            deleted_files.append(

                indexed_state

            )





    return SyncPlan(



        new_files=(

            new_files

        ),



        changed_files=(

            changed_files

        ),



        unchanged_files=(

            unchanged_files

        ),



        deleted_files=(

            deleted_files

        ),

    )





# =========================================================

# SQL-string escaping

# =========================================================





def sql_string_literal(

    value: str,

) -> str:

    """

    Quote a Python string for a Lance SQL filter.

    """



    escaped = (

        value.replace(

            "'",

            "''",

        )

    )





    return (

        f"'{escaped}'"

    )





def source_path_filter(

    source_paths: set[str] | list[str],

) -> str:

    """

    Build:



        source_path = '...' OR source_path = '...'

    """



    paths = sorted(

        set(

            source_paths

        )

    )





    if not paths:



        raise ValueError(

            "source_paths cannot be empty."

        )





    expressions = [



        (

            "source_path = "

            + sql_string_literal(

                path

            )

        )



        for path in paths

    ]





    return (

        "("

        + " OR ".join(

            expressions

        )

        + ")"

    )





# =========================================================

# Document preparation

# =========================================================





def prepare_document_records(

    current_state: CurrentDocumentState,

    embedder: OllamaEmbedder,

) -> tuple[

    list[dict],

    int,

    int,

]:

    """

    Parse, chunk, filter, and embed ONE PDF.



    Nothing is written to LanceDB until this function

    finishes successfully.

    """



    path = (

        current_state.path

    )





    # =====================================================

    # Parse

    # =====================================================



    document = (

        parse_pdf(

            path

        )

    )





    parsed_sha256 = (

        str(

            document.sha256

        )

        .strip()

        .lower()

    )





    # -----------------------------------------------------

    # Protect against a file changing while indexing.

    # -----------------------------------------------------



    if (

        parsed_sha256

        != current_state.sha256

    ):



        raise RuntimeError(

            "PDF changed while it was being indexed:\n"

            f"{path}\n\n"

            "No database change was made for this file."

        )





    # =====================================================

    # Chunk

    # =====================================================



    all_chunks = (

        chunk_document(

            document

        )

    )





    searchable_chunks = [



        chunk



        for chunk in all_chunks



        if (

            is_searchable_chunk(

                chunk

            )

        )

    ]





    if not searchable_chunks:



        raise RuntimeError(

            "No searchable chunks were produced for:\n"

            f"{path}\n\n"

            "The existing indexed copy, if any, was "

            "left untouched."

        )





    # =====================================================

    # Embed

    # =====================================================



    texts = [



        chunk.text



        for chunk in searchable_chunks

    ]





    vectors = (

        embedder.embed_document_batches(

            texts

        )

    )





    if (

        len(vectors)

        != len(searchable_chunks)

    ):



        raise RuntimeError(

            "Embedding count did not match chunk count "

            f"for:\n{path}"

        )





    # =====================================================

    # Build LanceDB records

    # =====================================================



    records = [



        make_record(

            document=(

                document

            ),



            chunk=(

                chunk

            ),



            vector=(

                vector

            ),

        )



        for (

            chunk,

            vector,

        ) in zip(

            searchable_chunks,

            vectors,

            strict=True,

        )

    ]





    return (

        records,

        len(

            all_chunks

        ),

        len(

            searchable_chunks

        ),

    )





# =========================================================

# New-document insertion

# =========================================================





def add_new_document(

    table,

    records: list[dict],

) -> None:

    """

    Add one entirely new PDF.



    Processing and embedding must already have succeeded.

    """



    if not records:



        raise ValueError(

            "Cannot add an empty record set."

        )





    table.add(

        records

    )





# =========================================================

# Changed-document atomic replacement

# =========================================================





def replace_changed_document(

    table,

    records: list[dict],

    old_state: IndexedDocumentState,

) -> None:

    """

    Atomically replace the rows belonging to one changed

    PDF.



    The new records have already been fully generated before

    this operation begins.



    merge_insert performs:



        matching chunk IDs    -> update

        new chunk IDs         -> insert

        old missing chunk IDs -> delete, but ONLY for this

                                 document's source_path

    """



    if not records:



        raise ValueError(

            "Cannot replace a document with zero records."

        )





    delete_scope = (

        source_path_filter(

            old_state.source_paths

        )

    )





    (

        table

        .merge_insert(

            "chunk_id"

        )

        .when_matched_update_all()

        .when_not_matched_insert_all()

        .when_not_matched_by_source_delete(

            condition=(

                delete_scope

            )

        )

        .execute(

            records

        )

    )





# =========================================================

# Deleted-document removal

# =========================================================





def remove_deleted_documents(

    table,

    deleted_states: list[IndexedDocumentState],

) -> list[str]:

    """

    Remove stale rows for PDFs no longer present on disk.



    This operation is deliberately optional at the public

    API level so a temporarily unavailable NAS/network mount

    cannot automatically erase the index.

    """



    if not deleted_states:



        return []





    all_source_paths: set[

        str

    ] = set()





    for state in deleted_states:



        all_source_paths.update(

            state.source_paths

        )





    if not all_source_paths:



        return []





    where = (

        source_path_filter(

            all_source_paths

        )

    )





    table.delete(

        where

    )





    return sorted(

        all_source_paths

    )





# =========================================================

# Main incremental sync

# =========================================================





def sync_semantic_index(

    document_root: str | Path = DOCUMENT_ROOT,

    *,

    dry_run: bool = False,

    sync_deletions: bool = False,

    optimize_after_write: bool = True,

    adopt_missing_manifest: bool = False,

) -> SyncSummary:

    """

    Synchronize the current PDF library with LanceDB.



    Default safety behavior:



        new PDFs       -> index

        changed PDFs   -> replace

        unchanged PDFs -> skip

        deleted PDFs   -> report only



    Use sync_deletions=True to remove stale rows for deleted

    PDFs.

    """



    # =====================================================

    # Open existing semantic table

    # =====================================================



    try:



        table = (

            open_semantic_table()

        )





    except Exception as exc:



        raise RuntimeError(

            "Could not open the existing semantic table.\n\n"

            "Run the normal full indexing pipeline once "

            "before using incremental indexing:\n\n"

            "    uv run python scripts\\index_pdfs.py\n"

        ) from exc





    # =====================================================

    # Index-pipeline compatibility

    # =====================================================



    ensure_index_manifest_compatible(

        adopt_missing_manifest=(

            adopt_missing_manifest

        ),

        # A dry run must never write files.

        persist_adoption=(

            not dry_run

        ),

    )



    # =====================================================

    # Plan

    # =====================================================



    plan = (

        build_sync_plan(

            table=(

                table

            ),



            document_root=(

                document_root

            ),

        )

    )





    new_paths = [



        str(

            item.path

        )



        for item in plan.new_files

    ]





    changed_paths = [



        str(

            item.path

        )



        for item in plan.changed_files

    ]





    unchanged_paths = [



        str(

            item.path

        )



        for item in plan.unchanged_files

    ]





    deleted_paths = sorted({



        source_path



        for state in plan.deleted_files



        for source_path in state.source_paths

    })





    # =====================================================

    # Dry run

    # =====================================================



    if dry_run:



        return SyncSummary(



            scanned_pdfs=(

                len(

                    new_paths

                )

                + len(

                    changed_paths

                )

                + len(

                    unchanged_paths

                )

            ),



            new_files=(

                new_paths

            ),



            changed_files=(

                changed_paths

            ),



            unchanged_files=(

                unchanged_paths

            ),



            deleted_detected=(

                deleted_paths

            ),



            indexed_files=[],



            deleted_removed=[],



            chunks_generated=0,



            searchable_chunks_written=0,



            optimized=False,



            optimize_error=None,



            dry_run=True,

        )





    # =====================================================

    # Nothing to modify

    # =====================================================



    modifications_needed = (



        bool(

            plan.new_files

        )



        or bool(

            plan.changed_files

        )



        or (

            sync_deletions

            and bool(

                plan.deleted_files

            )

        )

    )





    if not modifications_needed:



        return SyncSummary(



            scanned_pdfs=(

                len(

                    new_paths

                )

                + len(

                    changed_paths

                )

                + len(

                    unchanged_paths

                )

            ),



            new_files=(

                new_paths

            ),



            changed_files=(

                changed_paths

            ),



            unchanged_files=(

                unchanged_paths

            ),



            deleted_detected=(

                deleted_paths

            ),



            indexed_files=[],



            deleted_removed=[],



            chunks_generated=0,



            searchable_chunks_written=0,



            optimized=False,



            optimize_error=None,



            dry_run=False,

        )





    # =====================================================

    # Initialize embedder only when work is required

    # =====================================================



    embedder = (

        OllamaEmbedder()

    )





    indexed_files: list[

        str

    ] = []





    deleted_removed: list[

        str

    ] = []





    total_chunks_generated = 0



    total_searchable_written = 0





    # =====================================================

    # New PDFs

    # =====================================================



    for current_state in (

        plan.new_files

    ):



        records, chunk_count, searchable_count = (

            prepare_document_records(

                current_state=(

                    current_state

                ),



                embedder=(

                    embedder

                ),

            )

        )





        # ---------------------------------------------

        # No database mutation occurs until the complete

        # document is successfully prepared.

        # ---------------------------------------------



        add_new_document(

            table=(

                table

            ),



            records=(

                records

            ),

        )





        indexed_files.append(

            str(

                current_state.path

            )

        )





        total_chunks_generated += (

            chunk_count

        )





        total_searchable_written += (

            searchable_count

        )





    # =====================================================

    # Changed PDFs

    # =====================================================



    indexed_state = (

        load_indexed_document_state(

            table

        )

    )





    for current_state in (

        plan.changed_files

    ):



        old_state = (

            indexed_state.get(

                current_state.normalized_path

            )

        )





        if old_state is None:



            # ---------------------------------------------

            # Defensive fallback:

            # If state changed between planning and writing,

            # treat this as a new document rather than

            # deleting unrelated data.

            # ---------------------------------------------



            records, chunk_count, searchable_count = (

                prepare_document_records(

                    current_state=(

                        current_state

                    ),



                    embedder=(

                        embedder

                    ),

                )

            )





            add_new_document(

                table=(

                    table

                ),



                records=(

                    records

                ),

            )





        else:



            # ---------------------------------------------

            # Prepare complete replacement FIRST.

            # ---------------------------------------------



            records, chunk_count, searchable_count = (

                prepare_document_records(

                    current_state=(

                        current_state

                    ),



                    embedder=(

                        embedder

                    ),

                )

            )





            # ---------------------------------------------

            # Only now mutate the database.

            # ---------------------------------------------



            replace_changed_document(

                table=(

                    table

                ),



                records=(

                    records

                ),



                old_state=(

                    old_state

                ),

            )





        indexed_files.append(

            str(

                current_state.path

            )

        )





        total_chunks_generated += (

            chunk_count

        )





        total_searchable_written += (

            searchable_count

        )





    # =====================================================

    # Deleted PDFs

    # =====================================================



    if (

        sync_deletions

        and plan.deleted_files

    ):



        deleted_removed = (

            remove_deleted_documents(



                table=(

                    table

                ),



                deleted_states=(

                    plan.deleted_files

                ),

            )

        )





    # =====================================================

    # Refresh / optimize existing indexes

    # =====================================================



    optimized = False



    optimize_error: str | None = None





    if optimize_after_write:



        try:



            table.optimize()



            optimized = True





        except Exception as exc:



            # ---------------------------------------------

            # Data writes already succeeded.

            #

            # Do not falsely report the entire indexing run

            # as failed merely because post-write

            # optimization failed.

            # ---------------------------------------------



            optimize_error = (

                f"{type(exc).__name__}: "

                f"{exc}"

            )





    # =====================================================

    # Summary

    # =====================================================



    return SyncSummary(



        scanned_pdfs=(

            len(

                new_paths

            )

            + len(

                changed_paths

            )

            + len(

                unchanged_paths

            )

        ),



        new_files=(

            new_paths

        ),



        changed_files=(

            changed_paths

        ),



        unchanged_files=(

            unchanged_paths

        ),



        deleted_detected=(

            deleted_paths

        ),



        indexed_files=(

            indexed_files

        ),



        deleted_removed=(

            deleted_removed

        ),



        chunks_generated=(

            total_chunks_generated

        ),



        searchable_chunks_written=(

            total_searchable_written

        ),



        optimized=(

            optimized

        ),



        optimize_error=(

            optimize_error

        ),



        dry_run=False,

    )