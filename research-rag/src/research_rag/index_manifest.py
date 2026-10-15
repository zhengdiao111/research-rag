from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .config import (
    ALLOW_CROSS_PAGE_CHUNKS,
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    CHUNKER_PIPELINE_VERSION,
    EMBED_MODEL,
    INDEX_MANIFEST_FORMAT_VERSION,
    INDEX_MANIFEST_PATH,
    INDEX_RECORD_SCHEMA_VERSION,
    OCR_LANGUAGE,
    PARSER_PIPELINE_VERSION,
    SEARCHABLE_CHUNK_TYPES,
    SEMANTIC_TABLE_NAME,
    TOKENIZER_ENCODING,
    USE_OCR,
)


# =========================================================
# Data structure
# =========================================================


@dataclass(
    frozen=True
)
class IndexManifest:
    """
    Reproducibility metadata describing how the semantic
    index was produced.

    If any scientifically meaningful indexing parameter
    changes, the fingerprint should also change.
    """

    manifest_format_version: int

    record_schema_version: int

    semantic_table_name: str

    parser_version: str

    chunker_version: str

    settings: dict[
        str,
        Any,
    ]

    fingerprint: str


    def core_dict(
        self,
    ) -> dict:
        """
        Return the fingerprinted portion of the manifest.

        The fingerprint itself is excluded.
        """

        return {

            "manifest_format_version":
                self.manifest_format_version,

            "record_schema_version":
                self.record_schema_version,

            "semantic_table_name":
                self.semantic_table_name,

            "parser_version":
                self.parser_version,

            "chunker_version":
                self.chunker_version,

            "settings":
                self.settings,
        }


    def to_dict(
        self,
    ) -> dict:

        payload = (
            self.core_dict()
        )

        payload[
            "fingerprint"
        ] = self.fingerprint

        return payload


# =========================================================
# Canonical hashing
# =========================================================


def canonical_json(
    payload: dict,
) -> str:
    """
    Serialize deterministically so the same configuration
    always generates the same SHA-256.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


def fingerprint_payload(
    payload: dict,
) -> str:
    """
    Generate the pipeline fingerprint.
    """

    serialized = (
        canonical_json(
            payload
        )
    )

    return (
        hashlib
        .sha256(
            serialized.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# =========================================================
# Current pipeline description
# =========================================================


def current_index_settings(
) -> dict:
    """
    Settings that materially affect parsed text, chunking,
    filtering, or embeddings.

    Runtime-only performance settings such as batch size and
    timeout are intentionally excluded.
    """

    return {

        # ---------------------------------------------
        # PDF parsing
        # ---------------------------------------------

        "use_ocr":
            bool(
                USE_OCR
            ),

        "ocr_language":
            str(
                OCR_LANGUAGE
            ),


        # ---------------------------------------------
        # Chunking
        # ---------------------------------------------

        "tokenizer_encoding":
            str(
                TOKENIZER_ENCODING
            ),

        "chunk_min":
            int(
                CHUNK_MIN
            ),

        "chunk_target":
            int(
                CHUNK_TARGET
            ),

        "chunk_max":
            int(
                CHUNK_MAX
            ),

        "chunk_overlap":
            int(
                CHUNK_OVERLAP
            ),

        "allow_cross_page_chunks":
            bool(
                ALLOW_CROSS_PAGE_CHUNKS
            ),


        # ---------------------------------------------
        # Chunk filtering
        # ---------------------------------------------

        "searchable_chunk_types":
            sorted(
                str(item)
                for item
                in SEARCHABLE_CHUNK_TYPES
            ),


        # ---------------------------------------------
        # Embeddings
        # ---------------------------------------------

        "embedding_model":
            str(
                EMBED_MODEL
            ),
    }


def build_current_index_manifest(
) -> IndexManifest:
    """
    Construct the manifest corresponding to the pipeline
    configuration currently loaded by Python.
    """

    core = {

        "manifest_format_version":
            int(
                INDEX_MANIFEST_FORMAT_VERSION
            ),

        "record_schema_version":
            int(
                INDEX_RECORD_SCHEMA_VERSION
            ),

        "semantic_table_name":
            str(
                SEMANTIC_TABLE_NAME
            ),

        "parser_version":
            str(
                PARSER_PIPELINE_VERSION
            ),

        "chunker_version":
            str(
                CHUNKER_PIPELINE_VERSION
            ),

        "settings":
            current_index_settings(),
    }


    fingerprint = (
        fingerprint_payload(
            core
        )
    )


    return IndexManifest(

        manifest_format_version=(
            core[
                "manifest_format_version"
            ]
        ),

        record_schema_version=(
            core[
                "record_schema_version"
            ]
        ),

        semantic_table_name=(
            core[
                "semantic_table_name"
            ]
        ),

        parser_version=(
            core[
                "parser_version"
            ]
        ),

        chunker_version=(
            core[
                "chunker_version"
            ]
        ),

        settings=(
            core[
                "settings"
            ]
        ),

        fingerprint=(
            fingerprint
        ),
    )


# =========================================================
# Loading
# =========================================================


def load_index_manifest(
    path: str | Path = (
        INDEX_MANIFEST_PATH
    ),
) -> IndexManifest | None:
    """
    Load an existing manifest.

    Returns None when no manifest exists.

    A malformed or internally inconsistent manifest raises
    an error rather than being silently ignored.
    """

    path = (
        Path(path)
    )


    if not path.exists():

        return None


    try:

        payload = (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )


    except Exception as exc:

        raise RuntimeError(
            "Could not read index manifest:\n"
            f"{path}"
        ) from exc


    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            "Index manifest must contain a JSON object:\n"
            f"{path}"
        )


    required = {

        "manifest_format_version",
        "record_schema_version",
        "semantic_table_name",
        "parser_version",
        "chunker_version",
        "settings",
        "fingerprint",
    }


    missing = (
        required
        - set(
            payload
        )
    )


    if missing:

        raise RuntimeError(
            "Index manifest is missing required fields:\n"
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


    settings = (
        payload[
            "settings"
        ]
    )


    if not isinstance(
        settings,
        dict,
    ):

        raise RuntimeError(
            "Index manifest field 'settings' must be an object."
        )


    manifest = (
        IndexManifest(

            manifest_format_version=int(
                payload[
                    "manifest_format_version"
                ]
            ),

            record_schema_version=int(
                payload[
                    "record_schema_version"
                ]
            ),

            semantic_table_name=str(
                payload[
                    "semantic_table_name"
                ]
            ),

            parser_version=str(
                payload[
                    "parser_version"
                ]
            ),

            chunker_version=str(
                payload[
                    "chunker_version"
                ]
            ),

            settings=dict(
                settings
            ),

            fingerprint=str(
                payload[
                    "fingerprint"
                ]
            )
            .strip()
            .lower(),
        )
    )


    expected_fingerprint = (
        fingerprint_payload(
            manifest.core_dict()
        )
    )


    if (
        manifest.fingerprint
        != expected_fingerprint
    ):

        raise RuntimeError(
            "Index manifest fingerprint does not match "
            "its contents.\n\n"
            "The manifest may have been manually edited "
            "or corrupted:\n"
            f"{path}"
        )


    return manifest


# =========================================================
# Writing
# =========================================================


def write_index_manifest(
    manifest: IndexManifest,
    path: str | Path = (
        INDEX_MANIFEST_PATH
    ),
) -> Path:
    """
    Atomically write the index manifest.
    """

    path = (
        Path(path)
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary_path = (
        path.with_name(
            path.name
            + ".tmp"
        )
    )


    serialized = (
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


    temporary_path.write_text(
        serialized,
        encoding="utf-8",
    )


    temporary_path.replace(
        path
    )


    return path


# =========================================================
# Difference reporting
# =========================================================


def manifest_differences(
    stored: IndexManifest,
    current: IndexManifest,
) -> list[str]:
    """
    Human-readable differences between two manifests.
    """

    differences: list[
        str
    ] = []


    scalar_fields = [

        "manifest_format_version",
        "record_schema_version",
        "semantic_table_name",
        "parser_version",
        "chunker_version",
    ]


    for field_name in (
        scalar_fields
    ):

        old_value = getattr(
            stored,
            field_name,
        )

        new_value = getattr(
            current,
            field_name,
        )


        if old_value != new_value:

            differences.append(
                f"{field_name}: "
                f"{old_value!r} -> "
                f"{new_value!r}"
            )


    setting_names = sorted(
        set(
            stored.settings
        )
        | set(
            current.settings
        )
    )


    for setting_name in (
        setting_names
    ):

        old_value = (
            stored.settings.get(
                setting_name,
                "<missing>",
            )
        )

        new_value = (
            current.settings.get(
                setting_name,
                "<missing>",
            )
        )


        if old_value != new_value:

            differences.append(
                f"settings.{setting_name}: "
                f"{old_value!r} -> "
                f"{new_value!r}"
            )


    return differences


# =========================================================
# Compatibility gate
# =========================================================


def ensure_index_manifest_compatible(
    *,
    adopt_missing_manifest: bool = False,
    persist_adoption: bool = True,
) -> tuple[
    IndexManifest,
    bool,
]:
    """
    Ensure the existing semantic index was produced with the
    same pipeline configuration currently loaded.

    Returns:
        current_manifest,
        adopted_missing_manifest

    A missing manifest is NOT silently accepted.

    For the initial migration from an older index, the user
    may explicitly adopt the existing index by setting
    adopt_missing_manifest=True.

    A genuine fingerprint mismatch always fails closed.
    """

    current = (
        build_current_index_manifest()
    )


    stored = (
        load_index_manifest()
    )


    # =====================================================
    # First migration from pre-manifest index
    # =====================================================

    if stored is None:

        if not adopt_missing_manifest:

            raise RuntimeError(
                "No index manifest exists for the current "
                "LanceDB index.\n\n"
                "This index was probably created before "
                "pipeline fingerprinting was added.\n\n"
                "If you KNOW the existing index was built "
                "with the currently configured parser, "
                "chunker, filters, and embedding model, run:\n\n"
                "    uv run python scripts\\sync_index.py "
                "--adopt-current-index\n\n"
                "Otherwise perform a full rebuild instead."
            )


        if persist_adoption:

            write_index_manifest(
                current
            )


        return (
            current,
            True,
        )


    # =====================================================
    # Existing manifest
    # =====================================================

    if (
        stored.fingerprint
        == current.fingerprint
    ):

        return (
            current,
            False,
        )


    differences = (
        manifest_differences(
            stored=stored,
            current=current,
        )
    )


    difference_text = (
        "\n".join(
            f"  - {item}"
            for item
            in differences
        )
        if differences
        else (
            "  - fingerprint changed "
            "for an unknown reason"
        )
    )


    raise RuntimeError(
        "The semantic index is STALE relative to the "
        "current indexing pipeline.\n\n"
        "Changed index settings:\n"
        f"{difference_text}\n\n"
        "Incremental indexing has been stopped before "
        "any database modification.\n\n"
        "A full index rebuild is required before "
        "continuing."
    )