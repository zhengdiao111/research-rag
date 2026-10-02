import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_bool_env(name: str, default: bool = False) -> bool:
    """
    Read a boolean value from an environment variable.

    Accepted true values:
        true, 1, yes, y, on

    Accepted false values:
        false, 0, no, n, off
    """

    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip().lower()

    if value in {"true", "1", "yes", "y", "on"}:
        return True

    if value in {"false", "0", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name!r} must be a boolean. "
        f"Received: {value!r}"
    )


# ---------------------------------------------------------
# Document library
# ---------------------------------------------------------

_document_root = os.getenv("DOCUMENT_ROOT")

if not _document_root:
    raise RuntimeError(
        "DOCUMENT_ROOT is not defined.\n"
        "Add it to your .env file, for example:\n"
        r"DOCUMENT_ROOT=C:\Users\YourName\Documents\AI-Libraries\Research"
    )

DOCUMENT_ROOT = Path(_document_root).expanduser()


# ---------------------------------------------------------
# Generated project data
# ---------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

OCR_LANGUAGE = os.getenv(
    "OCR_LANGUAGE",
    "eng",
)

USE_OCR = get_bool_env(
    "USE_OCR",
    True,
)

SHOW_PROGRESS = get_bool_env(
    "SHOW_PROGRESS",
    True,
)

# ---------------------------------------------------------
# Research chunking
# ---------------------------------------------------------

CHUNK_TARGET = int(
    os.getenv(
        "CHUNK_TARGET",
        "550",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "100",
    )
)

CHUNK_MIN = int(
    os.getenv(
        "CHUNK_MIN",
        "180",
    )
)

CHUNK_MAX = int(
    os.getenv(
        "CHUNK_MAX",
        "700",
    )
)

TOKENIZER_ENCODING = os.getenv(
    "TOKENIZER_ENCODING",
    "cl100k_base",
)

ALLOW_CROSS_PAGE_CHUNKS = get_bool_env(
    "ALLOW_CROSS_PAGE_CHUNKS",
    False,
)


# ---------------------------------------------------------
# Chunking validation
# ---------------------------------------------------------

if CHUNK_MIN <= 0:
    raise ValueError(
        "CHUNK_MIN must be greater than 0."
    )

if CHUNK_TARGET < CHUNK_MIN:
    raise ValueError(
        "CHUNK_TARGET must be >= CHUNK_MIN."
    )

if CHUNK_MAX < CHUNK_TARGET:
    raise ValueError(
        "CHUNK_MAX must be >= CHUNK_TARGET."
    )

if CHUNK_OVERLAP < 0:
    raise ValueError(
        "CHUNK_OVERLAP cannot be negative."
    )

if CHUNK_OVERLAP >= CHUNK_TARGET:
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than "
        "CHUNK_TARGET."
    )

# ---------------------------------------------------------
# Ollama embeddings
# ---------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")


EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "nomic-embed-text",
)


EMBED_BATCH_SIZE = int(
    os.getenv(
        "EMBED_BATCH_SIZE",
        "32",
    )
)


EMBED_TIMEOUT_SECONDS = float(
    os.getenv(
        "EMBED_TIMEOUT_SECONDS",
        "120",
    )
)


# ---------------------------------------------------------
# LanceDB
# ---------------------------------------------------------

_lancedb_path = Path(
    os.getenv(
        "LANCEDB_PATH",
        "./data/lancedb",
    )
)


if _lancedb_path.is_absolute():

    LANCEDB_PATH = (
        _lancedb_path
    )

else:

    LANCEDB_PATH = (
        PROJECT_ROOT
        / _lancedb_path
    )


LANCEDB_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


SEMANTIC_TABLE_NAME = os.getenv(
    "SEMANTIC_TABLE_NAME",
    "research_chunks",
)


SEMANTIC_TOP_K = int(
    os.getenv(
        "SEMANTIC_TOP_K",
        "5",
    )
)


# ---------------------------------------------------------
# Searchable chunk types
# ---------------------------------------------------------

_searchable_chunk_types = os.getenv(
    "SEARCHABLE_CHUNK_TYPES",
    (
        "body,"
        "abstract,"
        "introduction,"
        "methods,"
        "results,"
        "discussion,"
        "conclusion,"
        "sidebar,"
        "caption,"
        "supplementary"
    ),
)


SEARCHABLE_CHUNK_TYPES = frozenset(

    item.strip()

    for item in (
        _searchable_chunk_types.split(
            ","
        )
    )

    if item.strip()
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

if EMBED_BATCH_SIZE <= 0:

    raise ValueError(
        "EMBED_BATCH_SIZE must be greater than 0."
    )


if EMBED_TIMEOUT_SECONDS <= 0:

    raise ValueError(
        "EMBED_TIMEOUT_SECONDS must be greater than 0."
    )


if SEMANTIC_TOP_K <= 0:

    raise ValueError(
        "SEMANTIC_TOP_K must be greater than 0."
    )


if not SEARCHABLE_CHUNK_TYPES:

    raise ValueError(
        "SEARCHABLE_CHUNK_TYPES cannot be empty."
    )