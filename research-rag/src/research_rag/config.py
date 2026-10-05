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


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3.5:4b",
)


LLM_TIMEOUT_SECONDS = float(
    os.getenv(
        "LLM_TIMEOUT_SECONDS",
        "300",
    )
)


LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.2",
    )
)


LLM_MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "LLM_MAX_OUTPUT_TOKENS",
        "1600",
    )
)


LLM_THINK = get_bool_env(
    "LLM_THINK",
    False,
)


# ---------------------------------------------------------
# RAG synthesis
# ---------------------------------------------------------


RAG_RETRIEVAL_K = int(
    os.getenv(
        "RAG_RETRIEVAL_K",
        "12",
    )
)


RAG_FINAL_K = int(
    os.getenv(
        "RAG_FINAL_K",
        "8",
    )
)


RAG_EVIDENCE_BUDGET = int(
    os.getenv(
        "RAG_EVIDENCE_BUDGET",
        "9000",
    )
)


RAG_CITATION_VERIFY = get_bool_env(
    "RAG_CITATION_VERIFY",
    True,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


if LLM_TIMEOUT_SECONDS <= 0:

    raise ValueError(
        "LLM_TIMEOUT_SECONDS must be greater than 0."
    )


if not (
    0.0
    <= LLM_TEMPERATURE
    <= 2.0
):

    raise ValueError(
        "LLM_TEMPERATURE must be between 0 and 2."
    )


if LLM_MAX_OUTPUT_TOKENS <= 0:

    raise ValueError(
        "LLM_MAX_OUTPUT_TOKENS must be greater than 0."
    )


if RAG_RETRIEVAL_K <= 0:

    raise ValueError(
        "RAG_RETRIEVAL_K must be greater than 0."
    )


if RAG_FINAL_K <= 0:

    raise ValueError(
        "RAG_FINAL_K must be greater than 0."
    )


if (
    RAG_FINAL_K
    > RAG_RETRIEVAL_K
):

    raise ValueError(
        "RAG_FINAL_K cannot exceed "
        "RAG_RETRIEVAL_K."
    )


if RAG_EVIDENCE_BUDGET <= 0:

    raise ValueError(
        "RAG_EVIDENCE_BUDGET must be greater than 0."
    )

    # ---------------------------------------------------------
# RAG grounding safeguards
# ---------------------------------------------------------

RAG_SUFFICIENCY_CHECK = get_bool_env(
    "RAG_SUFFICIENCY_CHECK",
    True,
)


RAG_CITATION_REPAIR = get_bool_env(
    "RAG_CITATION_REPAIR",
    True,
)

# ---------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------

RETRIEVAL_MODE = os.getenv(
    "RETRIEVAL_MODE",
    "hybrid",
).strip().lower()


FTS_COLUMN = os.getenv(
    "FTS_COLUMN",
    "text",
)


VECTOR_COLUMN = os.getenv(
    "VECTOR_COLUMN",
    "vector",
)


HYBRID_RRF_K = int(
    os.getenv(
        "HYBRID_RRF_K",
        "60",
    )
)


if RETRIEVAL_MODE not in {
    "vector",
    "hybrid",
}:

    raise ValueError(
        "RETRIEVAL_MODE must be either "
        "'vector' or 'hybrid'."
    )


if HYBRID_RRF_K <= 0:

    raise ValueError(
        "HYBRID_RRF_K must be greater than 0."
    )

# =========================================================
# Retrieval refinement — Milestone 5B
# =========================================================


RERANK_ENABLED = (
    os.getenv(
        "RERANK_ENABLED",
        "true",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


RETRIEVAL_CANDIDATE_MULTIPLIER = int(
    os.getenv(
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        "3",
    )
)


RERANK_CAPTION_WEIGHT = float(
    os.getenv(
        "RERANK_CAPTION_WEIGHT",
        "0.80",
    )
)


RERANK_METHODS_WEIGHT = float(
    os.getenv(
        "RERANK_METHODS_WEIGHT",
        "0.97",
    )
)


RERANK_SUPPLEMENTARY_WEIGHT = float(
    os.getenv(
        "RERANK_SUPPLEMENTARY_WEIGHT",
        "0.92",
    )
)


RERANK_MAX_CAPTIONS = int(
    os.getenv(
        "RERANK_MAX_CAPTIONS",
        "2",
    )
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


if RETRIEVAL_CANDIDATE_MULTIPLIER < 1:

    raise ValueError(
        "RETRIEVAL_CANDIDATE_MULTIPLIER "
        "must be at least 1."
    )


if not (
    0.0
    < RERANK_CAPTION_WEIGHT
    <= 1.0
):

    raise ValueError(
        "RERANK_CAPTION_WEIGHT must be "
        "greater than 0 and <= 1."
    )


if not (
    0.0
    < RERANK_METHODS_WEIGHT
    <= 1.0
):

    raise ValueError(
        "RERANK_METHODS_WEIGHT must be "
        "greater than 0 and <= 1."
    )


if not (
    0.0
    < RERANK_SUPPLEMENTARY_WEIGHT
    <= 1.0
):

    raise ValueError(
        "RERANK_SUPPLEMENTARY_WEIGHT must be "
        "greater than 0 and <= 1."
    )


if RERANK_MAX_CAPTIONS < 0:

    raise ValueError(
        "RERANK_MAX_CAPTIONS cannot be negative."
    )


# =========================================================
# Claim-to-citation verification — Milestone 5B.2
# =========================================================


CLAIM_SUPPORT_VERIFY = (
    os.getenv(
        "CLAIM_SUPPORT_VERIFY",
        "true",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


CLAIM_SUPPORT_REPAIR = (
    os.getenv(
        "CLAIM_SUPPORT_REPAIR",
        "true",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


CLAIM_SUPPORT_MAX_CLAIMS = int(
    os.getenv(
        "CLAIM_SUPPORT_MAX_CLAIMS",
        "24",
    )
)


CLAIM_SUPPORT_MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "CLAIM_SUPPORT_MAX_OUTPUT_TOKENS",
        "1800",
    )
)


if CLAIM_SUPPORT_MAX_CLAIMS <= 0:

    raise ValueError(
        "CLAIM_SUPPORT_MAX_CLAIMS "
        "must be greater than 0."
    )


if CLAIM_SUPPORT_MAX_OUTPUT_TOKENS <= 0:

    raise ValueError(
        "CLAIM_SUPPORT_MAX_OUTPUT_TOKENS "
        "must be greater than 0."
    )