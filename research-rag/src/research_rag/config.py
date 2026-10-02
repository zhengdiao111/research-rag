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