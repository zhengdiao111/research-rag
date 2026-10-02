# Research RAG

A local retrieval-augmented generation (RAG) system for scientific papers, review articles, textbooks, and other research PDFs.

The project is designed to answer research questions using a **locally stored scientific document library** while keeping retrieval, generation, and document processing under local control.

The current system supports:

- PDF parsing with page-aware text extraction
- research- and book-aware chunking
- local embeddings with Ollama
- LanceDB vector storage
- semantic retrieval
- BM25 full-text retrieval
- hybrid vector + lexical retrieval
- reciprocal-rank fusion
- grounded answer generation with a local LLM
- filename/page citations
- citation validation and repair
- evidence sufficiency checks
- fail-closed behavior for unsupported questions

The current primary models are:

- **LLM:** `qwen3.5:4b`
- **Embedding model:** `nomic-embed-text`

---

# Project Goals

The system is intended to provide a foundation for a local scientific research assistant that can:

1. Search a collection of locally stored research documents.
2. Retrieve passages relevant to a scientific question.
3. Combine semantic and exact-term retrieval.
4. Synthesize evidence across multiple papers and books.
5. Produce answers grounded only in retrieved evidence.
6. Cite the source PDF and page for supported claims.
7. Refuse to answer when the local library does not contain sufficient evidence.
8. Eventually support larger research workflows such as literature synthesis, comparison, extraction, and agentic research tasks.

The architecture is intentionally modular so retrieval, chunking, embeddings, reranking, generation, and the user interface can evolve independently.

---

# Current Architecture

```text
Local PDF library
        │
        ▼
PyMuPDF / PyMuPDF4LLM
        │
        ▼
Page-aware text extraction
        │
        ▼
Layout cleanup
        │
        ▼
Research-aware chunking
        │
        ├── body
        ├── abstract
        ├── introduction
        ├── methods
        ├── results
        ├── discussion
        ├── conclusion
        ├── captions
        ├── glossary
        ├── references
        ├── frontmatter
        ├── further reading
        ├── index
        └── metadata
        │
        ▼
nomic-embed-text
via Ollama
        │
        ▼
LanceDB
        │
        ├───────────────┐
        │               │
        ▼               ▼
Vector search        BM25 / FTS
        │               │
        └───────┬───────┘
                ▼
       Reciprocal Rank Fusion
                │
                ▼
         Hybrid retrieval
                │
                ▼
     Evidence deduplication
                │
                ▼
        Evidence budgeting
                │
                ▼
   Evidence sufficiency checks
                │
                ▼
          qwen3.5:4b
                │
                ▼
      Citation verification
                │
                ▼
 Deterministic filename/page
          citation rendering
                │
                ▼
        Grounded answer
```

---

# Current Milestones

## Milestone 1 — PDF Parsing

**Status: Complete**

The parser:

- discovers PDFs recursively
- calculates SHA-256 hashes
- extracts page-level text
- preserves page numbers
- detects captions
- removes page headers and footers when possible
- retains document metadata
- supports research papers and books
- saves parsed diagnostic output

The parser uses:

- `pymupdf`
- `pymupdf4llm`

---

## Milestone 2 — Research-Aware Chunking

**Status: Complete**

The chunker is aware of common scientific and book structures.

Examples include:

```text
abstract
introduction
methods
results
discussion
conclusion
body
sidebar
caption
supplementary
glossary

references
acknowledgments
frontmatter
further_reading
index
notes
metadata
```

The distinction matters because not every section should participate in semantic retrieval.

For example, the system normally excludes:

```text
frontmatter
references
acknowledgments
further_reading
index
notes
metadata
```

while retaining explanatory content such as:

```text
body
abstract
introduction
methods
results
discussion
conclusion
sidebar
caption
supplementary
glossary
```

Chunking is approximately controlled by:

```env
CHUNK_TARGET=550
CHUNK_OVERLAP=100
CHUNK_MIN=180
CHUNK_MAX=700
TOKENIZER_ENCODING=cl100k_base
ALLOW_CROSS_PAGE_CHUNKS=false
```

`cl100k_base` is used only as an approximate chunk-sizing tokenizer.

The actual embedding tokenizer is determined by `nomic-embed-text`.

---

## Milestone 3 — Embeddings and Semantic Retrieval

**Status: Complete**

Chunks are embedded locally using:

```text
nomic-embed-text
```

through Ollama.

The system stores:

- chunk text
- embeddings
- filename
- title
- author
- page range
- section
- chunk type
- chunk ID
- token count
- document metadata

in LanceDB.

The embedding layer includes protection for inputs that exceed the embedding model's context length.

Instead of silently truncating an oversized chunk, the system:

1. detects the embedding failure
2. recursively splits the text
3. embeds the smaller pieces
4. pools the embeddings
5. retains one embedding for the original RAG chunk

This preserves the original citation and retrieval unit.

---

## Milestone 4 — Grounded RAG Synthesis

**Status: Complete**

The system now performs full question answering:

```text
question
    ↓
retrieval
    ↓
evidence selection
    ↓
evidence sufficiency checks
    ↓
local LLM
    ↓
citation validation
    ↓
grounded answer
```

### Grounding safeguards

The RAG pipeline currently includes several safeguards.

#### Evidence overlap removal

Chunk overlap is useful for retrieval but can repeatedly send the same paragraph to the LLM.

Repeated paragraphs are removed before evidence is assembled.

#### Evidence budget

Retrieved evidence is limited by:

```env
RAG_RETRIEVAL_K=12
RAG_FINAL_K=8
RAG_EVIDENCE_BUDGET=9000
```

The system therefore retrieves more candidates than it ultimately sends to the LLM.

#### Lexical evidence sufficiency gate

The system detects distinctive terms in the query.

For example:

```text
pembrolizumab
BRM014
SMARCA2
reaction-diffusion
rs123456
```

If a distinctive query term occurs nowhere in the retrieved evidence, the system abstains rather than allowing the LLM to improvise.

Example:

```text
Question:
What are the clinical dosing guidelines for pembrolizumab?

Result:
The available evidence is insufficient to answer this question.
```

#### LLM evidence sufficiency gate

If the deterministic lexical check passes, Qwen performs a second classification:

```text
SUFFICIENT: YES
```

or:

```text
SUFFICIENT: NO
```

The LLM does not answer the research question during this step.

Its only job is to determine whether the retrieved evidence is actually capable of supporting an answer.

#### Citation-safe generation

The LLM cites evidence using internal IDs:

```text
[S1]
[S2]
[S3]
```

It does not generate filenames or page numbers.

Python then deterministically maps:

```text
[S1]
```

to something like:

```text
[Dunlop_2026_Shaping-Tissues-with-Defects_Science.pdf, p. 2]
```

This prevents the model from inventing citation metadata.

#### Citation validation

Generated source IDs are checked against the evidence packet.

If citation validation fails, the system performs one citation-repair attempt.

If the repair still fails, the answer is rejected.

#### Truncation protection

Ollama generation metadata is inspected.

If generation finishes with:

```text
done_reason='length'
```

the system does not expose the partial answer.

Instead, it automatically retries once with a substantially shorter answer prompt.

If the retry is also incomplete, the system abstains.

---

# Milestone 5 — Hybrid Retrieval

## Milestone 5A — BM25 + Vector Search + RRF

**Status: Implemented and tested**

The retrieval layer now supports:

```text
vector
```

and:

```text
hybrid
```

modes.

Hybrid retrieval combines:

```text
Nomic semantic vector search
            +
LanceDB BM25 full-text search
            ↓
 Reciprocal Rank Fusion
```

The FTS index is built directly on the existing LanceDB `text` column.

No separate lexical database is required.

Example `.env` configuration:

```env
RETRIEVAL_MODE=hybrid

FTS_COLUMN=text
VECTOR_COLUMN=vector

HYBRID_RRF_K=60
```

Hybrid retrieval has been tested successfully on queries including:

```text
Toner-Tu

Toner-Tu flocking theory

topological defects tissue shape

Navier-Stokes equations
```

The behavior is complementary:

```text
Vector retrieval
→ good for semantic similarity

BM25 retrieval
→ good for exact terminology

Hybrid retrieval
→ preserves both using reciprocal-rank fusion
```

---

# Current Project Structure

A representative project structure is:

```text
local-document-ai/
│
├── README.md
│
└── research-rag/
    │
    ├── .env
    ├── .gitignore
    ├── pyproject.toml
    ├── uv.lock
    │
    ├── data/
    │   ├── processed/
    │   └── lancedb/
    │
    ├── scripts/
    │   ├── ask.py
    │   ├── create_fts_index.py
    │   ├── index_pdfs.py
    │   ├── search.py
    │   ├── test_chunks.py
    │   ├── test_hybrid_search.py
    │   ├── test_llm.py
    │   ├── test_model.py
    │   └── test_pdf.py
    │
    ├── src/
    │   └── research_rag/
    │       ├── __init__.py
    │       ├── chunker.py
    │       ├── config.py
    │       ├── database.py
    │       ├── embeddings.py
    │       ├── llm.py
    │       ├── parser.py
    │       ├── rag.py
    │       └── retrieval.py
    │
    └── tests/
```

The exact script collection may change as development continues.

---

# Research Library

The current development setup keeps PDFs outside the Git repository.

Example:

```text
C:\Users\<username>\Documents\AI-Libraries\Research
```

This separation allows the code repository to remain small while the research library can eventually be moved to another disk or NAS.

The path is configured through `.env`.

---

# Requirements

The project currently assumes:

- Windows
- Python 3.12+
- `uv`
- Ollama
- LanceDB
- local PDF files

Ollama models:

```powershell
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

---

# Installation

From the parent projects directory:

```powershell
cd C:\Users\<username>\AI-Development\projects
```

Create the project:

```powershell
mkdir local-document-ai
cd local-document-ai
git init
```

Create the research RAG:

```powershell
uv init research-rag
cd research-rag
```

Install the major dependencies:

```powershell
uv add pymupdf pymupdf4llm
uv add python-dotenv
uv add tiktoken
uv add lancedb pyarrow pandas numpy httpx
```

Development dependencies:

```powershell
uv add --dev pytest
```

Then synchronize:

```powershell
uv sync
```

---

# Example `.env`

A representative configuration is:

```env
# =========================================================
# Document library
# =========================================================

DOCUMENT_ROOT=C:\Users\<username>\Documents\AI-Libraries\Research


# =========================================================
# PDF parsing
# =========================================================

OCR_LANGUAGE=eng
USE_OCR=false
SHOW_PROGRESS=true


# =========================================================
# Chunking
# =========================================================

CHUNK_TARGET=550
CHUNK_OVERLAP=100
CHUNK_MIN=180
CHUNK_MAX=700

TOKENIZER_ENCODING=cl100k_base

ALLOW_CROSS_PAGE_CHUNKS=false


# =========================================================
# Ollama
# =========================================================

OLLAMA_BASE_URL=http://localhost:11434


# =========================================================
# Embeddings
# =========================================================

EMBED_MODEL=nomic-embed-text

EMBED_BATCH_SIZE=32

EMBED_TIMEOUT_SECONDS=120


# =========================================================
# LanceDB
# =========================================================

LANCEDB_PATH=./data/lancedb

SEMANTIC_TABLE_NAME=research_chunks

SEMANTIC_TOP_K=5


# =========================================================
# Searchable chunk types
# =========================================================

SEARCHABLE_CHUNK_TYPES=body,abstract,introduction,methods,results,discussion,conclusion,sidebar,caption,supplementary,glossary


# =========================================================
# LLM
# =========================================================

LLM_MODEL=qwen3.5:4b

LLM_TIMEOUT_SECONDS=300

LLM_TEMPERATURE=0.2

LLM_MAX_OUTPUT_TOKENS=2500

LLM_THINK=false


# =========================================================
# Grounded RAG
# =========================================================

RAG_RETRIEVAL_K=12

RAG_FINAL_K=8

RAG_EVIDENCE_BUDGET=9000

RAG_CITATION_VERIFY=true

RAG_SUFFICIENCY_CHECK=true

RAG_CITATION_REPAIR=true


# =========================================================
# Hybrid retrieval
# =========================================================

RETRIEVAL_MODE=hybrid

FTS_COLUMN=text

VECTOR_COLUMN=vector

HYBRID_RRF_K=60
```

---

# Indexing the Research Library

To parse, chunk, embed, and index the research library:

```powershell
uv run python scripts\index_pdfs.py
```

The indexing pipeline performs approximately:

```text
discover PDFs
    ↓
parse documents
    ↓
create research-aware chunks
    ↓
filter searchable chunk types
    ↓
embed chunks
    ↓
write LanceDB table
```

---

# Creating the Full-Text Search Index

After the LanceDB table has been created:

```powershell
uv run python scripts\create_fts_index.py
```

A successful run should report an FTS index on:

```text
text
```

with all LanceDB rows indexed.

If the main LanceDB table is rebuilt from scratch, rebuild the FTS index afterward.

---

# Testing PDF Parsing

Run:

```powershell
uv run python scripts\test_pdf.py
```

or provide a specific PDF if supported by the current script.

The diagnostic output is useful for checking:

- headers
- footers
- captions
- section headings
- equation extraction
- multi-column layouts
- page boundaries

---

# Testing Chunking

Example:

```powershell
uv run python scripts\test_chunks.py "C:\path\to\paper.pdf"
```

Inspect:

- chunk type
- section
- page number
- approximate token count
- overlap
- section inheritance
- figure/caption classification

---

# Testing Semantic Search

Example:

```powershell
uv run python scripts\search.py "How do cells generate mechanical forces?"
```

Increase the number of results:

```powershell
uv run python scripts\search.py "How do cells generate mechanical forces?" --top-k 10
```

Show complete chunk text:

```powershell
uv run python scripts\search.py "How do cells generate mechanical forces?" --full
```

---

# Testing Hybrid Retrieval

Compare vector, BM25, and hybrid retrieval:

```powershell
uv run python scripts\test_hybrid_search.py "Toner-Tu"
```

Other useful examples:

```powershell
uv run python scripts\test_hybrid_search.py "Toner-Tu flocking theory"
```

```powershell
uv run python scripts\test_hybrid_search.py "topological defects tissue shape"
```

```powershell
uv run python scripts\test_hybrid_search.py "Navier-Stokes equations"
```

The diagnostic script prints:

```text
VECTOR SEARCH

LEXICAL / BM25 SEARCH

HYBRID SEARCH
```

so differences in ranking can be inspected directly.

---

# Asking Questions

Run:

```powershell
uv run python scripts\ask.py "How can topological defects control tissue shape?"
```

A successful answer should include deterministic source citations such as:

```text
[Dunlop_2026_Shaping-Tissues-with-Defects_Science.pdf, p. 2]
```

The command also reports diagnostics including:

```text
Retrieved candidates

Evidence chunks used

Evidence tokens

Evidence sufficiency

Citation validation
```

---

# Testing Out-of-Domain Abstention

A useful negative control is:

```powershell
uv run python scripts\ask.py "What are the clinical dosing guidelines for pembrolizumab?"
```

If the research library does not contain evidence about pembrolizumab, the expected response is:

```text
The available evidence is insufficient to answer this question.
```

This is intentional.

The system should not substitute pretrained LLM knowledge for missing library evidence.

---

# Citation Architecture

The citation system deliberately separates LLM citation selection from citation metadata.

The model receives:

```text
[S1]
[S2]
[S3]
```

and produces:

```text
Topological defects can produce localized stress fields [S1].
```

The Python application maps:

```text
S1
```

to stored metadata:

```text
filename
page_start
page_end
section
```

and renders:

```text
Topological defects can produce localized stress fields
[Dunlop_2026_Shaping-Tissues-with-Defects_Science.pdf, p. 2].
```

The model therefore never needs to generate its own filenames or page numbers.

---

# Current Retrieval Modes

The project supports:

## Vector

```env
RETRIEVAL_MODE=vector
```

Uses semantic similarity only.

Best for:

- conceptual questions
- paraphrases
- related biological mechanisms
- questions where the wording differs from the source

## Hybrid

```env
RETRIEVAL_MODE=hybrid
```

Uses:

```text
semantic vectors
+
BM25 lexical retrieval
+
reciprocal-rank fusion
```

Best for a research library containing:

- gene names
- protein names
- drug names
- mutations
- model names
- equations
- abbreviations
- author names
- specialized terminology

Hybrid is currently the preferred default.

---

# Known Limitations

## PDF layout extraction

Scientific PDFs are messy.

Remaining extraction artifacts can include:

- broken drop caps
- multi-column text interleaving
- equations with imperfect formatting
- figure labels interpreted as headings
- fragmented words
- unusual publisher layouts

The parser intentionally avoids aggressive automatic correction because changing scientific terminology incorrectly is worse than preserving a visible extraction artifact.

---

## Figure chunks

Figure captions sometimes rank highly because they contain exact scientific terminology.

For example, a query about:

```text
Navier-Stokes equations
```

may return figure captions before explanatory body sections.

This is not a hybrid-search failure, but it suggests a future opportunity for:

- chunk-type weighting
- diversity-aware retrieval
- reranking

---

## BM25 query behavior

Lexical retrieval depends on exact or morphologically related terms.

BM25 complements semantic search but does not replace it.

This is why hybrid retrieval is preferred over lexical-only search.

---

## RRF is rank-based

Reciprocal-rank-fusion scores are not probabilities.

Values such as:

```text
0.0325
```

should not be interpreted as confidence scores.

They represent combined ranking strength across retrieval branches.

---

## Corpus sufficiency

Vector databases always return nearest neighbors even when nothing is truly relevant.

This is why the system performs explicit evidence-sufficiency checks before generation.

---

## Current LLM size

`qwen3.5:4b` is intentionally lightweight enough to run locally.

Larger models may eventually improve:

- synthesis
- nuanced comparison
- long-answer coherence
- evidence evaluation
- citation discipline

but require more compute.

The architecture is model-independent enough to support future upgrades.

---

# Development Philosophy

Several design choices are intentional.

## Prefer explicit components

The project currently avoids large RAG frameworks such as LangChain or LlamaIndex.

Instead, the pipeline exposes its major operations directly:

```text
parser
chunker
embeddings
database
retrieval
RAG orchestration
LLM
```

This makes debugging and experimentation easier.

---

## Preserve source provenance

Page numbers, section names, filenames, and chunk IDs remain attached to text throughout the pipeline.

---

## Fail closed

If the system cannot establish that retrieved evidence supports the question, it should refuse to generate a scientific answer.

---

## Avoid silent truncation

Oversized embedding inputs are split and pooled rather than silently truncated.

Incomplete LLM output is rejected or retried.

---

## Retrieval and generation remain separate

Improving retrieval should not require rewriting the answer-generation system.

Likewise, changing the LLM should not require rebuilding the document index.

---

# Planned Next Steps

## Milestone 5B — Retrieval Refinement

Potential improvements include:

- chunk-type weighting
- caption/body preference
- candidate diversification
- document diversity
- neural reranking
- cross-encoder reranking
- query expansion
- acronym expansion
- scientific entity-aware retrieval

A neural reranker will be evaluated only if it produces enough improvement to justify additional compute and latency.

---

## Milestone 6 — Incremental Indexing

Currently the full library can be indexed in batch.

Future work will support:

```text
new PDF appears
      ↓
hash document
      ↓
determine whether new/changed
      ↓
parse only that document
      ↓
chunk
      ↓
embed
      ↓
update LanceDB
      ↓
update FTS index
```

This will make the system suitable for an actively growing research library or NAS.

---

## Milestone 7 — User Interface

A future UI may include:

- Streamlit or another local web interface
- question input
- streaming answers
- formatted Markdown
- LaTeX rendering
- clickable citations
- source previews
- retrieved evidence panels
- retrieval-mode selector
- model selector
- adjustable retrieval depth
- conversation history
- stop-generation control

---

## Future Research-Agent Features

Longer-term possibilities include:

### Literature synthesis

```text
question
   ↓
retrieve across many papers
   ↓
group evidence by claim/theme
   ↓
compare findings
   ↓
identify agreement/disagreement
   ↓
produce cited synthesis
```

### Paper comparison

Compare:

- methods
- experimental systems
- hypotheses
- results
- interpretations
- limitations

### Evidence tables

Convert literature evidence into structured forms such as:

```text
paper
system
method
perturbation
measurement
result
interpretation
```

### Research planning

Use retrieved literature to assist with:

- experimental design
- hypothesis generation
- mechanism comparison
- follow-up questions

### Agentic workflows

Potential future agents could:

- detect newly added papers
- summarize them
- connect them to existing topics
- update topic-level notes
- identify contradictory findings
- suggest papers for deeper reading

---

# Future Storage

The document library is intentionally stored separately from the Git repository.

This makes it possible to move the research collection later to:

- another local drive
- RAID storage
- a NAS
- network-mounted storage

while preserving the same application architecture.

Only the `.env` document path should need to change.

---

# Git

The parent directory can serve as the Git repository:

```text
local-document-ai/
├── research-rag/
└── personal-document-rag/
```

A recommended `.gitignore` includes:

```gitignore
__pycache__/
*.py[cod]

.pytest_cache/

.venv/

.env

data/processed/
data/lancedb/

*.pdf

.vscode/

Thumbs.db
.DS_Store
```

Large research PDFs, generated embeddings, LanceDB data, and secrets should not normally be committed.

---

# Related Planned Project

The parent repository is intended eventually to contain a second application:

```text
personal-document-rag/
```

This will remain separate from the research RAG because personal-document retrieval has different requirements, including:

- scanned documents
- OCR
- ID cards
- exact field extraction
- deterministic answers
- privacy-sensitive information
- image files
- forms
- receipts
- official records

The two applications may eventually share lower-level utilities while retaining separate retrieval and answer-generation behavior.

---

# Current Status

The core research RAG pipeline is functional.

```text
PDF parsing                         ✅
Research-aware chunking             ✅
Book-aware chunking                 ✅
Local embeddings                    ✅
LanceDB vector storage              ✅
Semantic retrieval                  ✅
BM25 / FTS retrieval                ✅
Hybrid retrieval                    ✅
Reciprocal-rank fusion              ✅
Evidence deduplication              ✅
Evidence budgeting                  ✅
Evidence sufficiency checking       ✅
Out-of-domain abstention            ✅
Grounded synthesis                  ✅
Citation validation                 ✅
Citation repair                     ✅
Filename/page citation rendering    ✅
Generation truncation protection    ✅
Automatic concise retry             ✅

Chunk-type retrieval weighting      ⏳
Neural reranking                    ⏳
Incremental indexing                ⏳
Automatic library watching          ⏳
Graph/agent workflows               ⏳
User interface                      ⏳
```

The current system therefore provides a solid local foundation for a more capable scientific literature research agent.