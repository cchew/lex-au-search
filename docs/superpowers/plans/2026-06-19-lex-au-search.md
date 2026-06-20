# lex-au-search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 2 of the AU Legislative Intelligence Stack — a hybrid search index over the lex-au AKN corpus, exposed via a FastAPI endpoint and an MCP stdio server for Claude Code.

**Architecture:** The lex-au corpus (`corpus/index.json` + `corpus/xml/*.xml`) is chunked by `<section>` element, embedded with FastEmbed (nomic-embed-text-v1.5 dense + BM25 sparse), and stored in Qdrant local disk. A FastAPI server and an MCP stdio server both delegate to a shared `Searcher` class; the CLI wires everything together.

**Tech Stack:** Python 3.12, lxml, qdrant-client[fastembed], FastAPI, uvicorn, mcp, click, pytest

## Global Constraints

- Python >= 3.12
- Qdrant local disk: `QdrantClient(path=str(storage_dir))` — no Docker, no external services
- Dense model: `nomic-ai/nomic-embed-text-v1.5` (768 dims) via FastEmbed
- Sparse model: `Qdrant/bm25` via FastEmbed
- nomic prefix required: `"search_document: " + text` for indexing; `"search_query: " + query` for search — never stored in payload
- Hybrid fusion: RRF (Qdrant default when both models configured)
- Collection name: `legislation`
- Entry point: `lex-au-search = "lexausearch.cli:cli"` in pyproject.toml
- Package name: `lex-au-search` (pip), import name: `lexausearch`
- Repo layout follows lex-au: `src/lexausearch/`, `tests/`, `pyproject.toml` at repo root
- AKN namespace: `http://docs.oasis-open.org/legaldocml/ns/akn/3.0`
- MIT licence, author Ching Chew <ching.chew@gmail.com>

---

## File Map

| File | Task |
|---|---|
| `pyproject.toml` | 1 |
| `CLAUDE.md` | 1 |
| `LICENSE` | 1 |
| `.gitignore` | 1 |
| `.env.example` | 1 |
| `src/lexausearch/__init__.py` | 1 |
| `src/lexausearch/models.py` | 1 |
| `tests/__init__.py` | 1 |
| `tests/conftest.py` | 1 |
| `tests/test_models.py` | 1 |
| `src/lexausearch/chunker.py` | 2 |
| `tests/test_chunker.py` | 2 |
| `src/lexausearch/indexer.py` | 3 |
| `src/lexausearch/searcher.py` | 3 |
| `tests/test_searcher.py` | 3 |
| `src/lexausearch/api.py` | 4 |
| `tests/test_api.py` | 4 |
| `src/lexausearch/mcp.py` | 5 |
| `tests/test_mcp.py` | 5 |
| `src/lexausearch/cli.py` | 6 |
| `README.md` | 6 |

---

## Task 1: Scaffold, pyproject.toml, models

**Files:**
- Create: `projects/lex-au-search/repo/` (all scaffold files listed below)
- Create: `src/lexausearch/__init__.py`
- Create: `src/lexausearch/models.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `Chunk(act_name, frbr_uri, eid, section_num, heading, text)` dataclass
  - `SearchResult(chunk: Chunk, score: float)` dataclass
  - `format_results(results: list[SearchResult]) -> str`

- [ ] **Step 1: Create repo directory**

```bash
mkdir -p projects/lex-au-search/repo/src/lexausearch
mkdir -p projects/lex-au-search/repo/tests
mkdir -p projects/lex-au-search/repo/docs/superpowers/plans
cd projects/lex-au-search/repo
git init
```

- [ ] **Step 2: Create pyproject.toml**

`projects/lex-au-search/repo/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=72"]
build-backend = "setuptools.build_meta"

[project]
name = "lex-au-search"
version = "0.1.0"
description = "Hybrid search over Australian Commonwealth legislation — Layer 2 of the AU Legislative Intelligence Stack"
authors = [{name = "Ching Chew", email = "ching.chew@gmail.com"}]
license = {text = "MIT"}
requires-python = ">=3.12"
dependencies = [
    "lxml>=5.3",
    "qdrant-client[fastembed]>=1.9",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "mcp>=1.0",
    "click>=8.1",
]

[project.scripts]
lex-au-search = "lexausearch.cli:cli"

[project.urls]
Repository = "https://github.com/cchew/lex-au-search"

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "httpx>=0.27",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Create CLAUDE.md**

`projects/lex-au-search/repo/CLAUDE.md`:
```markdown
# lex-au-search

Layer 2 of the AU Legislative Intelligence Stack. Hybrid search (dense + BM25 sparse) over the lex-au AKN 3.0 XML corpus.

## Stack position

Layer 1: lex-au (../lex-au/repo/) — AKN 3.0 XML corpus
Layer 2: lex-au-search (this repo) — hybrid search API + MCP
Layer 3: ClauseKit (../clause-kit/repo/) — rule extraction

## Setup

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

## CLI

lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/
lex-au-search serve
lex-au-search mcp  # requires LEXAU_SEARCH_STORAGE env var

## Tests

pytest
```

- [ ] **Step 4: Create LICENSE**

`projects/lex-au-search/repo/LICENSE`:
```
MIT License

Copyright (c) 2026 Ching Chew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Create .gitignore**

`projects/lex-au-search/repo/.gitignore`:
```
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.coverage
qdrant_storage/
corpus/
dist/
build/
.env
```

- [ ] **Step 6: Create .env.example**

`projects/lex-au-search/repo/.env.example`:
```
# Absolute path to Qdrant storage directory (required for MCP server)
LEXAU_SEARCH_STORAGE=/absolute/path/to/projects/lex-au-search/repo/qdrant_storage

# Path to lex-au corpus directory (optional convenience for scripts)
LEXAU_CORPUS_DIR=/absolute/path/to/projects/lex-au/repo/corpus
```

- [ ] **Step 7: Create src/lexausearch/__init__.py**

```python
```
(empty file)

- [ ] **Step 8: Write failing test for models**

`projects/lex-au-search/repo/tests/test_models.py`:
```python
from lexausearch.models import Chunk, SearchResult, format_results


def test_chunk_fields():
    chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-3",
        section_num="3",
        heading="Interpretation",
        text="In this Act personal information means information about an identified individual.",
    )
    assert chunk.act_name == "Privacy Act 1988"
    assert chunk.eid == "sec-3"
    assert chunk.heading == "Interpretation"


def test_chunk_heading_optional():
    chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-3",
        section_num="3",
        heading=None,
        text="Some text.",
    )
    assert chunk.heading is None


def test_format_results_contains_frbr_uri():
    chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-3",
        section_num="3",
        heading="Interpretation",
        text="In this Act personal information means information about an identified individual.",
    )
    result = SearchResult(chunk=chunk, score=0.85)
    output = format_results([result])
    assert "Privacy Act 1988" in output
    assert "sec-3" in output
    assert "0.85" in output
    assert chunk.text in output


def test_format_results_score_on_header_line():
    chunk = Chunk(
        act_name="Fair Work Act 2009",
        frbr_uri="/akn/au/act/2009/28/eng@2026-01-01",
        eid="part-1__sec-12",
        section_num="12",
        heading="Definitions",
        text="Some definitions here.",
    )
    result = SearchResult(chunk=chunk, score=0.71)
    output = format_results([result])
    lines = output.strip().split("\n")
    header = lines[0]
    assert "score=0.71" in header
    assert "part-1__sec-12" in header


def test_format_results_multiple():
    chunks = [
        Chunk("A Act 2000", "/akn/au/act/2000/1/eng@2026-01-01", "sec-1", "1", "Short title", "Text one."),
        Chunk("B Act 2001", "/akn/au/act/2001/2/eng@2026-01-01", "sec-2", "2", None, "Text two."),
    ]
    results = [SearchResult(c, 0.9 - i * 0.1) for i, c in enumerate(chunks)]
    output = format_results(results)
    assert "A Act 2000" in output
    assert "B Act 2001" in output
```

- [ ] **Step 9: Create tests/__init__.py**

```python
```
(empty file)

- [ ] **Step 10: Run test to verify it fails**

```bash
cd projects/lex-au-search/repo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'lexausearch'`

- [ ] **Step 11: Create src/lexausearch/models.py**

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Chunk:
    act_name: str
    frbr_uri: str
    eid: str
    section_num: str
    heading: str | None
    text: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def format_results(results: list[SearchResult]) -> str:
    parts = []
    for r in results:
        heading_line = f"\n{r.chunk.heading}" if r.chunk.heading else ""
        block = (
            f"[{r.chunk.act_name} — s {r.chunk.section_num} ({r.chunk.eid}) score={r.score:.2f}]"
            f"{heading_line}\n{r.chunk.text}"
        )
        parts.append(block)
    return "\n\n".join(parts)
```

- [ ] **Step 12: Create tests/conftest.py**

```python
import pytest
from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

PRIVACY_ACT_XML = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork>
          <FRBRthis value="/akn/au/act/1988/119/!main"/>
          <FRBRuri value="/akn/au/act/1988/119"/>
          <FRBRdate date="1988" name="Generation"/>
          <FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/>
          <FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/>
          <FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-3">
        <num>3</num>
        <heading>Interpretation</heading>
        <content>
          <p>In this Act personal information means information about an identified individual.</p>
        </content>
      </section>
      <section eId="sec-6">
        <num>6</num>
        <heading>Meaning of personal information</heading>
        <content>
          <p>Personal information means information or an opinion about an identified individual.</p>
          <p>The information need not be recorded in material form.</p>
        </content>
      </section>
    </body>
  </act>
</akomaNtoso>"""


@pytest.fixture(scope="module")
def privacy_chunks() -> list[Chunk]:
    return [
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-3",
            section_num="3",
            heading="Interpretation",
            text="3 Interpretation In this Act personal information means information about an identified individual.",
        ),
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-6",
            section_num="6",
            heading="Meaning of personal information",
            text="6 Meaning of personal information Personal information means information or an opinion about an identified individual. The information need not be recorded in material form.",
        ),
    ]
```

- [ ] **Step 13: Run tests and verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 5 tests PASS

- [ ] **Step 14: Commit**

```bash
git add .
git commit -m "feat(lex-au-search): scaffold, pyproject.toml, models, conftest"
```

---

## Task 2: Chunker

**Files:**
- Create: `src/lexausearch/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `Chunk` from `lexausearch.models`
- Produces:
  - `chunk_xml(xml_path: Path, act_name: str) -> list[Chunk]`
  - `chunk_corpus(corpus_dir: Path) -> list[Chunk]`

- [ ] **Step 1: Write failing tests**

`projects/lex-au-search/repo/tests/test_chunker.py`:
```python
import json
import pytest
import warnings
from pathlib import Path
from lxml import etree

from lexausearch.chunker import chunk_xml, chunk_corpus
from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _xml_bytes(xml_str: str) -> bytes:
    return xml_str.encode()


def test_chunk_xml_returns_one_chunk_per_section(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    assert len(chunks) == 2


def test_chunk_xml_frbr_uri_from_xml(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    assert all(c.frbr_uri == "/akn/au/act/1988/119/eng@2026-01-01" for c in chunks)


def test_chunk_xml_section_fields(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    s3 = next(c for c in chunks if c.eid == "sec-3")
    assert s3.act_name == "Privacy Act 1988"
    assert s3.section_num == "3"
    assert s3.heading == "Interpretation"
    assert "personal information" in s3.text


def test_chunk_xml_itertext_captures_all_paragraphs(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    s6 = next(c for c in chunks if c.eid == "sec-6")
    # Both <p> elements should be in the text
    assert "opinion" in s6.text
    assert "recorded in material form" in s6.text


def test_chunk_xml_missing_heading_is_none(tmp_path):
    xml = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork><FRBRthis value="/akn/au/act/2000/1/!main"/><FRBRuri value="/akn/au/act/2000/1"/>
          <FRBRdate date="2000" name="Generation"/><FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression><FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/><FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation><FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/><FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-1">
        <num>1</num>
        <content><p>Text without heading.</p></content>
      </section>
    </body>
  </act>
</akomaNtoso>"""
    xml_file = tmp_path / "test.xml"
    xml_file.write_bytes(xml.encode())
    chunks = chunk_xml(xml_file, "Test Act 2000")
    assert chunks[0].heading is None


def test_chunk_xml_warns_on_untagged_body_content(tmp_path):
    xml = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork><FRBRthis value="/akn/au/act/1988/119/!main"/><FRBRuri value="/akn/au/act/1988/119"/>
          <FRBRdate date="1988" name="Generation"/><FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression><FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/><FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation><FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/><FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-1"><num>1</num><content><p>Real section.</p></content></section>
      <p>Untagged schedule text not in a section element.</p>
    </body>
  </act>
</akomaNtoso>"""
    xml_file = tmp_path / "with-schedule.xml"
    xml_file.write_bytes(xml.encode())
    with pytest.warns(UserWarning, match="schedule"):
        chunk_xml(xml_file, "Privacy Act 1988")


def test_chunk_corpus(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    # Build a minimal corpus directory
    corpus_dir = tmp_path / "corpus"
    xml_dir = corpus_dir / "xml"
    xml_dir.mkdir(parents=True)
    xml_path = xml_dir / "privacy-act-1988.xml"
    xml_path.write_bytes(PRIVACY_ACT_XML.encode())
    index = {
        "acts": {
            "privacy-act-1988": {
                "name": "Privacy Act 1988",
                "xml_path": "xml/privacy-act-1988.xml",
            }
        }
    }
    (corpus_dir / "index.json").write_text(json.dumps(index))
    chunks = chunk_corpus(corpus_dir)
    assert len(chunks) == 2
    assert all(c.act_name == "Privacy Act 1988" for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ImportError: cannot import name 'chunk_xml' from 'lexausearch.chunker'`

- [ ] **Step 3: Create src/lexausearch/chunker.py**

```python
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

from lxml import etree

from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
AKN = f"{{{AKN_NS}}}"

logger = logging.getLogger(__name__)


def chunk_xml(xml_path: Path, act_name: str) -> list[Chunk]:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    frbr_uri_el = root.find(f".//{AKN}FRBRExpression/{AKN}FRBRuri")
    frbr_uri = frbr_uri_el.get("value") if frbr_uri_el is not None else ""

    chunks: list[Chunk] = []
    for section in root.iter(f"{AKN}section"):
        eid = section.get("eId", "")
        num_el = section.find(f"{AKN}num")
        heading_el = section.find(f"{AKN}heading")
        section_num = num_el.text.strip() if num_el is not None and num_el.text else ""
        heading = heading_el.text.strip() if heading_el is not None and heading_el.text else None
        text = " ".join("".join(section.itertext()).split())
        chunks.append(Chunk(
            act_name=act_name,
            frbr_uri=frbr_uri,
            eid=eid,
            section_num=section_num,
            heading=heading,
            text=text,
        ))

    # Warn if body-level non-section children exist (untagged schedule/preface content)
    body = root.find(f".//{AKN}body")
    if body is not None:
        for child in body:
            local = child.tag.replace(f"{{{AKN_NS}}}", "")
            if local != "section":
                warnings.warn(
                    f"{act_name}: body contains <{local}> element outside <section> — "
                    "schedule or preface content may be excluded from the index. "
                    "This is resolved in lex-au v0.1.1.",
                    UserWarning,
                    stacklevel=2,
                )
                break

    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    chunks: list[Chunk] = []
    for entry in index["acts"].values():
        xml_path = corpus_dir / entry["xml_path"]
        act_chunks = chunk_xml(xml_path, entry["name"])
        chunks.extend(act_chunks)
        logger.info(f"{entry['name']}: {len(act_chunks)} sections")
    return chunks
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lexausearch/chunker.py tests/test_chunker.py
git commit -m "feat(lex-au-search): chunker — AKN XML section parser with schedule warning"
```

---

## Task 3: Indexer + Searcher

**Files:**
- Create: `src/lexausearch/indexer.py`
- Create: `src/lexausearch/searcher.py`
- Test: `tests/test_searcher.py`

**Interfaces:**
- Consumes: `Chunk`, `SearchResult` from `lexausearch.models`; `QdrantClient` from `qdrant_client`
- Produces:
  - `configure_client(client: QdrantClient) -> QdrantClient` (sets models; shared by Indexer and Searcher)
  - `Indexer(client: QdrantClient).upsert(chunks: list[Chunk]) -> None`
  - `Searcher(client: QdrantClient).search(query: str, limit: int, act: str | None) -> list[SearchResult]`

Note: First call to `Indexer` or `Searcher` triggers FastEmbed model download (~270 MB for nomic-embed-text-v1.5, cached at `~/.cache/fastembed/`). Subsequent runs use the cache.

- [ ] **Step 1: Write failing tests**

`projects/lex-au-search/repo/tests/test_searcher.py`:
```python
import pytest
from qdrant_client import QdrantClient

from lexausearch.indexer import Indexer
from lexausearch.searcher import Searcher
from lexausearch.models import SearchResult


@pytest.fixture(scope="module")
def indexed_searcher(privacy_chunks):
    """Real in-memory Qdrant + FastEmbed. Downloads model on first run (~270 MB, cached)."""
    client = QdrantClient(":memory:")
    indexer = Indexer(client)
    indexer.upsert(privacy_chunks)
    return Searcher(client)


def test_search_returns_results(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_result_has_score(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert all(isinstance(r.score, float) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_result_chunk_fields(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    r = results[0]
    assert r.chunk.act_name == "Privacy Act 1988"
    assert r.chunk.frbr_uri == "/akn/au/act/1988/119/eng@2026-01-01"
    assert r.chunk.eid in ("sec-3", "sec-6")
    assert r.chunk.text != ""


def test_search_limit_respected(indexed_searcher):
    results = indexed_searcher.search("information", limit=1)
    assert len(results) <= 1


def test_search_act_filter_excludes_other_acts(privacy_chunks):
    from lexausearch.models import Chunk
    extra_chunk = Chunk(
        act_name="Fair Work Act 2009",
        frbr_uri="/akn/au/act/2009/28/eng@2026-01-01",
        eid="sec-12",
        section_num="12",
        heading="Definitions",
        text="12 Definitions In this Act employee means a person employed.",
    )
    client = QdrantClient(":memory:")
    indexer = Indexer(client)
    indexer.upsert(privacy_chunks + [extra_chunk])
    searcher = Searcher(client)
    results = searcher.search("information", limit=5, act="Fair Work Act 2009")
    assert all(r.chunk.act_name == "Fair Work Act 2009" for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_searcher.py -v
```

Expected: `ImportError: cannot import name 'Indexer'`

- [ ] **Step 3: Create src/lexausearch/indexer.py**

```python
from __future__ import annotations

from qdrant_client import QdrantClient

from lexausearch.models import Chunk

DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "legislation"


def configure_client(client: QdrantClient) -> QdrantClient:
    client.set_model(DENSE_MODEL)
    client.set_sparse_model(SPARSE_MODEL)
    return client


class Indexer:
    def __init__(self, client: QdrantClient) -> None:
        self._client = configure_client(client)

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._client.add(
            collection_name=COLLECTION,
            documents=["search_document: " + c.text for c in chunks],
            metadata=[
                {
                    "act_name": c.act_name,
                    "frbr_uri": c.frbr_uri,
                    "eid": c.eid,
                    "section_num": c.section_num,
                    "heading": c.heading,
                    "text": c.text,
                }
                for c in chunks
            ],
        )
```

- [ ] **Step 4: Create src/lexausearch/searcher.py**

```python
from __future__ import annotations

from qdrant_client import QdrantClient, models

from lexausearch.indexer import COLLECTION, configure_client
from lexausearch.models import Chunk, SearchResult


class Searcher:
    def __init__(self, client: QdrantClient) -> None:
        self._client = configure_client(client)

    def search(
        self,
        query: str,
        limit: int = 5,
        act: str | None = None,
    ) -> list[SearchResult]:
        query_filter = None
        if act:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="act_name",
                        match=models.MatchValue(value=act),
                    )
                ]
            )

        hits = self._client.query(
            collection_name=COLLECTION,
            query_text="search_query: " + query,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            SearchResult(
                chunk=Chunk(
                    act_name=hit.metadata["act_name"],
                    frbr_uri=hit.metadata["frbr_uri"],
                    eid=hit.metadata["eid"],
                    section_num=hit.metadata["section_num"],
                    heading=hit.metadata["heading"],
                    text=hit.metadata["text"],
                ),
                score=hit.score,
            )
            for hit in hits
        ]
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
pytest tests/test_searcher.py -v
```

Note: First run downloads FastEmbed models (~270 MB). Subsequent runs use the cache.

Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/lexausearch/indexer.py src/lexausearch/searcher.py tests/test_searcher.py
git commit -m "feat(lex-au-search): indexer + searcher — hybrid Qdrant with nomic prefix"
```

---

## Task 4: FastAPI Search Endpoint

**Files:**
- Create: `src/lexausearch/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Searcher` from `lexausearch.searcher`
- Produces: `create_app(searcher: Searcher) -> FastAPI`

- [ ] **Step 1: Write failing tests**

`projects/lex-au-search/repo/tests/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from lexausearch.api import create_app
from lexausearch.indexer import Indexer
from lexausearch.searcher import Searcher


@pytest.fixture(scope="module")
def api_client(privacy_chunks):
    client = QdrantClient(":memory:")
    Indexer(client).upsert(privacy_chunks)
    searcher = Searcher(client)
    app = create_app(searcher)
    return TestClient(app)


def test_search_returns_200(api_client):
    response = api_client.get("/search?q=personal+information")
    assert response.status_code == 200


def test_search_response_shape(api_client):
    response = api_client.get("/search?q=personal+information&limit=2")
    data = response.json()
    assert "results" in data
    result = data["results"][0]
    assert "act_name" in result
    assert "frbr_uri" in result
    assert "eid" in result
    assert "section_num" in result
    assert "heading" in result
    assert "text" in result
    assert "score" in result


def test_search_limit_parameter(api_client):
    response = api_client.get("/search?q=information&limit=1")
    data = response.json()
    assert len(data["results"]) <= 1


def test_search_act_filter(api_client):
    response = api_client.get("/search?q=information&act=Privacy+Act+1988")
    data = response.json()
    assert all(r["act_name"] == "Privacy Act 1988" for r in data["results"])


def test_search_missing_q_returns_422(api_client):
    response = api_client.get("/search")
    assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError: cannot import name 'create_app'`

- [ ] **Step 3: Create src/lexausearch/api.py**

```python
from __future__ import annotations

from fastapi import FastAPI

from lexausearch.searcher import Searcher


def create_app(searcher: Searcher) -> FastAPI:
    app = FastAPI(title="lex-au-search", version="0.1.0")

    @app.get("/search")
    def search(q: str, limit: int = 10, act: str | None = None) -> dict:
        results = searcher.search(q, limit=limit, act=act)
        return {
            "results": [
                {
                    "act_name": r.chunk.act_name,
                    "frbr_uri": r.chunk.frbr_uri,
                    "eid": r.chunk.eid,
                    "section_num": r.chunk.section_num,
                    "heading": r.chunk.heading,
                    "text": r.chunk.text,
                    "score": r.score,
                }
                for r in results
            ]
        }

    return app
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lexausearch/api.py tests/test_api.py
git commit -m "feat(lex-au-search): FastAPI search endpoint GET /search"
```

---

## Task 5: MCP stdio Server

**Files:**
- Create: `src/lexausearch/mcp.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `Searcher` from `lexausearch.searcher`; `format_results` from `lexausearch.models`
- Produces: `run_mcp_server(searcher: Searcher) -> None` (async, runs until stdin closes)

- [ ] **Step 1: Write failing tests**

`projects/lex-au-search/repo/tests/test_mcp.py`:
```python
import os
import pytest
from unittest.mock import MagicMock

from lexausearch.models import Chunk, SearchResult, format_results
from lexausearch.mcp import make_search_tool_handler


def _make_result(eid: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid=eid,
            section_num=eid.split("-")[-1],
            heading="Interpretation",
            text="In this Act personal information means...",
        ),
        score=score,
    )


def test_search_tool_output_contains_section_reference():
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [_make_result("sec-3", 0.85)]
    handler = make_search_tool_handler(mock_searcher)
    output = handler(query="personal information", limit=5, act=None)
    assert "Privacy Act 1988" in output
    assert "sec-3" in output


def test_search_tool_output_contains_score():
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [_make_result("sec-3", 0.85)]
    handler = make_search_tool_handler(mock_searcher)
    output = handler(query="personal information", limit=5, act=None)
    assert "0.85" in output


def test_search_tool_passes_act_filter():
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = []
    handler = make_search_tool_handler(mock_searcher)
    handler(query="test", limit=3, act="Privacy Act 1988")
    mock_searcher.search.assert_called_once_with(
        "test", limit=3, act="Privacy Act 1988"
    )


def test_missing_storage_env_var_raises():
    from lexausearch.mcp import get_storage_path
    original = os.environ.pop("LEXAU_SEARCH_STORAGE", None)
    try:
        with pytest.raises(SystemExit, match="LEXAU_SEARCH_STORAGE"):
            get_storage_path()
    finally:
        if original is not None:
            os.environ["LEXAU_SEARCH_STORAGE"] = original


def test_storage_env_var_returns_path():
    from lexausearch.mcp import get_storage_path
    os.environ["LEXAU_SEARCH_STORAGE"] = "/tmp/test_storage"
    try:
        path = get_storage_path()
        assert str(path) == "/tmp/test_storage"
    finally:
        del os.environ["LEXAU_SEARCH_STORAGE"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mcp.py -v
```

Expected: `ImportError: cannot import name 'make_search_tool_handler'`

- [ ] **Step 3: Create src/lexausearch/mcp.py**

```python
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from lexausearch.models import format_results
from lexausearch.searcher import Searcher


def get_storage_path() -> Path:
    storage = os.environ.get("LEXAU_SEARCH_STORAGE")
    if not storage:
        raise SystemExit(
            "LEXAU_SEARCH_STORAGE env var is not set. "
            "Set it to the absolute path of your qdrant_storage directory."
        )
    return Path(storage)


def make_search_tool_handler(searcher: Searcher) -> Callable:
    def handler(query: str, limit: int = 5, act: str | None = None) -> str:
        results = searcher.search(query, limit=limit, act=act)
        if not results:
            return "No results found."
        return format_results(results)
    return handler


async def run_mcp_server(searcher: Searcher) -> None:
    server = Server("lex-au-search")
    search_handler = make_search_tool_handler(searcher)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_legislation",
                description=(
                    "Search Australian Commonwealth legislation by topic or question. "
                    "Returns relevant sections with FRBR citations and relevance scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question or topic to search for",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of sections to return",
                        },
                        "act": {
                            "type": "string",
                            "description": "Optional: restrict search to a specific Act by name (e.g. 'Privacy Act 1988')",
                        },
                    },
                    "required": ["query"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "search_legislation":
            output = search_handler(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                act=arguments.get("act"),
            )
            return [types.TextContent(type="text", text=output)]
        raise ValueError(f"Unknown tool: {name}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_mcp.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lexausearch/mcp.py tests/test_mcp.py
git commit -m "feat(lex-au-search): MCP stdio server with search_legislation tool"
```

---

## Task 6: CLI, README, and Smoke Test

**Files:**
- Create: `src/lexausearch/cli.py`
- Create: `README.md`

**Interfaces:**
- Consumes: all modules
- Produces: `lex-au-search ingest / serve / mcp` CLI commands

- [ ] **Step 1: Create src/lexausearch/cli.py**

```python
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import click
from qdrant_client import QdrantClient

from lexausearch.chunker import chunk_corpus
from lexausearch.indexer import Indexer
from lexausearch.searcher import Searcher
from lexausearch.api import create_app
from lexausearch.mcp import get_storage_path, run_mcp_server


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--corpus-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to lex-au corpus directory (contains index.json and xml/)",
)
@click.option(
    "--storage-dir",
    default="./qdrant_storage",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to Qdrant local storage directory",
)
def ingest(corpus_dir: Path, storage_dir: Path) -> None:
    """Build Qdrant index from lex-au AKN corpus."""
    click.echo(f"Chunking corpus at {corpus_dir} ...")
    chunks = chunk_corpus(corpus_dir)
    click.echo(f"  {len(chunks)} sections found across all Acts.")
    click.echo(f"Indexing into {storage_dir} ...")
    client = QdrantClient(path=str(storage_dir))
    Indexer(client).upsert(chunks)
    click.echo(f"Done. {len(chunks)} chunks indexed.")


@cli.command()
@click.option(
    "--storage-dir",
    default="./qdrant_storage",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--port", default=8000, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
def serve(storage_dir: Path, port: int, host: str) -> None:
    """Run FastAPI search server."""
    import uvicorn
    client = QdrantClient(path=str(storage_dir))
    searcher = Searcher(client)
    app = create_app(searcher)
    uvicorn.run(app, host=host, port=port)


@cli.command()
def mcp() -> None:
    """Run MCP stdio server for Claude Code integration."""
    storage = get_storage_path()
    client = QdrantClient(path=str(storage))
    searcher = Searcher(client)
    asyncio.run(run_mcp_server(searcher))
```

- [ ] **Step 2: Reinstall package to pick up entry point**

```bash
pip install -e ".[dev]"
lex-au-search --help
```

Expected output:
```
Usage: lex-au-search [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  ingest  Build Qdrant index from lex-au AKN corpus.
  mcp     Run MCP stdio server for Claude Code integration.
  serve   Run FastAPI search server.
```

- [ ] **Step 3: Run smoke test against real lex-au corpus**

```bash
lex-au-search ingest \
  --corpus-dir ../../lex-au/repo/corpus/ \
  --storage-dir ./qdrant_storage
```

Expected output (values approximate):
```
Chunking corpus at ../../lex-au/repo/corpus/ ...
  1243 sections found across all Acts.
Indexing into qdrant_storage ...
Done. 1243 chunks indexed.
```

If section count is 0, check that `corpus/index.json` exists and `corpus/xml/*.xml` files are present (run `lexau build --all` in the lex-au repo first).

- [ ] **Step 4: Verify serve works**

```bash
lex-au-search serve --storage-dir ./qdrant_storage &
curl "http://localhost:8000/search?q=notification+obligations+data+breach&limit=3" | python -m json.tool
```

Expected: JSON response with 3 results, each containing `act_name`, `frbr_uri`, `eid`, `score`.

Kill the server: `kill %1`

- [ ] **Step 5: Create README.md**

`projects/lex-au-search/repo/README.md`:
```markdown
# lex-au-search

Hybrid semantic search over Australian Commonwealth legislation — Layer 2 of the AU Legislative Intelligence Stack.

**Status: v0.1.0** — local-first, no Docker, no external APIs.

## Stack position

```
Layer 1: lex-au        — AKN 3.0 XML corpus
Layer 2: lex-au-search — hybrid search API + MCP  ← this project
Layer 3: ClauseKit     — rule extraction JSON
Agent demo             — NL query → attributed answer (future)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

**1. Build the lex-au corpus first** (see [lex-au](https://github.com/cchew/lex-au)):

```bash
lexau build --all --corpus-dir corpus/
```

**2. Index the corpus:**

```bash
lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/
```

**3. Search:**

```bash
# HTTP API
lex-au-search serve
curl "http://localhost:8000/search?q=notification+obligations+data+breach&limit=5"

# MCP (Claude Code)
# Add to your Claude Code settings.json — see below
```

## MCP Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "lex-au-search": {
      "command": "lex-au-search",
      "args": ["mcp"],
      "env": {
        "LEXAU_SEARCH_STORAGE": "/absolute/path/to/lex-au-search/repo/qdrant_storage"
      }
    }
  }
}
```

Then ask Claude: *"What are my notification obligations if an AI system causes harm to a consumer?"*

## API

```
GET /search?q=<query>&limit=10&act=<optional act name>

Response:
{
  "results": [
    {
      "act_name": "Privacy Act 1988",
      "frbr_uri": "/akn/au/act/1988/119/eng@2026-01-01",
      "eid": "part-III__sec-16",
      "section_num": "16",
      "heading": "Interference with privacy of an individual",
      "text": "...",
      "score": 0.847
    }
  ]
}
```

## Known limits (v0.1.0)

- Schedule content (e.g. Privacy Act Schedule 1 — APPs) not indexed (lex-au v0.1.0 gap; resolved in lex-au v0.1.1)
- No auth on HTTP API (local use only)
- Section-level citations only; subsection-level requires lex-au v0.1.1 + re-ingest
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS (test_models, test_chunker, test_searcher, test_api, test_mcp)

- [ ] **Step 7: Copy plan into repo**

```bash
cp ../../../docs/superpowers/plans/2026-06-19-lex-au-search.md docs/superpowers/plans/
```

- [ ] **Step 8: Commit**

```bash
git add src/lexausearch/cli.py README.md docs/
git commit -m "feat(lex-au-search): CLI wiring, README, smoke test passing"
```

- [ ] **Step 9: Create GitHub repo and push**

```bash
gh repo create cchew/lex-au-search --public --description "Hybrid search over Australian Commonwealth legislation — Layer 2 of the AU Legislative Intelligence Stack"
git remote add origin https://github.com/cchew/lex-au-search.git
git push -u origin main
```

- [ ] **Step 10: Tag v0.1.0**

```bash
git tag -a v0.1.0 -m "v0.1.0 — local hybrid search, FastAPI, MCP stdio"
git push origin v0.1.0
```
