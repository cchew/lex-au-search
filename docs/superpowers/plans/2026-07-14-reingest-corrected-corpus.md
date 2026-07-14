# Re-ingest Corrected Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to run this plan task-by-task (subagent-driven-development is unnecessary here — no code changes, just pipeline commands with verification checkpoints).

**Goal:** Re-index the `lex-au` corpus into Qdrant after the `list-definition-truncation` fix lands, so search results carry complete (not truncated) definition text in chunks that include a `<def>`.

**Architecture:** No code changes in this repo. `chunk_corpus()`/`Indexer` already chunk and embed whatever text is in the corpus XML — the fix is entirely upstream in `lex-au`. This plan re-runs the existing `ingest` CLI command against a fresh storage directory and verifies the result.

**Tech Stack:** Python 3.12, Qdrant (local storage), existing `lex-au-search` CLI. No new dependencies.

**Dependency:** requires `lex-au`'s `docs/superpowers/plans/2026-07-14-list-definition-truncation.md` corpus rebuild to have actually run (its Task 4 stop point + a subsequent user-approved rebuild). **Do not start this plan until that corpus rebuild is confirmed done** — re-ingesting the current, still-truncated corpus would just index the same truncated content this whole effort is meant to fix.

## Global Constraints

- Python ≥ 3.12. Run with `pytest` from the repo root (venv: `source .venv/bin/activate`) for the one existing-test-suite check in Task 2.
- Full spec: `../../lex-au/repo/docs/superpowers/specs/2026-07-14-list-definition-truncation-design.md`.
- Commit after every task using `caveman-commit` conventions.
- Re-ingest into a **fresh** storage directory, not the existing `./qdrant_storage` in place — a collection drop + rebuild is the established pattern for corpus content changes in this repo (same approach used for the v0.4.0 embedding-model migration), avoiding stale/duplicate points from the old truncated chunks.

---

## Task 1: Re-ingest into fresh storage

**Files:** none modified — this task runs existing CLI commands only.

### Step 1: Confirm the lex-au corpus rebuild landed

Run:
```bash
ls -la ../lex-au/repo/corpus/xml/ | wc -l
grep -c "means <def>" ../lex-au/repo/corpus/xml/bankruptcy-act-1966.xml
```
Expected: file count is unchanged from the current 2,944 (this fix doesn't add/remove Acts, only completes existing `<def>` content), and the `bankruptcy-act-1966.xml` grep should show fewer truncated matches than before the fix (spot-check, not an exact assertion — exact expected count depends on the corpus-wide rebuild's real output, verify manually that `related entity`'s `<def>` in that file now contains "a relative of the person" rather than just "any of the following:").

If the corpus doesn't look rebuilt, STOP and confirm with the user before proceeding — do not re-ingest a still-truncated corpus.

### Step 2: Run the re-ingest into fresh storage

Run:
```bash
mv ./qdrant_storage ./qdrant_storage.pre-fix-backup
lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/ --storage-dir ./qdrant_storage
```
Expected: completes with a summary line `Done. N chunks + M Act records indexed.` — N and M should be in the same order of magnitude as the previous ingest (this fix changes `<def>` text length, not chunk count or Act count).

### Step 3: Commit nothing (no code changed) — log completion

This task produces no diff (storage directories are gitignored data, not tracked). Skip the commit step; proceed directly to Task 2's verification.

---

## Task 2: Verify complete definitions are now searchable

**Files:**
- Create: `scripts/verify_reingest.py`

### Step 1: Write the verification script

Create `scripts/verify_reingest.py`:

```python
"""One-off verification: confirm a chunk containing a previously-truncated
definition now carries the complete list content after re-ingest.
"""
from pathlib import Path
from qdrant_client import QdrantClient
from lexausearch.searcher import Searcher

STORAGE_DIR = Path("./qdrant_storage")


def main() -> None:
    client = QdrantClient(path=str(STORAGE_DIR))
    searcher = Searcher(client)

    results = searcher.search(
        "related entity means",
        limit=5,
        act="Bankruptcy Act 1966",
    )
    print(f"Results for 'related entity means' in Bankruptcy Act 1966: {len(results)}")
    for r in results:
        text = r.chunk.text
        has_lead_in_only = text.strip().endswith("any of the following:")
        has_list_content = "a relative of the person" in text
        print(f"  score={r.score:.3f} eid={r.chunk.eid}")
        print(f"    truncated (bug still present): {has_lead_in_only}")
        print(f"    complete (fix present): {has_list_content}")


if __name__ == "__main__":
    main()
```

### Step 2: Run it

Run: `python scripts/verify_reingest.py`

Expected: at least one result shows `complete (fix present): True`. If every result shows `truncated (bug still present): True` and none show the fix present, STOP — the re-ingest either ran against the wrong corpus path or the corpus rebuild (Task 1, Step 1) didn't actually land.

### Step 3: Run existing test suite (regression check, not new tests)

Run: `python -m pytest -q`
Expected: all existing tests still pass — this repo's own test suite doesn't test corpus content, so no test count change is expected, just confirmation nothing broke.

### Step 4: Commit the verification script, clean up backup storage

```bash
git add scripts/verify_reingest.py
git commit -m "test: add re-ingest verification script"
rm -rf ./qdrant_storage.pre-fix-backup
```

**STOP POINT.** Do not redeploy any consumer (e.g. the MCP server, any hosted search API) pointing at this storage directory without explicit user go-ahead. Report the verification script's output and stop here.
