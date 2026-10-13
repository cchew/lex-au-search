"""Restore the published lex-au corpus from Hugging Face into a local
directory, so this repo's ingest-delta pipeline has something to chunk
against in a fresh CI checkout (corpus/ is not checked into this repo).

Copied from lex-au-graph's script of the same name (same read-only-consumer
situation: this repo never writes back to cchew/lex-au, no sync stamp
needed). allow_patterns restricts the download to what chunk_corpus()
actually reads (index.json + xml/*.xml) -- lex-au-graph hit HF's rate limit
partway through an unfiltered snapshot_download from a GitHub Actions
runner (cchew/lex-au publishes 11,387 files total including reports/ and
images/); scoping the download avoids that class of failure here too.

Even filtered, this is still ~3,085 individual small files, not one big
blob -- observed live 2026-10-13 on a Colab GPU box taking noticeably
longer than a single multi-GB embed_cache.db download in the same
session, despite being the smaller transfer by total bytes. Per-file
request overhead dominates for many small files; snapshot_download's
own concurrency (max_workers, default 8) is the lever, not file size.
Bumped the default here -- still well short of the point where
lex-au-graph's rate-limit lesson would bite (that was about an
unfiltered 11,387-file pull, not concurrency on an already-filtered
~3,085-file one).
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

_ALLOW_PATTERNS = ["index.json", "xml/*.xml"]
_DEFAULT_MAX_WORKERS = 16


def restore(repo: str, local_dir: str, max_workers: int = _DEFAULT_MAX_WORKERS) -> str:
    return snapshot_download(
        repo_id=repo, repo_type="dataset", local_dir=local_dir,
        allow_patterns=_ALLOW_PATTERNS, max_workers=max_workers,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="cchew/lex-au", help="HF dataset repo")
    parser.add_argument("--local-dir", default="corpus")
    parser.add_argument("--max-workers", type=int, default=_DEFAULT_MAX_WORKERS,
                        help="Concurrent file downloads (HF default is 8)")
    args = parser.parse_args()

    path = restore(args.repo, args.local_dir, max_workers=args.max_workers)
    print(f"Restored corpus from {args.repo} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
