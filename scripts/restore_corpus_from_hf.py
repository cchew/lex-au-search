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
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

_ALLOW_PATTERNS = ["index.json", "xml/*.xml"]


def restore(repo: str, local_dir: str) -> str:
    return snapshot_download(
        repo_id=repo, repo_type="dataset", local_dir=local_dir, allow_patterns=_ALLOW_PATTERNS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="cchew/lex-au", help="HF dataset repo")
    parser.add_argument("--local-dir", default="corpus")
    args = parser.parse_args()

    path = restore(args.repo, args.local_dir)
    print(f"Restored corpus from {args.repo} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
