"""One-off helper: pull just shard 0's corpus slice from HF at a pinned revision.

Written for the v0.5.0 shard-0 recovery (2026-09-04/05): the pod's embed cache
was keyed to `cchew/lex-au` HEAD as it stood before a same-day corpus update,
so rebuilding shard 0's vector store from that cache required downloading the
corpus at the exact pre-update revision, not HEAD, to avoid spurious cache
misses. Edit PINNED_REV (and the shard index/size below) to reuse for a
different shard or a future revision-pin situation.
"""
import os, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from huggingface_hub import snapshot_download
from lexausearch.chunker import load_corpus_act_entries, shard_bounds

tok = os.environ["HF_CACHE_WRITE_TOKEN"]
PINNED_REV = "f0a89da476227b5f3dbb861be1b666d1234ac501"  # pre-run cchew/lex-au HEAD (2026-08-30 22:05:03 UTC)
t0 = time.time()
snapshot_download(repo_id="cchew/lex-au", repo_type="dataset", local_dir="corpus",
                  allow_patterns=["index.json"], token=tok, revision=PINNED_REV)
entries = load_corpus_act_entries(Path("corpus"))
start, end = shard_bounds(len(entries), 0, 300)
xml_paths = [e["xml_path"] for e in entries[start:end]]
print(f"shard 0 = Acts [{start}:{end}] of {len(entries)}; {len(xml_paths)} XML files")
snapshot_download(repo_id="cchew/lex-au", repo_type="dataset", local_dir="corpus",
                  allow_patterns=xml_paths, token=tok, revision=PINNED_REV)
print(f"corpus subset downloaded in {time.time() - t0:.0f}s")
