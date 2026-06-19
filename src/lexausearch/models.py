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
