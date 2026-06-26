from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Chunk:
    act_name: str
    frbr_uri: str
    eid: str
    provision_num: str        # "16" for sections; "3(1)" for subsections; "1" or "1.1" for schedule clauses
    provision_type: str       # "section", "subsection", or "schedule_clause"
    heading: str | None
    text: str
    refs: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)  # AKN <TLCTerm> eIds, e.g. "term-personal-information"


@dataclass
class ActRecord:
    act_name: str
    frbr_uri: str
    year: int                 # enactment year from FRBR URI (e.g. 1988)
    as_at_date: str           # compilation date from FRBRExpression FRBRdate (e.g. "2026-01-01")
    section_count: int
    schedule_clause_count: int


@dataclass
class ActSearchResult:
    record: ActRecord
    score: float


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _schedule_num_from_eid(eid: str) -> str:
    for part in eid.split("__"):
        if part.startswith("schedule-"):
            return part[len("schedule-"):]
    return "?"


def _as_at_from_frbr_uri(frbr_uri: str) -> str:
    # /akn/au/act/1988/119/eng@2026-01-01  →  "2026-01-01"
    if "@" in frbr_uri:
        return frbr_uri.split("@")[-1].split("/")[0]
    return ""


def _year_from_frbr_uri(frbr_uri: str) -> str:
    # /akn/au/act/1988/119/eng@2026-01-01  →  "1988"
    parts = frbr_uri.split("/")
    return parts[4] if len(parts) > 4 else ""


def format_results(results: list[SearchResult]) -> str:
    parts = []
    for r in results:
        c = r.chunk
        year = _year_from_frbr_uri(c.frbr_uri)
        as_at = _as_at_from_frbr_uri(c.frbr_uri)
        if c.provision_type == "schedule_clause":
            sched_num = _schedule_num_from_eid(c.eid)
            citation = f"{c.act_name} {year} Sch {sched_num} cl {c.provision_num}"
        else:
            citation = f"{c.act_name} {year} s {c.provision_num}"
        heading_line = f"\n{c.heading}" if c.heading else ""
        as_at_line = f" (as at {as_at})" if as_at else ""
        visible_refs = [ref for ref in c.refs if not ref.startswith("unresolved:")]
        refs_line = f"\nRefs: {', '.join(visible_refs)}" if visible_refs else ""
        block = (
            f"[{citation} ({c.eid}) score={r.score:.2f}{as_at_line}]"
            f"{heading_line}\n{c.text}{refs_line}"
        )
        parts.append(block)
    return "\n\n".join(parts)
