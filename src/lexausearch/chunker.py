from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from lxml import etree

from lexausearch.models import Chunk
from lexausearch.refs import extract_refs

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
AKN = f"{{{AKN_NS}}}"

logger = logging.getLogger(__name__)


def _element_text(el: etree._Element) -> str:
    """Recursively extract text from an AKN element.

    Tables use ' | ' between cells instead of no separator.
    authorialNote text is included (contextually relevant for embedding).
    """
    local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
    if local == "table":
        rows: list[str] = []
        for tr in el:
            tr_local = tr.tag.split("}")[-1] if "}" in tr.tag else tr.tag
            if tr_local == "tr":
                cells = [" ".join(c.itertext()).strip() for c in tr]
                row = " | ".join(c for c in cells if c)
                if row:
                    rows.append(row)
        return " ".join(rows)
    parts: list[str] = []
    if el.text and el.text.strip():
        parts.append(el.text.strip())
    for child in el:
        child_text = _element_text(child)
        if child_text:
            parts.append(child_text)
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return " ".join(parts)


def _extract_sections(
    root: etree._Element,
    frbr_uri: str,
    act_name: str,
    corpus_index: dict[str, str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in root.iter(f"{AKN}section"):
        eid = section.get("eId", "")
        num_el = section.find(f"{AKN}num")
        heading_el = section.find(f"{AKN}heading")
        provision_num = num_el.text.strip() if num_el is not None and num_el.text else ""
        heading = heading_el.text.strip() if heading_el is not None and heading_el.text else None
        text = " ".join(_element_text(section).split())
        chunks.append(Chunk(
            act_name=act_name,
            frbr_uri=frbr_uri,
            eid=eid,
            provision_num=provision_num,
            provision_type="section",
            heading=heading,
            text=text,
            refs=extract_refs(text, corpus_index),
        ))
    return chunks


def _section_num_from_parent(el: etree._Element) -> str:
    """Climb the element tree to find the enclosing <section>'s <num> text."""
    parent = el.getparent()
    while parent is not None:
        local = parent.tag.split("}")[-1] if "}" in parent.tag else parent.tag
        if local == "section":
            num_el = parent.find(f"{AKN}num")
            if num_el is not None and num_el.text:
                return num_el.text.strip()
            break
        parent = parent.getparent()
    return ""


def _subsec_num_from_eid(eid: str) -> str:
    """Extract the subsection ordinal from a compound eId, e.g. 'sec-3__subsec-1' → '(1)'."""
    for part in reversed(eid.split("__")):
        if part.startswith("subsec-"):
            return f"({part[len('subsec-'):]}"
    return ""


def _extract_subsections(
    root: etree._Element,
    frbr_uri: str,
    act_name: str,
    corpus_index: dict[str, str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for subsec in root.iter(f"{AKN}subsection"):
        eid = subsec.get("eId", "")
        text = " ".join(_element_text(subsec).split())
        if len(text) < 20:
            continue
        sec_num = _section_num_from_parent(subsec)
        subsec_suffix = _subsec_num_from_eid(eid)
        provision_num = f"{sec_num}{subsec_suffix}" if sec_num else subsec_suffix
        heading_el = subsec.find(f"{AKN}heading")
        heading = heading_el.text.strip() if heading_el is not None and heading_el.text else None
        chunks.append(Chunk(
            act_name=act_name,
            frbr_uri=frbr_uri,
            eid=eid,
            provision_num=provision_num,
            provision_type="subsection",
            heading=heading,
            text=text,
            refs=extract_refs(text, corpus_index),
        ))
    return chunks


def _extract_schedule_clauses(
    root: etree._Element,
    frbr_uri: str,
    act_name: str,
    corpus_index: dict[str, str],
) -> list[Chunk]:
    # lxml findall requires './/'-prefixed paths; bare '//' is not supported.
    schedule_clause_xpath = (
        f".//{AKN}attachments/{AKN}attachment"
        f"/{AKN}hcontainer[@name='schedule']"
        f"/{AKN}hcontainer[@name='clause']"
    )
    chunks: list[Chunk] = []
    for clause in root.findall(schedule_clause_xpath):
        eid = clause.get("eId", "")
        num_el = clause.find(f"{AKN}num")
        heading_el = clause.find(f"{AKN}heading")
        # Builder emits bare digit/decimal (e.g. "1", "1.1"), not "APP 1"
        provision_num = num_el.text.strip() if num_el is not None and num_el.text else ""
        heading = heading_el.text.strip() if heading_el is not None and heading_el.text else None
        text = " ".join(_element_text(clause).split())
        chunks.append(Chunk(
            act_name=act_name,
            frbr_uri=frbr_uri,
            eid=eid,
            provision_num=provision_num,
            provision_type="schedule_clause",
            heading=heading,
            text=text,
            refs=extract_refs(text, corpus_index),
        ))
    return chunks


def chunk_xml(
    xml_path: Path,
    act_name: str,
    corpus_index: dict[str, str] | None = None,
) -> list[Chunk]:
    if corpus_index is None:
        corpus_index = {}
    tree = etree.parse(xml_path)
    root = tree.getroot()

    frbr_uri_el = root.find(f".//{AKN}FRBRExpression/{AKN}FRBRuri")
    frbr_uri = frbr_uri_el.get("value") if frbr_uri_el is not None else ""

    sections = _extract_sections(root, frbr_uri, act_name, corpus_index)
    subsections = _extract_subsections(root, frbr_uri, act_name, corpus_index)
    clauses = _extract_schedule_clauses(root, frbr_uri, act_name, corpus_index)
    chunks = sections + subsections + clauses
    logger.info(
        f"{act_name}: {len(sections)} sections, {len(subsections)} subsections, {len(clauses)} schedule clauses"
    )
    return chunks


def load_corpus_act_names(corpus_dir: Path) -> set[str]:
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    return {entry["name"] for entry in index["acts"].values() if "name" in entry}


def missing_acts(corpus_act_names: set[str], indexed_act_names: set[str]) -> list[str]:
    return sorted(corpus_act_names - indexed_act_names)


def load_corpus_act_entries(corpus_dir: Path) -> list[dict]:
    """Ordered list of index.json's act entries (name + xml_path), in the
    dict's stable insertion order - the same order chunk_corpus() iterates,
    so shard slicing stays deterministic across runs of the same file."""
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    return [entry for entry in index["acts"].values() if "name" in entry]


def shard_count(total_acts: int, shard_size: int) -> int:
    """Number of shards (ceiling division) needed to cover total_acts."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return -(-total_acts // shard_size)


def shard_bounds(total_acts: int, shard_index: int, shard_size: int) -> tuple[int, int]:
    """[start, end) Act-list slice bounds for shard_index, clipped to total_acts."""
    start = min(shard_index * shard_size, total_acts)
    end = min(start + shard_size, total_acts)
    return (start, end)


def chunk_corpus_for_acts(corpus_dir: Path, act_names: set[str]) -> list[Chunk]:
    """Like chunk_corpus() but only parses/chunks Acts in act_names - bounds
    the in-memory chunk list to one shard's worth instead of the whole
    corpus. corpus_index for cross-Act ref resolution is still built from
    the FULL index.json (not just act_names), so refs to Acts outside this
    shard still resolve correctly."""
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    corpus_index = {
        entry["name"]: entry.get("frbr_expression_uri", "")
        for entry in index["acts"].values()
        if "name" in entry
    }
    chunks: list[Chunk] = []
    for entry in index["acts"].values():
        if "name" not in entry or entry["name"] not in act_names:
            continue
        xml_path = corpus_dir / entry["xml_path"]
        act_chunks = chunk_xml(xml_path, entry["name"], corpus_index)
        chunks.extend(act_chunks)
    return chunks


def compute_act_content_hashes(
    corpus_dir: Path, act_names: set[str] | None = None
) -> dict[str, str]:
    """SHA-256 hex digest of each Act's raw XML file bytes, keyed by Act
    name. act_names=None hashes every Act in the corpus; a set restricts to
    just those (mirrors chunk_corpus_for_acts's filtering shape). Hashes the
    raw file, not extracted chunk text, so both a legislative amendment and
    a lex-au re-extraction-only fix (same comp_num, different parser output)
    are detected uniformly."""
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    hashes: dict[str, str] = {}
    for entry in index["acts"].values():
        name = entry.get("name")
        if not name or (act_names is not None and name not in act_names):
            continue
        xml_path = corpus_dir / entry["xml_path"]
        hashes[name] = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    return hashes


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    # Build corpus_index for cross-Act ref resolution
    corpus_index = {
        entry["name"]: entry.get("frbr_expression_uri", "")
        for entry in index["acts"].values()
        if "name" in entry
    }
    chunks: list[Chunk] = []
    for entry in index["acts"].values():
        xml_path = corpus_dir / entry["xml_path"]
        act_chunks = chunk_xml(xml_path, entry["name"], corpus_index)
        chunks.extend(act_chunks)
    return chunks
