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


# lex-au index.json "structure_version" values this chunker has been tested
# against (lex-au corpus.py STRUCTURE_VERSION): 1 flat schedules, 2 nested
# part/division/clause and amendedAct/item/quotedStructure. An unknown value
# means lex-au changed shape again; refuse rather than silently drop content.
SUPPORTED_STRUCTURE_VERSIONS = frozenset({1, 2})


class UnsupportedCorpusStructure(RuntimeError):
    pass


def check_structure_version(index: dict) -> None:
    version = index.get("structure_version")
    if version is None:
        logger.warning(
            "corpus index.json has no structure_version (pre-2026-10-15 lex-au export); "
            "assuming a supported shape"
        )
        return
    if version not in SUPPORTED_STRUCTURE_VERSIONS:
        raise UnsupportedCorpusStructure(
            f"corpus structure_version {version!r} is not supported by this chunker "
            f"(supported: {sorted(SUPPORTED_STRUCTURE_VERSIONS)}). lex-au changed the AKN "
            "shape; update and re-test the chunker (tests/test_chunker_schedule_coverage.py) "
            "before ingesting."
        )


_SCHEDULE_CHUNK_MAX_WORDS = 400
# Windows packed by a child already carry its context prefix; allow that slack so a
# parent never re-cuts them mid-paragraph.
_SCHEDULE_CHUNK_SLACK_WORDS = 100


def _local(el: etree._Element) -> str:
    return el.tag.split("}")[-1] if isinstance(el.tag, str) and "}" in el.tag else str(el.tag)


def _is_schedule_clause(el: etree._Element) -> bool:
    return el.tag == f"{AKN}hcontainer" and el.get("name") == "clause"


def _is_chunked_in_attachment(el: etree._Element) -> bool:
    """True for elements that already become a section, subsection or clause chunk.

    Mirrors _extract_sections/_extract_subsections/_extract_schedule_clauses so
    the schedule-text pass only picks up what those three leave out.
    """
    if el.tag == f"{AKN}section" or _is_schedule_clause(el):
        return True
    if el.tag == f"{AKN}subsection":
        return len(" ".join(_element_text(el).split())) >= 20
    return False


def _words(text: str) -> int:
    return len(text.split())


def _window_words(text: str, prefix: str) -> list[str]:
    words = text.split()
    limit = _SCHEDULE_CHUNK_MAX_WORDS
    return [
        (prefix + " " if i and prefix else "") + " ".join(words[i : i + limit])
        for i in range(0, len(words), limit)
    ]


def _pack_texts(parts: list[str], prefix: str) -> list[str]:
    """Greedily join consecutive texts into windows of at most the word cap.

    A single part above the cap is split on word boundaries.
    """
    windows: list[str] = []
    current: list[str] = []
    count = 0
    for part in parts:
        n = _words(part)
        if n > _SCHEDULE_CHUNK_MAX_WORDS + _SCHEDULE_CHUNK_SLACK_WORDS:
            if current:
                windows.append(" ".join(current))
                current, count = [], 0
            windows.extend(_window_words(part, prefix))
            continue
        if current and count + n > _SCHEDULE_CHUNK_MAX_WORDS:
            windows.append(" ".join(current))
            current, count = [], 0
        current.append(part)
        count += n
    if current:
        windows.append(" ".join(current))
    return [(prefix + " " + w) if i and prefix else w for i, w in enumerate(windows)]


_BLOCK_TAGS = frozenset({
    "hcontainer", "paragraph", "subparagraph", "subsection", "section", "content",
    "p", "table", "tr", "td", "th", "blockList", "item", "list", "quotedStructure",
    "heading", "num", "intro", "wrapUp", "blockContainer", "tblock", "mod", "container",
})


def _split_subtree(node: etree._Element) -> list[str]:
    """Text units for a subtree with no chunked descendant, capped in size.

    An oversized node is split along its block children, keeping its own text
    and the text between children. A node whose children are all inline markup
    is split on word boundaries instead.
    """
    text = " ".join(_element_text(node).split())
    if _words(text) <= _SCHEDULE_CHUNK_MAX_WORDS:
        return [text] if text else []
    children = [c for c in node if isinstance(c.tag, str)]
    prefix = _context(node)
    if not any(_local(c) in _BLOCK_TAGS for c in children):
        return _window_words(text, prefix)
    parts: list[str] = []
    if node.text and node.text.strip():
        parts.append(" ".join(node.text.split()))
    for child in children:
        parts.extend(_split_subtree(child))
        if child.tail and child.tail.strip():
            parts.append(" ".join(child.tail.split()))
    return _pack_texts(parts, prefix)


def _context(node: etree._Element) -> str:
    bits = []
    for tag in ("num", "heading"):
        el = node.find(f"{AKN}{tag}")
        if el is not None:
            t = " ".join("".join(el.itertext()).split())
            if t:
                bits.append(t)
    return " ".join(bits)


def _nearest_eid(node: etree._Element) -> str:
    cur: etree._Element | None = node
    while cur is not None:
        eid = cur.get("eId")
        if eid:
            return eid
        cur = cur.getparent()
    return ""


def _schedule_text_units(
    attachment: etree._Element,
) -> list[tuple[etree._Element, str, str]]:
    """(node, text, eid_suffix) for attachment content no other pass chunks."""
    covered: set[etree._Element] = set()
    holds_chunked: set[etree._Element] = set()
    for el in attachment.iter():
        if not isinstance(el.tag, str) or not _is_chunked_in_attachment(el):
            continue
        covered.add(el)
        anc = el.getparent()
        while anc is not None and anc is not attachment.getparent() and anc not in holds_chunked:
            holds_chunked.add(anc)
            anc = anc.getparent()

    units: list[tuple[etree._Element, str, str]] = []
    outline: list[tuple[etree._Element, str]] = []

    def walk(node: etree._Element) -> None:
        children = [c for c in node if isinstance(c.tag, str)]
        if node not in holds_chunked:
            for i, text in enumerate(_split_subtree(node)):
                units.append((node, text, f"#t{i}" if i else ""))
            return
        residual = [
            c for c in children
            if c not in covered and c not in holds_chunked
            and _local(c) not in ("num", "heading")
        ]
        parts = [t for c in residual for t in _split_subtree(c)]
        header = _context(node)
        if header and not parts:
            outline.append((node, header))
        for i, text in enumerate(_pack_texts(parts, header)):
            full = text if (i or not header) else f"{header} {text}"
            units.append((node, full, f"#r{i}"))
        for c in children:
            if c in holds_chunked and c not in covered:
                walk(c)

    for child in attachment:
        if isinstance(child.tag, str) and child not in covered:
            outline.clear()
            walk(child)
            labels = [label for _, label in outline]
            for i, text in enumerate(_pack_texts(labels, "")):
                units.append((child, text, f"#outline{i}"))
    return units


def _extract_schedule_clauses(
    root: etree._Element,
    frbr_uri: str,
    act_name: str,
    corpus_index: dict[str, str],
) -> list[Chunk]:
    """Schedule chunks: every clause at any depth, plus schedule text no
    section/subsection/clause chunk covers (amendment items, part/division
    lead-in text, tables and prose in clause-less schedules).

    lex-au v0.10.0 nests schedules as schedule > part > division > clause and
    amending Acts as schedule > amendedAct > item > quotedStructure, so a
    direct-child clause match alone dropped ~7M words of schedule text.
    """
    chunks: list[Chunk] = []
    for attachment in root.iter(f"{AKN}attachment"):
        for clause in attachment.iter(f"{AKN}hcontainer"):
            if not _is_schedule_clause(clause):
                continue
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
        for node, text, suffix in _schedule_text_units(attachment):
            if not any(ch.isalpha() for ch in text):
                continue
            num_el = node.find(f"{AKN}num")
            heading_el = node.find(f"{AKN}heading")
            chunks.append(Chunk(
                act_name=act_name,
                frbr_uri=frbr_uri,
                eid=f"{_nearest_eid(node)}{suffix}",
                provision_num=(
                    " ".join("".join(num_el.itertext()).split()) if num_el is not None else ""
                ),
                provision_type="schedule_text",
                heading=(
                    " ".join("".join(heading_el.itertext()).split())
                    if heading_el is not None else None
                ) or None,
                text=text,
                refs=extract_refs(text, corpus_index),
            ))
    return chunks


_SCHEDULE_GAP_MIN_WORDS = 500
_SCHEDULE_GAP_MIN_RATIO = 0.5


def _warn_on_schedule_gap(
    root: etree._Element, act_name: str, chunks: list[Chunk]
) -> None:
    """Warn when an Act's attachments hold far more text than its chunks reach.

    Catches the next schema change that hides schedule content from the
    extractors, instead of letting it show up as a silent coverage drop. Chunk
    words are an upper bound (nested chunks overlap), so a low ratio is a real gap.
    """
    attachment_words = sum(
        _words(" ".join(_element_text(a).split()))
        for a in root.iter(f"{AKN}attachment")
    )
    if attachment_words < _SCHEDULE_GAP_MIN_WORDS:
        return
    covered_words = sum(
        _words(c.text) for c in chunks if c.eid.startswith("schedule-")
    )
    if covered_words < attachment_words * _SCHEDULE_GAP_MIN_RATIO:
        logger.warning(
            f"{act_name}: schedule coverage gap - {covered_words} chunked words "
            f"vs {attachment_words} attachment words"
        )


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
    _warn_on_schedule_gap(root, act_name, chunks)
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
    check_structure_version(index)
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
    check_structure_version(index)
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
