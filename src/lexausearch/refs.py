from __future__ import annotations

import re

# Patterns applied in specificity order (most specific first).
# 3-level must precede 2-level which must precede 1-level to avoid double-matching
# the same citation at different granularities.

_P1_3LEVEL = re.compile(r'\bs\s*(\d+[A-Z]*)\((\d+[A-Z]*)\)\(([a-z]+)\)')
_P2_2LEVEL = re.compile(r'\bs\s*(\d+[A-Z]*)\((\d+[A-Z]*)\)')
_P3_1LEVEL = re.compile(r'\b(?:section|s)\s+(\d+[A-Z]*)')
# Part/Division: restricted char class [\dIVX]+[A-Z]? avoids matching bare words
# Negative lookahead (?!\s+of\s+the) prevents false same-Act refs in cross-Act citations
_P4_PART_DIV = re.compile(r'\b(Part|Division)\s+([\dIVX]+[A-Z]?)(?!\s+of\s+the)')
# Schedule + clause/item — e.g. "Schedule 1, clause 3" or "Schedule 2, item 4"
_P5_SCHED_CLAUSE = re.compile(
    r'\bSchedule\s+(\d+)[,\s]+(?:clause|item|Part)\s+(\d+[A-Z]?)',
    re.IGNORECASE,
)
# Paragraph-only refs — "paragraph (a)" or "para (b)"
_P6_PARA = re.compile(r'\b(?:paragraph|para)\s+\(([a-z])\)')
# Cross-Act: "the Privacy Act 1988" — must follow subsidiary pattern to avoid overlap
_P7_CROSS_ACT = re.compile(r'\bthe\s+([A-Z][A-Za-z ]+?Act\s+\d{4})')
# Subsidiary legislation
_P8_SUBSIDIARY = re.compile(
    r'\bthe\s+([A-Z][A-Za-z ]+?(?:Regulation|Instrument|Order|Rules?)\s+\d{4})'
)


def extract_refs(text: str, corpus_index: dict[str, str]) -> list[str]:
    """Extract AU legislative citation strings from provision text.

    corpus_index maps act_name -> frbr_uri for cross-Act resolution.
    Returns deduplicated list. Unresolved refs are stored as 'unresolved:{Name}'.
    Filter 'unresolved:' entries before displaying to users.
    """
    refs: list[str] = []

    for m in _P1_3LEVEL.finditer(text):
        refs.append(f"#sec-{m.group(1)}__subsec-{m.group(2)}__para-{m.group(3)}")

    for m in _P2_2LEVEL.finditer(text):
        refs.append(f"#sec-{m.group(1)}__subsec-{m.group(2)}")

    for m in _P3_1LEVEL.finditer(text):
        refs.append(f"#sec-{m.group(1)}")

    for m in _P4_PART_DIV.finditer(text):
        prefix = "part" if m.group(1) == "Part" else "dvs"
        refs.append(f"#{prefix}-{m.group(2)}")

    for m in _P5_SCHED_CLAUSE.finditer(text):
        refs.append(f"#schedule-{m.group(1)}__clause-{m.group(2)}")

    for m in _P6_PARA.finditer(text):
        refs.append(f"#para-{m.group(1)}")

    for m in _P7_CROSS_ACT.finditer(text):
        name = m.group(1).strip()
        if name in corpus_index:
            refs.append(corpus_index[name])
        else:
            refs.append(f"unresolved:{name}")

    for m in _P8_SUBSIDIARY.finditer(text):
        refs.append(f"unresolved:{m.group(1).strip()}")

    # Deduplicate preserving order
    return list(dict.fromkeys(refs))
