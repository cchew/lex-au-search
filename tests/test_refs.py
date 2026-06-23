from lexausearch.refs import extract_refs

CORPUS = {"Privacy Act 1988": "/akn/au/act/1988/119/eng@2026-01-01"}


def test_3level_inline():
    refs = extract_refs("as provided in s 6(1)(a) of this Act", CORPUS)
    assert "#sec-6__subsec-1__para-a" in refs


def test_2level_inline():
    refs = extract_refs("under s 16(2) of this Act", CORPUS)
    assert "#sec-16__subsec-2" in refs


def test_1level_section():
    refs = extract_refs("see section 16 for details", CORPUS)
    assert "#sec-16" in refs


def test_1level_s_shorthand():
    refs = extract_refs("as required by s 26WA", CORPUS)
    assert "#sec-26WA" in refs


def test_part():
    refs = extract_refs("under Part III of this Act", CORPUS)
    assert "#part-III" in refs


def test_division():
    refs = extract_refs("as set out in Division 3", CORPUS)
    assert "#dvs-3" in refs


def test_schedule_clause():
    refs = extract_refs("as required by Schedule 1, clause 3", CORPUS)
    assert "#schedule-1__clause-3" in refs


def test_schedule_item():
    refs = extract_refs("see Schedule 2, item 4 of this Act", CORPUS)
    assert "#schedule-2__clause-4" in refs


def test_paragraph_ref():
    refs = extract_refs("under paragraph (a) of this subsection", CORPUS)
    assert "#para-a" in refs


def test_para_shorthand():
    refs = extract_refs("see para (b)", CORPUS)
    assert "#para-b" in refs


def test_cross_act_resolved():
    refs = extract_refs("under the Privacy Act 1988", CORPUS)
    assert "/akn/au/act/1988/119/eng@2026-01-01" in refs


def test_cross_act_unresolved():
    refs = extract_refs("under the Fair Work Act 2009", CORPUS)
    assert "unresolved:Fair Work Act 2009" in refs


def test_subsidiary_unresolved():
    refs = extract_refs("under the Privacy Regulation 2013", CORPUS)
    assert "unresolved:Privacy Regulation 2013" in refs


def test_deduplication():
    refs = extract_refs("section 6 and section 6 again", CORPUS)
    assert refs.count("#sec-6") == 1


def test_part_div_not_match_of_the():
    # "Part III of the Privacy Act 1988" — Part/Division regex must NOT match
    # (negative lookahead prevents false same-Act Part ref when "of the" follows)
    refs = extract_refs("under Part III of the Privacy Act 1988", CORPUS)
    assert "#part-III" not in refs


def test_empty_text():
    assert extract_refs("", CORPUS) == []


def test_empty_corpus():
    refs = extract_refs("under the Privacy Act 1988", {})
    assert "unresolved:Privacy Act 1988" in refs
