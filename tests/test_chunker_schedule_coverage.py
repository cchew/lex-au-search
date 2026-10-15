"""Schedule-content coverage for the chunker.

lex-au v0.10.0 (commit 3b1b9be, 2026-09-15) nests schedule content as
schedule > part > division > clause, and amending Acts as
schedule > amendedAct > item > quotedStructure. The chunker used to match only
schedule/clause as a direct child, so ~7M words of schedule text produced no
chunk. These tests pin the invariant: every non-empty <p> under an attachment
must appear in at least one chunk.
"""
from pathlib import Path

import pytest
from lxml import etree

from lexausearch.chunker import _element_text, chunk_xml

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
AKN = f"{{{AKN_NS}}}"
MAX_CHUNK_WORDS = 1200

_HEADER = """\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="@@NS@@">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork>
          <FRBRthis value="/akn/au/act/2000/1/!main"/>
          <FRBRuri value="/akn/au/act/2000/1"/>
          <FRBRdate date="2000" name="Generation"/>
          <FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/>
          <FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation>
          <FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/>
          <FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-1"><num>1</num><heading>Short title</heading>
        <content><p>This Act may be cited as the Test Act 2000.</p></content>
      </section>
    </body>
    <attachments>
      <attachment>
@@SCHEDULE@@
      </attachment>
    </attachments>
  </act>
</akomaNtoso>
"""

NESTED_CLAUSES = """\
<hcontainer name="schedule" eId="schedule-2">
  <num>2</num><heading>Detention</heading>
  <hcontainer name="part" eId="schedule-2__part-2">
    <num>2</num><heading>Detaining suspected illegal foreign fishers</heading>
    <hcontainer name="division" eId="schedule-2__part-2__dvs-4">
      <num>4</num><heading>Personal information</heading>
      <hcontainer name="clause" eId="schedule-2__part-2__dvs-4__clause-59">
        <num>59</num><heading>Disclosure of detainees personal information</heading>
        <content><p>Detainee records may be disclosed only to an authorised agency.</p></content>
      </hcontainer>
    </hcontainer>
  </hcontainer>
</hcontainer>"""

AMENDMENT_ITEMS = """\
<hcontainer name="schedule" eId="schedule-1">
  <num>1</num><heading>Amendments</heading>
  <hcontainer name="amendedAct" eId="schedule-1__amdact-1">
    <heading>Fair Work Act 2009</heading>
    <hcontainer name="item" eId="schedule-1__amdact-1__item-1">
      <num>1</num>
      <content><p>After section 23 insert the following provision.</p></content>
    </hcontainer>
    <hcontainer name="item" eId="schedule-1__amdact-1__item-2">
      <num>2</num>
      <content><p>Omit the words the Minister, substitute the Secretary.</p></content>
      <quotedStructure>
        <section eId="schedule-1__amdact-1__item-2__qstr-1__sec-23A">
          <num>23A</num><heading>Superannuation terms</heading>
          <content><p>A modern award must include a default fund term.</p></content>
        </section>
      </quotedStructure>
    </hcontainer>
  </hcontainer>
</hcontainer>"""

LOOSE_PROSE = """\
<hcontainer name="schedule" eId="schedule-3">
  <num>3</num><heading>Rates</heading>
  <hcontainer name="part" eId="schedule-3__part-1">
    <num>1</num>
    <content><p>The rate of levy is set by regulation for each financial year.</p></content>
    <hcontainer name="item" eId="schedule-3__part-1__item-1">
      <num>1</num>
      <content><p>Item one prescribes the base rate of levy.</p></content>
    </hcontainer>
  </hcontainer>
</hcontainer>"""

FLAT_CLAUSES = """\
<hcontainer name="schedule" eId="schedule-1">
  <heading>Principles</heading>
  <hcontainer name="clause" eId="schedule-1__clause-1">
    <num>1</num><heading>Openness</heading>
    <content><p>Entities must manage information openly.</p></content>
  </hcontainer>
  <hcontainer name="clause" eId="schedule-1__clause-2">
    <num>2</num><heading>Anonymity</heading>
    <content><p>Individuals may choose not to identify themselves.</p></content>
  </hcontainer>
</hcontainer>"""


def _build(tmp_path: Path, schedule: str) -> Path:
    path = tmp_path / "test-act-2000.xml"
    path.write_text(
        _HEADER.replace("@@NS@@", AKN_NS).replace("@@SCHEDULE@@", schedule),
        encoding="utf-8",
    )
    return path


def _paragraph_coverage(path: Path) -> tuple[int, int, list[str]]:
    """(covered, total, missing sample) over non-empty <p> under attachments."""
    chunks = chunk_xml(path, "Test Act 2000")
    haystack = [" ".join(c.text.split()) for c in chunks]
    root = etree.parse(str(path)).getroot()
    total = covered = 0
    missing: list[str] = []
    for att in root.iter(f"{AKN}attachment"):
        for p in att.iter(f"{AKN}p"):
            text = " ".join(_element_text(p).split())
            if not text:
                continue
            total += 1
            if any(text in h for h in haystack):
                covered += 1
            elif len(missing) < 5:
                missing.append(text[:80])
    return covered, total, missing


@pytest.mark.parametrize(
    "schedule",
    [NESTED_CLAUSES, AMENDMENT_ITEMS, LOOSE_PROSE, FLAT_CLAUSES],
    ids=["nested-clauses", "amendment-items", "loose-prose", "flat-clauses"],
)
def test_every_attachment_paragraph_is_covered(tmp_path, schedule):
    covered, total, missing = _paragraph_coverage(_build(tmp_path, schedule))
    assert total > 0
    assert covered == total, f"uncovered attachment paragraphs: {missing}"


def test_clause_nested_under_part_and_division_is_chunked(tmp_path):
    chunks = chunk_xml(_build(tmp_path, NESTED_CLAUSES), "Test Act 2000")
    hits = [c for c in chunks if "Detainee records" in c.text]
    assert len(hits) == 1
    assert hits[0].provision_type == "schedule_clause"
    assert hits[0].provision_num == "59"


def test_amendment_item_instruction_is_chunked(tmp_path):
    chunks = chunk_xml(_build(tmp_path, AMENDMENT_ITEMS), "Test Act 2000")
    texts = " ".join(c.text for c in chunks)
    assert "After section 23 insert the following provision" in texts
    assert "substitute the Secretary" in texts


def test_flat_clauses_still_produce_exactly_two_clause_chunks(tmp_path):
    chunks = chunk_xml(_build(tmp_path, FLAT_CLAUSES), "Test Act 2000")
    assert len([c for c in chunks if c.provision_type == "schedule_clause"]) == 2


def test_oversized_schedule_is_split_not_one_giant_chunk(tmp_path):
    paragraphs = "".join(
        f"<p>Rate row {i} " + "word " * 40 + "</p>" for i in range(200)
    )
    schedule = (
        '<hcontainer name="schedule" eId="schedule-9"><num>9</num>'
        f"<content>{paragraphs}</content></hcontainer>"
    )
    chunks = chunk_xml(_build(tmp_path, schedule), "Test Act 2000")
    sched = [c for c in chunks if "Rate row" in c.text]
    assert len(sched) > 1
    assert max(len(c.text.split()) for c in sched) <= MAX_CHUNK_WORDS
    covered, total, missing = _paragraph_coverage(_build(tmp_path, schedule))
    assert covered == total, missing


_CORPUS = Path(__file__).resolve().parent.parent / "hf-corpus"


@pytest.mark.skipif(not (_CORPUS / "xml").exists(), reason="hf-corpus not restored")
@pytest.mark.parametrize(
    "stem,needle,floor",
    [
        ("torres-strait-fisheries-act-1984", "detainees", 0.95),
        ("fair-work-amendment-act-2012", "default fund term", 0.90),
    ],
)
def test_real_act_schedule_coverage(stem, needle, floor):
    path = _CORPUS / "xml" / f"{stem}.xml"
    chunks = chunk_xml(path, stem)
    assert any(needle in c.text.lower() for c in chunks)
    covered, total, missing = _paragraph_coverage(path)
    assert total > 0
    assert covered / total >= floor, f"{covered}/{total} covered; e.g. {missing}"


def test_format_results_cites_schedule_text_as_a_schedule():
    from lexausearch.models import Chunk, SearchResult, format_results

    chunk = Chunk(
        act_name="Test Act", frbr_uri="/akn/au/act/2000/1/eng@2026-01-01",
        eid="schedule-1__amdact-1__item-3", provision_num="3",
        provision_type="schedule_text", heading=None, text="Omit the word, substitute another.",
    )
    out = format_results([SearchResult(chunk=chunk, score=0.5)])
    assert "Sch 1 3" in out
    assert " s 3" not in out


def test_chunk_xml_warns_when_attachment_text_is_unreachable(tmp_path, caplog, monkeypatch):
    import lexausearch.chunker as chunker

    monkeypatch.setattr(chunker, "_schedule_text_units", lambda attachment: [])
    big = "<content>" + "".join(f"<p>{'word ' * 40}row {i}</p>" for i in range(30)) + "</content>"
    schedule = f'<hcontainer name="schedule" eId="schedule-1"><num>1</num>{big}</hcontainer>'
    with caplog.at_level("WARNING", logger="lexausearch.chunker"):
        chunk_xml(_build(tmp_path, schedule), "Test Act 2000")
    assert any("schedule coverage gap" in r.message for r in caplog.records)


def test_clause_free_schedule_chunks_carry_the_schedule_eid(tmp_path):
    schedule = (
        '<hcontainer name="schedule" eId="schedule-4"><num>4</num>'
        "<heading>Exempt agencies</heading>"
        "<content><p>The Australian Secret Intelligence Service is an exempt agency.</p></content>"
        "</hcontainer>"
    )
    chunks = chunk_xml(_build(tmp_path, schedule), "Test Act 2000")
    sched = [c for c in chunks if c.provision_type == "schedule_text"]
    assert sched and all(c.eid.startswith("schedule-4") for c in sched)


def _own_text_uncovered(path: Path) -> list[str]:
    """Own text of every element under an attachment that no ancestor-eId chunk contains.

    Stricter than the paragraph check: covers headings, numbers and table cells.
    """
    import collections

    chunks = chunk_xml(path, "Test Act 2000")
    by: dict[str, list[str]] = collections.defaultdict(list)
    for c in chunks:
        by[c.eid.split("#")[0]].append(" ".join(c.text.split()))
    root = etree.parse(str(path)).getroot()
    missing: list[str] = []
    for att in root.iter(f"{AKN}attachment"):
        for el in att.iter():
            if not isinstance(el.tag, str):
                continue
            text = " ".join((el.text or "").split())
            if not any(ch.isalpha() for ch in text):
                continue
            anc: etree._Element | None = el
            found = False
            while anc is not None and not found:
                eid = anc.get("eId")
                found = bool(eid) and any(text in c for c in by.get(eid, ()))
                anc = anc.getparent()
            if not found:
                missing.append(text[:80])
    return missing


WRAPPER_HEADINGS = """\
<hcontainer name="schedule" eId="schedule-5">
  <num>5</num><heading>Procedures</heading>
  <hcontainer name="part" eId="schedule-5__part-1">
    <num>1</num><heading>Preliminary matters</heading>
    <hcontainer name="division" eId="schedule-5__part-1__dvs-1">
      <num>1</num><heading>Objects of this Schedule</heading>
      <hcontainer name="clause" eId="schedule-5__part-1__dvs-1__clause-1">
        <num>1</num><heading>Main object</heading>
        <content><p>The main object is to prescribe procedures.</p></content>
      </hcontainer>
    </hcontainer>
  </hcontainer>
</hcontainer>"""

QUOTED_HEADINGS = """\
<hcontainer name="schedule" eId="schedule-1">
  <num>1</num><heading>Amendments</heading>
  <hcontainer name="amendedAct" eId="schedule-1__amdact-1">
    <heading>Rules Act 2015</heading>
    <hcontainer name="item" eId="schedule-1__amdact-1__item-1">
      <num>1</num>
      <content><p>Insert:</p></content>
      <quotedStructure>
        <content><p>Forms</p></content>
        <section eId="schedule-1__amdact-1__item-1__qstr-1__sec-9">
          <num>9</num><heading>Approved forms</heading>
          <content><p>The Secretary may approve forms for this Act.</p></content>
        </section>
      </quotedStructure>
    </hcontainer>
  </hcontainer>
</hcontainer>"""


@pytest.mark.parametrize(
    "schedule",
    [NESTED_CLAUSES, AMENDMENT_ITEMS, LOOSE_PROSE, FLAT_CLAUSES, WRAPPER_HEADINGS, QUOTED_HEADINGS],
    ids=["nested", "items", "loose", "flat", "wrapper-headings", "quoted-headings"],
)
def test_every_element_text_under_an_attachment_is_covered(tmp_path, schedule):
    assert _own_text_uncovered(_build(tmp_path, schedule)) == []


def test_wrapper_headings_are_listed_in_a_schedule_outline(tmp_path):
    chunks = chunk_xml(_build(tmp_path, WRAPPER_HEADINGS), "Test Act 2000")
    outline = [c for c in chunks if "#outline" in c.eid]
    assert outline
    text = " ".join(c.text for c in outline)
    assert "Preliminary matters" in text and "Objects of this Schedule" in text


def test_short_quoted_heading_is_not_dropped(tmp_path):
    chunks = chunk_xml(_build(tmp_path, QUOTED_HEADINGS), "Test Act 2000")
    assert any("Forms" in c.text for c in chunks if c.provision_type == "schedule_text")


def test_packed_windows_are_not_recut_mid_paragraph(tmp_path):
    # 8 paragraphs of 50 words fill a child window to exactly the 400-word cap; the
    # child's context prefix then pushes later windows just over it. A parent that
    # re-cut those windows at 400 words would split a paragraph across two chunks.
    def part(i: int) -> str:
        paras = "".join(
            f"<p>Marker{i}x{j} " + "filler " * 49 + "</p>" for j in range(24)
        )
        return (
            f'<hcontainer name="part" eId="schedule-7__part-{i}"><num>{i}</num>'
            f"<heading>Long heading for part {i}</heading><content>{paras}</content>"
            "</hcontainer>"
        )

    schedule = (
        '<hcontainer name="schedule" eId="schedule-7"><num>7</num>'
        + part(1) + part(2) + "</hcontainer>"
    )
    assert _own_text_uncovered(_build(tmp_path, schedule)) == []
