"""Long provisions: the dense embedder reads only the first ~512 tokens of a chunk, so the rest of a
long section, subsection or clause was findable by keyword only. Each long chunk keeps its whole-provision
chunk (stable eId, citation) and gains tail windows `<eId>#wN` for text no child chunk already covers."""
from pathlib import Path

from lexausearch.chunker import chunk_xml

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _act(body: str, schedule: str = "") -> str:
    att = f"<attachments><attachment>{schedule}</attachment></attachments>" if schedule else ""
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{NS}"><act name="act"><meta><identification source="#lex-au">
<FRBRWork><FRBRthis value="/akn/au/act/2000/1/!main"/><FRBRuri value="/akn/au/act/2000/1"/><FRBRdate date="2000" name="Generation"/><FRBRauthor href="#p"/></FRBRWork>
<FRBRExpression><FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main"/><FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01"/><FRBRdate date="2026-01-01" name="Generation"/><FRBRauthor href="#p"/><FRBRlanguage language="eng"/></FRBRExpression>
<FRBRManifestation><FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/><FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/><FRBRdate date="2026-06-19" name="Generation"/><FRBRauthor href="#l"/></FRBRManifestation>
</identification></meta><body>{body}</body>{att}</act></akomaNtoso>"""


def _para(tag: str, n: int, words: int = 150) -> str:
    return f"<p>{tag}{n} " + " ".join(f"{tag}{n}w{i}" for i in range(words)) + "</p>"


def _chunks(tmp_path: Path, body: str, schedule: str = ""):
    p = tmp_path / "act.xml"
    p.write_text(_act(body, schedule))
    return chunk_xml(p, "Test Act 2000")


def test_long_section_without_subsections_gets_tail_windows(tmp_path):
    paras = "".join(_para("s", n) for n in range(8))  # 8 x 151 words = 1,208 words
    chunks = _chunks(tmp_path, f'<section eId="sec-5"><num>5</num><heading>Long thing</heading><content>{paras}</content></section>')
    whole = next(c for c in chunks if c.eid == "sec-5")
    windows = [c for c in chunks if c.eid.startswith("sec-5#w")]
    assert len(whole.text.split()) > 1000
    assert windows, "a 1,200-word section needs tail windows"
    assert [c.eid for c in windows] == [f"sec-5#w{i}" for i in range(1, len(windows) + 1)]
    assert all(c.provision_type == "section" and c.provision_num == "5" for c in windows)
    assert all(c.text.startswith("5 Long thing") for c in windows)  # context prefix
    assert all(len(c.text.split()) <= 520 for c in windows)
    # every word after the head window is in some tail window
    tail_words = set(" ".join(c.text for c in windows).split())
    assert {"s7w149", "s6w10", "s4w0"} <= tail_words


def test_short_section_gets_no_windows(tmp_path):
    chunks = _chunks(tmp_path, f'<section eId="sec-1"><num>1</num><heading>H</heading><content>{_para("a", 1, 300)}</content></section>')
    assert [c.eid for c in chunks] == ["sec-1"]


def test_long_section_made_of_subsections_is_not_duplicated(tmp_path):
    subs = "".join(
        f'<subsection eId="sec-9__subsec-{n}"><num>({n})</num><content>{_para("u", n)}</content></subsection>' for n in range(1, 8)
    )
    chunks = _chunks(tmp_path, f'<section eId="sec-9"><num>9</num><heading>Subs</heading>{subs}</section>')
    eids = [c.eid for c in chunks]
    assert "sec-9" in eids and sum(1 for e in eids if e.startswith("sec-9__subsec-")) == 7
    assert not [e for e in eids if e.startswith("sec-9#w")], "text covered by subsection chunks needs no windows"


def test_long_subsection_gets_windows(tmp_path):
    sub = f'<subsection eId="sec-2__subsec-1"><num>(1)</num><content>{"".join(_para("q", n) for n in range(6))}</content></subsection>'
    chunks = _chunks(tmp_path, f'<section eId="sec-2"><num>2</num><heading>H</heading>{sub}</section>')
    windows = [c for c in chunks if c.eid.startswith("sec-2__subsec-1#w")]
    assert windows and all(c.provision_type == "subsection" for c in windows)


def test_long_schedule_clause_gets_windows_and_keeps_schedule_num(tmp_path):
    clause = (
        '<hcontainer name="schedule" eId="schedule-2"><num>2</num>'
        f'<hcontainer name="clause" eId="schedule-2__clause-59"><num>59</num><heading>Disclosure</heading><content>{"".join(_para("c", n) for n in range(10))}</content></hcontainer>'
        "</hcontainer>"
    )
    chunks = _chunks(tmp_path, '<section eId="sec-1"><num>1</num><content><p>Short.</p></content></section>', clause)
    windows = [c for c in chunks if c.eid.startswith("schedule-2__clause-59#w")]
    assert windows
    assert all(c.provision_type == "schedule_clause" and c.schedule_num == "2" and c.provision_num == "59" for c in windows)
    assert any(c.eid == "schedule-2__clause-59" for c in chunks)
