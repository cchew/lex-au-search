from lexausearch.models import Chunk, SearchResult, format_results


def test_chunk_fields():
    chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-3",
        section_num="3",
        heading="Interpretation",
        text="In this Act personal information means information about an identified individual.",
    )
    assert chunk.act_name == "Privacy Act 1988"
    assert chunk.eid == "sec-3"
    assert chunk.heading == "Interpretation"


def test_chunk_heading_optional():
    chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-3",
        section_num="3",
        heading=None,
        text="Some text.",
    )
    assert chunk.heading is None


def test_format_results_contains_frbr_uri():
    chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-3",
        section_num="3",
        heading="Interpretation",
        text="In this Act personal information means information about an identified individual.",
    )
    result = SearchResult(chunk=chunk, score=0.85)
    output = format_results([result])
    assert "Privacy Act 1988" in output
    assert "sec-3" in output
    assert "0.85" in output
    assert chunk.text in output


def test_format_results_score_on_header_line():
    chunk = Chunk(
        act_name="Fair Work Act 2009",
        frbr_uri="/akn/au/act/2009/28/eng@2026-01-01",
        eid="part-1__sec-12",
        section_num="12",
        heading="Definitions",
        text="Some definitions here.",
    )
    result = SearchResult(chunk=chunk, score=0.71)
    output = format_results([result])
    lines = output.strip().split("\n")
    header = lines[0]
    assert "score=0.71" in header
    assert "part-1__sec-12" in header


def test_format_results_multiple():
    chunks = [
        Chunk("A Act 2000", "/akn/au/act/2000/1/eng@2026-01-01", "sec-1", "1", "Short title", "Text one."),
        Chunk("B Act 2001", "/akn/au/act/2001/2/eng@2026-01-01", "sec-2", "2", None, "Text two."),
    ]
    results = [SearchResult(c, 0.9 - i * 0.1) for i, c in enumerate(chunks)]
    output = format_results(results)
    assert "A Act 2000" in output
    assert "B Act 2001" in output
