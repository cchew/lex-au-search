import pytest
from lexausearch.models import (
    Chunk, ActRecord, ActSearchResult, SearchResult, format_results
)


def _chunk(**kwargs) -> Chunk:
    defaults = dict(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-6",
        provision_num="6",
        provision_type="section",
        heading="Definitions",
        text="Personal information means...",
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def test_chunk_has_provision_num():
    c = _chunk(provision_num="16")
    assert c.provision_num == "16"


def test_chunk_has_provision_type():
    c = _chunk(provision_type="section")
    assert c.provision_type == "section"


def test_chunk_refs_defaults_empty():
    c = _chunk()
    assert c.refs == []


def test_chunk_refs_stored():
    c = _chunk(refs=["#sec-3", "#part-III"])
    assert c.refs == ["#sec-3", "#part-III"]


def test_act_record_fields():
    r = ActRecord(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        year=1988,
        as_at_date="2026-01-01",
        section_count=42,
        schedule_clause_count=13,
    )
    assert r.year == 1988
    assert r.as_at_date == "2026-01-01"
    assert r.section_count == 42
    assert r.schedule_clause_count == 13


def test_act_search_result_fields():
    r = ActRecord(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        year=1988, as_at_date="2026-01-01",
        section_count=1, schedule_clause_count=0,
    )
    result = ActSearchResult(record=r, score=0.88)
    assert result.record.act_name == "Privacy Act 1988"
    assert result.score == 0.88


def test_format_results_section_citation():
    results = [SearchResult(chunk=_chunk(
        provision_num="6", provision_type="section", heading="Definitions"
    ), score=0.95)]
    output = format_results(results)
    assert "Privacy Act 1988 1988 s 6" in output
    assert "2026-01-01" in output


def test_format_results_schedule_clause_citation():
    results = [SearchResult(chunk=_chunk(
        eid="schedule-1__clause-2",
        provision_num="2",
        provision_type="schedule_clause",
        heading="Anonymity",
    ), score=0.90)]
    output = format_results(results)
    assert "Privacy Act 1988 1988 Sch 1 cl 2" in output


def test_format_results_filters_unresolved_refs():
    results = [SearchResult(chunk=_chunk(
        refs=["#sec-3", "unresolved:Privacy Regulation 2013"],
    ), score=0.9)]
    output = format_results(results)
    assert "#sec-3" in output
    assert "unresolved:" not in output


def test_format_results_no_heading():
    results = [SearchResult(chunk=_chunk(heading=None), score=0.8)]
    output = format_results(results)
    assert "Privacy Act 1988" in output


def test_format_results_no_refs_no_refs_line():
    results = [SearchResult(chunk=_chunk(refs=[]), score=0.8)]
    output = format_results(results)
    assert "Refs:" not in output
