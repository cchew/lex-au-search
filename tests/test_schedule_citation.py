"""Schedule citations use the printed Schedule number (<num>), not the eId ordinal.

lex-au numbers schedule eIds by position (schedule-1, schedule-2, ...), so an
Act with Schedules 1, 2, 2A has eIds schedule-1..3. Citations must read
"Sch 2A", not "Sch 3". Fallback to the ordinal only when <num> is missing
(e.g. an unnumbered Dictionary schedule).
"""
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from lexausearch.api import create_app
from lexausearch.cache import EmbedCache
from lexausearch.chunker import chunk_xml
from lexausearch.indexer import COLLECTION_SECTIONS, Indexer
from lexausearch.models import Chunk, SearchResult, format_results
from lexausearch.searcher import Searcher
from tests.conftest import AKN_NS

FRBR = "/akn/au/act/1920/50/eng@2024-10-14"
ACT = "Air Navigation Act 1920"

SUFFIXED_XML = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta><identification source="#lex-au"><FRBRExpression>
      <FRBRuri value="{FRBR}"/>
    </FRBRExpression></identification></meta>
    <body>
      <section eId="sec-1"><num>1</num><heading>Short title</heading>
        <content><p>This Act may be cited as the Air Navigation Act 1920.</p></content>
      </section>
    </body>
    <attachments>
      <attachment>
        <hcontainer name="schedule" eId="schedule-1">
          <num>1</num><heading>Convention</heading>
          <hcontainer name="clause" eId="schedule-1__clause-1">
            <num>1</num><heading>Scope</heading>
            <content><p>The convention applies to civil aircraft.</p></content>
          </hcontainer>
        </hcontainer>
      </attachment>
      <attachment>
        <hcontainer name="schedule" eId="schedule-2">
          <num>9A</num><heading>Protocol relating to an amendment</heading>
          <content><p>The protocol amends the convention in respect of airspace.</p></content>
        </hcontainer>
      </attachment>
      <attachment>
        <hcontainer name="schedule" eId="schedule-3">
          <num>2A</num><heading>Classification rules</heading>
          <hcontainer name="part" eId="schedule-3__part-2">
            <num>2</num><heading>Rules</heading>
            <intro><p>The following rules apply to medical devices.</p></intro>
            <hcontainer name="clause" eId="schedule-3__part-2__clause-1">
              <num>1.1</num><heading>Rule one</heading>
              <content><p>A device is class 1 unless otherwise stated.</p></content>
            </hcontainer>
          </hcontainer>
        </hcontainer>
      </attachment>
      <attachment>
        <hcontainer name="schedule" eId="schedule-4">
          <heading>Dictionary</heading>
          <content><p>aircraft means any machine that can derive support in the atmosphere.</p></content>
        </hcontainer>
      </attachment>
    </attachments>
  </act>
</akomaNtoso>"""


def _chunks(tmp_path):
    xml = tmp_path / "air.xml"
    xml.write_bytes(SUFFIXED_XML.encode())
    return chunk_xml(xml, ACT, {})


def _cite(c: Chunk) -> str:
    head = format_results([SearchResult(chunk=c, score=0.0)]).split("\n", 1)[0]
    return head[1 : head.index(f" ({c.eid}) score=")]


def _sched(**kw) -> Chunk:
    d = dict(act_name=ACT, frbr_uri=FRBR, heading=None, text="x")
    d.update(kw)
    return Chunk(**d)


# --- chunker records the printed label ---------------------------------------

def test_chunker_records_printed_schedule_num(tmp_path):
    by_eid = {c.eid: c for c in _chunks(tmp_path)}
    assert by_eid["schedule-1__clause-1"].schedule_num == "1"
    assert by_eid["schedule-2"].schedule_num == "9A"
    assert by_eid["schedule-3__part-2__clause-1"].schedule_num == "2A"
    assert by_eid["schedule-3__part-2#r0"].schedule_num == "2A"


def test_chunker_falls_back_to_ordinal_when_num_missing(tmp_path):
    by_eid = {c.eid: c for c in _chunks(tmp_path)}
    assert by_eid["schedule-4"].schedule_num == "4"


def test_body_chunks_have_no_schedule_num(tmp_path):
    by_eid = {c.eid: c for c in _chunks(tmp_path)}
    assert by_eid["sec-1"].schedule_num == ""


# --- citation strings --------------------------------------------------------

def test_citations_from_chunked_xml(tmp_path):
    cites = {c.eid: _cite(c) for c in _chunks(tmp_path)}
    assert cites["schedule-1__clause-1"] == f"{ACT} 1920 Sch 1 cl 1"
    assert cites["schedule-2"] == f"{ACT} 1920 Sch 9A"
    assert cites["schedule-3__part-2__clause-1"] == f"{ACT} 1920 Sch 2A cl 1.1"
    assert cites["schedule-3__part-2#r0"] == f"{ACT} 1920 Sch 2A 2"
    assert cites["schedule-4"] == f"{ACT} 1920 Sch 4"
    assert cites["sec-1"] == f"{ACT} 1920 s 1"


def test_schedule_clause_cites_printed_num():
    c = _sched(eid="schedule-3__clause-1", provision_num="1",
               provision_type="schedule_clause", schedule_num="2A")
    assert _cite(c) == f"{ACT} 1920 Sch 2A cl 1"


def test_schedule_level_text_does_not_repeat_the_number():
    c = _sched(eid="schedule-10", provision_num="9A",
               provision_type="schedule_text", schedule_num="9A")
    assert _cite(c) == f"{ACT} 1920 Sch 9A"


def test_split_schedule_level_text_drops_eid_suffix():
    c = _sched(eid="schedule-10#t1", provision_num="9A",
               provision_type="schedule_text", schedule_num="9A")
    assert _cite(c) == f"{ACT} 1920 Sch 9A"


def test_legacy_payload_without_schedule_num_uses_ordinal_without_suffix():
    c = _sched(eid="schedule-10#t1", provision_num="9A", provision_type="schedule_text")
    assert _cite(c) == f"{ACT} 1920 Sch 10"


def test_part_within_schedule_keeps_part_suffix():
    c = _sched(eid="schedule-3__part-2#r0", provision_num="2",
               provision_type="schedule_text", schedule_num="2A")
    assert _cite(c) == f"{ACT} 1920 Sch 2A 2"


# --- payload round trip (no re-embedding needed, field is payload only) -------

def _assert_payload_round_trip(indexer: Indexer, client: QdrantClient, tmp_path) -> None:
    indexer.upsert_chunks(_chunks(tmp_path))
    points, _ = client.scroll(COLLECTION_SECTIONS, limit=100, with_payload=True)
    by_eid = {p.payload["eid"]: p.payload for p in points}
    assert by_eid["schedule-3__part-2__clause-1"]["schedule_num"] == "2A"
    assert by_eid["sec-1"]["schedule_num"] == ""
    hit = Searcher(client).search("medical device class", limit=10, act=ACT)
    got = {r.chunk.eid: r.chunk.schedule_num for r in hit}
    assert got.get("schedule-3__part-2__clause-1") == "2A"


def test_payload_round_trip_without_cache(tmp_path):
    client = QdrantClient(":memory:")
    _assert_payload_round_trip(Indexer(client), client, tmp_path)


def test_payload_round_trip_with_cache(tmp_path):
    client = QdrantClient(":memory:")
    _assert_payload_round_trip(Indexer(client, cache=EmbedCache(":memory:")), client, tmp_path)


def test_searcher_tolerates_payload_without_schedule_num():
    client = QdrantClient(":memory:")
    Indexer(client).upsert_chunks([_sched(
        eid="schedule-1__clause-1", provision_num="1",
        provision_type="schedule_clause", text="legacy schedule clause text",
    )])
    points, _ = client.scroll(COLLECTION_SECTIONS, limit=1, with_payload=True)
    client.delete_payload(COLLECTION_SECTIONS, keys=["schedule_num"], points=[points[0].id])
    r = Searcher(client).search("legacy schedule clause", limit=1)[0]
    assert r.chunk.schedule_num == ""
    assert _cite(r.chunk) == f"{ACT} 1920 Sch 1 cl 1"


def test_api_exposes_schedule_num(tmp_path):
    client = QdrantClient(":memory:")
    Indexer(client).upsert_chunks(_chunks(tmp_path))
    api = TestClient(create_app(Searcher(client), client))
    res = api.get("/search", params={"q": "medical device class", "act": ACT}).json()["results"]
    assert {r["eid"]: r["schedule_num"] for r in res}["schedule-3__part-2__clause-1"] == "2A"
    secs = api.get(f"/legislation/{ACT}/sections").json()["chunks"]
    assert {c["eid"]: c["schedule_num"] for c in secs}["schedule-2"] == "9A"
