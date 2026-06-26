import json
import pytest
from pathlib import Path
from lxml import etree

from lexausearch.chunker import chunk_xml, chunk_corpus
from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def test_chunk_xml_returns_one_chunk_per_section(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    assert len(chunks) == 2


def test_chunk_xml_frbr_uri_from_xml(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    assert all(c.frbr_uri == "/akn/au/act/1988/119/eng@2026-01-01" for c in chunks)


def test_chunk_xml_section_fields(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    s3 = next(c for c in chunks if c.eid == "sec-3")
    assert s3.act_name == "Privacy Act 1988"
    assert s3.provision_num == "3"
    assert s3.heading == "Interpretation"
    assert "personal information" in s3.text


def test_chunk_xml_itertext_captures_all_paragraphs(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988")
    s6 = next(c for c in chunks if c.eid == "sec-6")
    # Both <p> elements should be in the text
    assert "opinion" in s6.text
    assert "recorded in material form" in s6.text


def test_chunk_xml_missing_heading_is_none(tmp_path):
    xml = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork><FRBRthis value="/akn/au/act/2000/1/!main"/><FRBRuri value="/akn/au/act/2000/1"/>
          <FRBRdate date="2000" name="Generation"/><FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression><FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/><FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation><FRBRthis value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/2000/1/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/><FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-1">
        <num>1</num>
        <content><p>Text without heading.</p></content>
      </section>
    </body>
  </act>
</akomaNtoso>"""
    xml_file = tmp_path / "test.xml"
    xml_file.write_bytes(xml.encode())
    chunks = chunk_xml(xml_file, "Test Act 2000")
    assert chunks[0].heading is None



def test_chunk_corpus(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML
    # Build a minimal corpus directory
    corpus_dir = tmp_path / "corpus"
    xml_dir = corpus_dir / "xml"
    xml_dir.mkdir(parents=True)
    xml_path = xml_dir / "privacy-act-1988.xml"
    xml_path.write_bytes(PRIVACY_ACT_XML.encode())
    index = {
        "acts": {
            "privacy-act-1988": {
                "name": "Privacy Act 1988",
                "xml_path": "xml/privacy-act-1988.xml",
            }
        }
    }
    (corpus_dir / "index.json").write_text(json.dumps(index))
    chunks = chunk_corpus(corpus_dir)
    assert len(chunks) == 2
    assert all(c.act_name == "Privacy Act 1988" for c in chunks)


def test_schedule_clauses_extracted(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V3, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V3.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    clause_chunks = [c for c in chunks if c.provision_type == "schedule_clause"]
    assert len(clause_chunks) == 2


def test_schedule_clause_fields(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V3, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V3.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    clause = next(c for c in chunks if c.eid == "schedule-1__clause-1")
    assert clause.provision_num == "1"
    assert clause.provision_type == "schedule_clause"
    assert clause.heading == "Open and transparent management"
    assert "openly" in clause.text


def test_section_provision_type(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V3, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V3.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    section_chunks = [c for c in chunks if c.provision_type == "section"]
    assert len(section_chunks) == 2


def test_authorial_note_in_section_text(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V3, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V3.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    sec6 = next(c for c in chunks if c.eid == "sec-6")
    # authorialNote text is included in section text (contextually relevant for embedding)
    assert "Note: See also section 3." in sec6.text


def test_table_cells_separated(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V3, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V3.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    sec7 = next(c for c in chunks if c.eid == "sec-7")
    # Cells must be separated by ' | ', not concatenated
    assert "Fee type | Amount" in sec7.text
    assert "Application fee | 100 penalty units" in sec7.text


def test_subsection_chunks_extracted(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V4, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V4.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    subsec_chunks = [c for c in chunks if c.provision_type == "subsection"]
    # sec-6__subsec-3 has text "Short." which is < 20 chars → skipped
    assert len(subsec_chunks) == 2


def test_subsection_eid_and_provision_num(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V4, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V4.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    ss1 = next(c for c in chunks if c.eid == "sec-6__subsec-1")
    assert ss1.provision_type == "subsection"
    assert ss1.provision_num == "6(1"


def test_subsection_text_non_empty(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V4, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V4.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    subsec_chunks = [c for c in chunks if c.provision_type == "subsection"]
    assert all(len(c.text) >= 20 for c in subsec_chunks)


def test_stub_subsections_skipped(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V4, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V4.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    # sec-6__subsec-3 text is "Short." — should not appear
    eids = [c.eid for c in chunks]
    assert "sec-6__subsec-3" not in eids


def test_section_chunks_alongside_subsections(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML_V4, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML_V4.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    section_chunks = [c for c in chunks if c.provision_type == "section"]
    subsec_chunks = [c for c in chunks if c.provision_type == "subsection"]
    assert len(section_chunks) == 1
    assert len(subsec_chunks) == 2


def test_chunk_terms_field_defaults_to_empty(tmp_path):
    from tests.conftest import PRIVACY_ACT_XML, CORPUS_INDEX
    xml_file = tmp_path / "privacy-act-1988.xml"
    xml_file.write_bytes(PRIVACY_ACT_XML.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    assert all(c.terms == [] for c in chunks)


def test_refs_populated_from_text(tmp_path):
    from tests.conftest import CORPUS_INDEX
    # Minimal XML with a cross-ref in a section
    AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork><FRBRthis value="/akn/au/act/1988/119/!main"/>
          <FRBRuri value="/akn/au/act/1988/119"/><FRBRdate date="1988" name="Generation"/>
          <FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression><FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/><FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation><FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/><FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-26WA">
        <num>26WA</num>
        <heading>Notification obligations</heading>
        <content><p>As required by section 6 of this Act.</p></content>
      </section>
    </body>
  </act>
</akomaNtoso>"""
    xml_file = tmp_path / "test.xml"
    xml_file.write_bytes(xml.encode())
    chunks = chunk_xml(xml_file, "Privacy Act 1988", CORPUS_INDEX)
    assert chunks[0].refs  # non-empty
    assert "#sec-6" in chunks[0].refs
