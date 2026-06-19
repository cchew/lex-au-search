import json
import pytest
import warnings
from pathlib import Path
from lxml import etree

from lexausearch.chunker import chunk_xml, chunk_corpus
from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _xml_bytes(xml_str: str) -> bytes:
    return xml_str.encode()


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
    assert s3.section_num == "3"
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


def test_chunk_xml_warns_on_untagged_body_content(tmp_path):
    xml = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork><FRBRthis value="/akn/au/act/1988/119/!main"/><FRBRuri value="/akn/au/act/1988/119"/>
          <FRBRdate date="1988" name="Generation"/><FRBRauthor href="#parliament"/>
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
      <section eId="sec-1"><num>1</num><content><p>Real section.</p></content></section>
      <p>Untagged schedule text not in a section element.</p>
    </body>
  </act>
</akomaNtoso>"""
    xml_file = tmp_path / "with-schedule.xml"
    xml_file.write_bytes(xml.encode())
    with pytest.warns(UserWarning, match="schedule"):
        chunk_xml(xml_file, "Privacy Act 1988")


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
