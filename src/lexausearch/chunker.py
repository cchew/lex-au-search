from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

from lxml import etree

from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
AKN = f"{{{AKN_NS}}}"

logger = logging.getLogger(__name__)


def chunk_xml(xml_path: Path, act_name: str) -> list[Chunk]:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    frbr_uri_el = root.find(f".//{AKN}FRBRExpression/{AKN}FRBRuri")
    frbr_uri = frbr_uri_el.get("value") if frbr_uri_el is not None else ""

    chunks: list[Chunk] = []
    for section in root.iter(f"{AKN}section"):
        eid = section.get("eId", "")
        num_el = section.find(f"{AKN}num")
        heading_el = section.find(f"{AKN}heading")
        provision_num = num_el.text.strip() if num_el is not None and num_el.text else ""
        heading = heading_el.text.strip() if heading_el is not None and heading_el.text else None
        text = " ".join("".join(section.itertext()).split())
        chunks.append(Chunk(
            act_name=act_name,
            frbr_uri=frbr_uri,
            eid=eid,
            provision_num=provision_num,
            provision_type="section",
            heading=heading,
            text=text,
            refs=[],
        ))

    # Warn if body-level non-section children exist (untagged schedule/preface content)
    body = root.find(f".//{AKN}body")
    if body is not None:
        for child in body:
            local = child.tag.replace(f"{{{AKN_NS}}}", "")
            if local != "section":
                warnings.warn(
                    f"{act_name}: body contains <{local}> element outside <section> — "
                    "schedule or preface content may be excluded from the index. "
                    "This is resolved in lex-au v0.1.1.",
                    UserWarning,
                    stacklevel=2,
                )
                break

    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    chunks: list[Chunk] = []
    for entry in index["acts"].values():
        xml_path = corpus_dir / entry["xml_path"]
        act_chunks = chunk_xml(xml_path, entry["name"])
        chunks.extend(act_chunks)
        logger.info(f"{entry['name']}: {len(act_chunks)} sections")
    return chunks
