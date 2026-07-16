from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException, Response
from qdrant_client import QdrantClient, models

from lexausearch.indexer import COLLECTION_ACTS, COLLECTION_SECTIONS
from lexausearch.searcher import Searcher


def _eid_sort_key(eid: str) -> tuple[int, ...]:
    """Natural sort key for eIds: extract all digit sequences as ints."""
    return tuple(int(n) for n in re.findall(r'\d+', eid))


def create_app(searcher: Searcher, client: QdrantClient) -> FastAPI:
    app = FastAPI(title="lex-au-search", version="0.4.2")

    @app.get(
        "/search",
        operation_id="search_legislation",
        description=(
            "Search Australian Commonwealth legislation by topic or question. "
            "Returns relevant sections and schedule clauses with FRBR citations. "
            "For 'what does term X mean' or cross-Act definition questions, call "
            "lex-au-graph's resolve_definition or find_all_definitions tool first (or "
            "alongside this one) — hybrid search here can return the wrong Act's use of "
            "a homonymous term, since it matches on semantic similarity, not authoritative "
            "definition scope."
        ),
    )
    def search(
        q: str,
        limit: int = 10,
        act: str | None = None,
        provision_type: str | None = None,
    ) -> dict:
        results = searcher.search(q, limit=limit, act=act, provision_type=provision_type)
        return {
            "results": [
                {
                    "act_name": r.chunk.act_name,
                    "frbr_uri": r.chunk.frbr_uri,
                    "eid": r.chunk.eid,
                    "provision_num": r.chunk.provision_num,
                    "provision_type": r.chunk.provision_type,
                    "heading": r.chunk.heading,
                    "text": r.chunk.text,
                    "refs": [ref for ref in r.chunk.refs if not ref.startswith("unresolved:")],
                    "score": r.score,
                }
                for r in results
            ]
        }

    @app.get(
        "/legislation/{act_name}",
        operation_id="get_act_metadata",
        description="Retrieve Act-level metadata by name (e.g. 'Privacy Act 1988').",
    )
    def get_act(act_name: str) -> dict:
        results = client.scroll(
            collection_name=COLLECTION_ACTS,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="act_name", match=models.MatchValue(value=act_name))]
            ),
            limit=1,
            with_payload=True,
        )
        points = results[0]
        if not points:
            raise HTTPException(status_code=404, detail=f"Act not found: {act_name}")
        payload = points[0].payload
        return {
            "act_name": payload["act_name"],
            "frbr_uri": payload["frbr_uri"],
            "year": payload["year"],
            "as_at_date": payload.get("as_at_date", ""),
            "section_count": payload.get("section_count", 0),
            "schedule_clause_count": payload.get("schedule_clause_count", 0),
        }

    @app.get(
        "/legislation/{act_name}/sections",
        operation_id="get_act_sections",
        description=(
            "Retrieve all indexed provisions for an Act, sorted by eId. Can be large for "
            "big Acts (Corporations Act, Migration Act, Income Tax Assessment Act each have "
            "1000+ provisions) — prefer search_legislation with a specific query where possible."
        ),
    )
    def get_sections(act_name: str) -> dict:
        results = client.scroll(
            collection_name=COLLECTION_SECTIONS,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="act_name", match=models.MatchValue(value=act_name))]
            ),
            limit=1000,
            with_payload=True,
        )
        points = sorted(
            results[0],
            key=lambda p: (
                0 if p.payload.get("provision_type") == "section" else 1,
                *_eid_sort_key(p.payload.get("eid", "")),
            ),
        )
        return {
            "chunks": [
                {
                    "eid": p.payload["eid"],
                    "provision_num": p.payload["provision_num"],
                    "provision_type": p.payload["provision_type"],
                    "heading": p.payload.get("heading"),
                    "text": p.payload["text"],
                }
                for p in points
            ]
        }

    @app.get(
        "/legislation/{act_name}/text",
        operation_id="get_act_text",
        description=(
            "Full Act text concatenated in provision order (plain text). WARNING: this can "
            "be tens of thousands of tokens for a large Act (e.g. Corporations Act, Migration "
            "Act, Income Tax Assessment Act) and will exceed most agent context windows. Use "
            "search_legislation for a targeted query or get_act_sections with filtering "
            "instead — only call this for genuinely small Acts or when the full text is "
            "explicitly required."
        ),
    )
    def get_text(act_name: str) -> Response:
        results = client.scroll(
            collection_name=COLLECTION_SECTIONS,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="act_name", match=models.MatchValue(value=act_name))]
            ),
            limit=1000,
            with_payload=True,
        )
        points = sorted(
            results[0],
            key=lambda p: (
                0 if p.payload.get("provision_type") == "section" else 1,
                *_eid_sort_key(p.payload.get("eid", "")),
            ),
        )
        text = "\n\n".join(p.payload["text"] for p in points)
        return Response(content=text, media_type="text/plain")

    return app
