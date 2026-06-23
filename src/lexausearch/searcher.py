from __future__ import annotations

from qdrant_client import QdrantClient, models

from lexausearch.indexer import COLLECTION_ACTS, COLLECTION_SECTIONS, configure_client
from lexausearch.models import ActRecord, ActSearchResult, Chunk, SearchResult


class Searcher:
    def __init__(self, client: QdrantClient) -> None:
        self._client = configure_client(client)

    def search(
        self,
        query: str,
        limit: int = 5,
        act: str | None = None,
        provision_type: str | None = None,
    ) -> list[SearchResult]:
        must: list[models.FieldCondition] = []
        if act:
            must.append(models.FieldCondition(
                key="act_name", match=models.MatchValue(value=act)
            ))
        if provision_type:
            must.append(models.FieldCondition(
                key="provision_type", match=models.MatchValue(value=provision_type)
            ))
        query_filter = models.Filter(must=must) if must else None

        hits = self._client.query(
            collection_name=COLLECTION_SECTIONS,
            query_text="search_query: " + query,
            query_filter=query_filter,
            limit=limit,
        )
        return [
            SearchResult(
                chunk=Chunk(
                    act_name=hit.metadata["act_name"],
                    frbr_uri=hit.metadata["frbr_uri"],
                    eid=hit.metadata["eid"],
                    provision_num=hit.metadata["provision_num"],
                    provision_type=hit.metadata["provision_type"],
                    heading=hit.metadata.get("heading"),
                    text=hit.metadata["text"],
                    refs=hit.metadata.get("refs", []),
                ),
                score=hit.score,
            )
            for hit in hits
        ]

    def search_acts(
        self,
        query: str,
        limit: int = 5,
    ) -> list[ActSearchResult]:
        hits = self._client.query(
            collection_name=COLLECTION_ACTS,
            query_text="search_query: " + query,
            limit=limit,
        )
        return [
            ActSearchResult(
                record=ActRecord(
                    act_name=hit.metadata["act_name"],
                    frbr_uri=hit.metadata["frbr_uri"],
                    year=hit.metadata["year"],
                    as_at_date=hit.metadata.get("as_at_date", ""),
                    section_count=hit.metadata.get("section_count", 0),
                    schedule_clause_count=hit.metadata.get("schedule_clause_count", 0),
                ),
                score=hit.score,
            )
            for hit in hits
        ]
