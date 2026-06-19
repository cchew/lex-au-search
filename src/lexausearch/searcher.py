from __future__ import annotations

from qdrant_client import QdrantClient, models

from lexausearch.indexer import COLLECTION, configure_client
from lexausearch.models import Chunk, SearchResult


class Searcher:
    def __init__(self, client: QdrantClient) -> None:
        self._client = configure_client(client)

    def search(
        self,
        query: str,
        limit: int = 5,
        act: str | None = None,
    ) -> list[SearchResult]:
        query_filter = None
        if act:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="act_name",
                        match=models.MatchValue(value=act),
                    )
                ]
            )

        hits = self._client.query(
            collection_name=COLLECTION,
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
                    section_num=hit.metadata["section_num"],
                    heading=hit.metadata["heading"],
                    text=hit.metadata["text"],
                ),
                score=hit.score,
            )
            for hit in hits
        ]
