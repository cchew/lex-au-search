from __future__ import annotations

from qdrant_client import QdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

from lexausearch.indexer import COLLECTION_ACTS, COLLECTION_SECTIONS, configure_client
from lexausearch.models import ActRecord, ActSearchResult, Chunk, SearchResult


class Searcher:
    def __init__(self, client: QdrantClient) -> None:
        self._client = configure_client(client)

    def _hybrid_query(
        self,
        collection_name: str,
        query_text: str,
        query_filter: models.Filter | None,
        limit: int,
    ) -> list:
        """Dense + BM25 sparse query merged with RRF via query_batch_points.

        Replicates the deprecated client.query() behaviour without invoking the
        deprecated method, so no deprecation warnings are emitted.
        """
        _, dense_vector = next(
            self._client._embed_documents(
                [query_text],
                embedding_model_name=self._client.embedding_model_name,
                embed_type="query",
            )
        )
        sparse_sv = next(
            self._client._sparse_embed_documents(
                [query_text],
                embedding_model_name=self._client.sparse_embedding_model_name,
            )
        )
        sparse_vector = models.SparseVector(
            indices=sparse_sv.indices,
            values=sparse_sv.values,
        )

        dense_req = models.QueryRequest(
            query=dense_vector,
            using=self._client.get_vector_field_name(),
            filter=query_filter,
            limit=limit * 2,
            with_payload=True,
        )
        sparse_req = models.QueryRequest(
            query=sparse_vector,
            using=self._client.get_sparse_vector_field_name(),
            filter=query_filter,
            limit=limit * 2,
            with_payload=True,
        )

        dense_resp, sparse_resp = self._client.query_batch_points(
            collection_name=collection_name,
            requests=[dense_req, sparse_req],
        )
        return reciprocal_rank_fusion(
            [dense_resp.points, sparse_resp.points], limit=limit
        )

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

        hits = self._hybrid_query(
            COLLECTION_SECTIONS, query, query_filter, limit
        )
        return [
            SearchResult(
                chunk=Chunk(
                    act_name=hit.payload["act_name"],
                    frbr_uri=hit.payload["frbr_uri"],
                    eid=hit.payload["eid"],
                    provision_num=hit.payload["provision_num"],
                    provision_type=hit.payload["provision_type"],
                    heading=hit.payload.get("heading"),
                    text=hit.payload["text"],
                    refs=hit.payload.get("refs", []),
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
        hits = self._hybrid_query(
            COLLECTION_ACTS, query, None, limit
        )
        return [
            ActSearchResult(
                record=ActRecord(
                    act_name=hit.payload["act_name"],
                    frbr_uri=hit.payload["frbr_uri"],
                    year=hit.payload["year"],
                    as_at_date=hit.payload.get("as_at_date", ""),
                    section_count=hit.payload.get("section_count", 0),
                    schedule_clause_count=hit.payload.get("schedule_clause_count", 0),
                ),
                score=hit.score,
            )
            for hit in hits
        ]
