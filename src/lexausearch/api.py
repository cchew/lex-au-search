from __future__ import annotations

from fastapi import FastAPI

from lexausearch.searcher import Searcher


def create_app(searcher: Searcher) -> FastAPI:
    app = FastAPI(title="lex-au-search", version="0.1.0")

    @app.get("/search")
    def search(q: str, limit: int = 10, act: str | None = None) -> dict:
        results = searcher.search(q, limit=limit, act=act)
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
                    "refs": r.chunk.refs,
                    "score": r.score,
                }
                for r in results
            ]
        }

    return app
