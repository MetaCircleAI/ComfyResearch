from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from comfy_research.engine.analysis.pca_run import iter_pca_events
from comfy_research.schemas.graph import Edge, Node

router = APIRouter(prefix="/api", tags=["pca"])


class PcaRequest(BaseModel):
    pca_node_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


def _ndjson_encode(event: dict) -> bytes:
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


@router.post("/pca")
def post_pca(body: PcaRequest) -> StreamingResponse:
    """Streams NDJSON: `progress` (step/total), then one `complete` event."""

    # Materialize events before StreamingResponse so HTTPException becomes a normal 4xx JSON body
    # (Starlette may commit 200 before the stream; errors mid-stream surface as a client "network error").
    events = list(iter_pca_events(body.nodes, body.edges, body.pca_node_id))

    def generate():
        for event in events:
            yield _ndjson_encode(event)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
