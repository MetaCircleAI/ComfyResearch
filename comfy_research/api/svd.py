from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from comfy_research.engine.analysis.svd_run import iter_svd_events
from comfy_research.schemas.graph import Edge, Node

router = APIRouter(prefix="/api", tags=["svd"])


class SvdRequest(BaseModel):
    svd_node_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


def _ndjson_encode(event: dict) -> bytes:
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


@router.post("/svd")
def post_svd(body: SvdRequest) -> StreamingResponse:
    """Streams NDJSON: `progress` (step/total), then one `complete` event."""
    events = list(iter_svd_events(body.nodes, body.edges, body.svd_node_id))

    def generate():
        for event in events:
            yield _ndjson_encode(event)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
