from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from comfy_research.schemas.graph import GraphDocument

GraphFileExportTier = Literal["small", "medium", "large"]
LibraryOrigin = Literal["combined_model"]


class SavedGraphEntry(BaseModel):
    id: str
    name: str
    tier: GraphFileExportTier
    document: GraphDocument
    savedAt: float = Field(..., description="Unix time in milliseconds")
    libraryOrigin: Optional[LibraryOrigin] = None
