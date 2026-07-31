"""Label schema for Learning to Parse Training Dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from comfy_research.lmech.lmn.primitives import NAME_TO_ID, PRIMITIVE_NAMES

# Mechanism types aligned with LMN primitives (+ no-object for DETR padding).
SEGMENT_TYPES: List[str] = list(PRIMITIVE_NAMES)
NUM_SEGMENT_TYPES = len(SEGMENT_TYPES)
# DETR-style: last class index is "no object" for unmatched queries.
NO_OBJECT_CLASS = NUM_SEGMENT_TYPES
NUM_CLASSES = NUM_SEGMENT_TYPES + 1

TYPE_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(SEGMENT_TYPES)}
ID_TO_TYPE: Dict[int, str] = {i: name for name, i in TYPE_TO_ID.items()}


@dataclass
class SegmentAnnotation:
    type: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentAnnotation":
        return cls(type=str(d["type"]), start=float(d["start"]), end=float(d["end"]))


@dataclass
class CurveSample:
    id: str
    t: List[float]
    y: List[float]
    segments: List[SegmentAnnotation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "t": self.t,
            "y": self.y,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CurveSample":
        return cls(
            id=str(d["id"]),
            t=[float(x) for x in d["t"]],
            y=[float(x) for x in d["y"]],
            segments=[SegmentAnnotation.from_dict(s) for s in d["segments"]],
        )


def segment_type_id(type_name: str) -> int:
    if type_name in TYPE_TO_ID:
        return TYPE_TO_ID[type_name]
    return NAME_TO_ID.get(type_name, 0)
