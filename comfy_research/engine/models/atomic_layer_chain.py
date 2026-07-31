"""Atomic layer nodes (linear / activation / LayerNorm) composed as nn.Sequential for the trainer."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from fastapi import HTTPException

from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.engine.models.afno_lite_spatiotemporal_model import (
    afno_encoder_block_layer_from_canvas_md,
    afno_patch_decode_layer_from_canvas_md,
    afno_patch_embed_layer_from_canvas_md,
    afno_spectral_mixer_layer_from_canvas_md,
)
from comfy_research.engine.models.local_mixing import CausalLocalMixingResidual
from comfy_research.engine.models.positional_embedding_layers import AbsolutePositionalEmbedding, RotaryEmbedding
from comfy_research.engine.models.rms_norm_module import RMSNorm
from comfy_research.schemas.graph import Edge, Node, NodeKind

SEQUENTIAL_MODEL_TYPES = frozenset(NodeKind(node_type) for node_type in node_types_with_capability("atomic_layer_model"))

# Direct children of ``combined_model`` that participate in the inner tensor chain (atomic layers and nested wrappers).
COMBINED_CHAIN_MEMBER_TYPES = frozenset(SEQUENTIAL_MODEL_TYPES | {NodeKind.combined_model})


def _scalar_int(x: Any, default: int = 0) -> int:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return int(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _scalar_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return float(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _scalar_str(x: Any, default: str = "") -> str:
    if isinstance(x, list):
        if not x:
            return default
        return str(x[0])
    return str(x) if x is not None else default


def _scalar_bool(x: Any, default: bool = True) -> bool:
    if isinstance(x, list):
        x = x[0] if x else None
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def _clamp_leaky_p(x: float) -> float:
    """Match frontend Activation layer: LeakyReLU ``negative_slope`` in [-1, 1]."""
    return max(-1.0, min(1.0, float(x)))


def _activation_module(name: str, leaky_negative_slope: float | None = None) -> nn.Module:
    m: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "identity": nn.Identity,
    }
    cls = m.get(name)
    if cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown activation: {name}")
    if name == "leaky_relu":
        p = _clamp_leaky_p(leaky_negative_slope if leaky_negative_slope is not None else 0.0)
        return nn.LeakyReLU(p)
    return cls()


def collect_atomic_layer_chain_front_to_back(
    tip: Node,
    edges: list[Edge],
    nmap: dict[str, Node],
) -> list[Node]:
    """Walk predecessors via ``tensor`` ← ``tensor`` until the first layer (legacy ``in``/``model`` still accepted)."""
    rev: list[Node] = []
    cur: Node | None = tip
    seen: set[str] = set()
    while cur is not None:
        if cur.id in seen:
            raise HTTPException(status_code=400, detail="Cycle detected in atomic model layer chain.")
        seen.add(cur.id)
        if cur.type not in SEQUENTIAL_MODEL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Trainer model chain tip must be an atomic layer node, got {cur.type}.",
            )
        rev.append(cur)
        preds: list[Node] = []
        for e in edges:
            if e.target != cur.id:
                continue
            th = (e.targetHandle or "").strip()
            if th not in ("tensor", "tensor_in", "in", ""):
                continue
            sh = (e.sourceHandle or "").strip()
            if sh not in ("tensor", "tensor_out", "model", ""):
                continue
            src = nmap.get(e.source)
            if src is not None:
                preds.append(src)
        if len(preds) > 1:
            raise HTTPException(
                status_code=400,
                detail="Each atomic layer accepts at most one incoming chain link (tensor → tensor).",
            )
        nxt = preds[0] if preds else None
        if nxt is not None and nxt.type not in SEQUENTIAL_MODEL_TYPES:
            if nxt.type == NodeKind.combined_model:
                nxt = None
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Atomic model chain: upstream of the left tensor socket must be "
                        f"linear_layer, activation_layer, layer_norm_layer, rms_norm_layer, embedding_layer, "
                        f"unembedding_layer, absolute_pos_embed_layer, rotary_embed_layer, local_mixing_layer, "
                        f"or AFNO atomic layers "
                        f"(got {nxt.type})."
                    ),
                )
        cur = nxt
    rev.reverse()
    return rev


def _is_tensor_chain_edge_for_members(e: Edge, member_ids: set[str]) -> bool:
    """Tensor→tensor link between sibling chain members (matches frontend ``isTensorChainEdge``)."""
    if e.source not in member_ids or e.target not in member_ids:
        return False
    sh = (e.sourceHandle or "").strip()
    th = (e.targetHandle or "").strip()
    sh_ok = sh in ("tensor", "tensor_out", "model", "")
    th_ok = th in ("tensor", "tensor_in", "in", "")
    return sh_ok and th_ok


def order_sequential_chain_member_ids(member_ids: set[str], edges: list[Edge]) -> list[str] | None:
    """Topological order of one tensor chain through ``member_ids`` (unique path covering all members)."""
    indeg: dict[str, int] = {mid: 0 for mid in member_ids}
    for e in edges:
        if not _is_tensor_chain_edge_for_members(e, member_ids):
            continue
        indeg[e.target] = indeg.get(e.target, 0) + 1
    heads = sorted([mid for mid in member_ids if indeg.get(mid, 0) == 0])
    if len(heads) != 1:
        return None
    out: list[str] = []
    seen: set[str] = set()
    cur: str | None = heads[0]
    while cur is not None:
        if cur in seen:
            return None
        seen.add(cur)
        out.append(cur)
        outs: list[str] = []
        for e in edges:
            if e.source != cur:
                continue
            if not _is_tensor_chain_edge_for_members(e, member_ids):
                continue
            outs.append(e.target)
        if len(outs) == 0:
            cur = None
        elif len(outs) > 1:
            return None
        else:
            cur = outs[0]
    if len(out) != len(member_ids):
        return None
    return out


def collect_flat_atomic_chain_under_combined(
    combined: Node,
    nodes: list[Node],
    edges: list[Edge],
) -> list[Node]:
    """Ordered atomic layers under a combined-model shell, recursively expanding nested ``combined_model`` children."""
    if combined.type != NodeKind.combined_model:
        return []
    nmap = {n.id: n for n in nodes}
    cid = combined.id
    members = [n for n in nodes if n.parentId == cid and n.type in COMBINED_CHAIN_MEMBER_TYPES]
    if not members:
        return []
    member_ids = {n.id for n in members}
    ordered_ids = order_sequential_chain_member_ids(member_ids, edges)
    if not ordered_ids:
        return []
    flat: list[Node] = []
    for oid in ordered_ids:
        n = nmap.get(oid)
        if n is None:
            return []
        if n.type in SEQUENTIAL_MODEL_TYPES:
            flat.append(n)
        elif n.type == NodeKind.combined_model:
            sub = collect_flat_atomic_chain_under_combined(n, nodes, edges)
            if not sub:
                return []
            flat.extend(sub)
        else:
            return []
    return flat


def resolve_combined_model_sequential_tip(
    combined: Node,
    nodes: list[Node],
    edges: list[Edge],
) -> Node | None:
    """Rightmost atomic layer in the flattened tensor chain under ``combined_model`` (trainer / activation wiring)."""
    if combined.type != NodeKind.combined_model:
        return None
    flat = collect_flat_atomic_chain_under_combined(combined, nodes, edges)
    if not flat:
        return None
    return flat[-1]


def _is_frozen(d: dict[str, Any]) -> bool:
    """Return True if any freeze marker is set in the node data dict.

    Supported freeze markers (any one triggers freeze):
      - freeze: true
      - trainable: false
      - requiresGrad: false
      - requires_grad: false
    """
    if _scalar_bool(d.get("freeze"), False):
        return True
    if not _scalar_bool(d.get("trainable"), True):
        return True
    if not _scalar_bool(d.get("requiresGrad"), True):
        return True
    if not _scalar_bool(d.get("requires_grad"), True):
        return True
    return False


def build_atomic_layer_module(node: Node) -> nn.Module:
    d: dict[str, Any] = node.data or {}
    module: nn.Module
    if node.type == NodeKind.linear_layer:
        inf = _scalar_int(d.get("inFeatures"), 1)
        outf = _scalar_int(d.get("outFeatures"), 1)
        bias = _scalar_bool(d.get("bias"), True)
        if inf < 1 or outf < 1:
            raise HTTPException(status_code=400, detail="linear_layer inFeatures and outFeatures must be >= 1.")
        module = nn.Linear(inf, outf, bias=bias)
    elif node.type == NodeKind.activation_layer:
        act = _scalar_str(d.get("activation"), "relu")
        lp_raw = d.get("leakyP")
        if lp_raw is None:
            lp_raw = d.get("leaky_p")
        leaky_slope = _clamp_leaky_p(_scalar_float(lp_raw, 0.0))
        module = _activation_module(act, leaky_slope if act == "leaky_relu" else None)
    elif node.type == NodeKind.layer_norm_layer:
        shape = _scalar_int(d.get("normalizedShape"), 1)
        eps = _scalar_float(d.get("eps"), 1e-5)
        aff = _scalar_bool(d.get("elementwiseAffine"), True)
        if shape < 1:
            raise HTTPException(status_code=400, detail="layer_norm_layer normalizedShape must be >= 1.")
        module = nn.LayerNorm(shape, eps=eps, elementwise_affine=aff)
    elif node.type == NodeKind.rms_norm_layer:
        shape = _scalar_int(d.get("normalizedShape"), 1)
        eps = _scalar_float(d.get("eps"), 1e-6)
        aff = _scalar_bool(d.get("elementwiseAffine"), True)
        if shape < 1:
            raise HTTPException(status_code=400, detail="rms_norm_layer normalizedShape must be >= 1.")
        module = RMSNorm(shape, eps=eps, elementwise_affine=aff)
    elif node.type == NodeKind.embedding_layer:
        num = _scalar_int(d.get("numEmbeddings"), 1)
        dim = _scalar_int(d.get("embeddingDim"), 1)
        pad = _scalar_int(d.get("paddingIdx"), -1)
        sgf = _scalar_bool(d.get("scaleGradByFreq"), False)
        if num < 1 or dim < 1:
            raise HTTPException(status_code=400, detail="embedding_layer numEmbeddings and embeddingDim must be >= 1.")
        pd = pad if pad >= 0 else None
        module = nn.Embedding(num, dim, padding_idx=pd, scale_grad_by_freq=sgf)
    elif node.type == NodeKind.unembedding_layer:
        inf = _scalar_int(d.get("inFeatures"), 1)
        outf = _scalar_int(d.get("outFeatures"), 1)
        bias = _scalar_bool(d.get("bias"), True)
        if inf < 1 or outf < 1:
            raise HTTPException(status_code=400, detail="unembedding_layer inFeatures and outFeatures must be >= 1.")
        module = nn.Linear(inf, outf, bias=bias)
    elif node.type == NodeKind.absolute_pos_embed_layer:
        msl = _scalar_int(d.get("maxSeqLen"), 512)
        ed = _scalar_int(d.get("embeddingDim"), 64)
        if msl < 1 or ed < 1:
            raise HTTPException(
                status_code=400,
                detail="absolute_pos_embed_layer maxSeqLen and embeddingDim must be >= 1.",
            )
        module = AbsolutePositionalEmbedding(msl, ed)
    elif node.type == NodeKind.rotary_embed_layer:
        rd = _scalar_int(d.get("rotaryDim"), 64)
        base = _scalar_float(d.get("thetaBase"), 10000.0)
        if rd < 2 or rd % 2 != 0:
            raise HTTPException(
                status_code=400,
                detail="rotary_embed_layer rotaryDim must be an even integer >= 2.",
            )
        if base <= 0.0:
            raise HTTPException(status_code=400, detail="rotary_embed_layer thetaBase must be > 0.")
        module = RotaryEmbedding(rd, base=base)
    elif node.type == NodeKind.local_mixing_layer:
        mdim = _scalar_int(d.get("modelDim"), 64)
        ks = _scalar_int(d.get("kernelSize"), 5)
        if mdim < 1:
            raise HTTPException(status_code=400, detail="local_mixing_layer modelDim must be >= 1.")
        if ks < 3:
            ks = 3
        if ks % 2 == 0:
            ks += 1
        module = CausalLocalMixingResidual(mdim, ks)
    elif node.type == NodeKind.afno_patch_embed_layer:
        module = afno_patch_embed_layer_from_canvas_md(d)
    elif node.type == NodeKind.afno_spectral_mixer_layer:
        module = afno_spectral_mixer_layer_from_canvas_md(d)
    elif node.type == NodeKind.afno_encoder_block_layer:
        module = afno_encoder_block_layer_from_canvas_md(d)
    elif node.type == NodeKind.afno_patch_decode_layer:
        module = afno_patch_decode_layer_from_canvas_md(d)
    else:
        raise HTTPException(status_code=400, detail=f"Not an atomic layer node: {node.type}")
    if _is_frozen(d):
        module.requires_grad_(False)
    return module


def build_sequential_from_atomic_tip(tip: Node, edges: list[Edge], nmap: dict[str, Node]) -> nn.Sequential:
    chain = collect_atomic_layer_chain_front_to_back(tip, edges, nmap)
    return nn.Sequential(*[build_atomic_layer_module(n) for n in chain])


def build_sequential_from_flat_atomic_chain(chain: list[Node]) -> nn.Sequential:
    return nn.Sequential(*[build_atomic_layer_module(n) for n in chain])
