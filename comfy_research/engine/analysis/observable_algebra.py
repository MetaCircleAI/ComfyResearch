"""Observable algebra: per-axis, local-flatten, or global-flatten reductions to a scalar."""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np

from comfy_research.engine.analysis.observable_user_eval import _stat_reduce_1d

TensorScope = Literal["single", "all_matching"]
FlattenMode = Literal["none", "local", "global", "sv_entropy"]
ObservableSource = Literal["weight", "representation"]

FLATTEN_MODES: tuple[str, ...] = ("none", "local", "global", "sv_entropy")

_MEMBER_KEY_RE = re.compile(r"[^a-zA-Z0-9_]+")


def normalize_flatten_mode(raw: str | None) -> str:
    mode = (raw or "none").strip().lower()
    if mode not in FLATTEN_MODES:
        raise ValueError(f"flatten_mode must be one of {FLATTEN_MODES!r}; got {raw!r}.")
    return mode


def _family_suffix(tensor_name: str) -> str | None:
    """Suffix shared by layer-indexed params (``0.weight``, ``body.2.bias``, …)."""
    trimmed = (tensor_name or "").strip()
    m = re.match(r"^(?:body\.)?\d+\.(.+)$", trimmed)
    if m:
        return m.group(1)
    m = re.match(r"^\d+\.(.+)$", trimmed)
    if m:
        return m.group(1)
    return None


def family_pattern_from_tensor_name(tensor_name: str) -> str:
    """From ``0.weight`` / ``body.0.bias`` → ``*.weight`` / ``*.bias``."""
    suf = _family_suffix(tensor_name)
    if suf:
        return f"*.{suf}"
    return (tensor_name or "").strip() or "tensor"


def _representation_io_from_id(representation_id: str) -> str | None:
    trimmed = (representation_id or "").strip()
    if "::" not in trimmed:
        return None
    _, io = trimmed.rsplit("::", 1)
    return io.strip() or None


def representation_all_matching_name_base(representation_id: str) -> str:
    """IO family without wildcard (global flatten names)."""
    io = _representation_io_from_id(representation_id)
    if io:
        return io.lower()
    trimmed = (representation_id or "").strip()
    m = re.match(r"^h\d+_(.+)$", trimmed)
    if m:
        return f"h*_{m.group(1)}"
    return trimmed or "rep"


def representation_all_matching_label_base(representation_id: str) -> str:
    """All-matching label base — ``*.input`` / ``*.output`` (multiple tensors)."""
    kind = representation_all_matching_name_base(representation_id)
    if kind.startswith("h*_"):
        return kind
    return f"*.{kind}"


def family_pattern_from_representation_id(representation_id: str) -> str:
    """From ``body.4::output`` → ``*.output``; legacy ``h2_preact`` → ``h*_preact``."""
    trimmed = (representation_id or "").strip()
    if not trimmed:
        return "representation"
    io = _representation_io_from_id(trimmed)
    if io:
        return f"*.{io.lower()}"
    m = re.match(r"^h\d+_(.+)$", trimmed)
    if m:
        return f"h*_{m.group(1)}"
    return trimmed


def matching_representation_ids(
    all_ids: list[str],
    selected: str,
    scope: str,
) -> list[str]:
    """Return representation ids included by *scope* (sorted)."""
    sel = (selected or "").strip()
    if not sel:
        return []
    if scope != "all_matching":
        return [sel] if sel in all_ids else []
    sel_io = _representation_io_from_id(sel)
    if sel_io:
        matched = sorted(n for n in all_ids if _representation_io_from_id(n) == sel_io)
        return matched if matched else ([sel] if sel in all_ids else [])
    m = re.match(r"^h\d+_(.+)$", sel)
    if m:
        suf = m.group(1)
        pat = re.compile(r"^h\d+" + re.escape("_" + suf) + r"$")
        matched = sorted(n for n in all_ids if pat.match(n))
        return matched if matched else ([sel] if sel in all_ids else [])
    return [sel] if sel in all_ids else []


def matching_parameter_names(
    all_names: list[str],
    selected: str,
    scope: str,
) -> list[str]:
    """Return parameter names included by *scope* (sorted)."""
    sel = (selected or "").strip()
    if not sel:
        return []
    if scope != "all_matching":
        return [sel] if sel in all_names else []
    suffix = _family_suffix(sel)
    if suffix:
        pats = (
            re.compile(r"^\d+\." + re.escape(suffix) + r"$"),
            re.compile(r"^body\.\d+\." + re.escape(suffix) + r"$"),
        )
        matched = sorted(
            n for n in all_names if any(p.match(n) for p in pats) or n.endswith("." + suffix)
        )
        return matched if matched else ([sel] if sel in all_names else [])
    return [sel] if sel in all_names else []


def member_storage_key(tensor_name: str) -> str:
    """Safe suffix for ``paired::member::{key}`` history keys."""
    s = _MEMBER_KEY_RE.sub("_", (tensor_name or "").strip()).strip("_")
    return s or "tensor"


def member_display_label(storage_key: str) -> str:
    """Undo ``member_storage_key`` for viz legends (best-effort)."""
    return storage_key.replace("_", ".")


class AxisReductionSpec:
    __slots__ = ("axis_index", "axis_label", "op")

    def __init__(self, axis_index: int, axis_label: str, op: str) -> None:
        self.axis_index = int(axis_index)
        self.axis_label = str(axis_label)
        self.op = str(op)


def parse_axis_reductions(
    raw: list[dict[str, Any]],
    *,
    flatten_mode: str = "none",
    tensor_shape: list[int] | None = None,
) -> list[AxisReductionSpec]:
    mode = normalize_flatten_mode(flatten_mode)
    out: list[AxisReductionSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "").strip()
        if not op:
            continue
        try:
            axis_index = int(item.get("axis_index", item.get("axisIndex", -1)))
        except (TypeError, ValueError):
            continue
        axis_label = str(item.get("axis_label") or item.get("axisLabel") or f"dim{axis_index}").strip()
        out.append(AxisReductionSpec(axis_index, axis_label, op))
    if not out:
        if mode == "sv_entropy":
            return [AxisReductionSpec(0, "sv", "entropy")]
        raise ValueError("At least one reduction is required.")
    if mode in ("local", "global", "sv_entropy"):
        if mode == "sv_entropy":
            return out if out else [AxisReductionSpec(0, "sv", "entropy")]
        if len(out) != 1:
            raise ValueError(
                f"flatten_mode={mode!r} requires exactly one 1D reduction op; got {len(out)}."
            )
        return out
    rank = len(tensor_shape or [])
    if rank > 0:
        if len(out) != rank:
            raise ValueError(
                f"Rank-{rank} tensor requires exactly {rank} per-axis reduction(s); got {len(out)}."
            )
        indices = sorted(s.axis_index for s in out)
        if indices != list(range(rank)):
            raise ValueError(f"Reductions must cover axes 0..{rank - 1} exactly once (got {indices}).")
        for spec in out:
            if spec.axis_index < 0 or spec.axis_index >= rank:
                raise ValueError(f"Axis index {spec.axis_index} out of range for rank {rank}.")
    return out


def flat_op_from_reductions(reductions: list[AxisReductionSpec]) -> str:
    if not reductions:
        raise ValueError("At least one reduction is required.")
    return str(reductions[0].op)


def _flatten_chain_label(flatten_mode: str, op: str) -> str:
    mode = normalize_flatten_mode(flatten_mode)
    if mode == "local":
        return f"flatten → {op}(flat)"
    if mode == "global":
        return f"concat → flatten → {op}(global)"
    if mode == "sv_entropy":
        return "SVD → singular value entropy"
    return ""


def _human_representation_label(representation_id: str, layer_index: int = 0, layer_io: str = "") -> str:
    rid = (representation_id or "").strip()
    if layer_io == "input" and layer_index > 0:
        return f"layer {layer_index} input"
    if layer_io == "output" and layer_index > 0:
        return f"layer {layer_index} output"
    if rid == "input":
        return "input"
    if rid == "output":
        return "output"
    m = re.match(r"^h(\d+)_preact$", rid)
    if m:
        return f"layer {m.group(1)} preact"
    m = re.match(r"^h(\d+)_postact$", rid)
    if m:
        return f"layer {m.group(1)} postact"
    return rid or "representation"


def _array_loader_expr(*, observable_source: str, subject_id: str) -> str:
    if (observable_source or "weight").strip().lower() == "representation":
        return f"activation_representation_as_numpy({subject_id!r})"
    return f"named_parameter_as_numpy({subject_id!r})"


def _format_spec_shape(shape: list[int]) -> str:
    parts: list[str] = []
    for i, x in enumerate(shape):
        xi = int(x)
        if i == 0 and xi < 0:
            parts.append("batch")
        else:
            parts.append(str(xi))
    return "×".join(parts)


def format_algebra_human_chain(
    *,
    tensor_name: str,
    tensor_shape: list[int],
    reductions: list[AxisReductionSpec],
    source_model_node_id: str = "",
    tensor_scope: str = "single",
    flatten_mode: str = "none",
    observable_source: str = "weight",
    representation_id: str = "",
    layer_index: int = 0,
    layer_io: str = "",
    matched_tensor_names: list[str] | None = None,
) -> str:
    mode = normalize_flatten_mode(flatten_mode)
    shape_s = _format_spec_shape(tensor_shape) if tensor_shape else "?"
    if mode in ("local", "global", "sv_entropy"):
        chain = _flatten_chain_label(mode, flat_op_from_reductions(reductions) if mode != "sv_entropy" else "")
    else:
        ordered = sorted(reductions, key=lambda r: r.axis_index)
        chain = " → ".join(f"{r.op}({r.axis_label})" for r in ordered)
    src = f"model {source_model_node_id[:8]}… · " if source_model_node_id else ""
    if (observable_source or "weight").strip().lower() == "representation":
        if tensor_scope == "all_matching":
            if mode == "global":
                kind = global_flatten_representation_kind(representation_id or tensor_name)
                members = matched_tensor_names or []
                if members:
                    member_s = ", ".join(members)
                    return f"{src}rep {kind} → {chain} · [{member_s}]"
                return f"{src}rep {kind} → {chain}"
            pat = family_pattern_from_representation_id(representation_id or tensor_name)
            members = matched_tensor_names or []
            if members:
                member_s = ", ".join(members)
                return f"{src}rep {pat} [{shape_s}] × [{member_s}] → {chain}"
            return f"{src}rep {pat} [{shape_s}] → {chain}"
        subj = _human_representation_label(representation_id or tensor_name, layer_index, layer_io)
        return f"{src}rep {subj} [{shape_s}] → {chain}"
    if tensor_scope == "all_matching":
        if mode == "global":
            kind = global_flatten_label_base(tensor_name)
            members = matched_tensor_names or []
            if members:
                member_s = ", ".join(members)
                return f"{src}{kind} → {chain} · [{member_s}]"
            return f"{src}{kind} → {chain}"
        pat = family_pattern_from_tensor_name(tensor_name)
        members = matched_tensor_names or []
        if members:
            member_s = ", ".join(members)
            return f"{src}{pat} [{shape_s}] × [{member_s}] → {chain}"
        return f"{src}{pat} [{shape_s}] → {chain}"
    return f"{src}{tensor_name} [{shape_s}] → {chain}"


def format_algebra_definition_code(
    *,
    tensor_name: str,
    reductions: list[AxisReductionSpec],
    flatten_mode: str = "none",
    member_tensor_names: list[str] | None = None,
    observable_source: str = "weight",
) -> str:
    mode = normalize_flatten_mode(flatten_mode)
    loader = _array_loader_expr(observable_source=observable_source, subject_id=tensor_name)
    if mode == "sv_entropy":
        return "\n".join(
            [
                f"arr = {loader}",
                "metric = float(singular_value_entropy(np.asarray(arr)))",
            ]
        )
    op = flat_op_from_reductions(reductions)
    if mode == "global":
        names = list(member_tensor_names or [tensor_name])
        lines: list[str] = ["parts = []"]
        for name in names:
            expr = _array_loader_expr(observable_source=observable_source, subject_id=name)
            lines.append(f"parts.append(np.asarray({expr.split('= ', 1)[1]}).reshape(-1))")
        lines.append("arr = np.concatenate(parts)")
        lines.append(f"metric = float(flat_stat_reduce(arr, {op!r}))")
        return "\n".join(lines)
    if mode == "local":
        return "\n".join(
            [
                f"arr = {loader}",
                "arr = np.asarray(arr).reshape(-1)",
                f"metric = float(flat_stat_reduce(arr, {op!r}))",
            ]
        )
    ordered = sorted(reductions, key=lambda r: r.axis_index)
    lines = [f"arr = {loader}"]
    for spec in ordered:
        lines.append(f"arr = reduce_along_axis(arr, 0, {spec.op!r})")
    lines.append("metric = float(np.asarray(arr).reshape(-1)[0])")
    return "\n".join(lines)


def flat_stat_reduce(arr: np.ndarray, op: str) -> float:
    """Row-major flatten then apply a 1D statistics op."""
    return _stat_reduce_1d(np.asarray(arr, dtype=np.float64).reshape(-1), op)


def reduce_tensor_algebra(
    arr: np.ndarray,
    reductions: list[AxisReductionSpec],
    *,
    flatten_mode: str = "none",
) -> float:
    mode = normalize_flatten_mode(flatten_mode)
    if mode == "local":
        return flat_stat_reduce(arr, flat_op_from_reductions(reductions))
    if mode == "sv_entropy":
        from comfy_research.engine.analysis.tensor_metrics import singular_value_entropy

        return singular_value_entropy(arr)
    if mode == "global":
        raise ValueError("Global flatten requires multiple tensors; use reduce_global_flatten_algebra.")
    work = np.asarray(arr, dtype=np.float64)
    ordered = sorted(reductions, key=lambda r: r.axis_index)
    for spec in ordered:
        if work.ndim < 1:
            raise ValueError("Tensor rank fell below 1 before all reductions were applied.")
        work = np.apply_along_axis(lambda row: _stat_reduce_1d(row, spec.op), 0, work)
    if work.ndim == 0:
        return float(work)
    flat = work.reshape(-1)
    if flat.size != 1:
        raise ValueError(f"Reduction chain did not collapse to a scalar (remaining shape {work.shape}).")
    return float(flat[0])


def reduce_global_flatten_algebra(
    arrays: list[np.ndarray],
    reductions: list[AxisReductionSpec],
) -> float:
    if not arrays:
        raise ValueError("Global flatten requires at least one tensor.")
    parts = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    merged = np.concatenate(parts)
    return flat_stat_reduce(merged, flat_op_from_reductions(reductions))


def global_flatten_label_base(tensor_name: str) -> str:
    """Parameter kind for global flatten (``body.4.weight`` → ``weight``), not a wildcard family."""
    trimmed = (tensor_name or "").strip() or "tensor"
    if "." in trimmed:
        return trimmed.rsplit(".", 1)[-1]
    return trimmed


def global_flatten_representation_kind(representation_id: str) -> str:
    return representation_all_matching_name_base(representation_id)


def auto_algebra_label(
    *,
    tensor_name: str,
    reductions: list[AxisReductionSpec],
    tensor_scope: str = "single",
    flatten_mode: str = "none",
    observable_source: str = "weight",
    representation_id: str = "",
    layer_index: int = 0,
    layer_io: str = "",
) -> str:
    mode = normalize_flatten_mode(flatten_mode)
    if (observable_source or "weight").strip().lower() == "representation":
        if tensor_scope == "all_matching":
            base = representation_all_matching_label_base(representation_id or tensor_name)
        else:
            base = _human_representation_label(representation_id or tensor_name, layer_index, layer_io).replace(" ", "_")
        if mode == "sv_entropy":
            return f"{base}.sv_entropy"
        if mode == "global":
            op = flat_op_from_reductions(reductions)
            kind = (
                global_flatten_representation_kind(representation_id or tensor_name)
                if tensor_scope == "all_matching"
                else base
            )
            return f"{kind}.{op}_global"
        if mode == "local":
            op = flat_op_from_reductions(reductions)
            return f"{base}.{op}_flat"
        ordered = sorted(reductions, key=lambda r: r.axis_index)
        suffix = ".".join(f"{r.op}_{r.axis_label}" for r in ordered)
        return f"{base}.{suffix}" if suffix else base
    if mode == "global":
        op = flat_op_from_reductions(reductions)
        return f"{global_flatten_label_base(tensor_name)}.{op}_global"
    base = family_pattern_from_tensor_name(tensor_name) if tensor_scope == "all_matching" else tensor_name.strip() or "tensor"
    if mode == "local":
        op = flat_op_from_reductions(reductions)
        return f"{base}.{op}_flat"
    ordered = sorted(reductions, key=lambda r: r.axis_index)
    suffix = ".".join(f"{r.op}_{r.axis_label}" for r in ordered)
    return f"{base}.{suffix}" if suffix else base


def suggest_axis_labels(shape: list[int], tensor_name: str = "") -> list[str]:
    rank = len(shape)
    if rank == 0:
        return []
    name = (tensor_name or "").lower()
    if rank == 2 and "weight" in name:
        return ["out", "in"]
    if rank == 3 and "weight" in name:
        return ["out", "in", "kernel"]
    if rank == 4 and "weight" in name:
        return ["out", "in", "h", "w"]
    return [f"dim{i}" for i in range(rank)]
