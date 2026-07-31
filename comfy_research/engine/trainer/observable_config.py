"""Observable config readers, hist-key conventions, and resume-restore helpers (extracted from trainer_run)."""

import re
from typing import Any, Literal, cast

from comfy_research.engine.models.attention_only_model import AttentionOnlyModel
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.analysis.token_lm_observable_support import TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel
from comfy_research.engine.trainer.scalar import _scalar_bool
from comfy_research.schemas.graph import Node, NodeKind

_EMBEDDING_OBSERVABLE_MODEL_TYPES = (
    AttentionOnlyModel,
    TokenTransformerModel,
    MultiTokenTransformerModel,
    *TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES,
)

OBS_ENCODER_LAYER_SERIES_NODEKINDS = frozenset(
    {
        NodeKind.observable_attention_entropy_mean,
        NodeKind.observable_attention_max_weight_mean,
        NodeKind.observable_attention_head_sink_max,
        NodeKind.observable_attention_position_bias_ratio,
        NodeKind.observable_embedding_effective_rank,
        NodeKind.observable_embedding_feature_drift,
        NodeKind.observable_activation_norm_mean,
        NodeKind.observable_activation_outlier_ratio,
    }
)


def _trainer_disable_extra_observables(td: dict[str, Any]) -> bool:
    v = td.get("disableExtraObservables")
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("true", "1", "yes", "on")


def _observable_metrics_log_enabled(on: Node, disable_extra: bool) -> bool:
    """When debugging, skip expensive wired observables except accuracy (loss is always logged)."""
    if not disable_extra:
        return True
    t = on.type
    return t == NodeKind.observable_accuracy or t == "observable_accuracy"


def _log_step_needs_attention_cache(
    observable_nodes: list[Node],
    disable_extra: bool,
) -> bool:
    """True when any enabled wired observable reads attention maps during ``record()``."""
    attn_types = (
        NodeKind.observable_sink_attention_mass,
        NodeKind.observable_attention_entropy_mean,
        NodeKind.observable_attention_max_weight_mean,
        NodeKind.observable_attention_head_sink_max,
        NodeKind.observable_attention_position_bias_ratio,
        "observable_sink_attention_mass",
        "observable_attention_entropy_mean",
        "observable_attention_max_weight_mean",
        "observable_attention_head_sink_max",
        "observable_attention_position_bias_ratio",
        "observable_attention_relation_score",
    )
    for on in observable_nodes:
        if not _observable_metrics_log_enabled(on, disable_extra):
            continue
        if on.type in attn_types:
            return True
    return False


def _sanitize_observable_hist_segment(name: str, *, max_len: int = 120) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    s = "".join(out).strip("_") or "layer"
    return s[:max_len]


def _gradient_norm_segments_restored_from_hist(oid: str, omh: dict[str, list[float]]) -> list[str]:
    prefix = f"{oid}::top::"
    return sorted({k[len(prefix) :] for k in omh if isinstance(k, str) and k.startswith(prefix)})


def _observable_gradient_norm_normalized(data: dict[str, Any] | None) -> bool:
    """When True (default), divide gradient L2 norms by ``sqrt(# scalar params in that scope)``."""
    return _scalar_bool((data or {}).get("gradientNormNormalized"), True)


def _observable_l2_aggregation(data: dict[str, Any] | None) -> Literal["global", "top_level_module", "tensor"]:
    """How to break out L2 norms on ``observable_gradient_norm`` / ``observable_weight_l2`` nodes."""
    d = data or {}
    raw = d.get("normAggregation")
    if isinstance(raw, str):
        s = raw.strip()
        if s in ("global", "top_level_module", "tensor"):
            return cast(Literal["global", "top_level_module", "tensor"], s)
    if _scalar_bool(d.get("perTopLevel"), False):
        return "top_level_module"
    return "global"


def _sink_attention_mass_layer_mode(data: dict[str, Any] | None) -> Literal["global", "all_layers"]:
    """``observable_sink_attention_mass``: one averaged scalar vs per-encoder-layer series."""
    d = data or {}
    raw = d.get("sinkAttentionMassLayers")
    if isinstance(raw, str) and raw.strip() == "all_layers":
        return "all_layers"
    return "global"


def _obs_encoder_layer_mode(data: dict[str, Any] | None) -> Literal["global", "all_layers"]:
    """``observableEncoderLayers`` on several observables: one scalar vs ``paired::layer::<i>`` series."""
    d = data or {}
    raw = d.get("observableEncoderLayers")
    if isinstance(raw, str) and raw.strip() == "all_layers":
        return "all_layers"
    return "global"


def _activation_stats_layer_mode(data: dict[str, Any] | None) -> Literal["global", "all_layers"]:
    """``observable_activation_stats``: bucket-average scalar vs per-bucket mean/std series."""
    d = data or {}
    raw = d.get("activationStatsLayers")
    if isinstance(raw, str) and raw.strip() == "all_layers":
        return "all_layers"
    return "global"


_ACTIVATION_STATS_LAYER_SUBMOD_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _activation_stats_bucket_from_module_name(name: str) -> str:
    """Group hooked Linear/Conv outputs by encoder layer index, else ``rest`` (embedding, heads, …)."""
    m = _ACTIVATION_STATS_LAYER_SUBMOD_RE.search(name)
    if m:
        return str(int(m.group(1)))
    return "rest"


def _activation_stats_ordered_bucket_keys(buckets: dict[str, tuple[float, float]]) -> list[str]:
    nums: list[int] = []
    has_rest = False
    for k in buckets:
        if k == "rest":
            has_rest = True
        else:
            try:
                nums.append(int(k))
            except ValueError:
                nums.append(-999999)
    nums.sort()
    out = [str(i) for i in nums]
    if has_rest:
        out.append("rest")
    return out


def _observable_hist_subseries_suffixes(oid: str, tag: str, omh: dict[str, list[float]]) -> list[str]:
    prefix = f"{oid}::{tag}::"
    return sorted({k[len(prefix) :] for k in omh if isinstance(k, str) and k.startswith(prefix)})


def _gradient_norm_tensor_segments_restored_from_hist(oid: str, omh: dict[str, list[float]]) -> list[str]:
    return _observable_hist_subseries_suffixes(oid, "tensor", omh)
