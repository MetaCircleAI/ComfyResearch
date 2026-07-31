"""Structural boundary guards for the ``comfy_research.engine.trainer`` package.

These tests cover the trainer_run lifecycle extraction.

AST/import-based and fast. Behavioral coverage lives in the golden tests
(test_trainer_materialize_golden.py, test_train_smoke_golden.py) and is
deliberately not duplicated here.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
TRAINER_RUN_PATH = ENGINE_DIR / "runs" / "trainer_run.py"
TRAINER_PKG_DIR = ENGINE_DIR / "trainer"
TRAINER_PKG_PREFIX = "comfy_research.engine.trainer."

# PR-ordered registry: module basename -> symbols moved out of trainer_run.
# Order matters: a module may only import trainer-package modules listed
# BEFORE it here (one-way intra-package dependency direction). Extended by
# one entry per extraction PR.
MOVED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "scalar": (
        "_scalar_int",
        "_scalar_float",
        "_scalar_str",
        "_scalar_bool",
    ),
    "graph": (
        "_node_map",
        "_incoming",
        "_incoming_all",
        "_incoming_to_target",
        "_require",
        "_require_dataset",
        "_require_optimizer",
        "_DATASET_NODE_TYPES",
        "_DATASET_NODE_TYPE_LABEL",
        "_OPTIMIZER_NODE_TYPES",
        "_OPTIMIZER_NODE_TYPE_LABEL",
    ),
    "tensor_norms": (
        "_weight_l2_norm",
        "_weight_l1_norm",
        "_weight_l2_norm_per_top_level",
        "_gradient_l2_norm_global",
        "_gradient_l2_norm_per_top_level",
        "_parameter_l2_norms_named",
        # Pulled forward from observable_config (spec escape hatch): default
        # arg of _parameter_l2_norms_named.
        "MAX_OBSERVABLE_L2_TENSOR_SERIES",
    ),
    "device_runtime": (
        "_resolve_trainer_compute_device",
        "_optimizer_state_to_device",
        "_trainer_force_single_thread_cpu",
        "_trainer_configure_reproducible_torch",
        "_trainer_master_rng_seed",
    ),
    "checkpoint": (
        "_pack_checkpoint_b64",
        "_apply_checkpoint_b64",
        "load_model_weights_from_checkpoint_b64",
    ),
    "model_payloads": (
        "apply_parameter_tensor_payloads_from_node",
        "apply_parameter_tensor_payloads_from_atomic_chain",
    ),
    "model_helpers": (
        "_numeric_transformer_core",
        "_multi_token_transformer_core",
        "_numeric_hyena_core",
        "_token_lm_vocab_seq_from_model",
        "_mpp_spatiotemporal_core",
        "_flatten_features_for_mse",
        "_regression_batch_for_model",
        "_forward_reg",
        "_prepare_x_for_atomic_sequential",
        "_find_first_embedding",
        "_atomic_chain_dataset_dims",
        "_count_nonlinear_relu_neurons",
        "_ATOMIC_CHAIN_FIRST",
        "_ATOMIC_CHAIN_LAST",
    ),
    "loss_terms": (
        "_trainer_primary_loss_tensor",
        "_trainer_loss_wiring",
        "_kan_reg_loss_term",
        "_weight_reg_loss_term",
        "_weight_reg_loss_additions",
        "_extra_loss_additions",
        "_l2_projection_target_norm",
        "_apply_l2_weight_projection",
        "_KAN_REG_METRICS",
        "_PRIMARY_TRAINER_LOSS_KINDS",
        "_WEIGHT_REG_TRAINER_LOSS_KINDS",
        "_LOSS_SOCKET_AUX_KINDS",
    ),
    "observable_config": (
        "_trainer_disable_extra_observables",
        "_observable_metrics_log_enabled",
        "_log_step_needs_attention_cache",
        "_sanitize_observable_hist_segment",
        "_gradient_norm_segments_restored_from_hist",
        "_gradient_norm_tensor_segments_restored_from_hist",
        "_observable_gradient_norm_normalized",
        "_observable_l2_aggregation",
        "_sink_attention_mass_layer_mode",
        "_obs_encoder_layer_mode",
        "_activation_stats_layer_mode",
        "_activation_stats_bucket_from_module_name",
        "_activation_stats_ordered_bucket_keys",
        "_observable_hist_subseries_suffixes",
        "OBS_ENCODER_LAYER_SERIES_NODEKINDS",
        "_EMBEDDING_OBSERVABLE_MODEL_TYPES",
        "_ACTIVATION_STATS_LAYER_SUBMOD_RE",
    ),
    "observable_metrics": (
        "_activation_mean_std_bucketed",
        "_activation_mean_std_averages",
        "_activation_norm_mean_and_outlier_ratio",
        "_activation_norm_mean_and_outlier_per_bucket",
        "_effective_rank_max_per_encoder_layer",
        "_layer_bucket_flat_concat_vectors",
        "_unwrap_model_block_loop_for_attention",
        "_softmax_attention_probs_or_none",
        "_softmax_attention_probs_all_layers_or_none",
        "_attn_sink_mass",
        "_attn_entropy_mean",
        "_attn_max_weight_mean",
        "_attn_head_sink_max",
        "_attn_position_bias_ratio",
        "_hessian_loss_eigenvalues",
        "_observable_multi_series_l2_payload_from_hist",
        "_activation_stats_layer_series_payload_from_hist",
        "_paired_layer_series_payload_from_hist",
        "_paired_member_series_payload_from_hist",
        "HESSIAN_PARAM_LIMIT",
        "HESSIAN_FORCE_MAX_PARAMS",
        "SPECTRAL_POWER_MAX_ELEMS",
    ),
    "attention_relation_dsl": (
        "AttentionRelationDslError",
        "AttentionRelationPredicate",
        "compile_attention_relation_predicate",
    ),
    "attention_relation_metrics": (
        "MAX_SELECTED_PAIRS",
        "attention_relation_score",
    ),
    "attention_map_history": (),
    "observable_viz": (
        "find_loss_visualization_targets",
        "_trainer_observable_input_handle_ok",
        "_observable_connected_to_trainer",
        "_coerce_node_type_value",
        "_default_observable_viz_variant_for_obs",
        "_obs_viz_stream_row",
        "observable_viz_metric_updates",
        "_LEGACY_VIZ_TO_OBS",
        "_VARIANT_TO_OBS",
    ),
    "teacher_helpers": (
        "_resolve_teacher_sampler",
        "_sample_teacher_input_tensor",
        "_build_teacher_mlp_eval",
        "_resolve_teacher_model_eval",
    ),
    "dataset_helpers": (
        "_vision_flatten_enabled",
        "_resolve_linear_dataset_mlp_dims",
        "_sample_base_vector_dataset_arrays",
        "_dataset_sampling_mode_raw",
        "_trainer_train_dataset_streaming",
        "_take_train_minibatch",
        "_VISION_DATASET_TYPES",
        "_PDE_FIELD_DATASET_TYPES",
        "_LINEAR_LIKE_DATASET_TYPES",
        "_TOY_LANGUAGE_TOKEN_DATASETS",
        "_TOKEN_CLASSIFICATION_DATASET_TYPES",
        "_VECTOR_REGRESSION_DATASET_TYPES",
        "_DIFFUSION_NOISE_DATASET_TYPES",
        "_DATASET_MIXER_TYPES",
        "_TEXT_HEAVY_TOY_LANGUAGE_DATASET_TYPES",
    ),
    # ---- preparation modules (empty tuple = registered for import-order and
    # existence guards without relocation claims) ----
    "context": ("TrainerRunContext",),
    "prepare_state": (),
    "prepare_graph": (
        # Pulled forward with their only consumer, resolve_graph_stage.
        "_TRAINER_REGISTERED_MODEL_TYPES",
        "_TRAINER_MODEL_TYPE_LABEL",
    ),
    # Batched eval helpers re-exported for path tools (parametric
    # path sampler imports them via trainer_run).
    "eval_batches": (
        "_batched_primary_loss_mean",
        "_batched_classification_accuracy",
    ),
    # Cyclic schedule node-data helpers (new module, no relocation claims).
    "cyclic_helpers": (),
    "prepare_config": ("MAX_TRAINING_STEPS",),
    "user_observable_helpers": (
        "_user_observable_path_anchor",
        "_user_observable_definition_code",
        "_user_observable_algebra_rec",
        "_log_step_needs_representation_cache",
    ),
    "dataset_arrays": (),
    "provider_types": (),
    "dataset_materialize": (),
    "prepare_build_vision": (),
    "prepare_build_atomic": (),
    "prepare_build_vector": (),
    "prepare_build_token": (),
    "prepare_finalize": (),
    "recorder": (),
    "minibatch_sampler": (),
    "training_loop": (
        "_trainer_lr_mult_for_step",
        "_loss_plot_png_b64",
    ),
}

# Modules that must not hardcode a device. Exemptions (spec invariant 5,
# verified against pre-refactor source): device_runtime (resolves devices by
# design), checkpoint (cpu map_location defaults), loss_terms
# (_kan_reg_loss_term CPU fallback), observable_metrics (existing ``.cpu()``
# host transfers for NumPy).
NO_HARDCODED_DEVICE: tuple[str, ...] = (
    "scalar",
    "graph",
    "tensor_norms",
    "model_payloads",
    "model_helpers",
    "observable_config",
    "observable_viz",
    "teacher_helpers",
    "dataset_helpers",
)

_HARDCODED_DEVICE_PATTERNS = ('torch.device("cpu")', "torch.device('cpu')", ".cpu()")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _package_parts(path: Path) -> list[str]:
    """Dotted package parts of the module at *path*, relative to the repo root."""
    repo_root = ENGINE_DIR.parents[1]
    return list(path.resolve().relative_to(repo_root).parts[:-1])


def _imported_modules(path: Path) -> set[str]:
    """Absolute module names imported by *path*.

    Relative imports are resolved against the file's own package so that
    ``from .scalar import _scalar_int`` inside the trainer package registers
    as ``comfy_research.engine.trainer.scalar`` — otherwise the one-way and
    intra-package direction guards could be bypassed by switching to relative
    imports.
    """
    pkg_parts = _package_parts(path)
    mods: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    mods.add(node.module)
                continue
            base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
            if node.module:
                mods.add(".".join([*base, node.module]))
            else:
                # ``from . import scalar`` — each alias is a submodule.
                for alias in node.names:
                    mods.add(".".join([*base, alias.name]))
    return mods


def _top_level_local_definitions(path: Path) -> set[str]:
    """Names bound at module top level by def/class/assignment.

    ``ImportFrom`` aliases are deliberately NOT counted: the re-export lines
    in trainer_run.py bind moved names via import, and that must not trip the
    no-local-def guard (spec invariant 4).
    """
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _module_path(mod: str) -> Path:
    return TRAINER_PKG_DIR / f"{mod}.py"


def test_trainer_pkg_modules_do_not_import_trainer_run():
    for mod in MOVED_SYMBOLS:
        offenders = [m for m in _imported_modules(_module_path(mod)) if "trainer_run" in m]
        assert not offenders, f"trainer/{mod}.py imports trainer_run: {offenders}"


def test_intra_package_imports_flow_backward_only():
    order = list(MOVED_SYMBOLS)
    for idx, mod in enumerate(order):
        allowed = set(order[:idx])
        for imported in _imported_modules(_module_path(mod)):
            if imported.startswith(TRAINER_PKG_PREFIX):
                dep = imported.removeprefix(TRAINER_PKG_PREFIX).split(".")[0]
                assert dep in allowed, (
                    f"trainer/{mod}.py imports trainer.{dep}, "
                    f"which is not earlier in the PR order {order}"
                )


def test_trainer_run_has_no_local_defs_for_moved_symbols():
    local = _top_level_local_definitions(TRAINER_RUN_PATH)
    for mod, symbols in MOVED_SYMBOLS.items():
        leaked = sorted(set(symbols) & local)
        assert not leaked, (
            f"trainer_run.py still locally defines {leaked} (moved to trainer/{mod}.py)"
        )


def test_trainer_run_reexports_are_the_moved_objects():
    pytest.importorskip("torch")
    trainer_run = importlib.import_module("comfy_research.engine.runs.trainer_run")
    for mod, symbols in MOVED_SYMBOLS.items():
        source = importlib.import_module(f"comfy_research.engine.trainer.{mod}")
        for name in symbols:
            assert hasattr(trainer_run, name), f"trainer_run lost re-export: {name}"
            assert getattr(trainer_run, name) is getattr(source, name), (
                f"trainer_run.{name} is not the object defined in trainer.{mod}"
            )


def test_moved_callables_report_new_module():
    pytest.importorskip("torch")
    trainer_run = importlib.import_module("comfy_research.engine.runs.trainer_run")
    for mod, symbols in MOVED_SYMBOLS.items():
        expected = f"comfy_research.engine.trainer.{mod}"
        for name in symbols:
            obj = getattr(trainer_run, name)
            if isinstance(obj, type) or (callable(obj) and hasattr(obj, "__module__")):
                assert obj.__module__ == expected, (
                    f"{name}.__module__ == {getattr(obj, '__module__', None)!r}, "
                    f"expected {expected!r}"
                )


def test_no_hardcoded_device():
    for mod in NO_HARDCODED_DEVICE:
        src = _module_path(mod).read_text(encoding="utf-8")
        for pattern in _HARDCODED_DEVICE_PATTERNS:
            assert pattern not in src, (
                f"trainer/{mod}.py contains hardcoded device: {pattern}"
            )


# ---- preparation-stage boundary guards ----

PREPARE_STAGE_MODULES: tuple[str, ...] = (
    "prepare_graph",
    "prepare_config",
    "prepare_build_vision",
    "prepare_build_atomic",
    "prepare_build_vector",
    "prepare_build_token",
    "prepare_finalize",
)

INJECTED_GETTER_MODULES: tuple[str, ...] = ("user_observable_helpers", "recorder")


def test_prepare_state_is_package_private():
    pkg_root = ENGINE_DIR.parents[0]  # comfy_research/
    for path in pkg_root.rglob("*.py"):
        if TRAINER_PKG_DIR in path.parents:
            continue
        exempt = (
            Path(__file__).resolve(),
            TRAINER_RUN_PATH.resolve(),
            # trainer_run.py hosts the prepare_trainer_run orchestrator (the
            # facade); the RNG-sequence golden drives the stages manually to
            # inject its RecordingGenerator (spec) — both are legitimate
            # out-of-package consumers.
            (Path(__file__).parent / "test_trainer_rng_sequence_golden.py").resolve(),
        )
        if path.resolve() in exempt:
            continue
        offenders = [
            m for m in _imported_modules(path)
            if m.endswith("trainer.prepare_state") or m == "prepare_state"
        ]
        assert not offenders, f"{path} imports prepare_state (package-private): {offenders}"


def test_stage_functions_declare_reads_writes():
    for mod in PREPARE_STAGE_MODULES:
        for node in _tree(_module_path(mod)).body:
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_stage"):
                doc = ast.get_docstring(node) or ""
                assert "Reads:" in doc and "Writes:" in doc, (
                    f"trainer/{mod}.py::{node.name} docstring must declare Reads:/Writes:"
                )


def test_user_observable_getter_is_injected_only():
    for mod in INJECTED_GETTER_MODULES:
        offenders = [m for m in _imported_modules(_module_path(mod)) if "user_observables" in m]
        assert not offenders, (
            f"trainer/{mod}.py imports comfy_research.api.user_observables — "
            f"the getter must arrive by injection (patch seam)."
        )


def test_prepare_state_uses_slots():
    src = _module_path("prepare_state").read_text(encoding="utf-8")
    assert "slots=True" in src, "PrepareState must be @dataclass(slots=True)"


# ---- Observable record registry guards ----

# Explicit expected set locks the current behavior
# branches. Never derive this from the registry or from capabilities.
EXPECTED_OBSERVABLE_HANDLER_KINDS: frozenset[str] = frozenset({
    "observable_weight_l2", "observable_weight_l1", "observable_capacity",
    "observable_hessian_eigenvalues", "observable_weight_product_sv", "observable_user",
    "observable_relu_nonlinear_count", "observable_accuracy",
    "observable_gradient_norm", "observable_train_test_gap",
    "observable_activation_stats", "observable_sink_attention_mass",
    "observable_attention_entropy_mean", "observable_attention_max_weight_mean",
    "observable_attention_head_sink_max", "observable_attention_position_bias_ratio",
    "observable_attention_relation_score",
    "observable_attention_map",
    "observable_activation_norm_mean", "observable_activation_outlier_ratio",
    "observable_embedding_effective_rank", "observable_embedding_feature_drift",
    "observable_fourier_component",
    "observable_information_plane",
    "observable_layer_spectral_norm",
    "observable_embedding_evolution", "observable_embedding_trajectory",
    "observable_neuron_trajectory_2d", "observable_weight_displacement",
    "observable_last_layer_weight_norm",
    "observable_representation_rankme", "observable_representation_alpha_req",
    "kan_reg",
})


def test_observable_registry_complete_and_str_keyed():
    """迁移台账：手写集与 NodeDef 定义集不相交，并集恒等于 EXPECTED 总账。

    EXPECTED_OBSERVABLE_HANDLER_KINDS 保持全量清单不变(总账);手写集随迁移单调缩小,
    guard 本身即台账。"""
    pytest.importorskip("torch")
    import inspect

    from comfy_research.engine.trainer.recorder import (
        _HAND_WRITTEN_HANDLERS,
        OBSERVABLE_RECORD_HANDLERS,
        ObservableRecorder,
    )
    from comfy_research.nodes.registry import defs_recorders

    hand = set(_HAND_WRITTEN_HANDLERS)
    defs = set(defs_recorders())
    assert not (hand & defs), f"kind(s) in BOTH channels: {sorted(hand & defs)}"
    assert hand | defs == set(EXPECTED_OBSERVABLE_HANDLER_KINDS), (
        f"ledger drift: missing={sorted(set(EXPECTED_OBSERVABLE_HANDLER_KINDS) - (hand | defs))}, "
        f"extra={sorted((hand | defs) - set(EXPECTED_OBSERVABLE_HANDLER_KINDS))}"
    )
    assert set(OBSERVABLE_RECORD_HANDLERS) == hand | defs
    for k in OBSERVABLE_RECORD_HANDLERS:
        assert type(k) is str, f"registry key {k!r} must be a plain str"
    record_methods = {
        n for n, _ in inspect.getmembers(ObservableRecorder, inspect.isfunction)
        if n.startswith("_record_")
    }
    registered_hand = {f.__name__ for f in _HAND_WRITTEN_HANDLERS.values()}
    assert registered_hand == record_methods, (
        f"unregistered methods: {sorted(record_methods - registered_hand)}; "
        f"registered non-methods: {sorted(registered_hand - record_methods)}"
    )
    assert len(set(OBSERVABLE_RECORD_HANDLERS.values())) == len(OBSERVABLE_RECORD_HANDLERS)
    # legacy 终态:recorder 手写集 == 缓迁登记表。语义限定:只指 recorder
    # provider hand set;prepare_graph/prepare_finalize/observable_viz/canvas 的
    # type-keyed 逻辑是 documented exceptions(见 registry.py 注释),不在此台账。
    from comfy_research.nodes.registry import KNOWN_NODEDEF_MIGRATION_DEFERRED

    assert hand == set(KNOWN_NODEDEF_MIGRATION_DEFERRED), (
        f"recorder hand set != deferred ledger: hand={sorted(hand)}, "
        f"deferred={sorted(KNOWN_NODEDEF_MIGRATION_DEFERRED)}"
    )


# ---- Criterion ownership guards ----

def test_criterion_is_written_only_by_build_criterion_stage():
    """Earlier criterion stores were dead because the criterion is rebuilt for
    every task before any reader can observe them. Keep them dead:
    no build module may assign .criterion. Scope is CONSTRUCTION ownership —
    place_on_device_stage's post-construction ``s.criterion = criterion.to(device)``
    write-back is device placement, not construction, and stays."""
    for mod in ("prepare_build_vision", "prepare_build_atomic",
                "prepare_build_vector", "prepare_build_token"):
        for node in ast.walk(_tree(_module_path(mod))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "criterion":
                        raise AssertionError(
                            f"trainer/{mod}.py assigns .criterion at line {node.lineno}; "
                            f"criterion CONSTRUCTION belongs to build_criterion_stage alone"
                        )


def test_provider_registries_have_exact_build_branch_keys():
    """DATASET_PROVIDERS and MODEL_PROVIDERS are real branch-level
    registries — exactly the four BuildBranch keys, distinct callables."""
    pytest.importorskip("torch")
    from comfy_research.engine.trainer.dataset_materialize import DATASET_PROVIDERS
    from comfy_research.engine.trainer.prepare_finalize import MODEL_PROVIDERS

    expected = {"vision", "atomic", "vector", "token"}
    for name, reg in (("DATASET_PROVIDERS", DATASET_PROVIDERS), ("MODEL_PROVIDERS", MODEL_PROVIDERS)):
        assert set(reg) == expected, f"{name} keys {sorted(reg)} != {sorted(expected)}"
        assert len(set(reg.values())) == 4, f"{name} has duplicate provider callables"


def test_providers_do_not_import_prepare_state():
    """Providers receive Requests, never PrepareState; the state
    judgment lives in the prepare_finalize stages alone."""
    for mod in ("provider_types", "dataset_materialize", "prepare_build_vision",
                "prepare_build_atomic", "prepare_build_vector", "prepare_build_token"):
        offenders = [m for m in _imported_modules(_module_path(mod)) if m.endswith("prepare_state")]
        assert not offenders, f"trainer/{mod}.py imports prepare_state: {offenders}"


def test_model_build_request_carries_no_tensors():
    """Temporary x_t/y_t fields are retired; the model-provider
    boundary is DatasetArrays -> ModelBuildRequest -> ModelBuildResult."""
    import dataclasses

    from comfy_research.engine.trainer.provider_types import ModelBuildRequest

    names = {f.name for f in dataclasses.fields(ModelBuildRequest)}
    assert not (names & {"x_t", "y_t"}), names


def test_record_dispatch_chain_is_gone():
    src_text = _module_path("recorder").read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservableRecorder")
    record = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "record")

    # 1) no ``on.type == ...`` comparisons remain in ``record()``
    for node in ast.walk(record):
        if isinstance(node, ast.Compare):
            left = node.left
            assert not (
                isinstance(left, ast.Attribute) and left.attr == "type"
                and isinstance(left.value, ast.Name) and left.value.id == "on"
            ), f"record() still compares on.type at line {node.lineno}"

    # 2) none of the observable kind strings appear as constants in ``record()``
    #    (Only observable kind strings are forbidden; event names are allowed.)
    consts = {
        n.value for n in ast.walk(record)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    leaked = consts & set(EXPECTED_OBSERVABLE_HANDLER_KINDS)
    assert not leaked, f"record() still hardcodes kind strings: {sorted(leaked)}"

    # 3) no direct ``self._record_*`` call remains in ``record()``
    for node in ast.walk(record):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_record_"):
            raise AssertionError(f"record() calls {node.attr} directly at line {node.lineno}")


# ---- Initialization socket guards ----

def test_initialization_socket_supported_kinds():
    """The initialization-socket allowlist contains only real NodeKind members."""
    from comfy_research.engine.trainer.prepare_config import SUPPORTED_INITIALIZATION_NODE_KINDS
    from comfy_research.schemas.graph import NodeKind

    expected = {
        NodeKind.idnns_initialization,
        NodeKind.mup_initialization,
        NodeKind.saxe_initialization,
        NodeKind.symmetrized_mlp_init,
    }
    assert set(SUPPORTED_INITIALIZATION_NODE_KINDS) == expected


# ---- Observable wiring guards use runtime helpers rather than AST scans ----

# kind -> one-line reason why a wiring surface intentionally misses it.
KNOWN_OBSERVABLE_WIRING_GAPS: dict[str, str] = {}


def test_observable_handlers_match_allowed_obs():
    """主 guard:record registry 与 trainer 校验 allowlist 一致(gaps 显式登记)。"""
    pytest.importorskip("torch")
    from comfy_research.engine.trainer.prepare_graph import allowed_observable_kinds
    from comfy_research.engine.trainer.recorder import observable_record_handler_kinds

    diff = observable_record_handler_kinds() ^ allowed_observable_kinds()
    unexplained = {k for k in diff if k not in KNOWN_OBSERVABLE_WIRING_GAPS}
    assert not unexplained, f"unexplained handler/allowlist drift: {sorted(unexplained)}"


def test_observable_viz_variants_resolvable():
    """viz guard:manifest 里带 vizVariant/spawnsVizNode 的 observable 必须能被后端默认 variant 识别。"""
    from comfy_research.engine.trainer.observable_viz import observable_viz_variant_map
    from comfy_research.generated.node_manifest import load_node_manifest

    variant_map = observable_viz_variant_map()
    missing = []
    for spec in load_node_manifest():
        obs = spec.get("observable") or {}
        if obs.get("vizVariant") or obs.get("spawnsVizNode"):
            t = spec["type"]
            if t not in variant_map and t not in KNOWN_OBSERVABLE_WIRING_GAPS:
                missing.append(t)
    assert not missing, f"observables with viz metadata unknown to backend viz: {sorted(missing)}"


def test_observable_user_variant_whitelist_covers_user_defaults():
    """user-variant guard:默认 variant 为 'user' 的 kind 必须在 user 白名单里。

    两处内联白名单已合一为 _USER_VARIANT_OBSERVABLE_KINDS（相等性在构造上成立，
    旧 guard 的两 literal 比对随之退役)。"""
    from comfy_research.engine.trainer.observable_viz import (
        observable_user_variant_kinds,
        observable_viz_variant_map,
    )

    user_kinds = {k for k, v in observable_viz_variant_map().items() if v == "user"}
    missing = {
        k for k in user_kinds
        if k not in observable_user_variant_kinds() and k not in KNOWN_OBSERVABLE_WIRING_GAPS
    }
    assert not missing, f"user-variant kinds absent from whitelist: {sorted(missing)}"
