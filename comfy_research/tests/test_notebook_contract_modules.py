"""contract 模块守卫。

三层 shim 分类:
1. notebook/codegen permanent(14,前端 codegen 生成的 notebook import 合同);
2. cross-machine exec permanent(2,ssh 在远端 `python -m` 的跨机协议);
3. migration temporary: empty after engine.remote_train_session_ipc moved——
   内部 importer 全走新路径,旧远端 checkout 用旧真身、新 checkout 经 CLI shim,
   franken 混合场景不是受支持的真实用例。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# notebook/codegen 合同(永久;前端 codegen 生成的 notebook import 行字面引用,
# 恰等测试从前端源码派生——数字自证,防口径漂移。linear_dataset_sampling 在列:
# 普查时即为 14 之一(当时是断路径,#123 建档修复,不新增名额)。
NOTEBOOK_CONTRACT_SHIMS = (
    "comfy_research.engine.afno_lite_spatiotemporal_model",
    "comfy_research.engine.hyena_like_numeric_model",
    "comfy_research.engine.information_bottleneck_dataset",
    "comfy_research.engine.linear_dataset_sampling",
    "comfy_research.engine.matrix_preconditioner_optimizers",
    "comfy_research.engine.mpp_spatiotemporal_model",
    "comfy_research.engine.pde_field_dataset_runtime",
    "comfy_research.engine.positional_embedding_layers",
    "comfy_research.engine.random_input_distribution_runtime",
    "comfy_research.engine.symbolic_dataset_compile",
    "comfy_research.engine.teacher_dataset_runtime",
    "comfy_research.engine.toy_language_common",
    "comfy_research.engine.transformer_encoder_custom",
    "comfy_research.engine.vision_datasets_runtime",
    "comfy_research.engine.vision_models",
)

# 跨机 exec 合同(永久;remote/ssh.py 的远端命令字符串有意指旧路径)
EXEC_CONTRACT_SHIMS = (
    "comfy_research.remote_train_cli",
    "comfy_research.remote_train_control_cli",
)
# 新跨机 CLI 无旧 checkout 存量 → ssh 命令字符串直接引用 canonical
# 新路径,不入旧路径 shim 集(不新增 EXEC_CONTRACT_SHIMS 名额)。
CANONICAL_EXEC_CLIS = ("comfy_research.remote.parametric_path_cli",)
def _run_module_help(module: str) -> None:
    r = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert r.returncode == 0, (module, r.stderr[-500:])
    assert "usage:" in r.stdout.lower(), (module, r.stdout[:200])


def test_exec_contract_shims_run_via_python_dash_m() -> None:
    """subprocess 直测 `python -m`(runpy 不覆盖真实跨机入口语义)。"""
    for m in EXEC_CONTRACT_SHIMS:
        _run_module_help(m)


def test_canonical_cli_modules_also_run_via_python_dash_m() -> None:
    """新 canonical 路径同样可 -m(本地开发用)。"""
    for m in ("comfy_research.remote.train_cli", "comfy_research.remote.control_cli"):
        _run_module_help(m)


def test_ssh_remote_command_strings_pin_old_cli_paths() -> None:
    """跨机协议:本地永远向远端发旧模块路径(远端旧 checkout 跑旧真身,
    新 checkout 经 shim——两边都通)。改成新路径会打爆滞后的远端。"""
    src = (ROOT / "comfy_research" / "remote" / "ssh.py").read_text()
    for m in EXEC_CONTRACT_SHIMS:
        assert f'"{m}"' in src, m


def test_canonical_exec_clis_import_and_pin_new_ssh_command_strings() -> None:
    """新跨机合同:CLI 读 stdin JSON(无 argparse/--help)→ import 级断言
    main 存在;并钉 ssh.py 命令字符串引用 canonical 新路径(禁止旧路径形态)。"""
    import importlib

    src = (ROOT / "comfy_research" / "remote" / "ssh.py").read_text()
    for m in CANONICAL_EXEC_CLIS:
        mod = importlib.import_module(m)
        assert callable(getattr(mod, "main", None)), m
        assert f'"{m}"' in src, m
        # 新合同不得存在旧顶层路径文件(无 shim)。
        old_name = m.rsplit(".", 1)[1]
        assert not (ROOT / "comfy_research" / f"remote_{old_name}.py").exists(), m


def test_notebook_contract_whitelist_matches_frontend_census() -> None:
    """白名单恰等守卫(提前落地):从前端源码派生 import-line 普查,
    与枚举常量逐字恰等——多一个少一个都爆,数字口径不再靠注释。"""
    import re

    census: set[str] = set()
    for p in (ROOT / "frontend" / "src").rglob("*.ts*"):
        for m in re.finditer(r"(?:from|import) (comfy_research\.engine\.[a-z_0-9]+)", p.read_text()):
            census.add(m.group(1))
    assert census == set(NOTEBOOK_CONTRACT_SHIMS), census.symmetric_difference(set(NOTEBOOK_CONTRACT_SHIMS))
    assert len(NOTEBOOK_CONTRACT_SHIMS) == 15


MOVED_MODEL_MODULES = (
    "afno_lite_spatiotemporal_model", "attention_only_model", "diagonal_ssm_token_model",
    "diffusion_score_model", "hyena_like_conv_model", "hyena_like_numeric_model",
    "kan_model_build", "linear_attention_model", "mpp_spatiotemporal_model",
    "multi_token_transformer_model", "numeric_transformer_model", "residual_ln_model",
    "rwkv_time_mix_token_model", "slot_attention_token_model", "token_transformer_model",
    "model_loop_expand", "positional_embedding_layers", "local_mixing", "rms_norm_module",
    "atomic_layer_chain",
)
# 白名单成员(旧路径 = 显式 re-export shim);其余 18 旧路径不存在。
MODEL_COMPATIBILITY_SHIMS = frozenset({
    "afno_lite_spatiotemporal_model", "hyena_like_numeric_model", "mpp_spatiotemporal_model",
    "positional_embedding_layers", "transformer_encoder_custom", "vision_models",
})
# 符号级合同覆盖集（models + datasets 白名单 shim）；linear_dataset_sampling
# 属 permanent notebook shim 但非 moved module，由独立测试锁定合同。
SHIMMED_WITH_SYMBOL_CONTRACT = MODEL_COMPATIBILITY_SHIMS | frozenset({
    "information_bottleneck_dataset", "pde_field_dataset_runtime", "random_input_distribution_runtime", "symbolic_dataset_compile",
    "teacher_dataset_runtime", "toy_language_common", "vision_datasets_runtime",
    "matrix_preconditioner_optimizers",  # Shampoo/SOAP
})


def test_model_modules_moved_with_shim_discipline() -> None:
    """真实文件名常量断言——新路径存在;旧路径仅白名单 shim。"""
    for m in MOVED_MODEL_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "models" / f"{m}.py").exists(), m
        old = ROOT / "comfy_research" / "engine" / f"{m}.py"
        if m in MODEL_COMPATIBILITY_SHIMS:
            src = old.read_text()
            assert f"from comfy_research.engine.models.{m} import" in src, m
        else:
            assert not old.exists(), f"{m}: old path must be gone (no shim)"


MOVED_DATASET_MODULES = (
    "dataset_runtime", "dataset_preview_helpers", "information_bottleneck_dataset", "pde_field_dataset_runtime",
    "random_input_distribution_runtime", "synthetic_dataset_builders", "teacher_dataset_runtime",
    "toy_language_common", "toy_language_dyck_runtime", "toy_language_external_runtime",
    "toy_language_formal_runtime", "toy_language_inspect", "toy_language_ngram_runtime",
    "toy_language_pcfg_runtime", "toy_language_physics_lm_runtime", "vision_datasets_runtime",
    "symbolic_dataset_compile", "trainer_dataset_streaming", "streaming_seed",
)
DATASET_COMPATIBILITY_SHIMS = frozenset({
    "information_bottleneck_dataset", "pde_field_dataset_runtime", "random_input_distribution_runtime", "symbolic_dataset_compile",
    "teacher_dataset_runtime", "toy_language_common", "vision_datasets_runtime",
})


def test_dataset_modules_moved_with_shim_discipline() -> None:
    for m in MOVED_DATASET_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "datasets" / f"{m}.py").exists(), m
        old = ROOT / "comfy_research" / "engine" / f"{m}.py"
        if m in DATASET_COMPATIBILITY_SHIMS:
            src = old.read_text()
            assert f"from comfy_research.engine.datasets.{m} import" in src, m
        else:
            assert not old.exists(), f"{m}: old path must be gone (no shim)"


MOVED_OPTIMIZER_MODULES = (
    "optimizer_builders", "matrix_preconditioner_optimizers", "muon_optimizer",
    "signsgd_optimizer", "mup_init", "saxe_init",
)
MOVED_LOSS_MODULES = ("loss_builders", "loss_selection")
OPTIMIZER_COMPATIBILITY_SHIMS = frozenset({"matrix_preconditioner_optimizers"})


def test_optimizer_loss_modules_moved_with_shim_discipline() -> None:
    for m in MOVED_OPTIMIZER_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "optimizers" / f"{m}.py").exists(), m
        old = ROOT / "comfy_research" / "engine" / f"{m}.py"
        if m in OPTIMIZER_COMPATIBILITY_SHIMS:
            src = old.read_text()
            assert f"from comfy_research.engine.optimizers.{m} import" in src, m
        else:
            assert not old.exists(), m
    for m in MOVED_LOSS_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "losses" / f"{m}.py").exists(), m
        assert not (ROOT / "comfy_research" / "engine" / f"{m}.py").exists(), m


def test_builder_registry_paths_exact_by_category() -> None:
    """registry implementation 路径按类恰等(7 optimizer/3 loss/
    model 行保持 models），不同类别不得混面。"""
    from comfy_research.engine.node_builder_registry import REGISTERED_BUILDER_IMPLEMENTATIONS

    want = {
        "adam_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "adamw_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "sgd_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "signsgd_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "muon_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "shampoo_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "soap_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
        "cross_entropy_loss": "comfy_research.engine.losses.loss_builders",
        "diffusion_mse_loss": "comfy_research.engine.losses.loss_builders",
        "mse_loss": "comfy_research.engine.losses.loss_builders",
    }
    got = {k: v for k, v in REGISTERED_BUILDER_IMPLEMENTATIONS.items() if k in want}
    assert got == want, {k: v for k, v in got.items() if want.get(k) != v}


MOVED_ANALYSIS_MODULES = (
    "pca_run", "svd_run", "tensor_metrics", "tensor_selector_resolve", "tensor_slicing",
    "canvas_tensor_chain", "observable_algebra", "observable_user_eval",
    "model_weight_materialize", "activation_collect", "activation_run_store",
    "encoder_attention_extract", "representation_specs", "kan_visualize",
)
MOVED_CRL_MODULES = ("crl_buffer", "crl_networks", "crl_point_u4_env", "crl_port_map", "crl_run")


def test_analysis_crl_modules_moved_hard_cut() -> None:
    """19 模块全硬切(零白名单成员,零 shim)——旧路径必须不存在。"""
    for m in MOVED_ANALYSIS_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "analysis" / f"{m}.py").exists(), m
        assert not (ROOT / "comfy_research" / "engine" / f"{m}.py").exists(), m
    for m in MOVED_CRL_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "crl" / f"{m}.py").exists(), m
        assert not (ROOT / "comfy_research" / "engine" / f"{m}.py").exists(), m


MOVED_RUNS_MODULES = (
    "cuda_devices", "probe_input_selection",
    "sweep_control", "train_control", "train_coordinate_descent", "train_sweep",
    "trainer_run", "ai4science_alias",
)


def test_runs_and_straggler_modules_moved_hard_cut() -> None:
    for m in MOVED_RUNS_MODULES:
        assert (ROOT / "comfy_research" / "engine" / "runs" / f"{m}.py").exists(), m
        assert not (ROOT / "comfy_research" / "engine" / f"{m}.py").exists(), m
    assert (ROOT / "comfy_research" / "engine" / "analysis" / "token_lm_observable_support.py").exists()
    assert not (ROOT / "comfy_research" / "engine" / "token_lm_observable_support.py").exists()


def test_builder_registry_fallback_is_runs_trainer_run() -> None:
    """registry 默认 implementation(legacy_trainer_run 面)= 新路径。"""
    src = (ROOT / "comfy_research" / "engine" / "node_builder_registry.py").read_text()
    assert '"comfy_research.engine.runs.trainer_run"' in src
    assert '"comfy_research.engine.trainer_run"' not in src


def test_linear_dataset_sampling_has_no_shim_chain() -> None:
    """notebook shim 直指 datasets 真身,不得链式经过旧
    engine.random_input_distribution_runtime shim。"""
    from comfy_research.engine import linear_dataset_sampling as shim
    from comfy_research.engine.datasets import random_input_distribution_runtime as real

    assert shim.sample_inputs is real.sample_inputs
    src = (ROOT / "comfy_research" / "engine" / "linear_dataset_sampling.py").read_text()
    assert "from comfy_research.engine.datasets.random_input_distribution_runtime import" in src


def test_shim_symbols_cover_frontend_notebook_imports() -> None:
    """Compatibility shims re-export the exact symbols imported by notebooks."""
    import importlib
    import re

    need: dict[str, set[str]] = {}
    # 括号多行 import 全量捕获(初版单换行截断制造过 build_ 假符号)。
    pat = re.compile(
        r"from (comfy_research\.engine\.[a-z_0-9]+) import (\([^)]*\)|[^\n)]*)",
        re.S,
    )
    for p in (ROOT / "frontend" / "src").rglob("*.ts*"):
        for m in pat.finditer(p.read_text()):
            mod, syms = m.group(1), m.group(2)
            short = mod.rsplit(".", 1)[1]
            if short in SHIMMED_WITH_SYMBOL_CONTRACT:
                # 每个逗号段取首个标识符(括号内注释词不算);TS 插值 ${...}
                # 是动态符号,交由下方 INTERPOLATED 显式集覆盖。
                for frag in syms.strip("()").split(","):
                    frag = frag.split("#")[0].strip()
                    if not frag or "${" in frag:
                        continue
                    mm = re.match(r"([A-Za-z_][A-Za-z_0-9]*)", frag)
                    if mm:
                        need.setdefault(short, set()).add(mm.group(1))
    # afnoAtomicLayerSpecCode.ts 经 BUILDER_BY_KIND 插值出 4 个动态符号——
    # 显式钉死(值集与 TS 字面表恰等由源锚断言)。
    afno_src = (ROOT / "frontend" / "src" / "graph" / "specCode" / "afnoAtomicLayerSpecCode.ts").read_text()
    INTERPOLATED = {
        "afno_lite_spatiotemporal_model": {
            "afno_patch_embed_layer_from_canvas_md",
            "afno_spectral_mixer_layer_from_canvas_md",
            "afno_encoder_block_layer_from_canvas_md",
            "afno_patch_decode_layer_from_canvas_md",
        },
    }
    for short, syms in INTERPOLATED.items():
        for sym in syms:
            assert f'"{sym}"' in afno_src, sym  # 源锚:TS 字面表仍含该值
        need.setdefault(short, set()).update(syms)
    assert set(need) == SHIMMED_WITH_SYMBOL_CONTRACT, set(need).symmetric_difference(SHIMMED_WITH_SYMBOL_CONTRACT)
    for short, syms in sorted(need.items()):
        sub = "models" if short in MODEL_COMPATIBILITY_SHIMS else (
            "optimizers" if short in OPTIMIZER_COMPATIBILITY_SHIMS else "datasets"
        )
        shim = importlib.import_module(f"comfy_research.engine.{short}")
        real = importlib.import_module(f"comfy_research.engine.{sub}.{short}")
        assert syms, short
        for s in sorted(syms):
            assert getattr(shim, s) is getattr(real, s), (short, s)


def test_r1_smoke_imports() -> None:
    """trainer/model import graph 早爆哨。"""
    import importlib

    importlib.import_module("comfy_research.engine.models.model_builders")
    importlib.import_module("comfy_research.engine.analysis.activation_collect")
    importlib.import_module("comfy_research.engine.crl.crl_run")
    importlib.import_module("comfy_research.engine.runs.trainer_run")
    importlib.import_module("comfy_research.engine.runs.train_sweep")
    importlib.import_module("comfy_research.engine.analysis.token_lm_observable_support")
    importlib.import_module("comfy_research.engine.optimizers.optimizer_builders")
    importlib.import_module("comfy_research.engine.losses.loss_builders")
    importlib.import_module("comfy_research.engine.node_builder_registry")


def test_linear_dataset_sampling_import_works() -> None:
    """生成的 symbolic_func notebook cell 的 import 行必须可用(既有断路修复)。"""
    from comfy_research.engine.linear_dataset_sampling import sample_inputs

    assert callable(sample_inputs)


def test_linear_dataset_sampling_is_the_real_function() -> None:
    from comfy_research.engine import linear_dataset_sampling, random_input_distribution_runtime

    assert linear_dataset_sampling.sample_inputs is random_input_distribution_runtime.sample_inputs


def test_codegen_still_emits_the_contract_import() -> None:
    """codegen 字符串不动(golden 恒等由 vitest 门兜;此处钉合同行存在,
    防"顺手把 codegen 改成新路径"让本模块变孤儿)。"""
    src = (ROOT / "frontend" / "src" / "graph" / "extraDatasetCodegen.ts").read_text()
    assert "from comfy_research.engine.linear_dataset_sampling import sample_inputs" in src
