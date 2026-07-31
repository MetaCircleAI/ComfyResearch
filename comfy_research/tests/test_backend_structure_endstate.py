"""Backend package structure contracts.

engine/ 顶层只保留注册入口和永久 notebook 合同 shim；runtime 全部位于
七个领域子包 + trainer/;remote/ 收编跨机聚类。五类旧路径引用
(dotted import / 包形式 import / patch 字符串 / 路径 pin / __module__)
全仓归零,例外仅限逐项枚举的 golden-sensitive 生成注释与合同常量。
"""
from __future__ import annotations

import re
from pathlib import Path

from comfy_research.tests.test_notebook_contract_modules import (
    EXEC_CONTRACT_SHIMS,
    MOVED_ANALYSIS_MODULES,
    MOVED_CRL_MODULES,
    MOVED_DATASET_MODULES,
    MOVED_LOSS_MODULES,
    MOVED_MODEL_MODULES,
    MOVED_OPTIMIZER_MODULES,
    MOVED_RUNS_MODULES,
    NOTEBOOK_CONTRACT_SHIMS,
)

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "comfy_research" / "engine"

SUBPACKAGES = ("models", "datasets", "optimizers", "losses", "analysis", "crl", "runs", "trainer")

ALL_MOVED = (MOVED_MODEL_MODULES + MOVED_DATASET_MODULES + MOVED_OPTIMIZER_MODULES
             + MOVED_LOSS_MODULES + MOVED_ANALYSIS_MODULES + MOVED_CRL_MODULES
             + MOVED_RUNS_MODULES + ("token_lm_observable_support", "remote_train_session_ipc"))

# golden-sensitive 生成注释/readme 的旧路径字样(非 import 合同;改=金标漂移)。
# 逐项 file→modules 列,任何未列出的旧路径引用 fail。
FRONTEND_COMMENT_ALLOWLIST: dict[str, frozenset[str]] = {
    "components/nodes/CrlEnvConfigNode.tsx": frozenset({"crl_run"}),
    "components/nodes/CrlResidualMlpNode.tsx": frozenset({"crl_networks"}),
    "components/nodes/CrlTrainerNode.tsx": frozenset({"crl_run"}),
    "graph/extraDatasetCodegen.ts": frozenset({"crl_run", "information_bottleneck_dataset", "random_input_distribution_runtime", "symbolic_dataset_compile", "trainer_run"}),
    "graph/generatedNodeSpecTypes.ts": frozenset({"activation_collect"}),
    "graph/observableNotebookStub.ts": frozenset({"observable_user_eval", "trainer_run"}),
    "graph/optimizerCodegen.ts": frozenset({"matrix_preconditioner_optimizers"}),
    "graph/specCode/absolutePosEmbedLayerSpecCode.ts": frozenset({"positional_embedding_layers"}),
    "graph/specCode/afnoAtomicLayerSpecCode.ts": frozenset({"afno_lite_spatiotemporal_model"}),
    "graph/specCode/afnoLiteSpatiotemporalModelSpecCode.ts": frozenset({"afno_lite_spatiotemporal_model"}),
    "graph/specCode/alternativeArchNotebookSpecCode.ts": frozenset({"local_mixing"}),
    "graph/specCode/attentionOnlyModelSpecCode.ts": frozenset({"attention_only_model"}),
    "graph/specCode/diffusionScoreNotebookSpecCode.ts": frozenset({"diffusion_score_model"}),
    "graph/specCode/inContextAssociativeRecallDatasetSpecCode.ts": frozenset({"trainer_run"}),
    "graph/specCode/inputSamplerSpecCode.ts": frozenset({"random_input_distribution_runtime"}),
    "graph/specCode/localMixingLayerSpecCode.ts": frozenset({"local_mixing"}),
    "graph/specCode/mppSpatiotemporalModelSpecCode.ts": frozenset({"mpp_spatiotemporal_model"}),
    "graph/specCode/notebookImplementationReadme.ts": frozenset({
        "attention_only_model", "diagonal_ssm_token_model", "diffusion_score_model",
        "hyena_like_conv_model", "linear_attention_model", "rwkv_time_mix_token_model",
        "slot_attention_token_model"}),
    "graph/specCode/numericHyenaModelSpecCode.ts": frozenset({"hyena_like_numeric_model"}),
    "graph/specCode/pdeFieldDatasetSpecCode.ts": frozenset({"pde_field_dataset_runtime"}),
    "graph/specCode/randomInputDistributionSpecCode.ts": frozenset({"random_input_distribution_runtime"}),
    "graph/specCode/rotaryEmbedLayerSpecCode.ts": frozenset({"positional_embedding_layers"}),
    "graph/specCode/teacherDatasetSpecCode.ts": frozenset({"teacher_dataset_runtime"}),
    "graph/specCode/tokenPredictionDatasetSpecCode.ts": frozenset({"trainer_run"}),
    "graph/specCode/toyLanguageDyckNotebookEmbed.ts": frozenset({"toy_language_common", "toy_language_dyck_runtime"}),
    "graph/specCode/toyLanguageExternalNotebookEmbed.ts": frozenset({"toy_language_common", "toy_language_external_runtime"}),
    "graph/specCode/toyLanguageFormalNotebookEmbed.ts": frozenset({"toy_language_common", "toy_language_formal_runtime"}),
    "graph/specCode/toyLanguageNgramNotebookEmbed.ts": frozenset({"toy_language_common", "toy_language_ngram_runtime"}),
    "graph/specCode/toyLanguagePcfgNotebookEmbed.ts": frozenset({"toy_language_common", "toy_language_pcfg_runtime"}),
    "graph/specCode/toyLanguagePhysicsLmNotebookEmbed.ts": frozenset({"toy_language_common", "toy_language_physics_lm_runtime"}),
    "graph/specCode/toyLanguageRuntimeNotebookEmbeds.generated.ts": frozenset({"toy_language_common"}),
    "graph/specCode/visionDatasetSpecCode.ts": frozenset({"vision_datasets_runtime"}),
    "graph/visionDatasetDefinitionCode.ts": frozenset({"vision_datasets_runtime"}),
}


def test_engine_top_level_is_exactly_registry_plus_contract_shims() -> None:
    """engine/ 顶层恰等:__init__ + node_builder_registry(唯一真身)+ 15 notebook
    shim。临时层为零。"""
    got = sorted(p.name for p in ENGINE.glob("*.py"))
    shims = sorted(m.rsplit(".", 1)[1] + ".py" for m in NOTEBOOK_CONTRACT_SHIMS)
    want = sorted(["__init__.py", "node_builder_registry.py", *shims])
    assert got == want, set(got).symmetric_difference(set(want))


def test_subpackages_exist_with_lightweight_inits() -> None:
    """七子包 + trainer 存在;__init__ 轻量(docstring/future 之外零 import——
    包 import 不得提前拉重依赖)。"""
    for sub in SUBPACKAGES:
        init = ENGINE / sub / "__init__.py"
        assert init.exists(), sub
        if sub == "trainer":
            continue  # trainer 是既有包，自持纪律。
        body = init.read_text()
        assert not re.search(r"^(from|import) (?!__future__)", body, re.M), sub


def test_remote_package_holds_the_cross_machine_cluster() -> None:
    for name in ("train_cli", "control_cli", "session_registry", "ssh", "session_ipc"):
        assert (ROOT / "comfy_research" / "remote" / f"{name}.py").exists(), name
    for m in EXEC_CONTRACT_SHIMS:
        assert (ROOT / "comfy_research" / (m.rsplit(".", 1)[1] + ".py")).exists(), m


def test_no_old_engine_paths_anywhere_in_python_trees() -> None:
    """五类扫描固化:全仓 python 树零旧路径引用(合同常量测试文件与 shim 自身豁免)。"""
    alt = "|".join(ALL_MOVED)
    pats = [
        re.compile(r"comfy_research\.engine\.(" + alt + r")\b"),
        re.compile(r"comfy_research/engine/(" + alt + r")\.py"),
        re.compile(r'"engine" / "(' + alt + r')\.py"'),
        re.compile(r'_ENGINE / "(' + alt + r')\.py"'),
    ]
    exempt_names = {m.rsplit(".", 1)[1] for m in NOTEBOOK_CONTRACT_SHIMS}
    # 跨版本 harness fallback(窄豁免:仅此一文件、仅此一串——它以 HEAD scripts
    # 驱动旧布局的 base worktree；旧路径 import 是其存在意义。
    HARNESS_FALLBACK = ("scripts/differential_harness.py", "comfy_research.engine.trainer_run")
    offenders: list[str] = []
    for root in ("comfy_research", "tests", "scripts"):
        for p in (ROOT / root).rglob("*.py"):
            sp = str(p)
            if "__pycache__" in sp:
                continue
            if p.name == "test_notebook_contract_modules.py" or sp.endswith("test_backend_structure_endstate.py"):
                continue
            if p.parent == ENGINE and p.stem in exempt_names:
                continue  # shim 文件自身(docstring 提及)
            text = p.read_text(errors="ignore")
            for pat in pats:
                for m in pat.finditer(text):
                    if sp.endswith(HARNESS_FALLBACK[0]) and m.group(0) == HARNESS_FALLBACK[1]:
                        continue
                    offenders.append(f"{sp}: {m.group(0)}")
    assert not offenders, offenders[:20]


def test_frontend_old_path_mentions_are_enumerated() -> None:
    """前端旧路径字样恰在 allowlist 内(import 合同由 whitelist 恰等测试独立钉;
    此处管非 import 的生成注释/readme——任何新增旧路径提及必须显式登记)。"""
    alt = "|".join(ALL_MOVED)
    pat = re.compile(r"comfy_research[./]engine[./](" + alt + r")\b")
    fe = ROOT / "frontend" / "src"
    got: dict[str, set[str]] = {}
    for p in fe.rglob("*.ts*"):
        rel = str(p.relative_to(fe))
        for m in pat.finditer(p.read_text(errors="ignore")):
            got.setdefault(rel, set()).add(m.group(1))
    want = {k: set(v) for k, v in FRONTEND_COMMENT_ALLOWLIST.items() if v}
    assert got == want, {
        "unexpected": {k: sorted(v - want.get(k, set())) for k, v in got.items() if v - want.get(k, set())},
        "missing": {k: sorted(v - got.get(k, set())) for k, v in want.items() if v - got.get(k, set())},
    }
