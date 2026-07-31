from pathlib import Path
import re


README = Path("README.md")
CANONICAL_REPOSITORY = "https://github.com/MetaCircleAI/ComfyResearch"
LEGACY_REPOSITORY = "KindXiaoming" + "/ComfyResearch"


def test_readme_uses_canonical_repository() -> None:
    source = README.read_text()

    assert CANONICAL_REPOSITORY in source
    assert LEGACY_REPOSITORY not in source


def test_readme_local_markdown_targets_exist() -> None:
    source = README.read_text()
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source)
    missing = []

    for target in targets:
        if target.startswith(("https://", "http://", "#")):
            continue
        local_target = target.split("#", 1)[0]
        if local_target and not Path(local_target).exists():
            missing.append(target)

    assert missing == []


def test_readme_routes_docs_to_rendered_site() -> None:
    source = README.read_text()
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source)

    assert (
        "[Documentation](https://docs.comfy-research.com/en/0.1.0/introduction/)"
        in source
    )
    assert (
        "[中文文档](https://docs.comfy-research.com/zh/0.1.0/introduction/)"
        in source
    )
    assert not [
        target
        for target in targets
        if target.startswith("docs/en/") and not target.startswith("docs/en/_images/")
    ]


def test_readme_uses_onboarding_overview_before_reproduction_result() -> None:
    source = README.read_text()
    overview = "docs/en/_images/app/overview-stable-workbench.png"
    result = "docs/en/_images/app/edge-of-stability-cpu.png"

    assert Path(overview).is_file()
    assert overview in source
    assert result in source
    assert source.index(overview) < source.index(result)


def test_readme_states_maturity_and_first_run_success() -> None:
    source = README.read_text()

    assert "Status-Pre--release" in source
    assert "do not yet promise compatibility" in source
    assert "## Why ComfyResearch" in source
    assert "## How it works" in source
    assert "### Success checkpoint" in source
    assert "Trainer reaches 80 steps" in source


def test_readme_commands_match_supported_entrypoints() -> None:
    source = README.read_text()

    for required in (
        "### macOS and Linux",
        "<summary><h3>Windows PowerShell</h3></summary>",
        "npm --prefix frontend ci",
        "npm --prefix frontend run build",
        "python app.py --host 127.0.0.1 --port 8042 --open",
        "python -m ruff check comfy_research scripts tests --select F821,F822,F823",
        "python -m pytest -q comfy_research/tests tests",
        "## Core checks",
    ):
        assert required in source

    assert "## Common validation commands" not in source


def test_readme_has_minimal_support_route_and_bounded_length() -> None:
    source = README.read_text()

    assert "## Getting help" in source
    assert "https://github.com/MetaCircleAI/ComfyResearch/issues" in source
    assert "<details>" in source
    assert "<summary>Repository layout</summary>" in source
    assert len(source.splitlines()) <= 180
