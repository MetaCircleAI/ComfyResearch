import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from docs_test_helpers import rendered_text
from scripts import build_docs


def test_backend_ci_installs_docs_dependencies_before_running_docs_tests() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    install_docs = "pip install -r docs/requirements.txt"
    run_tests = "run: python -m pytest -q comfy_research/tests tests"

    assert install_docs in workflow
    assert workflow.index(install_docs) < workflow.index(run_tests)


def test_build_docs_writes_the_docs_site(tmp_path: Path) -> None:
    site = tmp_path / "site"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_docs.py",
            "--output",
            str(site),
        ],
        check=True,
    )

    assert (site / ".nojekyll").is_file()
    routes = [
        "introduction",
        "get-started",
        "user-guide",
        "examples",
        "extend",
        "reference",
    ]
    for route in routes:
        assert (site / "en/dev" / route / "index.html").is_file()

    for route in ("first-graph", "development-mode"):
        assert (site / "en/dev/get-started" / route / "index.html").is_file()
    assert (site / "en/dev/extend/add-observable/index.html").is_file()

    for route in (
        "build-and-run-graphs",
        "observables",
        "projects-and-artifacts",
        "remote-gpu",
        "reproducibility",
        "troubleshooting",
        "execution-model",
    ):
        assert (site / "en/dev/user-guide" / route / "index.html").is_file()

    for route in (
        "application",
        "training-api",
        "data-contracts",
        "node-contracts",
        "support-status",
    ):
        assert (site / "en/dev/reference" / route / "index.html").is_file()

    assert "en/dev/introduction/" in (site / "index.html").read_text()
    assert "introduction/" in (site / "en/dev/index.html").read_text()
    assert (site / build_docs.OUTPUT_MARKER).is_file()
    assert (site / "en/dev/_static/comfyresearch.css").is_file()
    assert (site / "en/dev/_static/comfyresearch.js").is_file()

    introduction_html = (site / "en/dev/introduction/index.html").read_text()
    assert 'data-page="introduction"' in introduction_html
    assert 'class="navbar-brand cr-brand"' in introduction_html
    assert 'class="cr-brand-logo cr-brand-logo--light"' in introduction_html
    assert 'class="cr-brand-logo cr-brand-logo--dark"' in introduction_html
    assert 'class="cr-brand-logo cr-brand-logo--compact"' in introduction_html
    assert "_static/app-logo.svg" in introduction_html
    assert "_static/app-logo-dark.svg" in introduction_html
    assert "_static/app-icon.svg" in introduction_html
    assert 'class="cr-brand-connector"' not in introduction_html
    assert 'class="cr-header-search' in introduction_html
    assert 'class="cr-version-switch"' in introduction_html
    assert 'class="cr-version-button"' in introduction_html
    assert "theme-switch-container" in introduction_html
    assert "theme-switch-button" in introduction_html
    assert '>dev<' in introduction_html
    assert 'class="cr-search-form"' in introduction_html
    assert 'class="cr-search-input"' in introduction_html
    assert 'class="cr-search-results"' in introduction_html
    assert 'aria-label="Search the documentation"' in introduction_html
    assert 'class="btn search-button-field' not in introduction_html
    assert "search-button__kbd-shortcut" in introduction_html
    assert "cr-search-shortcut" not in introduction_html
    assert "cr-version-selector" not in introduction_html
    assert "cr-language-selector" not in introduction_html
    assert "DOCUMENTATION_OPTIONS.search_as_you_type = false" in introduction_html
    assert 'href="../get-started/"' in introduction_html
    assert 'href="../get-started/first-graph/"' in introduction_html
    assert 'href="../examples/"' in introduction_html
    assert introduction_html.count("cr-link-card") == 4
    assert "Reproductions of classic learning-mechanics" in introduction_html
    assert "The graph is more than a picture" in introduction_html
    assert "../_images/overview-stable-workbench.png" in introduction_html
    assert "Product screenshot pending · IMG-01" not in introduction_html
    assert 'class="cr-page-identity"' in introduction_html
    assert "cr-page-badge--dev" not in introduction_html
    assert ">Overview<" in introduction_html
    assert ">Stable core<" in introduction_html
    assert (
        "https://github.com/MetaCircleAI/ComfyResearch/edit/main/docs/en/introduction.md"
        not in introduction_html
    )
    assert "https://github.com/MetaCircleAI/ComfyResearch/issues/new" not in introduction_html
    assert "Edit on GitHub" not in introduction_html
    assert "Report a docs issue" not in introduction_html
    assert "copybutton.js" in introduction_html

    for generated_page in ("search/index.html", "genindex/index.html"):
        generated_html = (site / "en/dev" / generated_page).read_text()
        assert 'class="cr-page-identity"' not in generated_html
        assert 'cr-page-badge--"' not in generated_html

    get_started_page = site / "en/dev/get-started/index.html"
    get_started_html = get_started_page.read_text()
    get_started_text = rendered_text(get_started_page)
    assert "copybutton.css" in get_started_html
    assert "copybutton.js" in get_started_html
    assert 'class="highlight-bash notranslate"' in get_started_html
    assert "copybtn" in (site / "en/dev/_static/copybutton.js").read_text()
    assert (site / "en/dev/_static/clipboard.min.js").is_file()
    assert "Python 3.10 or newer" in get_started_text
    assert "npm --prefix frontend ci" in get_started_text
    assert "/api/health" in get_started_text

    first_graph_page = site / "en/dev/get-started/first-graph/index.html"
    first_graph_html = first_graph_page.read_text()
    first_graph_text = rendered_text(first_graph_page)
    assert "Edge of Stability (CPU)" in first_graph_text
    assert "80 steps" in first_graph_text
    assert "Success checkpoints" in first_graph_text
    assert 'aria-label="Section Navigation"' in first_graph_html
    assert 'aria-label="Breadcrumb"' in first_graph_html
    assert 'class="prev-next-footer d-print-none"' in first_graph_html
    assert 'id="pst-secondary-sidebar"' in first_graph_html
    assert ">Tutorial<" in first_graph_html
    assert ">Stable core<" in first_graph_html

    user_guide_page = site / "en/dev/user-guide/index.html"
    user_guide_html = user_guide_page.read_text()
    user_guide_text = rendered_text(user_guide_page)
    assert "Work with experiments" in user_guide_text
    assert "Build and run graphs" in user_guide_text
    assert 'class="cr-section-nav"' not in user_guide_html
    assert 'href="##' not in user_guide_html
    assert 'id="pst-secondary-sidebar"' not in user_guide_html
    assert (
        'id="pst-secondary-sidebar"'
        in (site / "en/dev/user-guide/build-and-run-graphs/index.html").read_text()
    )

    artifacts_text = rendered_text(
        site / "en/dev/user-guide/projects-and-artifacts/index.html"
    )
    for tier in ("Small", "Medium", "Large"):
        assert tier in artifacts_text
    assert "Record the experiment contract" in rendered_text(
        site / "en/dev/user-guide/reproducibility/index.html"
    )

    stylesheet = (site / "en/dev/_static/comfyresearch.css").read_text()
    assert (
        '--cr-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", '
        "Helvetica, Arial, sans-serif;"
    ) in stylesheet
    assert "--pst-font-family-base: var(--cr-sans);" in stylesheet
    assert "--pst-font-family-heading: var(--cr-sans);" in stylesheet
    assert "font-family: var(--cr-serif);" not in stylesheet
    assert "font-size: 16px;" in stylesheet
    assert "line-height: 1.6;" in stylesheet
    assert "letter-spacing: -0.02em;" in stylesheet
    assert "max-width: 72ch;" in stylesheet
    assert "margin: 40px 0 14px;" in stylesheet
    assert "margin: 20px 0 22px;" in stylesheet
    assert "margin: 20px 0 24px;" in stylesheet
    assert (
        "body.cr-docs :where(a, button, summary, input, select, textarea):focus-visible"
        in stylesheet
    )
    assert ".bd-article section:target > :is(h2, h3, h4)" in stylesheet
    assert ".cr-example-grid {" in stylesheet
    assert "background-color: transparent !important;" in stylesheet
    assert "margin-left: 0 !important;" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
    assert ".bd-article section > p:not(.rubric)" in stylesheet
    assert "min-width: 0;" in stylesheet
    assert "max-width: none;" in stylesheet
    assert ".bd-main .bd-sidebar-secondary {\n    display: none;" not in stylesheet
    assert "font-size: clamp(36px, 3vw, 44px);" in stylesheet
    assert "@media (max-width: 1279px)" in stylesheet
    assert 'html[data-theme="light"] .bd-article div.highlight span' in stylesheet

    extend_text = rendered_text(site / "en/dev/extend/index.html")
    assert "Extend Comfy Research" in extend_text
    assert "Add a scalar Observable" in extend_text
    assert "Node contracts" not in extend_text
    assert "observable_def" not in extend_text
    tutorial_text = rendered_text(site / "en/dev/extend/add-observable/index.html")
    assert "observable_def" in tutorial_text
    assert "recorder_for" in tutorial_text
    assert "generate:node-manifest" in tutorial_text
    assert "Product screenshot pending" not in tutorial_text

    for source in Path("docs/en").rglob("*.md"):
        source_text = source.read_text().replace("Start on CPU · 10–15 min", "")
        assert "—" not in source_text
        assert "–" not in source_text

    reference_text = rendered_text(site / "en/dev/reference/index.html")
    assert "Runtime and file contracts" in reference_text
    assert "Application" in reference_text
    assert "Training API" in reference_text
    assert "Node contracts" in reference_text
    assert "application/x-ndjson" in rendered_text(
        site / "en/dev/reference/training-api/index.html"
    )
    assert "data/workspace.json" in rendered_text(
        site / "en/dev/reference/application/index.html"
    )
    assert "/en/dev/" in rendered_text(
        site / "en/dev/reference/support-status/index.html"
    )

    stable_docs_text = "\n".join(
        rendered_text(site / "en/dev" / route / "index.html")
        for route in (
            "introduction",
            "get-started",
            "user-guide",
            "extend",
            "reference",
        )
    )
    for excluded_product_area in ("Architecture IQ", "LMech", "PVI"):
        assert excluded_product_area not in stable_docs_text

    examples_html = (site / "en/dev/examples/index.html").read_text()
    assert examples_html.count("cr-example-card") == 16
    assert "Wang Jinxin" not in examples_html
    assert "Author attribution appears inside each article" in rendered_text(
        site / "en/dev/examples/index.html"
    )
    assert "max-width: 560px;" not in stylesheet
    assert "min-height: 248px;" not in stylesheet

    example_routes = [
        "edge-of-stability-cpu",
        "lazy-vs-rich-regime",
        "staggered-singular-value-dynamics",
        "edge-of-stability-eos",
        "in-context-associative-recall",
        "information-bottleneck-figure3",
        "jastrzebski-fig1-cyclic-cbs-vs-clr",
        "keskar-fig2-3-sb-lb",
        "random-label-memorization-figure1a",
        "spectral-bias-figure1a",
        "diffusion-reproducibility",
        "linear-mode-connectivity",
        "rank-collapse-figure5",
        "rank-collapse-tinyshakespeare",
    ]
    for route in example_routes:
        page = site / "en/dev/examples/reproductions" / route / "index.html"
        assert page.is_file()
        html = page.read_text()
        assert ">Author<" in html
        assert ">Owner<" not in html
        assert "Phenomenon reproduction" in html
        assert "Limitations" in html
        assert "cr-article-meta" in html
        assert "cr-abstract" in html

    forbidden_screenshot_hashes = {
        "afed5be9d3cc79c3b9ede8f988b3d73c52e8f1813f5d629d51a4a3da9a94c238",
        "5b780832a1dcb49efdea3aa224516473707f688bf19cd3d87d2ce48887f3d541",
    }
    published_hashes = {
        sha256(path.read_bytes()).hexdigest()
        for path in site.rglob("*")
        if path.is_file()
    }
    assert forbidden_screenshot_hashes.isdisjoint(published_hashes)

    examples_text = rendered_text(site / "en/dev/examples/index.html")
    jastrzebski_text = rendered_text(
        site
        / "en/dev/examples/reproductions"
        / "jastrzebski-fig1-cyclic-cbs-vs-clr"
        / "index.html"
    )
    actual_jastrzebski_id = "repro-jastrzbski-fig1-vgg11"
    stale_jastrzebski_id = "repro-jastrzebski-fig1-vgg11"
    assert actual_jastrzebski_id in examples_text
    assert actual_jastrzebski_id in jastrzebski_text
    assert stale_jastrzebski_id not in examples_text
    assert stale_jastrzebski_id not in jastrzebski_text

    keskar_text = rendered_text(
        site / "en/dev/examples/reproductions" / "keskar-fig2-3-sb-lb" / "index.html"
    )
    assert "repro: Keskar Fig 2+3 SB/LB" in keskar_text
    assert "repro: Keskar Fig 2+3 SB/LB - with plot" not in keskar_text
    assert (
        "trainable parameters are interpolated; evaluation uses per-batch "
        "BatchNorm statistics."
    ) in keskar_text
    assert "complete model state" not in keskar_text
    assert "including buffers" not in keskar_text


def test_build_docs_refuses_unowned_output(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    sentinel = site / "sentinel"
    sentinel.write_text("preserve me")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_docs.py",
            "--output",
            str(site),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert sentinel.read_text() == "preserve me"


def test_build_docs_refuses_file_output(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.write_text("preserve me")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_docs.py",
            "--output",
            str(site),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert site.read_text() == "preserve me"


def test_build_docs_refuses_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / build_docs.OUTPUT_MARKER).write_bytes(build_docs.MARKER_CONTENT)
    sentinel = target / "sentinel"
    sentinel.write_text("preserve me")
    site = tmp_path / "site"
    site.symlink_to(target, target_is_directory=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_docs.py",
            "--output",
            str(site),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert site.is_symlink()
    assert sentinel.read_text() == "preserve me"


def test_build_docs_rebuilds_tool_owned_output(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / build_docs.OUTPUT_MARKER).write_bytes(build_docs.MARKER_CONTENT)
    stale_file = site / "stale"
    stale_file.write_text("replace me")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_docs.py",
            "--output",
            str(site),
        ],
        check=True,
    )

    assert (site / build_docs.OUTPUT_MARKER).is_file()
    assert not stale_file.exists()
    assert (site / "en/dev/introduction/index.html").is_file()


def test_failed_build_preserves_tool_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / build_docs.OUTPUT_MARKER).write_bytes(build_docs.MARKER_CONTENT)
    sentinel = site / "sentinel"
    sentinel.write_text("preserve me")

    def fail_build(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "sphinx")

    monkeypatch.setattr(build_docs.subprocess, "run", fail_build)

    with pytest.raises(subprocess.CalledProcessError):
        build_docs.build(site)

    assert sentinel.read_text() == "preserve me"
    assert (site / build_docs.OUTPUT_MARKER).read_bytes() == build_docs.MARKER_CONTENT


def test_cleanup_failure_does_not_fail_completed_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / build_docs.OUTPUT_MARKER).write_bytes(build_docs.MARKER_CONTENT)
    (site / "old").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / build_docs.OUTPUT_MARKER).write_bytes(build_docs.MARKER_CONTENT)
    (staging / "new").write_text("new")

    def fail_cleanup(path: Path) -> None:
        raise OSError(path)

    monkeypatch.setattr(build_docs.shutil, "rmtree", fail_cleanup)

    build_docs.replace_output(site, staging)

    assert (site / "new").read_text() == "new"
