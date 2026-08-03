from pathlib import Path
from re import findall

from docs_test_helpers import front_matter


def test_docs_pages_publish_identity_and_governance() -> None:
    requirements = Path("docs/requirements.txt").read_text()
    config = Path("docs/conf.py").read_text()

    assert "sphinx-copybutton==0.5.2" in requirements
    assert '"sphinx_copybutton"' in config
    assert '"article_header_start":' in config
    assert '"page-status.html"' in config
    assert '"template-link.html"' in config
    assert '"article_footer_items": []' in config
    assert '"use_edit_page_button": True' not in config
    assert "html_context = {" not in config

    expected_types = {
        "introduction.md": "overview",
        "get-started/index.md": "tutorial",
        "get-started/first-graph.md": "tutorial",
        "get-started/development-mode.md": "how-to",
        "user-guide/index.md": "overview",
        "user-guide/build-and-run-graphs.md": "how-to",
        "user-guide/observables.md": "how-to",
        "user-guide/projects-and-artifacts.md": "how-to",
        "user-guide/remote-gpu.md": "how-to",
        "user-guide/reproducibility.md": "how-to",
        "user-guide/troubleshooting.md": "how-to",
        "user-guide/execution-model.md": "explanation",
        "examples/index.md": "overview",
        "examples/reproductions/diffusion-reproducibility.md": "reproduction",
        "examples/reproductions/edge-of-stability-cpu.md": "reproduction",
        "examples/reproductions/lazy-vs-rich-regime.md": "reproduction",
        "examples/reproductions/staggered-singular-value-dynamics.md": "reproduction",
        "examples/reproductions/edge-of-stability-eos.md": "reproduction",
        "examples/reproductions/in-context-associative-recall.md": "reproduction",
        "examples/reproductions/information-bottleneck-figure3.md": "reproduction",
        "examples/reproductions/Neural_Mechanics.md": "reproduction",
        "examples/reproductions/random-label-memorization-figure1a.md": "reproduction",
        "examples/reproductions/saad_solla_plateau_reproduction.md": "reproduction",
        "examples/reproductions/spectral-bias-figure1a.md": "reproduction",
        "examples/reproductions/jastrzebski-fig1-cyclic-cbs-vs-clr.md": "reproduction",
        "examples/reproductions/keskar-fig2-3-sb-lb.md": "reproduction",
        "examples/reproductions/linear-mode-connectivity.md": "reproduction",
        "examples/reproductions/rank-collapse-figure5.md": "reproduction",
        "examples/reproductions/rank-collapse-tinyshakespeare.md": "reproduction",
        "extend/index.md": "overview",
        "extend/add-observable.md": "tutorial",
        "reference/index.md": "overview",
        "reference/application.md": "reference",
        "reference/training-api.md": "reference",
        "reference/data-contracts.md": "reference",
        "reference/node-contracts.md": "reference",
        "reference/support-status.md": "reference",
    }
    allowed_types = {
        "overview",
        "tutorial",
        "how-to",
        "explanation",
        "reference",
        "reproduction",
    }

    source_root = Path("docs/en")
    actual_pages = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.md")
        if path != source_root / "index.md"
    }
    assert actual_pages == set(expected_types)

    for relative_path, expected_type in expected_types.items():
        metadata = front_matter(source_root / relative_path)
        assert metadata["doc_type"] == expected_type
        assert metadata["doc_type"] in allowed_types
        expected_status = (
            "phenomenon" if expected_type == "reproduction" else "stable-core"
        )
        assert metadata["doc_status"] == expected_status
        assert "owner" not in metadata
        assert "last_reviewed" not in metadata

    status_template = Path("docs/_templates/page-status.html").read_text()
    assert "cr-page-identity" in status_template
    assert "cr_ui.doc_status_stable_core" in status_template
    assert "cr_ui.doc_status_phenomenon" in status_template
    assert "doc_version_dev" not in status_template

    template_link = Path("docs/_templates/template-link.html").read_text()
    assert "cr-template-link" in template_link
    assert "templateId={{ online_template_id }}" in template_link
    assert 'target="_blank"' in template_link
    assert 'rel="noopener"' in template_link
    assert "cr_ui.open_template" in template_link

    expected_template_ids = {
        "examples/reproductions/jastrzebski-fig1-cyclic-cbs-vs-clr.md": "repro-jastrzbski-fig1-vgg11",
        "examples/reproductions/edge-of-stability-cpu.md": "edge-of-stability-cpu",
        "examples/reproductions/edge-of-stability-eos.md": "97ff01a8-1724-43ec-8c38-fdb62bbe5faf",
        "examples/reproductions/keskar-fig2-3-sb-lb.md": "repro-keskar-fig23-sb-lb",
        "examples/reproductions/lazy-vs-rich-regime.md": "d2e8cdd7-d14c-42c3-94dc-ba2b419c07f9",
        "examples/reproductions/staggered-singular-value-dynamics.md": "a04f21b3-e31c-4ba3-b8b9-d3af752f77d4",
        "examples/reproductions/spectral-bias-figure1a.md": "repro-spectral-bias-fig1a",
        "examples/reproductions/saad_solla_plateau_reproduction.md": "e399fd7d-e107-44d0-94b6-7e2159392253",
        "examples/reproductions/rank-collapse-figure5.md": "repro-rank-collapse-figure5-linear-ce",
        "examples/reproductions/rank-collapse-tinyshakespeare.md": "repro-rank-collapse-tinyshakespeare-pretraining",
        "examples/reproductions/information-bottleneck-figure3.md": "dafb8339-a932-4b10-b3b6-185fc53a5a4f",
        "examples/reproductions/diffusion-reproducibility.md": "repro-diffusion-same-init-different-seed",
        "examples/reproductions/random-label-memorization-figure1a.md": "repro-random-label-memorization-fig1a",
        "examples/reproductions/linear-mode-connectivity.md": "repro-linear-mode-connectivity-cifar10",
        "examples/reproductions/Neural_Mechanics.md": "b38ae9dd-735b-46c4-973f-a850a2a55544",
        "examples/reproductions/in-context-associative-recall.md": "5d1a2ab4-825d-4251-8a5a-d7b83b42c1d7",
        "get-started/first-graph.md": "edge-of-stability-cpu",
    }
    expected_online_template_ids = {
        **expected_template_ids,
        "examples/reproductions/Neural_Mechanics.md": "b15a6036-0e38-4f8f-84ba-8b763c408dc9",
        "examples/reproductions/saad_solla_plateau_reproduction.md": "de684a36-a2d5-440f-bb2b-3c249abb8270",
    }
    template_root = Path("data/graph_library/templates")
    for relative_path, template_id in expected_template_ids.items():
        metadata = front_matter(source_root / relative_path)
        assert metadata["template_id"] == template_id
        assert (template_root / f"{template_id}.json").is_file()
        assert metadata.get("online_template_id", template_id) == expected_online_template_ids[relative_path]
    assert '"doc_status_stable_core": "Stable core"' in config
    assert '"doc_status_phenomenon": "Phenomenon reproduction"' in config

    assert not Path("docs/_templates/page-footer.html").exists()
    assert "report_docs_issue" not in config
    assert "footer_aria_label" not in config


def test_examples_index_surfaces_a_claim_legend_before_the_catalog() -> None:
    examples = Path("docs/en/examples/index.md").read_text()
    category_headings = (
        "## Optimization Stability and Sharpness",
        "## Learning Phases and Feature Formation",
        "## Representation Compression and Information Flow",
        "## Generalization, Memorization, and Reproducibility",
        "## Loss Geometry and Parameter Symmetry",
        "## In-Context Learning",
    )
    reproduction_links = (
        "reproductions/jastrzebski-fig1-cyclic-cbs-vs-clr",
        "reproductions/edge-of-stability-cpu",
        "reproductions/edge-of-stability-eos",
        "reproductions/keskar-fig2-3-sb-lb",
        "reproductions/lazy-vs-rich-regime",
        "reproductions/staggered-singular-value-dynamics",
        "reproductions/spectral-bias-figure1a",
        "reproductions/saad_solla_plateau_reproduction",
        "reproductions/rank-collapse-figure5",
        "reproductions/rank-collapse-tinyshakespeare",
        "reproductions/information-bottleneck-figure3",
        "reproductions/diffusion-reproducibility",
        "reproductions/random-label-memorization-figure1a",
        "reproductions/linear-mode-connectivity",
        "reproductions/Neural_Mechanics",
        "reproductions/in-context-associative-recall",
    )

    assert ":::{div} cr-claim-legend" in examples
    assert examples.index("cr-claim-legend") < examples.index(category_headings[0])
    assert [examples.index(heading) for heading in category_headings] == sorted(
        examples.index(heading) for heading in category_headings
    )
    assert "PHENOMENON" in examples
    assert "PAPER-FAITHFUL" in examples
    assert "current collection contains sixteen phenomenon" in examples
    assert examples.count("::::{grid} 1 1 2 3") == len(category_headings)
    assert all(examples.count(f":link: {link}") == 1 for link in reproduction_links)
    assert "## Learning Mechanics" not in examples
    assert "## Physics of AI" not in examples
    assert "cr-card-author" not in examples
    assert "cr-card-owner" not in examples
    assert "cr-avatar" not in examples


def test_reproduction_pages_use_their_primary_phenomenon_category() -> None:
    category_files = {
        "Optimization Stability and Sharpness": (
            "jastrzebski-fig1-cyclic-cbs-vs-clr.md",
            "edge-of-stability-cpu.md",
            "edge-of-stability-eos.md",
            "keskar-fig2-3-sb-lb.md",
        ),
        "Learning Phases and Feature Formation": (
            "lazy-vs-rich-regime.md",
            "staggered-singular-value-dynamics.md",
            "spectral-bias-figure1a.md",
            "saad_solla_plateau_reproduction.md",
        ),
        "Representation Compression and Information Flow": (
            "rank-collapse-figure5.md",
            "rank-collapse-tinyshakespeare.md",
            "information-bottleneck-figure3.md",
        ),
        "Generalization, Memorization, and Reproducibility": (
            "diffusion-reproducibility.md",
            "random-label-memorization-figure1a.md",
        ),
        "Loss Geometry and Parameter Symmetry": (
            "linear-mode-connectivity.md",
            "Neural_Mechanics.md",
        ),
        "In-Context Learning": ("in-context-associative-recall.md",),
    }
    source_root = Path("docs/en/examples/reproductions")

    for category, filenames in category_files.items():
        for filename in filenames:
            scope = (
                "Spectral audit"
                if filename == "rank-collapse-tinyshakespeare.md"
                else "Phenomenon reproduction"
            )
            source = (source_root / filename).read_text()
            expected = f"{category} · {scope}"
            assert ":::{div} cr-eyebrow\n" + expected + "\n:::" in source


def test_screenshot_capture_plan_tracks_all_planned_captures() -> None:
    plan = Path("docs/screenshot-plan.md").read_text()
    expected_ids = [
        *(f"IMG-{index:02d}" for index in range(1, 11)),
        *(f"IMG-{index:02d}" for index in range(12, 16)),
    ]

    assert findall(r"^## (IMG-\d{2}):", plan, flags=8) == expected_ids


def test_img08_is_a_manual_capture_of_two_single_canvas_projects() -> None:
    plan = Path("docs/screenshot-plan.md").read_text()
    img08 = plan.split("## IMG-08:", 1)[1].split("## IMG-09:", 1)[0]
    plan_text = plan.lower()
    img08_text = "\n".join(
        line for line in img08.splitlines() if "**Final file:**" not in line
    ).lower()

    assert "capture each image manually" in plan_text
    assert "from the running application" in plan_text
    for forbidden in (
        "automated capture",
        "automatic capture",
        "automatically capture",
        "programmatic capture",
        "scripted capture",
        "screenshot automation",
    ):
        assert forbidden not in plan_text

    assert all(term in img08_text for term in ("baseline", "comparison", "project tabs"))
    assert any(
        requirement in img08_text
        for requirement in ("each project owns one canvas", "one canvas per project")
    )
    assert "one project's canvas is open" in img08_text
    for obsolete in (
        "project tree",
        "expanded canvas",
        "pair of canvases",
        "both canvases",
        "more than one canvas",
    ):
        assert obsolete not in img08_text
    assert not findall(
        r"\b(?:two|2|multiple)\b[^.\n]{0,50}\bcanvases\b",
        img08_text,
    )


def test_training_api_documents_workspace_v3() -> None:
    training_api = Path("docs/en/reference/training-api.md").read_text()

    assert "Load the version-3 workspace snapshot" in training_api
    assert "version-2 workspace" not in training_api


def test_docs_theme_uses_native_navigation_and_search() -> None:
    config = Path("docs/conf.py").read_text()

    for required_setting in (
        '"show_prev_next": True',
        '"navigation_with_keys": True',
        '"search_as_you_type": False',
        '"disable_search": True',
        '"navbar_persistent": []',
        '"article_header_start":',
        '"**": ["sidebar-nav-bs.html"]',
        '"introduction": []',
        '"examples/index": []',
    ):
        assert required_setting in config

    assert '"version-chip.html"' not in config
    assert '"language-chip.html"' not in config

    github_template = Path("docs/_templates/github-link.html")
    assert github_template.is_file()
    github_link = github_template.read_text()
    assert '"github-link.html"' in config
    assert 'href="https://github.com/MetaCircleAI/ComfyResearch"' in github_link
    assert 'aria-label="{{ cr_ui.github_repo_aria_label }}"' in github_link
    assert 'class="fa-brands fa-github' in github_link
    assert 'target="_blank"' in github_link
    assert 'rel="noopener"' in github_link
    assert '"github_repo_aria_label": "Open the GitHub repository"' in config
    assert '"github_repo_aria_label": "打开 GitHub 仓库"' in config

    search_template = Path("docs/_templates/search-button-design.html").read_text()
    assert 'class="cr-header-search' in search_template
    assert 'class="cr-search-form"' in search_template
    assert 'class="cr-search-input"' in search_template
    assert 'type="search"' in search_template
    assert 'name="q"' in search_template
    assert 'class="cr-search-results"' in search_template
    assert 'class="cr-search-status"' in search_template
    assert "search-button__kbd-shortcut" in search_template
    assert 'kbd-shortcut__modifier">Ctrl' in search_template
    assert "search-button__button" not in search_template
    assert "<kbd>/</kbd>" not in search_template
    assert search_template.index('class="cr-search-input"') < search_template.index(
        'class="cr-search-submit"'
    )

    version_template = Path("docs/_templates/version-switch.html").read_text()
    assert 'class="cr-version-button"' in version_template
    assert 'type="button"' in version_template
    assert "documentation_versions" in version_template
    assert '"version-switch.html"' in config
    assert config.index('"version-switch.html"') < config.index(
        '"search-button-design.html"'
    )

    script = Path("docs/_static/comfyresearch.js").read_text()
    assert 'event.key === "/"' not in script
    assert 'querySelectorAll(".cr-header-search")' in script
    assert 'querySelectorAll(".cr-version-switch")' in script
    assert 'setAttribute("aria-controls"' in script
    assert 'setAttribute("aria-expanded"' in script
    assert 'root.addEventListener("keydown"' in script
    assert 'event.key === "Escape"' in script
    assert 'event.key === "Enter"' in script
    assert 'event.key === "ArrowDown"' in script
    assert "Search.performSearch(" in script
    assert "showModal" not in script
    assert 'querySelectorAll("button.copybtn")' in script
    for localized_copy in (
        "No results for",
        "Type to search documentation.",
        "Search results are ready.",
        "Search is unavailable",
        "Copy code",
        "Copied to clipboard.",
    ):
        assert localized_copy not in script

    for data_attribute in (
        "data-search-empty",
        "data-search-loading",
        "data-search-ready",
        "data-search-unavailable",
        "data-search-no-results",
        "data-copy-code",
        "data-copied-label",
        "data-copied-status",
    ):
        assert data_attribute in search_template


def test_introduction_surfaces_the_two_beginner_actions() -> None:
    introduction = Path("docs/en/introduction.md").read_text()

    assert ":::{button-ref} get-started/index" in introduction
    assert "Start on CPU · 10–15 min" in introduction
    assert ":::{button-ref} get-started/first-graph" in introduction
    assert "Already installed? Run the first graph" in introduction


def test_product_screenshots_replace_all_pending_captures() -> None:
    expected_figures = {
        "get-started/first-graph.md": (
            "first-graph-template.png",
            "first-graph-trainer.png",
            "first-graph-results.png",
        ),
        "user-guide/build-and-run-graphs.md": (
            "add-node-from-library.png",
            "stable-training-core.png",
        ),
        "user-guide/observables.md": ("observable-and-visualization.png",),
        "user-guide/projects-and-artifacts.md": (
            "project-canvas-tree.png",
            "save-artifact-tiers.png",
        ),
        "user-guide/remote-gpu.md": ("remote-gpu-configuration.png",),
    }
    all_sources = "\n".join(
        source.read_text() for source in Path("docs/en").rglob("*.md")
    )

    assert "Product screenshot pending" not in all_sources
    assert "cr-screenshot-placeholder" not in all_sources
    assert "Capture specification" not in all_sources
    assert "IMG-11" not in all_sources

    for relative_path, image_names in expected_figures.items():
        source = Path("docs/en") / relative_path
        source_text = source.read_text()
        assert source_text.count(":class: cr-product-screenshot") == len(image_names)
        for image_name in image_names:
            assert f"../_images/app/{image_name}" in source_text
            assert (Path("docs/en/_images/app") / image_name).is_file()

    plan = Path("docs/screenshot-plan.md").read_text()
    expected_ids = [
        *(f"IMG-{index:02d}" for index in range(1, 11)),
        *(f"IMG-{index:02d}" for index in range(12, 16)),
    ]
    assert findall(r"^## (IMG-\d{2}):", plan, flags=8) == expected_ids
    assert "IMG-11" not in plan

    jastrzebski = Path(
        "docs/en/examples/reproductions/jastrzebski-fig1-cyclic-cbs-vs-clr.md"
    ).read_text()
    assert "Product screenshot pending · IMG-12" not in jastrzebski
    assert "../../_images/app/jastrzebski-cbs-clr-curves.png" in jastrzebski
    assert "cr-compute-warning" in jastrzebski
    assert "Phenomenon reproduction" in jastrzebski
    assert "## Limitations" in jastrzebski

    keskar = Path("docs/en/examples/reproductions/keskar-fig2-3-sb-lb.md").read_text()
    assert "Product screenshot pending · IMG-13" not in keskar
    assert "Product screenshot pending · IMG-14" not in keskar
    assert "../../_images/app/keskar-sb-lb-results.png" in keskar
    assert "Phenomenon reproduction" in keskar
    assert "## Limitations" in keskar

    edge_of_stability = Path(
        "docs/en/examples/reproductions/edge-of-stability-cpu.md"
    ).read_text()
    assert "Product screenshot pending · IMG-15" not in edge_of_stability
    assert "../../_images/app/edge-of-stability-cpu.png" in edge_of_stability
    assert "Hessian" in edge_of_stability
    assert "## Paper Experiment and Reproduction Boundary" in edge_of_stability
    assert "self-stabilization analysis" in edge_of_stability
    assert "## Interpretation" in edge_of_stability
    assert "## Limitations" in edge_of_stability


def test_get_started_tutorial_contract() -> None:
    landing = Path("docs/en/get-started/index.md").read_text()
    tutorial = Path("docs/en/get-started/first-graph.md").read_text()
    development = Path("docs/en/get-started/development-mode.md").read_text()

    assert "python app.py --host 127.0.0.1 --port 8042 --open" in landing
    assert "get-started/first-graph" not in landing
    assert "first-graph" in landing
    assert "development-mode" in landing

    for required_text in (
        "Edge of Stability (CPU)",
        "80 steps",
        "Success checkpoints",
        "Make one controlled change",
        "../_images/app/first-graph-template.png",
        "../_images/app/first-graph-trainer.png",
        "../_images/app/first-graph-results.png",
    ):
        assert required_text in tutorial

    assert "COMFYRESEARCH_PORT=8042 npm --prefix frontend run dev" in development
    assert "npm --prefix frontend run build" in development


def test_user_guide_page_contracts() -> None:
    guide_dir = Path("docs/en/user-guide")
    pages = {
        name: (guide_dir / f"{name}.md").read_text()
        for name in (
            "build-and-run-graphs",
            "observables",
            "projects-and-artifacts",
            "remote-gpu",
            "reproducibility",
            "troubleshooting",
            "execution-model",
        )
    }

    landing = (guide_dir / "index.md").read_text()
    for name in pages:
        assert name in landing

    graphs = pages["build-and-run-graphs"]
    assert "Cartesian product" in graphs
    assert "one varying axis" in graphs
    assert "../_images/app/add-node-from-library.png" in graphs
    assert "../_images/app/stable-training-core.png" in graphs

    observables = pages["observables"]
    assert "measurement cost" in observables
    assert "log frequency" in observables
    assert "../_images/app/observable-and-visualization.png" in observables

    artifacts = pages["projects-and-artifacts"]
    for tier in ("Small", "Medium", "Large"):
        assert tier in artifacts
    assert "../_images/app/project-canvas-tree.png" in artifacts
    assert "../_images/app/save-artifact-tiers.png" in artifacts

    remote = pages["remote-gpu"]
    assert ".comfyresearch/remote_train_config.json" in remote
    assert "plain JSON" in remote
    assert "127.0.0.1" in remote
    assert "../_images/app/remote-gpu-configuration.png" in remote

    reproducibility = pages["reproducibility"]
    assert "does not prove" in reproducibility
    assert "number of seeds" in reproducibility


def test_reference_page_contracts() -> None:
    reference_dir = Path("docs/en/reference")
    pages = {
        name: (reference_dir / f"{name}.md").read_text()
        for name in (
            "application",
            "training-api",
            "data-contracts",
            "node-contracts",
            "support-status",
        )
    }

    landing = (reference_dir / "index.md").read_text()
    for name in pages:
        assert name in landing

    application = pages["application"]
    for required_text in (
        "0.0.0.0",
        "8000",
        "data/workspace.json",
        "Version-3 projects, graph documents, and viewports",
        "data/graph_library/templates/",
        ".comfyresearch/remote_train_config.json",
        "COMFYRESEARCH_TRAIN_REMOTE_HOST",
        "COMFYRESEARCH_TRAIN_REMOTE_USER",
        "COMFYRESEARCH_TRAIN_REMOTE_PATH",
        "COMFYRESEARCH_TRAIN_REMOTE_PYTHON",
        "COMFYRESEARCH_TRAIN_REMOTE_IDENTITY",
        "COMFYRESEARCH_TRAIN_REMOTE_PASSWORD",
        "COMFYRESEARCH_TRAIN_REMOTE_EXTRA_OPTS",
        "COMFYRESEARCH_TRAIN_REMOTE_UPLOAD_DATASET",
    ):
        assert required_text in application
    assert "data/graph.json" not in application
    assert "Version-2 projects" not in application

    training = pages["training-api"]
    for required_text in (
        "POST /api/train",
        "POST /api/train/control",
        "application/x-ndjson",
        "`progress`",
        "`phase`",
        "`complete`",
        "`paused`",
        "`aborted`",
        "`error`",
    ):
        assert required_text in training

    contracts = pages["data-contracts"]
    assert "Graph document version: `1`" in contracts
    assert "Workspace snapshot version: `3`" in contracts
    assert "legacy `data/graph.json`" not in contracts
    assert "migrates that graph" not in contracts
    assert "200" in contracts

    nodes = pages["node-contracts"]
    assert "npm --prefix frontend run generate:node-manifest" in nodes
    assert "npm --prefix frontend run verify:node-manifest" in nodes
    assert "Do not edit generated files" in nodes
    for node_channel in (
        "DatasetDef",
        "ModelDef",
        "LossDef",
        "OptimizerDef",
        "TrainerDef",
        "InitializationDef",
        "ObservableDef",
        "FrontendNodeDef",
    ):
        assert node_channel in nodes
    assert "../extend/add-observable.md" in nodes

    support = pages["support-status"]
    for product_area in (
        "PVI",
        "LMech",
        "Apps",
        "Test",
        "Architecture IQ",
    ):
        assert product_area in support


def test_reference_pages_advertise_only_the_current_graph_library_surface() -> None:
    support = Path("docs/en/reference/support-status.md").read_text()
    stable_workflow = support.split("## Documented stable workflow", 1)[1].split(
        "## ", 1
    )[0]
    application = Path("docs/en/reference/application.md").read_text()
    training_api = Path("docs/en/reference/training-api.md").read_text()

    assert "Templates, graph files, and export tiers;" in stable_workflow
    assert "Assets" not in stable_workflow
    assert "## Experimental product areas" not in support
    assert "| Auto |" not in support
    assert "| Three Body |" not in support
    assert "data/graph_library/assets.json" not in application
    assert "List `workflows` and `templates`" in training_api


def test_reader_docs_do_not_advertise_the_removed_asset_library() -> None:
    introduction = Path("docs/en/introduction.md").read_text()
    projects = Path("docs/en/user-guide/projects-and-artifacts.md").read_text()
    contracts = Path("docs/en/reference/data-contracts.md").read_text()

    assert "Assets" not in introduction
    assert "Asset" not in projects
    assert "Asset JSON" not in contracts
    assert "reusable **Templates**" in introduction
    assert "graph file or Template" in projects
    assert "Workflow JSON lists" in contracts


def test_extend_information_architecture_and_examples_layout() -> None:
    landing = Path("docs/en/extend/index.md").read_text()
    tutorial = Path("docs/en/extend/add-observable.md").read_text()
    node_contracts = Path("docs/en/reference/node-contracts.md").read_text()

    assert "# Extend Comfy Research" in landing
    assert "add-observable" in landing
    assert "::::{grid} 1 1 1 1" in landing
    assert landing.count(":::{grid-item-card}") == 1
    assert ":link: ../reference/node-contracts" not in landing
    assert "Node contracts" not in landing
    assert "WEIGHT_MEAN_ABS = observable_def(" not in landing
    assert "| Dataset | `DatasetDef`" not in landing

    for required_text in (
        "# Add a scalar Observable",
        "WEIGHT_MEAN_ABS = observable_def(",
        "@recorder_for(WEIGHT_MEAN_ABS)",
        "npm --prefix frontend run generate:node-manifest",
        "npm --prefix frontend run verify:node-manifest",
        "When a custom frontend is justified",
        "Contribution checklist",
    ):
        assert required_text in tutorial
    assert tutorial.count("../reference/node-contracts.md") == 2

    assert "| Dataset | `DatasetDef`" in node_contracts
    assert "| Observable | `ObservableDef`" in node_contracts
    assert "Node contracts" in Path("docs/en/reference/index.md").read_text()


def test_cloudflare_pages_git_integration_is_documented() -> None:
    deployment = Path("docs/cloudflare-pages.md").read_text()

    for required_text in (
        "Connect to Git",
        "Framework preset: `None`",
        "Production branch: `main`",
        "Root directory: leave blank",
        "PYTHON_VERSION",
        "3.11",
        "python -m pip install -r docs/requirements.txt && make docs-test && make docs",
        "docs/_build/html",
        "preview deployment",
    ):
        assert required_text in deployment

    for retired_setting in (
        "CLOUDFLARE_PROJECT_NAME",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "wrangler",
        ".github/workflows/docs-cloudflare.yml",
    ):
        assert retired_setting not in deployment

    assert not Path(".github/workflows/docs-pages.yml").exists()
    assert not Path(".github/workflows/docs-cloudflare.yml").exists()
