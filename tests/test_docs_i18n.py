import importlib.util
import posixpath
import re
import subprocess
import sys
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import pytest
from babel.messages.pofile import read_po

from docs_test_helpers import css_block
from scripts import build_docs


ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / "docs/en"
ZH_CATALOGS = ROOT / "docs/locales/zh_CN/LC_MESSAGES"
ZH_REPRODUCTION_CATALOGS = ZH_CATALOGS / "examples/reproductions"

MACHINE_TRANSLATION_MARKERS = (
    "中央处理器",
    "抽象的",
    "繁殖",
    "舒适研究",
    "物品",
    "损耗障碍",
    "检查点",
    "贝塞尔",
)

ENGLISH_IDENTITY_ALLOWLIST = {
    "WJ",
    "Dataset",
    "Model",
    "Loss",
    "Optimizer",
    "Observable",
    "Trainer",
    "CPU",
    "GPU",
    "Hessian",
    "Transformer",
    "BatchNorm",
    "ReLU",
    "SGD",
    "MLP",
    "SVD",
    "Fourier",
    "RankMe",
    "alphaReQ",
    "checkpoint",
    "epoch",
    "seed",
    "observable",
    "Bezier",
    "attention",
    "pretraining",
    "CBS",
    "CLR",
    "SB",
    "LB",
    "Objective",
    "Cross-entropy",
    "Batch size",
    "Python",
    "Node.js",
    "Git",
    "Comfy Research",
    "GraphDocument",
    "string",
    "number",
    "PVI",
    "LMech",
    "Apps",
    "Test",
    "Architecture IQ",
    "LPIPS",
    "RP",
    "MAE",
    "GL",
}


def load_docs_config():
    spec = importlib.util.spec_from_file_location("comfyresearch_docs_conf", ROOT / "docs/conf.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBuilder:
    @staticmethod
    def get_target_uri(pagename: str) -> str:
        return f"{pagename}/"


def page_context(language: str, pagename: str) -> dict:
    config = load_docs_config()
    assert callable(getattr(config, "add_page_context", None))
    context: dict = {}
    app = SimpleNamespace(
        builder=FakeBuilder(),
        config=SimpleNamespace(language=language),
    )
    config.add_page_context(app, pagename, "page.html", context, None)
    return context


def test_docs_requirements_pin_i18n_toolchain() -> None:
    requirements = (ROOT / "docs/requirements.txt").read_text()

    assert "sphinx-intl==2.3.2" in requirements
    assert "Babel==2.17.0" in requirements


def test_docs_config_enables_non_compact_gettext_catalogs() -> None:
    config = load_docs_config()

    assert getattr(config, "locale_dirs", None) == ["../locales"]
    assert getattr(config, "gettext_compact", None) is False


@pytest.mark.parametrize(
    ("language", "expected_locale", "alternate_label"),
    [
        ("en", "en", "中文"),
        ("zh_CN", "zh_CN", "EN"),
    ],
)
def test_page_context_injects_localized_ui(
    language: str,
    expected_locale: str,
    alternate_label: str,
) -> None:
    context = page_context(language, "introduction")
    required_ui_keys = {
        "search_placeholder",
        "search_aria_label",
        "search_button_aria_label",
        "search_empty",
        "search_loading",
        "search_ready",
        "search_unavailable",
        "search_no_results",
        "copy_code",
        "copied_label",
        "copied_status",
        "doc_status_aria_label",
        "doc_type_overview",
        "doc_type_tutorial",
        "doc_type_how_to",
        "doc_type_explanation",
        "doc_type_reference",
        "doc_type_reproduction",
        "doc_status_stable_core",
        "doc_status_phenomenon",
        "github_repo_aria_label",
        "version_switch_aria_label",
        "language_switch_label",
        "language_switch_aria_label",
    }

    assert context["current_locale"] == expected_locale
    assert context["alternate_language_label"] == alternate_label
    assert required_ui_keys <= context["cr_ui"].keys()
    if language == "zh_CN":
        assert context["cr_ui"]["search_placeholder"] == "搜索文档"
        assert context["cr_ui"]["language_switch_label"] == "EN"
    else:
        assert context["cr_ui"]["search_placeholder"] == "Search the docs"
        assert context["cr_ui"]["language_switch_label"] == "中文"
    assert "%(query)s" in context["cr_ui"]["search_no_results"]
    assert context["alternate_language_label"] == context["cr_ui"]["language_switch_label"]
    assert context["current_docs_version"] == "v0.1.0"
    assert context["documentation_versions"] == [
        {"label": "v0.1.0", "url": "./", "is_current": True}
    ]


@pytest.mark.parametrize(
    ("language", "pagename", "expected"),
    [
        ("en", "get-started/first-graph", "../../../../zh/0.1.0/get-started/first-graph/"),
        ("zh_CN", "get-started/first-graph", "../../../../en/0.1.0/get-started/first-graph/"),
        ("en", "search", "../../../zh/0.1.0/search/"),
        ("zh_CN", "genindex", "../../../en/0.1.0/genindex/"),
    ],
)
def test_page_context_links_to_the_same_page_in_the_other_language(
    language: str,
    pagename: str,
    expected: str,
) -> None:
    context = page_context(language, pagename)

    assert context["alternate_language_url"] == expected
    target = posixpath.normpath(
        posixpath.join(
            f"{'en' if language == 'en' else 'zh'}/0.1.0/{pagename}",
            context["alternate_language_url"],
        )
    )
    assert target == f"{'zh' if language == 'en' else 'en'}/0.1.0/{pagename}"


def test_docs_config_registers_page_context_handler() -> None:
    config = load_docs_config()
    assert callable(getattr(config, "setup", None))
    connected = []
    app = SimpleNamespace(connect=lambda event, callback: connected.append((event, callback)))

    result = config.setup(app)

    assert ("html-page-context", config.add_page_context) in connected
    assert ("doctree-read", config.preserve_author_names) in connected
    assert result["parallel_read_safe"] is True


def fake_sphinx_build(command: list[str], **_: object) -> None:
    definition = next(
        (command[index + 1] for index, part in enumerate(command) if part == "-D"),
        "language=en",
    )
    language = definition.split("=", 1)[1]
    destination = Path(command[-1])
    (destination / "introduction").mkdir(parents=True)
    (destination / "introduction/index.html").write_text(
        f'<html lang="{language}"></html>'
    )
    (destination / "searchindex.js").write_text(f"Search.setIndex({language!r})")


def test_build_runs_both_locales_in_one_staging_tree_and_writes_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []

    def record_build(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        fake_sphinx_build(command, **kwargs)

    monkeypatch.setattr(build_docs.subprocess, "run", record_build)
    output = tmp_path / "site"

    build_docs.build(output)

    assert [
        next(
            (
                command[index + 1]
                for index, part in enumerate(command)
                if part == "-D"
            ),
            None,
        )
        for command in commands
    ] == [
        "language=en",
        "language=zh_CN",
    ]
    assert commands[0][-1].endswith("/en/0.1.0")
    assert commands[1][-1].endswith("/zh/0.1.0")
    assert (output / "en/0.1.0/introduction/index.html").is_file()
    assert (output / "zh/0.1.0/introduction/index.html").is_file()
    assert "en/0.1.0/introduction/" in (output / "index.html").read_text()
    assert "introduction/" in (output / "en/0.1.0/index.html").read_text()
    assert "introduction/" in (output / "zh/0.1.0/index.html").read_text()
    assert (output / "en/0.1.0/searchindex.js").read_text() != (
        output / "zh/0.1.0/searchindex.js"
    ).read_text()


def test_build_keeps_previous_output_when_either_locale_entry_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "site"
    output.mkdir()
    (output / build_docs.OUTPUT_MARKER).write_bytes(build_docs.MARKER_CONTENT)
    (output / "previous.txt").write_text("keep me")

    def omit_chinese_entry(command: list[str], **kwargs: object) -> None:
        if "language=en" in command:
            fake_sphinx_build(command, **kwargs)

    monkeypatch.setattr(build_docs.subprocess, "run", omit_chinese_entry)

    with pytest.raises(FileNotFoundError):
        build_docs.build(output)

    assert (output / "previous.txt").read_text() == "keep me"
    assert not (output / "en").exists()
    assert not (output / "zh").exists()


def test_language_switch_is_one_accessible_link() -> None:
    template_path = ROOT / "docs/_templates/language-switch.html"
    assert template_path.is_file()
    template = template_path.read_text()
    config = (ROOT / "docs/conf.py").read_text()

    assert template.count("<a ") == 1
    assert "<details" not in template
    assert "alternate_language_url" in template
    assert "alternate_language_label" in template
    assert "language-switch.html" in config
    assert not (ROOT / "docs/_templates/language-chip.html").exists()
    assert not (ROOT / "docs/_templates/version-chip.html").exists()


def test_language_switch_css_keeps_a_44px_compact_target() -> None:
    stylesheet = (ROOT / "docs/_static/comfyresearch.css").read_text()

    assert ".cr-selector" not in stylesheet
    assert ".cr-example-section" not in stylesheet
    assert ".cr-language-switch" in stylesheet
    switch_block = stylesheet.split(".cr-language-switch {", 1)[1].split("}", 1)[0]
    assert "min-width: 44px;" in switch_block
    assert "min-height: 44px;" in switch_block
    assert re.findall(r"@media \(max-width: (\d+)px\)", stylesheet) == [
        "1279",
        "639",
    ]


def test_language_switch_visited_state_keeps_component_text_color() -> None:
    stylesheet = (ROOT / "docs/_static/comfyresearch.css").read_text()
    switch_link_states = css_block(
        stylesheet,
        ".cr-language-switch:link,\n.cr-language-switch:visited",
    )

    assert "color: var(--cr-ink);" in switch_link_states


def test_makefile_has_reproducible_i18n_update_target() -> None:
    makefile = (ROOT / "Makefile").read_text()
    assert "docs-i18n-update:" in makefile
    target = makefile.split("docs-i18n-update:", 1)[1].split("\n\n", 1)[0]

    assert "docs-i18n-update" in makefile.split(".PHONY:", 1)[1].splitlines()[0]
    assert (
        "$(PYTHON) -m sphinx -b gettext -c docs docs/en docs/_build/gettext"
        in target
    )
    assert (
        "$(PYTHON) -m sphinx_intl update -p docs/_build/gettext "
        "-d docs/locales -l zh_CN --no-obsolete"
        in target
    )
    assert "scripts/build_docs.py" in makefile.split("docs:", 1)[1].split("\n\n", 1)[0]


def test_chinese_catalogs_cover_every_public_page_and_have_no_pending_entries() -> None:
    expected_catalogs = {
        page.relative_to(DOCS_SOURCE).with_suffix(".po")
        for page in DOCS_SOURCE.rglob("*.md")
    }
    actual_catalogs = {
        catalog.relative_to(ZH_CATALOGS)
        for catalog in ZH_CATALOGS.rglob("*.po")
    }

    assert actual_catalogs == expected_catalogs

    pending = []
    for catalog_path in sorted(ZH_CATALOGS.rglob("*.po")):
        with catalog_path.open() as catalog_file:
            catalog = read_po(catalog_file)
        for message in catalog:
            if not message.id:
                continue
            if "fuzzy" in message.flags or not message.string:
                pending.append((catalog_path.relative_to(ROOT), message.id))

    assert pending == []


def test_chinese_catalogs_have_no_translation_placeholders() -> None:
    placeholders = []
    for catalog_path in sorted(ZH_CATALOGS.rglob("*.po")):
        with catalog_path.open() as catalog_file:
            catalog = read_po(catalog_file)
        for message in catalog:
            if "【待翻译】" in message.string:
                placeholders.append((catalog_path.relative_to(ROOT), message.id))

    assert placeholders == []


def test_reproduction_catalogs_avoid_machine_translation_markers() -> None:
    offenders = []
    for catalog_path in sorted(ZH_REPRODUCTION_CATALOGS.glob("*.po")):
        with catalog_path.open() as catalog_file:
            catalog = read_po(catalog_file)
        for message in catalog:
            for marker in MACHINE_TRANSLATION_MARKERS:
                if marker in message.string:
                    offenders.append(
                        (catalog_path.relative_to(ROOT), marker, message.id)
                    )

    assert offenders == []


def test_chinese_catalogs_do_not_leave_reader_prose_in_english() -> None:
    untranslated = []
    for catalog_path in sorted(ZH_CATALOGS.rglob("*.po")):
        with catalog_path.open() as catalog_file:
            catalog = read_po(catalog_file)
        for message in catalog:
            if not message.id or message.string != message.id:
                continue
            outside_code = re.sub(r"`[^`]*`", "", message.id)
            code_only = message.id.startswith("`") and not re.search(
                r"[A-Za-z]", outside_code
            )
            math_block = (
                message.id.startswith("\n")
                and message.id.endswith("\n")
                and "\\" in message.id
            )
            if (
                message.id in ENGLISH_IDENTITY_ALLOWLIST
                or code_only
                or math_block
                or re.fullmatch(r"[A-Z]{2}", message.id) is not None
                or re.fullmatch(r"[0-9][0-9,./ ]*", message.id) is not None
                or message.id.startswith("{bdg-secondary}")
                or message.id.startswith("$")
                or message.id.lstrip().startswith("\\")
                or message.id.isdigit()
            ):
                continue
            untranslated.append((catalog_path.relative_to(ROOT), message.id))

    assert untranslated == []


def test_chinese_catalogs_keep_markdown_labels_parseable() -> None:
    broken_labels = []
    for catalog_path in sorted(ZH_CATALOGS.rglob("*.po")):
        with catalog_path.open() as catalog_file:
            catalog = read_po(catalog_file)
        for message in catalog:
            if not message.id or not message.string:
                continue
            if re.search(r"\*\*[^*]+：\*\*\S", message.string):
                broken_labels.append((catalog_path.relative_to(ROOT), message.string))

    assert broken_labels == []


def test_chinese_introduction_cta_translations_preserve_internal_links() -> None:
    with (ZH_CATALOGS / "introduction.po").open() as catalog_file:
        catalog = read_po(catalog_file)

    assert catalog.get("Start on CPU · 10–15 min").string == (
        "[在 CPU 上开始 · 10–15 分钟](get-started/index.md)"
    )
    assert catalog.get("Already installed? Run the first graph").string == (
        "[已经安装？运行第一个图](get-started/first-graph.md)"
    )


@pytest.fixture(scope="module")
def built_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("bilingual-docs") / "site"
    for language, route in build_docs.LOCALES:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "--keep-going",
                "-b",
                "dirhtml",
                "-D",
                f"language={language}",
                "-c",
                "docs",
                "docs/en",
                str(output / route),
            ],
            check=True,
            cwd=ROOT,
        )
    return output


def language_switch_href(page: Path) -> str:
    match = re.search(
        r'<a class="cr-language-switch"\s+href="([^"]+)"',
        page.read_text(),
    )
    assert match is not None
    return unescape(match.group(1))


def test_real_build_has_bilingual_routes_and_language_metadata(
    built_site: Path,
) -> None:
    english = built_site / "en/0.1.0/introduction/index.html"
    chinese = built_site / "zh/0.1.0/introduction/index.html"

    assert english.is_file()
    assert chinese.is_file()
    assert '<html lang="en"' in english.read_text()
    assert '<html lang="zh-CN"' in chinese.read_text()
    assert "搜索" in (built_site / "zh/0.1.0/search/index.html").read_text()


def test_real_build_preserves_introduction_cta_components_in_both_languages(
    built_site: Path,
) -> None:
    english = (built_site / "en/0.1.0/introduction/index.html").read_text()
    chinese = (built_site / "zh/0.1.0/introduction/index.html").read_text()

    assert "Start on CPU · 10–15 min" in english
    assert "Already installed? Run the first graph" in english
    assert "在 CPU 上开始 · 10–15 分钟" in chinese
    assert "已经安装？运行第一个图" in chinese
    for page in (english, chinese):
        assert 'href="../get-started/"' in page
        assert 'href="../get-started/first-graph/"' in page


@pytest.mark.parametrize(
    ("route", "author"),
    [
        ("jastrzebski-fig1-cyclic-cbs-vs-clr", "王金鑫 (Wang Jinxin)"),
        ("rank-collapse-figure5", "郭绍阳 (Guo Shaoyang)"),
    ],
)
def test_real_build_preserves_author_names_in_both_languages(
    built_site: Path,
    route: str,
    author: str,
) -> None:
    for language in ("en", "zh"):
        page = (
            built_site
            / language
            / "0.1.0/examples/reproductions"
            / route
            / "index.html"
        )
        assert author in page.read_text()


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("en/0.1.0/introduction/index.html", "../../../zh/0.1.0/introduction/"),
        ("zh/0.1.0/introduction/index.html", "../../../en/0.1.0/introduction/"),
        ("en/0.1.0/search/index.html", "../../../zh/0.1.0/search/"),
        ("zh/0.1.0/search/index.html", "../../../en/0.1.0/search/"),
        ("en/0.1.0/genindex/index.html", "../../../zh/0.1.0/genindex/"),
        ("zh/0.1.0/genindex/index.html", "../../../en/0.1.0/genindex/"),
    ],
)
def test_real_build_switches_to_the_same_page(
    built_site: Path,
    route: str,
    expected: str,
) -> None:
    assert language_switch_href(built_site / route) == expected


def test_real_build_writes_independent_search_indexes(built_site: Path) -> None:
    english_index = built_site / "en/0.1.0/searchindex.js"
    chinese_index = built_site / "zh/0.1.0/searchindex.js"

    assert english_index.is_file()
    assert chinese_index.is_file()
    assert english_index.resolve() != chinese_index.resolve()
    assert english_index.read_bytes() != chinese_index.read_bytes()
