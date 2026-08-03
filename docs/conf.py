import posixpath

from docutils import nodes
from docs.versioning import (
    CURRENT_DOCS_PATH,
    CURRENT_DOCS_VERSION,
    LOCALE_PREFIXES,
    LOCALE_ROUTES,
    PUBLISHED_DOCS_VERSIONS,
)

project = "Comfy Research"
html_title = "Comfy Research"

extensions = ["myst_parser", "sphinx_design", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence", "dollarmath"]
locale_dirs = ["../locales"]
gettext_compact = False

UI_TEXT = {
    "en": {
        "search_placeholder": "Search the docs",
        "search_aria_label": "Search the documentation",
        "search_button_aria_label": "Search documentation",
        "search_empty": "Type to search documentation.",
        "search_loading": "Searching documentation...",
        "search_ready": "Search results are ready.",
        "search_unavailable": (
            "Search is unavailable. Press Enter to open the full search page."
        ),
        "search_no_results": "No results for “%(query)s”.",
        "copy_code": "Copy code",
        "copied_label": "Copied to clipboard",
        "copied_status": "Copied to clipboard.",
        "doc_status_aria_label": "Documentation status",
        "doc_type_overview": "Overview",
        "doc_type_tutorial": "Tutorial",
        "doc_type_how_to": "How-to",
        "doc_type_explanation": "Explanation",
        "doc_type_reference": "Reference",
        "doc_type_reproduction": "Reproduction",
        "doc_status_stable_core": "Stable core",
        "doc_status_phenomenon": "Phenomenon reproduction",
        "open_template": "Open template ↗",
        "github_repo_aria_label": "Open the GitHub repository",
        "version_switch_aria_label": "Switch documentation version",
        "language_switch_label": "中文",
        "language_switch_aria_label": "Switch to Chinese",
    },
    "zh_CN": {
        "search_placeholder": "搜索文档",
        "search_aria_label": "搜索文档",
        "search_button_aria_label": "搜索文档",
        "search_empty": "输入内容以搜索文档。",
        "search_loading": "正在搜索文档……",
        "search_ready": "搜索结果已就绪。",
        "search_unavailable": "搜索暂不可用。按回车键打开完整搜索页面。",
        "search_no_results": "未找到“%(query)s”的相关结果。",
        "copy_code": "复制代码",
        "copied_label": "已复制到剪贴板",
        "copied_status": "已复制到剪贴板。",
        "doc_status_aria_label": "文档状态",
        "doc_type_overview": "概述",
        "doc_type_tutorial": "教程",
        "doc_type_how_to": "操作指南",
        "doc_type_explanation": "说明",
        "doc_type_reference": "参考",
        "doc_type_reproduction": "复现",
        "doc_status_stable_core": "稳定核心",
        "doc_status_phenomenon": "现象复现",
        "open_template": "在线打开模板 ↗",
        "github_repo_aria_label": "打开 GitHub 仓库",
        "version_switch_aria_label": "切换文档版本",
        "language_switch_label": "EN",
        "language_switch_aria_label": "切换到英文",
    },
}

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["comfyresearch.css"]
html_js_files = ["comfyresearch.js"]

html_theme_options = {
    "navbar_start": ["brand.html"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "version-switch.html",
        "search-button-design.html",
        "language-switch.html",
        "github-link.html",
        "theme-switcher",
    ],
    "navbar_persistent": [],
    "article_header_start": [
        "breadcrumbs",
        "page-status.html",
        "template-link.html",
    ],
    "article_footer_items": [],
    "secondary_sidebar_items": ["page-toc"],
    "show_prev_next": True,
    "navigation_with_keys": True,
    "search_as_you_type": False,
    "disable_search": True,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "footer_start": [],
    "footer_center": [],
    "footer_end": [],
    "primary_sidebar_end": [],
}

html_sidebars = {
    "introduction": [],
    "examples/index": [],
    "**": ["sidebar-nav-bs.html"],
}


def add_page_context(app, pagename, templatename, context, doctree):
    locale = app.config.language
    if locale not in LOCALE_ROUTES:
        locale = "en"
    alternate_locale = "zh_CN" if locale == "en" else "en"
    target_uri = app.builder.get_target_uri(pagename)
    current_page = posixpath.join(LOCALE_ROUTES[locale], target_uri)
    alternate_page = posixpath.join(LOCALE_ROUTES[alternate_locale], target_uri)
    alternate_url = posixpath.relpath(
        alternate_page,
        posixpath.dirname(current_page),
    )
    if target_uri.endswith("/"):
        alternate_url += "/"

    version_options = []
    for label, version_path in PUBLISHED_DOCS_VERSIONS:
        version_page = posixpath.join(
            LOCALE_PREFIXES[locale], version_path, target_uri
        )
        version_url = posixpath.relpath(
            version_page,
            posixpath.dirname(current_page),
        )
        if target_uri.endswith("/"):
            version_url += "/"
        version_options.append(
            {
                "label": label,
                "url": version_url,
                "is_current": version_path == CURRENT_DOCS_PATH,
            }
        )

    page_meta = context.get("meta") or {}
    template_id = page_meta.get("template_id")
    if not template_id and doctree is not None:
        template_id = doctree.get("template_id")

    context.update(
        {
            "cr_ui": UI_TEXT[locale],
            "current_locale": locale,
            "alternate_language_url": alternate_url,
            "alternate_language_label": UI_TEXT[locale]["language_switch_label"],
            "current_docs_version": CURRENT_DOCS_VERSION,
            "documentation_versions": version_options,
            "template_id": template_id,
        }
    )


def preserve_author_names(app, doctree):
    for person in doctree.findall(nodes.container):
        if "cr-meta-person" not in person.get("classes", []):
            continue
        for paragraph in person.findall(nodes.paragraph):
            if any(isinstance(child, nodes.strong) for child in paragraph.children):
                paragraph["translatable"] = False


def setup(app):
    app.connect("html-page-context", add_page_context)
    app.connect("doctree-read", preserve_author_names)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
