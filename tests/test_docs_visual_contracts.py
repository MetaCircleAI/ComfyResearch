from pathlib import Path

from docs_test_helpers import css_block, css_block_with, css_blocks


def contrast_ratio(foreground: str, background: str) -> float:
    def relative_luminance(color: str) -> float:
        channels = tuple(int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))
        linear = tuple(
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        )
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_product_screenshot_style() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    screenshots = css_block(
        stylesheet,
        ".bd-article figure:has(> img.cr-product-screenshot),\n"
        ".bd-article figure:has(> img.cr-reproduction-screenshot)",
    )
    screenshot_images = css_block(
        stylesheet,
        ".bd-article img.cr-product-screenshot,\n"
        ".bd-article img.cr-reproduction-screenshot",
    )

    assert "margin: 24px 0 32px;" in screenshots
    assert "width: 100%;" in screenshot_images
    assert "border: 1px solid var(--cr-line);" in screenshot_images
    assert ".cr-screenshot-placeholder" not in stylesheet
    assert ".cr-capture-spec" not in stylesheet


def test_portrait_product_screenshots_are_narrow_and_centered() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    portrait = css_block(
        stylesheet,
        ".bd-article figure:has(> img.cr-product-screenshot-portrait)",
    )

    assert "max-width: 360px;" in portrait
    assert "margin: 24px auto 32px;" in portrait


def test_docs_theme_uses_the_brand_palette() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text().lower()

    for token, value in {
        "--cr-brand-ink": "#29252d",
        "--cr-brand-primary": "#7714cf",
        "--cr-brand-secondary": "#998dc7",
        "--cr-brand-soft": "#d4c2fc",
    }.items():
        assert f"{token}: {value};" in stylesheet

    for retired_color in ("#98632d", "#7f4f24", "#f1e5d2"):
        assert retired_color not in stylesheet


def test_docs_theme_keeps_content_surfaces_neutral_and_accents_scoped() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    for token, value in {
        "--cr-paper": "#ffffff",
        "--cr-line": "#e5e7eb",
        "--cr-neutral-soft": "#f6f7f8",
    }.items():
        assert f"{token}: {value};" in stylesheet

    for selector, expected_background in (
        ("html", "background: var(--cr-paper);"),
        ("body.cr-docs", "background: var(--cr-paper);"),
        (".bd-article table th", "background: var(--cr-neutral-soft);"),
        (".bd-article code.literal", "background: var(--cr-neutral-soft);"),
        (".cr-example-card .sd-badge", "background: var(--cr-neutral-soft);"),
        (".cr-card-tags code", "background: var(--cr-neutral-soft);"),
        (
            "#pst-primary-sidebar-modal[open],\n  #pst-secondary-sidebar-modal[open]",
            "background: var(--cr-neutral-soft);",
        ),
    ):
        block = css_block_with(stylesheet, selector, expected_background)
        assert expected_background in block
        assert "rgba(212, 194, 252" not in block

    card_block = css_block_with(
        stylesheet,
        ".cr-link-grid .sd-card,\n.cr-example-card.sd-card",
        "background: var(--cr-surface);",
    )
    assert any(
        background in card_block
        for background in (
            "background: var(--cr-paper);",
            "background: var(--cr-surface);",
            "background: var(--cr-neutral-soft);",
        )
    )

    assert "rgba(212, 194, 252" not in stylesheet
    assert "background: var(--cr-accent);" in stylesheet
    assert "outline: 3px solid var(--cr-accent);" in stylesheet


def test_docs_theme_has_dark_tokens_and_a_theme_aware_header() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    dark_theme = css_block(stylesheet, 'html[data-theme="dark"]')

    for token, value in {
        "--cr-paper": "#141116",
        "--cr-surface": "#1c181f",
        "--cr-header-bg": "#242424",
        "--cr-neutral-soft": "#24202a",
        "--cr-ink": "#f7f3fa",
        "--cr-text": "#d8d0dc",
        "--cr-accent": "#b768ff",
        "--cr-on-accent": "#141116",
        "--cr-on-ink": "#141116",
    }.items():
        assert f"{token}: {value};" in dark_theme

    for token in (
        "--cr-line:",
        "--cr-status-stable-bg:",
        "--cr-code:",
        "--cr-backdrop:",
        "--pst-color-text-base:",
        "--pst-color-surface:",
        "--pst-color-warning-bg:",
    ):
        assert token in dark_theme

    switcher = css_block(stylesheet, ".theme-switch-container .theme-switch-button")
    assert "width: 44px;" in switcher
    assert "height: 44px;" in switcher
    assert "border: 1px solid var(--cr-line);" in switcher
    assert ".theme-switch-container .dropdown-menu" in stylesheet
    assert "background: var(--cr-backdrop);" in stylesheet


def test_dark_header_and_table_rows_stay_neutral_and_legible() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    root_theme = css_block(stylesheet, ":root")
    dark_theme = css_block(stylesheet, 'html[data-theme="dark"]')
    header = css_block(stylesheet, ".bd-header")
    assert "--cr-header-bg: #ffffff;" in root_theme
    assert "--cr-header-bg: #242424;" in dark_theme
    assert "--pst-color-on-background: var(--cr-surface);" in root_theme
    assert "--pst-color-on-background: var(--cr-surface);" in dark_theme
    assert "background: var(--cr-header-bg);" in header
    assert "background-color: var(--cr-header-bg) !important;" in header

    dark_cells = css_block(
        stylesheet,
        'html[data-theme="dark"] .bd-article table.table tbody > tr > *',
    )
    assert "color: var(--cr-text);" in dark_cells
    assert "background-color: var(--cr-surface);" in dark_cells
    assert "box-shadow: none;" in dark_cells

    dark_odd_cells = css_block(
        stylesheet,
        'html[data-theme="dark"] .bd-article table.table tbody > tr.row-odd > *',
    )
    assert "background-color: var(--cr-neutral-soft);" in dark_odd_cells

    assert contrast_ratio("#d8d0dc", "#1c181f") >= 4.5
    assert contrast_ratio("#d8d0dc", "#24202a") >= 4.5
    assert contrast_ratio("#b4a9bb", "#242424") >= 4.5


def test_docs_theme_uses_accessible_semantic_control_foregrounds() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    root_theme = css_block(stylesheet, ":root")
    dark_theme = css_block(stylesheet, 'html[data-theme="dark"]')

    assert "--cr-on-accent: #fff;" in root_theme
    assert "--cr-on-ink: #fff;" in root_theme

    for foreground, background in (
        ("#141116", "#b768ff"),
        ("#141116", "#dfc4ff"),
        ("#141116", "#f7f3fa"),
    ):
        assert contrast_ratio(foreground, background) >= 4.5

    for selector, token in (
        (
            ".bd-header .bd-navbar-elements .current > .nav-link,\n"
            ".bd-header .bd-navbar-elements .active > .nav-link",
            "--cr-on-accent",
        ),
        (".cr-intro-actions p:first-child > a", "--cr-on-accent"),
        (".cr-intro-actions p:first-child > a:hover", "--cr-on-accent"),
        (".cr-intro-actions p:nth-child(3) > a", "--cr-on-ink"),
        (".cr-intro-actions p:nth-child(3) > a:hover", "--cr-on-accent"),
        (".cr-avatar", "--cr-on-ink"),
        ("#pst-back-to-top", "--cr-on-accent"),
    ):
        assert f"color: var({token})" in css_block(stylesheet, selector)

    copy_button_hover = css_block(
        stylesheet,
        ".bd-article div.highlight button.copybtn:hover,\n"
        ".bd-article div.highlight button.copybtn:focus-visible,\n"
        ".bd-article div.highlight button.copybtn.success",
    )
    assert "color: #fff;" in copy_button_hover
    assert "--cr-on-accent: #141116;" in dark_theme
    assert "--cr-on-ink: #141116;" in dark_theme


def test_informational_surfaces_do_not_reuse_the_action_accent() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    assert "--cr-neutral-strong: #9ca3af;" in stylesheet
    for selector in (
        ".bd-article .cr-abstract.admonition",
        ".admonition,\ndiv.admonition",
        ".bd-article .admonition.note",
        ".bd-article blockquote",
        ".cr-avatar",
    ):
        block = css_block(stylesheet, selector)
        assert "var(--cr-accent" not in block

    warning_block = css_block(
        stylesheet, ".bd-article .cr-compute-warning.admonition"
    )
    assert "background: var(--cr-status-phenomenon-bg);" in warning_block
    assert "border-left: 4px solid #d97706;" in warning_block

    syntax_keyword_block = css_block(
        stylesheet,
        'html[data-theme="light"] .bd-article div.highlight .k,\n'
        'html[data-theme="light"] .bd-article div.highlight .kc,\n'
        'html[data-theme="light"] .bd-article div.highlight .kd,\n'
        'html[data-theme="light"] .bd-article div.highlight .kn,\n'
        'html[data-theme="light"] .bd-article div.highlight .kp,\n'
        'html[data-theme="light"] .bd-article div.highlight .kr,\n'
        'html[data-theme="light"] .bd-article div.highlight .ne,\n'
        'html[data-theme="light"] .bd-article div.highlight .ow',
    )
    assert "var(--cr-brand" not in syntax_keyword_block


def test_landing_top_spacing_is_compact() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    landing_block = css_block_with(
        stylesheet,
        'body[data-page="introduction"] .bd-content,\n'
        'body[data-page="examples/index"] .bd-content',
        "padding: 48px 48px 110px;",
    )
    assert "padding: 48px 48px 110px;" in landing_block


def test_page_identity_badges_remain_legible() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    badge_block = css_block(stylesheet, ".cr-page-badge")
    assert "font-size: 12px;" in badge_block


def test_current_navigation_uses_only_the_purple_brand_palette() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    section_current = css_block(
        stylesheet,
        ".bd-docs-nav .active > .nav-link,\n.bd-docs-nav .nav-link.active",
    )
    assert "color: var(--cr-accent) !important;" in section_current
    assert "border-left: 3px solid var(--cr-accent);" in section_current

    section_current_theme = css_block(
        stylesheet, ".bd-docs-nav .bd-toc-item .toctree-l1.current > a"
    )
    assert "color: var(--cr-accent) !important;" in section_current_theme
    assert "border-left: 3px solid var(--cr-accent) !important;" in section_current_theme
    assert "box-shadow: none !important;" in section_current_theme

    toc_current = css_block(
        stylesheet,
        ".bd-sidebar-secondary .toc-entry.active > a,\n"
        ".bd-sidebar-secondary .toc-entry > a.active",
    )
    assert "background: transparent !important;" in toc_current
    assert "border-left-color: var(--cr-accent) !important;" in toc_current
    assert "var(--cr-accent-soft)" in toc_current

    previous_title = css_block(stylesheet, ".prev-next-footer .prev-next-title")
    assert "color: var(--cr-accent) !important;" in previous_title


def test_examples_cards_use_the_original_multicolumn_layout_without_authors() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    card_bodies = css_blocks(stylesheet, ".cr-example-card .sd-card-body")
    card_body = css_block_with(
        stylesheet, ".cr-example-card .sd-card-body", "min-height: 275px;"
    )

    assert "display: flex;" in card_body
    assert "min-height: 275px;" in card_body
    assert "padding: 24px;" in card_body
    assert "flex-direction: column;" in card_body
    assert "grid-template-columns:" not in card_body
    assert all("grid-template-areas:" not in block for block in card_bodies)

    section_heading = css_block(
        stylesheet, 'body[data-page="examples/index"] .bd-article h2'
    )
    assert "margin-top: 34px;" in section_heading


def test_header_search_keeps_its_boundary_without_adding_a_focus_halo() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    form = css_block_with(stylesheet, ".cr-search-form", "border-radius: 10px;")
    assert "background: var(--cr-surface);" in form
    assert "border: 1px solid var(--cr-line);" in form
    assert "border-radius: 10px;" in form
    assert "box-shadow: none;" in form

    active = css_block(
        stylesheet,
        ".cr-search-form:hover,\n.cr-header-search:focus-within .cr-search-form",
    )
    assert "background: var(--cr-surface);" in active
    assert "border-color: var(--cr-line);" in active
    assert "box-shadow: none;" in active

    assert ".cr-search-input:focus-visible {" in stylesheet
    assert ".cr-search-input:focus::placeholder {" in stylesheet
    assert "caret-color: var(--cr-ink);" in stylesheet
    assert ".cr-header-search:focus-within .search-button__kbd-shortcut {" in stylesheet


def test_native_search_visual_integration() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    assert ".cr-header-search" in stylesheet
    assert ".cr-search-panel" in stylesheet
    hidden_theme_dialog = css_block(stylesheet, "#pst-search-dialog")
    assert "display: none !important;" in hidden_theme_dialog


def test_docs_theme_has_one_responsive_and_accessibility_contract() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    assert "@media (max-width: 1279px)" in stylesheet
    assert "@media (max-width: 1199px)" not in stylesheet
    assert "@media (max-width: 1099px)" not in stylesheet
    assert "@media (max-width: 639px)" in stylesheet

    compact = css_block(stylesheet, "@media (max-width: 1279px)")
    for selector, source in (
        (".bd-header .primary-toggle", compact),
        (".bd-header .secondary-toggle", compact),
        (".cr-search-submit", stylesheet),
        (".cr-intro-actions a", stylesheet),
    ):
        selector_blocks = css_blocks(source, selector)
        assert any(
            ("min-width: 44px;" in block or "width: 44px;" in block)
            and ("min-height: 44px;" in block or "height: 44px;" in block)
            for block in selector_blocks
        )

    assert "#pst-primary-sidebar" in compact
    assert "#pst-secondary-sidebar" in compact
    assert "#pst-primary-sidebar-modal[open]" in compact
    assert "#pst-secondary-sidebar-modal[open]" in compact
    assert ".sidebar-primary__section" in compact
    assert ".bd-main .bd-sidebar-secondary {\n    display: none;" not in compact

    assert "font-size: clamp(36px, 3vw, 44px);" in stylesheet
    assert "font-size: 52px;" in stylesheet
    assert "font-size: 28px;" in stylesheet
    assert "max-width: 72ch;" in stylesheet
    assert ".bd-article h2 {\n  margin:" in stylesheet
    for h2_block in css_blocks(stylesheet, ".bd-article h2"):
        assert "border-bottom:" not in h2_block
    assert ".sd-card:has(.sd-stretched-link:focus-visible)" in stylesheet

    for token in (
        "--cr-status-stable-bg:",
        "--cr-status-phenomenon-bg:",
    ):
        assert token in stylesheet
    assert ".cr-page-identity" in stylesheet
    assert ".cr-page-footer" not in stylesheet
    assert ".cr-intro-actions a" in stylesheet
    assert ".cr-intro-actions a span" in stylesheet
    assert ".cr-intro-actions p:first-child > a" in stylesheet
    assert ".cr-intro-actions p:nth-child(2) > a" in stylesheet
    assert ".cr-github-link" in stylesheet
    github_link = css_block(stylesheet, ".cr-github-link")
    assert "width: 44px;" in github_link
    assert "height: 44px;" in github_link
    assert "button.copybtn:focus-visible" in stylesheet


def test_compact_and_narrow_layout_rules_win_the_theme_cascade() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    compact = css_block(stylesheet, "@media (max-width: 1279px)")
    narrow = css_block(stylesheet, "@media (max-width: 639px)")

    secondary_toggle = css_block(compact, ".bd-header .secondary-toggle")
    assert "display: grid !important;" in secondary_toggle

    compact_logo = css_block(narrow, ".cr-brand-logo--compact")
    assert "display: block;" in compact_logo
    assert "width: 44px;" in compact_logo
    assert "height: 44px;" in compact_logo

    narrow_header = css_block(narrow, ".bd-header .bd-header__inner")
    assert "grid-template-columns: minmax(52px, 1fr) auto auto;" in narrow_header

    primary_toggle = css_block(narrow, ".bd-header .primary-toggle")
    assert "grid-column: 3;" in primary_toggle
    assert "margin-right: 8px;" in primary_toggle

    secondary_toggle = css_block(narrow, ".bd-header .secondary-toggle")
    assert "display: none !important;" in secondary_toggle
    assert "display: none;" in css_block(narrow, ".cr-version-switch")

    assert "padding: 0 8px;" in css_block(
        narrow, ".bd-header .navbar-header-items__start"
    )
    assert "gap: 4px;" in css_block(narrow, ".bd-header .navbar-header-items__end")
    search_overlay = css_block(narrow, ".cr-header-search:focus-within")
    for declaration in (
        "position: fixed;",
        "top: 10px;",
        "right: 8px;",
        "left: 8px;",
        "width: auto;",
    ):
        assert declaration in search_overlay

    narrow_content = css_block(
        narrow,
        ".bd-main > .bd-content,\n"
        'body[data-page="introduction"] .bd-content,\n'
        'body[data-page="examples/index"] .bd-content',
    )
    assert "padding: 38px 20px 64px;" in narrow_content


def test_brand_uses_the_complete_app_logo_svg() -> None:
    template = Path("docs/_templates/brand.html").read_text()
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    app_logo = Path("docs/_static/app-logo.svg")
    dark_logo = Path("docs/_static/app-logo-dark.svg")

    assert app_logo.is_file()
    assert app_logo.read_text().startswith("<svg")
    assert 'class="cr-brand-logo cr-brand-logo--light"' in template
    assert 'class="cr-brand-logo cr-brand-logo--dark"' in template
    assert 'class="cr-brand-logo cr-brand-logo--compact"' in template
    assert "app-logo.svg" in template
    assert "app-logo-dark.svg" in template
    assert "app-icon.svg" in template
    assert "alt=\"\"" in template
    assert "M884 0H0V272H884V0Z" not in app_logo.read_text()
    assert dark_logo.is_file()
    assert dark_logo.read_text().startswith("<svg")
    light_svg = app_logo.read_text()
    dark_svg = dark_logo.read_text()
    assert '<image' not in dark_svg
    assert "filter" not in dark_svg
    assert "href=" not in dark_svg
    assert 'fill="black"' not in dark_svg
    assert dark_svg.count('fill="#F7F3FA"') == 1
    assert dark_svg.replace('fill="#F7F3FA"', 'fill="black"') == light_svg
    assert "cr-brand-icon" not in template
    assert "cr-brand-name" not in template
    assert "cr-brand-mark" not in template
    assert "cr-brand-node" not in template
    assert "cr-brand-connector" not in stylesheet
    assert "cr-brand-node" not in stylesheet
    logo_blocks = css_blocks(stylesheet, ".cr-brand-logo")
    assert any("width: 200px;" in block and "height: auto;" in block for block in logo_blocks)
    assert ".cr-brand-logo--dark" in stylesheet


def test_avatar_initials_are_centered_without_paragraph_margins() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    avatar = css_block(stylesheet, ".cr-avatar")
    initials = css_block(stylesheet, ".cr-avatar > p")

    assert "display: flex;" in avatar
    assert "align-items: center;" in avatar
    assert "justify-content: center;" in avatar
    assert "display: flex;" in initials
    assert "height: 100%;" in initials
    assert "margin: 0;" in initials
    assert "align-items: center;" in initials
    assert "line-height: 1;" in initials


def test_brand_logo_scales_at_compact_widths() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    compact = css_block(stylesheet, "@media (max-width: 1279px)")

    logo = css_block(compact, ".cr-brand-logo")
    assert "width: 180px;" in logo


def test_version_switch_uses_a_native_button_and_menu() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()

    button_blocks = css_blocks(stylesheet, ".cr-version-button")
    menu = css_block(stylesheet, ".cr-version-menu")
    assert any("min-height: 44px;" in block for block in button_blocks)
    assert "position: absolute;" in menu


def test_extend_examples_use_the_full_content_width() -> None:
    stylesheet = Path("docs/_static/comfyresearch.css").read_text()
    example_grid = css_block(stylesheet, ".cr-example-grid")
    example_card_body = css_block_with(
        stylesheet, ".cr-example-card .sd-card-body", "min-height: 275px;"
    )

    assert "width: 100%;" in example_grid
    assert "max-width: none;" in example_grid
    assert "max-width: 560px;" not in example_grid
    assert "min-height: 248px;" not in example_card_body
