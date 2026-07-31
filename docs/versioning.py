"""Documentation version labels and their published path segments."""

# Release branches change these values together. For v0.1.0, use:
# CURRENT_DOCS_VERSION = "v0.1.0"
# CURRENT_DOCS_PATH = "0.1.0"
CURRENT_DOCS_VERSION = "dev"
CURRENT_DOCS_PATH = "dev"

# List every version deployed under the same static-site root. Each item is a
# (label, path) pair so the UI can keep a release label such as v0.1.0 while
# publishing it at /en/0.1.0/ and /zh/0.1.0/.
PUBLISHED_DOCS_VERSIONS = ((CURRENT_DOCS_VERSION, CURRENT_DOCS_PATH),)

LOCALE_PREFIXES = {
    "en": "en",
    "zh_CN": "zh",
}

LOCALE_ROUTES = {
    locale: f"{prefix}/{CURRENT_DOCS_PATH}"
    for locale, prefix in LOCALE_PREFIXES.items()
}
