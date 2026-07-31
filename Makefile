# Local CI helpers for paper-repro templates (and related backend checks).
# Usage:
#   make ci-repro          # schema + compile + manual rebuild + determinism for the two templates
#   make ci-repro-quick    # schema/params only (no prepare/train)
#   make ci-repro-cyclic   # cyclic schedule unit tests (η/S phase, data-epoch)

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help ci-repro ci-repro-quick ci-repro-cyclic docs docs-test docs-i18n-update

help:
	@echo "Targets:"
	@echo "  make ci-repro         Schema/compile + manual rebuild + determinism for the two repro templates"
	@echo "  make ci-repro-quick   Schema/params + on-disk slug checks only"
	@echo "  make ci-repro-cyclic  Cyclic schedule unit tests (eta/S phase, data-epoch)"
	@echo "  make docs             Build the Sphinx documentation"
	@echo "  make docs-test        Test the documentation build"
	@echo "  make docs-i18n-update Refresh the zh_CN gettext catalog"

ci-repro:
	$(PYTEST) -q \
		comfy_research/tests/test_repro_templates.py \
		comfy_research/tests/test_repro_template_manual_rebuild.py \
		comfy_research/tests/test_repro_template_determinism.py \
		comfy_research/tests/test_cyclic_schedules.py \
		comfy_research/tests/test_cyclic_trainer_compile.py \
		comfy_research/tests/test_parametric_path_sampler.py \
		--tb=short

ci-repro-quick:
	$(PYTEST) -q \
		comfy_research/tests/test_repro_templates.py::test_repro_template_saved_graph_entry \
		comfy_research/tests/test_repro_templates.py::test_repro_template_node_kinds_and_params \
		--tb=short

ci-repro-cyclic:
	$(PYTEST) -q \
		comfy_research/tests/test_cyclic_schedules.py \
		comfy_research/tests/test_cyclic_trainer_compile.py \
		--tb=short

docs:
	$(PYTHON) scripts/build_docs.py

docs-i18n-update:
	$(PYTHON) -m sphinx -b gettext -c docs docs/en docs/_build/gettext
	$(PYTHON) -m sphinx_intl update -p docs/_build/gettext -d docs/locales -l zh_CN --no-obsolete

docs-test:
	$(PYTEST) -q \
		tests/test_docs_source_contracts.py \
		tests/test_docs_build_contracts.py \
		tests/test_docs_visual_contracts.py \
		tests/test_docs_translation_source.py \
		tests/test_docs_i18n.py
