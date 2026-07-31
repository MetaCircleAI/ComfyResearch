from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_api_is_registered() -> None:
    from comfy_research.main import create_app

    api_paths = create_app().openapi()["paths"]
    assert "/api/workspace" in api_paths
    assert "/api/graph" not in api_paths


@pytest.mark.filterwarnings(
    r"ignore:^Using `httpx` with `starlette\.testclient` is deprecated; install `httpx2` instead\.$:"
    r"starlette.exceptions.StarletteDeprecationWarning"
)
def test_graph_library_accepts_only_supported_kinds() -> None:
    from fastapi.testclient import TestClient

    from comfy_research.main import create_app

    client = TestClient(create_app())

    assert client.get("/api/graph-library/assets").status_code == 400
    assert client.get("/api/graph-library/workflows").status_code == 200
    assert client.get("/api/graph-library/templates").status_code == 200


def test_graph_report_pdf_cli_renders_an_existing_template(tmp_path) -> None:
    template = (
        ROOT
        / "data"
        / "graph_library"
        / "templates"
        / "569dffac-0c43-450f-b02b-8a4e732a8ff8.json"
    )
    template_source = template.read_text(encoding="utf-8")
    assert '"stepTicks"' in template_source
    assert any(
        f'"{field}"' in template_source
        for field in (
            "lossHistory",
            "plotPngBase64",
            "valueHistory",
            "observableMetricHistories",
        )
    )

    output = tmp_path / "sparse-attention-plot.pdf"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "graph_report_pdf.py"),
            "--input",
            str(template),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"%PDF-")


def test_graph_report_pdf_uses_generated_node_hints() -> None:
    from scripts import graph_report_pdf

    manifest = graph_report_pdf._parse_node_manifest_hints()

    assert manifest["absolute_pos_embed_layer"].startswith("Learned [max_seq, dim] bias")
    assert graph_report_pdf._node_library_info_text("absolute_pos_embed_layer") == manifest["absolute_pos_embed_layer"]
