import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path


SOURCE_ROOT = Path("docs/en")
EXPECTED_REPRODUCTION_AUTHORS = {
    "Neural_Mechanics": "赵浩然 (Zhao Haoran)",
    "diffusion-reproducibility": "王星运 (Wang Xingyun)",
    "edge-of-stability-cpu": "王金鑫 (Wang Jinxin)",
    "edge-of-stability-eos": "屈清宇 (Qu Qingyu)",
    "in-context-associative-recall": "屈清宇 (Qu Qingyu)",
    "information-bottleneck-figure3": "熊程宇 (Xiong Chengyu)",
    "jastrzebski-fig1-cyclic-cbs-vs-clr": "王金鑫 (Wang Jinxin)",
    "keskar-fig2-3-sb-lb": "王金鑫 (Wang Jinxin)",
    "lazy-vs-rich-regime": "屈清宇 (Qu Qingyu)",
    "linear-mode-connectivity": "王星运 (Wang Xingyun)",
    "random-label-memorization-figure1a": "熊程宇 (Xiong Chengyu)",
    "rank-collapse-figure5": "郭绍阳 (Guo Shaoyang)",
    "rank-collapse-tinyshakespeare": "郭绍阳 (Guo Shaoyang)",
    "saad_solla_plateau_reproduction": "赵浩然 (Zhao Haoran)",
    "spectral-bias-figure1a": "熊程宇 (Xiong Chengyu)",
    "staggered-singular-value-dynamics": "屈清宇 (Qu Qingyu)",
}
READER_HTML = re.compile(
    r"<(?:details|summary|div|p|span|strong|code)(?:\s|>)",
    flags=re.IGNORECASE,
)


class ClassTextParser(HTMLParser):
    def __init__(self, target_class: str) -> None:
        super().__init__()
        self.target_class = target_class
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "").split()
        if self.depth:
            self.depth += 1
        elif self.target_class in classes:
            self.depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self.depth:
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def class_text(html: str, class_name: str) -> str:
    parser = ClassTextParser(class_name)
    parser.feed(html)
    return parser.text


def test_reader_facing_copy_uses_translatable_myst_nodes() -> None:
    offenders = {
        str(page): sorted(set(READER_HTML.findall(page.read_text())))
        for page in SOURCE_ROOT.rglob("*.md")
        if READER_HTML.search(page.read_text())
    }

    assert offenders == {}


def test_product_screenshots_do_not_leave_capture_placeholders() -> None:
    sources = "\n".join(page.read_text() for page in SOURCE_ROOT.rglob("*.md"))

    assert "Product screenshot pending" not in sources
    assert "Capture specification" not in sources
    assert sources.count(":class: cr-product-screenshot") == 9


def test_translatable_components_preserve_their_rendered_structure(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "dirhtml",
            "-c",
            "docs",
            "docs/en",
            str(site),
        ],
        check=True,
    )

    first_graph_page = (site / "get-started/first-graph/index.html").read_text()
    assert first_graph_page.count("cr-product-screenshot") == 3
    for image_name in (
        "first-graph-template.png",
        "first-graph-trainer.png",
        "first-graph-results.png",
    ):
        assert image_name in first_graph_page
    assert "cr-screenshot-placeholder" not in first_graph_page

    reproduction_routes = {
        page.stem
        for page in (SOURCE_ROOT / "examples" / "reproductions").glob("*.md")
    }
    assert reproduction_routes == EXPECTED_REPRODUCTION_AUTHORS.keys()

    for route, author in EXPECTED_REPRODUCTION_AUTHORS.items():
        html = (
            site / "examples/reproductions" / route / "index.html"
        ).read_text()
        meta = class_text(html, "cr-article-meta")
        assert f"Author {author}" in meta
        assert "Scope Phenomenon reproduction" in meta
