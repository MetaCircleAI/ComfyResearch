from html import unescape
from pathlib import Path
from re import compile, escape, sub


def rendered_text(page: Path) -> str:
    text = sub(r"<[^>]+>", "", page.read_text())
    return " ".join(unescape(text).split())


def front_matter(page: Path) -> dict[str, str]:
    source = page.read_text()
    assert source.startswith("---\n"), f"{page} is missing YAML front matter"
    metadata_block = source.split("---\n", 2)[1]
    return {
        key.strip(): value.strip()
        for line in metadata_block.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def css_blocks(stylesheet: str, header: str) -> list[str]:
    pattern = compile(r"\s+".join(escape(part) for part in header.split()) + r"\s*\{")
    blocks = []
    for match in pattern.finditer(stylesheet):
        depth = 1
        cursor = match.end()
        while cursor < len(stylesheet) and depth:
            if stylesheet[cursor] == "{":
                depth += 1
            elif stylesheet[cursor] == "}":
                depth -= 1
            cursor += 1
        assert depth == 0, f"unclosed CSS block: {header}"
        blocks.append(stylesheet[match.end() : cursor - 1])
    assert blocks, f"missing CSS block: {header}"
    return blocks


def css_block(stylesheet: str, header: str) -> str:
    blocks = css_blocks(stylesheet, header)
    assert len(blocks) == 1, f"ambiguous CSS block: {header}"
    return blocks[0]


def css_block_with(stylesheet: str, header: str, declaration: str) -> str:
    blocks = [block for block in css_blocks(stylesheet, header) if declaration in block]
    assert len(blocks) == 1, f"missing or ambiguous declaration in CSS block: {header}"
    return blocks[0]
