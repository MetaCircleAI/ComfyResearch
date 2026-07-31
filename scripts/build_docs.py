import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.versioning import LOCALE_ROUTES

DEFAULT_OUTPUT = DOCS / "_build" / "html"
OUTPUT_MARKER = ".comfyresearch-docs-build"
MARKER_CONTENT = b"Comfy Research docs site\n"
LOCALES = tuple(LOCALE_ROUTES.items())


def check_existing_output(output: Path) -> None:
    if output.is_symlink():
        raise ValueError(f"refusing to replace symlinked output: {output}")
    if not output.exists():
        return

    marker = output / OUTPUT_MARKER
    if (
        not output.is_dir()
        or marker.is_symlink()
        or not marker.is_file()
        or marker.stat().st_size != len(MARKER_CONTENT)
        or marker.read_bytes() != MARKER_CONTENT
    ):
        raise ValueError(f"refusing to replace unowned output: {output}")


def replace_output(output: Path, staging: Path) -> None:
    check_existing_output(output)
    if not output.exists():
        staging.replace(output)
        return

    previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    output.replace(previous)
    try:
        staging.replace(output)
    except BaseException:
        previous.replace(output)
        raise
    try:
        shutil.rmtree(previous)
    except OSError:
        pass


def build(output: Path) -> None:
    output = Path(os.path.abspath(output))
    check_existing_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))

    try:
        for language, route in LOCALES:
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
                    str(staging / route),
                ],
                check=True,
                cwd=ROOT,
            )

        for _, route in LOCALES:
            introduction = staging / route / "introduction/index.html"
            if not introduction.is_file():
                raise FileNotFoundError(introduction)

        (staging / ".nojekyll").write_text("")
        default_route = LOCALE_ROUTES["en"]
        (staging / "index.html").write_text(
            f'<meta http-equiv="refresh" content="0; url={default_route}/introduction/">\n'
        )
        for _, route in LOCALES:
            (staging / route / "index.html").write_text(
                '<meta http-equiv="refresh" content="0; url=introduction/">\n'
            )
        (staging / OUTPUT_MARKER).write_bytes(MARKER_CONTENT)
        replace_output(output, staging)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
