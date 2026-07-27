from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script_section(readme: str, script: str) -> str:
    heading = f"### `scripts/{script}`"
    start = readme.index(heading)
    following = readme.find("\n### ", start + len(heading))
    return readme[start:] if following < 0 else readme[start:following]


def test_readme_documents_every_script_argument() -> None:
    readme = (ROOT / "README.md").read_text()
    for script in (
        "cache_latents.py",
        "cache_features.py",
        "train.py",
        "sample.py",
        "eval.py",
        "compare.py",
    ):
        path = ROOT / "scripts" / script
        spec = importlib.util.spec_from_file_location(
            f"readme_cli_{script.removesuffix('.py')}", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.build_parser()
        section = _script_section(readme, script)
        for action in parser._actions:
            if action.dest == "help":
                continue
            name = (
                max(action.option_strings, key=len)
                if action.option_strings
                else action.dest
            )
            assert f"`{name}" in section, (
                f"README section for {script} does not document {name}"
            )
