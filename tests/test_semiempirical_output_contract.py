from __future__ import annotations

import ast
from pathlib import Path


_SEMIEMPIRICAL_ROOT = (
    Path(__file__).resolve().parents[1] / "python" / "vibeqc" / "semiempirical"
)


def _print_call_sites() -> list[tuple[str, int]]:
    sites: list[tuple[str, int]] = []
    for path in sorted(_SEMIEMPIRICAL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_SEMIEMPIRICAL_ROOT).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                sites.append((rel, node.lineno))
    return sites


def test_semiempirical_modules_do_not_print_user_output():
    offenders = _print_call_sites()
    assert offenders == [], (
        "semiempirical modules must emit user-facing text through "
        "vibeqc.output.write(), not print(): "
        f"{offenders}"
    )
