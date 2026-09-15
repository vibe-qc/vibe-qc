"""Guard: the C++ core must not write user-facing output.

vibe-qc formats every user-facing byte in Python. The C++ core computes
numbers, fills result structs, and hands them back across pybind11; the
``vibeqc.output`` document layer decides how they render. That is what
lets precision, units, and table layout be configured in one place
(``vibeqc.output.document.FormatPolicy``) instead of being frozen into a
``printf`` somewhere in a Fock build.

Until this test existed the rule held only by accident: 231 C++ sources
happened to contain exactly one output call. This pins it, so a stray
``std::cout`` in a new kernel fails CI instead of quietly opening a
second, unformattable output channel that no policy can reach.

If you need a value visible from C++, put it on the result struct and
render it in Python. If you need a temporary debug print, gate it behind
a verbosity flag, send it to stderr (never stdout, which would corrupt
piped output), and add it to ``_ALLOWED`` below with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_CPP_ROOT = Path(__file__).resolve().parents[1] / "cpp"

# Output calls that would reach a user's terminal. ``snprintf`` and
# ``sprintf`` are excluded by the word boundary: they build strings and
# write to no stream.
_EMITTERS = re.compile(
    r"""
    \bstd::cout\b
  | \bstd::cerr\b
  | \bstd::clog\b
  | \bstd::puts\b
  | \bputchar\b
  | (?<![a-zA-Z0-9_])printf\s*\(
  | (?<![a-zA-Z0-9_])fprintf\s*\(
    """,
    re.VERBOSE,
)

# path -> why this one call is allowed to stay.
_ALLOWED: dict[str, str] = {
    "src/lobpcg.cpp": (
        "One verbosity-gated fprintf(stderr, ...) reporting inner "
        "eigensolver residuals. Diagnostic only, never on stdout, emits "
        "no energy or other user-facing quantity."
    ),
}


def _sources() -> list[Path]:
    files = [
        p
        for suffix in ("*.cpp", "*.hpp", "*.h", "*.cc", "*.cxx")
        for p in _CPP_ROOT.rglob(suffix)
    ]
    assert files, f"no C++ sources found under {_CPP_ROOT}"
    return sorted(files)


def _offenders() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in _sources():
        rel = path.relative_to(_CPP_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            code = line.split("//", 1)[0]
            if _EMITTERS.search(code):
                found.append((rel, lineno, line.strip()))
    return found


def test_cpp_core_writes_no_user_facing_output():
    offenders = _offenders()
    if offenders:
        listing = "\n".join(f"  {f}:{n}  {src}" for f, n, src in offenders)
        pytest.fail(
            "C++ sources must not emit output directly; return the value on "
            "the result struct and render it via vibeqc.output instead.\n"
            f"{listing}"
        )


def test_allowlisted_files_still_exist():
    # An allowlist entry for a deleted file is dead weight that would
    # silently re-permit output if the path were ever recreated.
    for rel in _ALLOWED:
        assert (_CPP_ROOT / rel).is_file(), f"stale _ALLOWED entry: {rel}"


def test_allowlisted_file_still_only_writes_to_stderr():
    # The lobpcg exemption is narrow: stderr, not stdout. If it grows a
    # std::cout, the exemption no longer describes what the file does.
    text = (_CPP_ROOT / "src/lobpcg.cpp").read_text(encoding="utf-8")
    assert "std::cout" not in text
    assert "stdout" not in text
