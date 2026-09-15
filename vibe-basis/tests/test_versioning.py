"""vibe-basis carries its own version line — keep its two copies honest.

``VERSIONING.md`` makes ``pyproject.toml``'s ``[project] version`` and
``src/vibe_basis/__init__.py``'s ``__version__`` a single number stored
in two places. ``vb --version`` reads the second one (``cli.py``'s
``click.version_option``), so a drift between them does not fail loudly
— it just makes the CLI report a version nobody shipped.

These tests are metadata-only: no install, no network, no subprocess.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import vibe_basis

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"

# PEP 440 release segment, optionally with a .devN / rcN suffix. We do
# not accept an arbitrary PEP 440 string: VERSIONING.md commits to
# MAJOR.MINOR.PATCH SemVer, and a local or epoch segment would silently
# break the >= pin vibe-qc's [basisopt] extra resolves against.
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?$")


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_pyproject_and_dunder_version_agree():
    declared = _pyproject()["project"]["version"]
    assert declared == vibe_basis.__version__, (
        f"version drift: pyproject.toml says {declared!r} but "
        f"vibe_basis.__version__ is {vibe_basis.__version__!r}. "
        "Bump both in the same commit — see VERSIONING.md."
    )


def test_version_is_semver():
    assert _SEMVER.match(vibe_basis.__version__), (
        f"{vibe_basis.__version__!r} is not MAJOR.MINOR.PATCH. "
        "VERSIONING.md commits vibe-basis to SemVer."
    )


def test_cli_version_option_reports_the_package_version():
    """``vb --version`` must print the number the package carries."""
    from click.testing import CliRunner

    from vibe_basis.cli import main

    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0, result.output
    assert vibe_basis.__version__ in result.output


def _is_editable_install(dist) -> bool:
    """True when ``dist`` was installed with ``pip install -e``.

    PEP 610 records this in ``direct_url.json`` as
    ``{"dir_info": {"editable": true}}``.
    """
    import json

    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return False
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable"))
    except (ValueError, AttributeError):
        return False


def test_distribution_metadata_matches_when_installed():
    """A NON-editable install's metadata must agree with the module.

    Editable installs are exempt, and deliberately so. ``pip install -e``
    records the version once, at install time, and never revisits it --
    but ``vibe_basis.__version__`` is read live from the source tree. So
    every version bump makes an existing editable install disagree until
    someone reinstalls, which is a routine and harmless dev condition,
    not a defect in the tree.

    This test originally asserted unconditionally and duly went red on a
    correct tree once the version moved 0.9.0 -> 0.10.0 under an editable
    install (2026-08-12). A gate that fails on a normal working state
    trains people to ignore it, so the editable case now skips with the
    remedy in the message. For wheel installs the assertion stands: there
    a mismatch is genuinely wrong.

    Also skipped where vibe-basis is not installed at all -- a bare
    ``pytest`` inside vibe-basis/ runs off ``conftest.py``'s ``sys.path``
    entry, with no distribution to disagree with.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    import pytest

    try:
        dist = distribution("vibe-basis")
    except PackageNotFoundError:
        pytest.skip("vibe-basis is not installed in this environment")

    installed = dist.version

    if installed != vibe_basis.__version__ and _is_editable_install(dist):
        pytest.skip(
            f"editable install records {installed!r} while the source tree "
            f"says {vibe_basis.__version__!r}; editable metadata is a "
            "snapshot taken at install time. Re-run "
            "`pip install -e vibe-basis/` to refresh it."
        )

    assert installed == vibe_basis.__version__, (
        f"installed distribution reports {installed!r} but the imported "
        f"module says {vibe_basis.__version__!r} — for a non-editable "
        "install that means the wheel was built from a different tree, "
        "or a bump never reached pyproject.toml."
    )


def test_versioning_doc_exists_and_records_this_version():
    doc = PACKAGE_ROOT / "VERSIONING.md"
    assert doc.exists(), "VERSIONING.md is the scheme's source of truth"
    assert vibe_basis.__version__ in doc.read_text(encoding="utf-8"), (
        f"VERSIONING.md has no History row for {vibe_basis.__version__} — "
        "add one in the same commit as the bump."
    )


def test_python_floor_matches_vibe_qc():
    """vibe-basis is reachable via vibe-qc's ``[basisopt]`` extra.

    A stricter floor here than vibe-qc's own would make
    ``uv pip install -e '.[basisopt]'`` unresolvable on the older
    interpreters vibe-qc still supports, and it would surface as an
    opaque resolver error rather than as "wrong Python".
    """
    requires = _pyproject()["project"]["requires-python"]
    assert requires == ">=3.11", (
        f"vibe-basis requires-python is {requires!r}; it must match "
        "vibe-qc's >=3.11 floor while the [basisopt] extra exists."
    )
    # tomllib — the newest stdlib module this package uses — is 3.11+,
    # so the floor is honest as well as compatible.
    assert sys.version_info >= (3, 11)
