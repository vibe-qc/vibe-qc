#!/usr/bin/env python3
"""Regenerate ``qvf_manifest_v2.schema.json`` from the v1 schema.

The v2 manifest schema is v1 plus one addition: ``SectionReactionPath``
gains an optional ``lattice`` binary member (+ ``dim`` in its metadata)
so periodic reaction paths carry the cell needed to render them. That
delta lives in exactly one place --
``vibeqc.output.formats.qvf._derive_v2_schema`` -- and this script
renders it to disk.

The runtime validator does *not* read the generated file: it derives v2
in memory. The file exists so external validators (and anyone resolving
the ``$id``) can fetch a standalone v2 schema. Keeping it a generated
artefact is what stops it re-forking: v2 was previously a hand-copied
snapshot of v1 and silently fell behind ten section kinds, four root
properties, the Section ``critical`` flag, and the fat-band
``projections`` member.

Usage::

    .venv/bin/python scripts/gen_qvf_v2_schema.py           # rewrite the file
    .venv/bin/python scripts/gen_qvf_v2_schema.py --check    # verify, don't write

``--check`` exits 1 if the on-disk file is stale, so it is safe to wire
into CI or a pre-commit hook. ``tests/test_qvf_v2_schema_drift.py``
asserts the same invariant.

Run it from the checkout that ``pip install -e`` was run against. From a
git worktree the editable install's meta-path finder resolves ``vibeqc``
to the *other* checkout, and the script refuses to run rather than read
one tree's v1 and overwrite another tree's v2 -- see ``_assert_same_tree``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from vibeqc.output.formats import qvf as _qvf  # noqa: E402

_SCHEMA_PATH_V1 = _qvf._SCHEMA_PATH_V1
_SCHEMA_PATH_V2 = _qvf._SCHEMA_PATH_V2
_derive_v2_schema = _qvf._derive_v2_schema


def _assert_same_tree() -> None:
    """Refuse to run against a `vibeqc` imported from a different checkout.

    The scikit-build editable install registers a `ScikitBuildRedirectingFinder`
    on `sys.meta_path`, which resolves `vibeqc` to whichever checkout was
    `pip install -e`'d -- ahead of the `sys.path` entry above, and immune to
    `PYTHONPATH`. Run from a git worktree, this script would therefore read the
    *main checkout's* v1 schema and, without `--check`, write the regenerated v2
    back into the *main checkout* too: a silent cross-tree write of a file
    derived from the wrong source. Fail loudly instead.
    """
    imported = Path(_qvf.__file__).resolve()
    if _REPO_ROOT in imported.parents:
        return
    raise SystemExit(
        f"refusing to run: `vibeqc` resolved to {imported}\n"
        f"which is outside this checkout ({_REPO_ROOT}).\n\n"
        "The editable install's meta-path finder pins `vibeqc` to the checkout "
        "it was installed from, so this script would read that tree's v1 schema "
        "and write that tree's v2 file. Either run it from the installed "
        "checkout, or drop the finder before importing:\n\n"
        '    python -c "import sys; sys.meta_path[:] = [f for f in sys.meta_path '
        "if type(f).__name__ != 'ScikitBuildRedirectingFinder']; "
        'exec(open(\'scripts/gen_qvf_v2_schema.py\').read())"\n'
    )


def render() -> str:
    """Return the v2 schema JSON text derived from the v1 file on disk."""
    with open(_SCHEMA_PATH_V1, encoding="utf-8") as f:
        v1 = json.load(f)
    return json.dumps(_derive_v2_schema(v1), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the on-disk v2 schema is stale; write nothing",
    )
    args = ap.parse_args()
    _assert_same_tree()

    expected = render()
    current = (
        _SCHEMA_PATH_V2.read_text(encoding="utf-8")
        if _SCHEMA_PATH_V2.is_file()
        else None
    )

    if current == expected:
        print(f"up to date: {_SCHEMA_PATH_V2.relative_to(_REPO_ROOT)}")
        return 0

    if args.check:
        print(
            f"STALE: {_SCHEMA_PATH_V2.relative_to(_REPO_ROOT)} does not match the "
            "v1 + ReactionPathLattice derivation.\n"
            "Regenerate with: python scripts/gen_qvf_v2_schema.py",
            file=sys.stderr,
        )
        return 1

    _SCHEMA_PATH_V2.write_text(expected, encoding="utf-8")
    print(f"wrote: {_SCHEMA_PATH_V2.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
