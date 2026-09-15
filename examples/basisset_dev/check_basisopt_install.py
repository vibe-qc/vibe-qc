"""Preflight: is the external-program basis-optimization path usable here?

Run this before a basis-set optimization campaign. It answers one
question with a clear yes/no: can this environment reach ``vibe_basis``
and the ``vibeqc.basis_optimization`` modules that drive external SCF
programs?

    .venv/bin/python examples/basisset_dev/check_basisopt_install.py

Why this exists
---------------
``vibe-basis`` is co-located at ``vibe-basis/`` in the checkout but is
NOT installed by a plain ``pip install -e .``. Six modules that ship
inside the vibe-qc wheel import it at module top level, so on an
environment without it they raise ``ModuleNotFoundError`` at import —
long after you have set up a campaign, and with a message that does not
say "install the extra". On 2026-07-26 that was true of every vibe-qc
environment on the maintainer's machine.

The fix is vibe-qc's ``[basisopt]`` extra::

    uv pip install -e '.[basisopt]'          # uv, resolves vibe-basis/
    pip install -e . && pip install -e vibe-basis/   # pip, two steps

See ``docs/installation.md`` § "Optional basis-set optimization driver"
and ``handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md``.

Exit status
-----------
``0`` if the path is fully usable, ``1`` otherwise — so this is usable
as a campaign-script guard::

    .venv/bin/python examples/basisset_dev/check_basisopt_install.py || exit 1
"""

from __future__ import annotations

import importlib
import sys

# The modules under vibeqc that import vibe_basis at module top level.
# Anything driving an external SCF program goes through one of these.
EXTERNAL_PATH_MODULES = (
    "vibeqc.basis_optimization.calculators",
    "vibeqc.basis_optimization.recipes.crystal_objective",
    "vibeqc.basis_optimization.recipes.crystal_stage1",
    "vibeqc.basis_optimization.recipes.crystal_stage2",
    "vibeqc.basis_optimization.recipes.crystal_stage3",
    "vibeqc.basis_optimization.recipes.production",
)

# The in-process molecular path needs none of the above. Reported
# separately so a "no" answer still tells you what you *can* run.
IN_PROCESS_MODULES = (
    "vibeqc.basis_optimization.parametrise",
    "vibeqc.basis_optimization.bdiis",
    "vibeqc.basis_optimization.recipes.molecular",
)

INSTALL_HINT = (
    "    uv pip install -e '.[basisopt]'\n"
    "  or, under pip (which ignores [tool.uv.sources]):\n"
    "    pip install -e . && pip install -e vibe-basis/"
)


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"


def _try_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def main() -> int:
    print("vibe-qc basis-optimization preflight")
    print("=" * 52)
    print(f"python      : {sys.version.split()[0]}")
    print(f"vibe-qc     : {_version('vibe-qc')}")
    print(f"vibe-basis  : {_version('vibe-basis')}")
    print()

    vibe_basis_ok, vibe_basis_err = _try_import("vibe_basis")
    if vibe_basis_ok:
        import vibe_basis

        print(f"vibe_basis importable from {vibe_basis.__file__}")
    else:
        print(f"vibe_basis NOT importable  ({vibe_basis_err})")
    print()

    print("In-process molecular path (needs no vibe-basis):")
    in_process_ok = True
    for name in IN_PROCESS_MODULES:
        ok, err = _try_import(name)
        in_process_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -- {err}"))
    print()

    print("External-program path (needs vibe-basis):")
    external_ok = True
    for name in EXTERNAL_PATH_MODULES:
        ok, err = _try_import(name)
        external_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -- {err}"))
    print()

    if external_ok:
        print("READY: both paths are usable in this environment.")
        return 0

    print("NOT READY: the external-program path is unavailable here.")
    if not vibe_basis_ok:
        print("Cause: vibe-basis is not installed. Install it with:")
        print(INSTALL_HINT)
    else:
        print(
            "vibe_basis imports, so this is not a missing-extra problem —\n"
            "read the per-module error above."
        )
    if in_process_ok:
        print(
            "\nThe in-process molecular path still works: see\n"
            "docs/user_guide/basis_optimization.md for "
            "optimize_molecular_basis."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
