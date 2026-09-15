"""DF SCF SIGSEGV smoke / reproducer (v0.7.3 finding).

The v0.7.2 density-fitting flagship (Boys' Crucible) added
``density_fit=True`` + ``aux_basis=...`` to RHFOptions / UHFOptions /
RKSOptions / UKSOptions / MP2Options / UMP2Options. The v0.7.3 runner
wiring (Whitten's Bridge) plumbed it into the regression suite. Doing
so revealed that ``vq.run_rhf`` segfaults inside the DF SCF on
``def2-svp`` for at least two molecules (H2O and benzene), with at
least two aux bases (``def2-svp-jk`` auto-detected and
``def2-universal-jkfit`` explicit). The v0.7.2 subprocess-isolation
layer captures the crash as a clean ``error`` row, so the regression
suite doesn't brick — but the underlying DF SCF bug remains.

This script is the **minimal hand-off reproducer** for whoever
debugs the DF kernel next. Each block runs one combination from
scratch in the current process, so a SIGSEGV here aborts the whole
script — that's the *intended* behaviour: the very next line of
output past ``calling run_rhf...`` is the segfault, with whatever
state the caller can capture (gdb, address sanitiser, valgrind,
core dump). The regression-suite layer would mask all that.

Run::

    .venv/bin/python examples/debug/df_smoke.py

Or with a debugger to catch the crash::

    .venv/bin/python -X dev examples/debug/df_smoke.py
    gdb --args .venv/bin/python examples/debug/df_smoke.py   # then `run`, `bt full`

Outputs the active vibe-qc / aux-basis info before each call so the
crash output is self-documenting.

For the non-DF baseline (which works), see
``examples/regression/output/<run_id>/verbose/h2o__sto-3g__rhf__mol.log``
in any successful regression-suite run.
"""
from __future__ import annotations

import os
import sys

import vibeqc as vq

print(f"vibe-qc {vq.__version__}  (path={vq.__file__})")
print(f"python  {sys.version.split()[0]}  ({sys.executable})")
print()


# Szabo & Ostlund H2O geometry (in Å, converted on the fly).
ANG_TO_BOHR = 1.0 / 0.529177210903
H2O_ATOMS = (
    (8, ( 0.0,        0.0,       0.0)),
    (1, ( 0.7569503,  0.5858823, 0.0)),
    (1, (-0.7569503,  0.5858823, 0.0)),
)


def _build_h2o() -> "vq.Molecule":
    return vq.Molecule([
        vq.Atom(z, [c * ANG_TO_BOHR for c in xyz])
        for z, xyz in H2O_ATOMS
    ])


def run_rhf_df(label: str, basis_name: str, aux_basis: str) -> None:
    """One DF SCF attempt; the segfault aborts the whole process here."""
    print("=" * 78)
    print(f"[{label}] basis={basis_name!r}  aux={aux_basis!r}")
    print("=" * 78)

    mol = _build_h2o()
    print(f"  mol: {len(mol.atoms)} atoms, {int(mol.n_electrons())} electrons",
          flush=True)

    basis = vq.make_basis(mol, basis_name)
    print(f"  orbital basis: {basis_name} → {basis.nbasis} bf / {basis.nshells} shells",
          flush=True)

    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-9
    opts.max_iter = 50
    opts.density_fit = True
    opts.aux_basis = aux_basis

    print(f"  opts.density_fit = True", flush=True)
    print(f"  opts.aux_basis   = {opts.aux_basis!r}", flush=True)
    print(f"  → calling vq.run_rhf(mol, basis, opts) ...", flush=True)

    result = vq.run_rhf(mol, basis, opts)

    print(f"  ✓ converged={bool(result.converged)}  "
          f"E={float(result.energy):.10f} Ha  "
          f"n_iter={int(result.n_iter)}",
          flush=True)
    print()


def main() -> None:
    # Combination 1: auto-detected JK aux for def2-svp.
    auto_jk = vq.default_aux_basis_for("def2-svp", kind="jk")
    run_rhf_df("def2-svp + auto-jk", "def2-svp", auto_jk)

    # Combination 2: explicit def2-universal-jkfit (universal aux).
    run_rhf_df("def2-svp + def2-universal-jkfit", "def2-svp",
               "def2-universal-jkfit")

    # Sanity: same molecule, sto-3g + auto-detected JK aux.
    # (Smaller orbital basis, smaller aux — useful as a "does ANY DF
    # combo converge?" probe.)
    try:
        sto_jk = vq.default_aux_basis_for("sto-3g", kind="jk")
        run_rhf_df("sto-3g + auto-jk", "sto-3g", sto_jk)
    except Exception as exc:
        print(f"[sto-3g + auto-jk] aux autodetect failed: {exc}")
        print()

    print("All DF probes finished without segfault — bug may be fixed.")


if __name__ == "__main__":
    main()
