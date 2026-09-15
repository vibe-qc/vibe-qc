"""Tighter reproducer for the v0.7.3-class DF SCF SIGSEGV.

The existing handoff reproducer ``examples/debug/df_smoke.py`` runs
the entire DF SCF on H₂O / def2-svp / def2-svp-jk to surface the
segfault. This one bypasses every layer above libint and crashes
inside ``compute_2c_eri`` itself — three lines, no SCF.

Discovered while running an ADFT spike (see POSTMORTEM.md in this
directory): every aux basis with a p-shell or higher angular momentum
segfaults during ``compute_2c_eri`` evaluation. s-only auxes (sto-3g,
sto-6g, 6-31g) work fine.

NOT a libint build-config issue. The vendored libint at
``third_party/libint/install/include/libint2/config.h`` is built with::

    LIBINT_MAX_AM           = 5
    LIBINT_ERI_MAX_AM_LIST  = 5,4,3   (deriv 0/1/2)
    LIBINT_ERI2_MAX_AM_LIST = 5,4,3
    LIBINT_ERI3_MAX_AM_LIST = 5,4,3

— way more than enough for cc-pvdz (max_l=1 on H) or def2-svp-jk
(max_l=2). The bug is in ``cpp/src/df.cpp``'s ``compute_2c_eri`` or
its libint2::Engine usage, not in libint itself.

Run::

    .venv/bin/python examples/experimental/coulomb-method-zoo/bug_compute_2c_eri_pshell.py

Or with a debugger to catch the crash::

    gdb --args .venv/bin/python examples/experimental/coulomb-method-zoo/bug_compute_2c_eri_pshell.py

Each block exits 139 (SIGSEGV) for any ``aux_name`` whose H shells
include l ≥ 1.
"""
from __future__ import annotations

import sys

import vibeqc as vq

print(f"vibe-qc {vq.__version__}  (path={vq.__file__})")
print(f"python  {sys.version.split()[0]}")
print()

mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])], 0, 1)


def probe(aux_name: str) -> None:
    """One probe; segfault during compute_2c_eri kills the whole process."""
    print("=" * 60, flush=True)
    print(f"  aux = {aux_name!r}", flush=True)
    aux = vq.BasisSet(mol, aux_name)
    max_l = max(sh.l for sh in aux.shells())
    print(f"  loaded:  n_aux={aux.nbasis}  max_l={max_l}  "
          f"nshells={aux.nshells}", flush=True)
    print(f"  → calling vq.compute_2c_eri(aux) ...", flush=True)
    A = vq.compute_2c_eri(aux)
    print(f"  ✓ OK, A shape = {A.shape}", flush=True)
    print()


# These work (s-only):
probe("sto-3g")
probe("sto-6g")
probe("6-31g")

# This crashes (max_l=1):
probe("cc-pvdz")

# (Anything below this point doesn't run; SIGSEGV terminated the process.)
probe("def2-svp")           # max_l=1
probe("def2-svp-jk")        # max_l=2
probe("def2-tzvp-jk")       # max_l=2

print("All probes finished without segfault — bug may be fixed.")
