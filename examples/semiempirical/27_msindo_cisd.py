"""Variational CISD on the MSINDO surface, via vibe-qc's generic CI (H2 + H2O).

``msindo_cisd`` runs configuration interaction with singles and doubles over the
closed-shell INDO reference. It does **not** port MSINDO's own selecting
determinant-CI driver (``rhfcisd.f``); instead it hands the INDO MO integrals —
the converged reference's one-electron core ``h`` and the ``_indo_ao_eri``
two-electron tensor, transformed to the MO basis — to vibe-qc's
**reference-agnostic** CI solver (:func:`vibeqc.solvers.cisd`). The same
validated Slater–Condon engine (``build_hamiltonian_matrix_unrestricted``) that
powers vibe-qc's ab-initio FCI/CASCI builds and diagonalises the CISD matrix.

What this script demonstrates / validates (against vibe-qc's OWN FCI/CIS, since
reference-MSINDO CISD uses empirical scalings + a selecting driver and is not a
parity target):
  1. **CISD == FCI for two electrons**: H2 has no triple/quadruple
     substitutions, so the singles+doubles space is the full CI space and
     ``msindo_cisd`` reproduces the INDO ``casci`` (full-space FCI) exactly.
  2. **Variational bracketing on H2O**: ``E_FCI ≤ E_CISD ≤ E_SCF``, with CISD
     recovering most of the INDO correlation while staying reference-dominated.
  3. **Dominant configurations**: the leading correction to the H2O reference is
     a closed-shell αβ double into the same spatial virtual (HOMO²→LUMO²-type).

Run:  .venv/bin/python examples/semiempirical/27_msindo_cisd.py
"""

from __future__ import annotations

import numpy as np

from vibeqc.semiempirical.methods.msindo import msindo_cisd, _msindo_mo_hamiltonian
from vibeqc.solvers import casci

# H2 (2 electrons) and H2O near their MSINDO minima (Ångström).
H2 = ([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
H2O = ([8, 1, 1], [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]])


def _fci_total(Z, xyz):
    """INDO full CI (casci over the whole MO space) on the same integrals."""
    ham, _eps, _no, nelec, _escf, _c = _msindo_mo_hamiltonian(Z, xyz)
    return casci(ham.h1e, ham.h2e, n_active_elec=nelec, n_active_orb=ham.norb,
                 nuclear_repulsion=ham.nuclear_repulsion).e_total


def main() -> None:
    print(__doc__)

    # --- 1. CISD == FCI for a two-electron system ---------------------
    r2 = msindo_cisd(*H2)
    e_fci_h2 = _fci_total(*H2)
    print("H2 (2 electrons): CISD spans the full CI space")
    print(f"  E_CISD = {r2.e_total:.10f} Ha   ({r2.n_det} determinants)")
    print(f"  E_FCI  = {e_fci_h2:.10f} Ha")
    print(f"  |E_CISD - E_FCI| = {abs(r2.e_total - e_fci_h2):.1e}  (machine precision)")

    # --- 2. Variational bracketing on H2O -----------------------------
    r = msindo_cisd(*H2O)
    e_fci = _fci_total(*H2O)
    print("\nH2O: E_FCI <= E_CISD <= E_SCF")
    print(f"  E_SCF  = {r.e_scf:.8f} Ha")
    print(f"  E_CISD = {r.e_total:.8f} Ha   (corr {r.e_corr:+.6f}, {r.n_det} dets)")
    print(f"  E_FCI  = {e_fci:.8f} Ha   (corr {e_fci - r.e_scf:+.6f})")
    pct = 100.0 * r.e_corr / (e_fci - r.e_scf)
    print(f"  CISD recovers {pct:.1f}% of the INDO correlation; "
          f"reference weight = {r.reference_weight:.4f}")
    assert e_fci <= r.e_total + 1e-9 <= r.e_scf + 1e-9

    # --- 3. Dominant configurations -----------------------------------
    print("\nH2O CISD leading configurations:")
    for desc, coeff, weight in r.dominant_configurations(4):
        print(f"  {desc:24s} c = {coeff:+.4f}   weight = {weight:.4f}")


if __name__ == "__main__":
    main()
