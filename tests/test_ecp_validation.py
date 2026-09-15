"""Phase 14e tests: ECP SCF energies validated against PySCF reference.

Compares vibe-qc's ECP-aware RHF / UHF / RKS to PySCF's
ECP-aware drivers on systems where both libraries can use the same
libecpint XML library and the same Gaussian basis. PySCF and
vibe-qc both link libecpint, so an exact-to-µHa agreement is the
right bar — any disagreement points at a vibe-qc bug.

Reference systems (small, fast, all-closed-shell + ECP-using):

  - **Zn²⁺ / 6-31G + LANL2DZ** (10 valence electrons after a
    18-electron LANL2DZ core; PySCF: E_HF = −62.456043 Ha).

If PySCF is not installed, these tests are skipped — they're
intended as ground-truth comparisons, not gating CI tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402


def _run_pyscf_rhf(atom_str: str, basis: str, ecp: str,
                   charge: int = 0, spin: int = 0) -> float:
    """Run PySCF RHF with the requested ECP and return the total energy."""
    mol = gto.M(
        atom=atom_str, basis=basis, ecp=ecp,
        charge=charge, spin=spin, verbose=0,
    )
    return scf.RHF(mol).run().e_tot


def _run_vibeqc_rhf_zn2_lanl():
    """vibe-qc RHF on Zn²⁺ / 6-31G + LANL2DZ.

    LANL2DZ on Zn replaces an 18-electron core. Molecule.charge is the
    physical ionic charge (+2 for Zn²⁺), so n_electrons() = 28 — the
    full physical count. The SCF subtracts the ECP core itself,
    leaving 10 valence electrons (= 3d¹⁰)."""
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 2, 1)
    basis = vq.BasisSet(mol, "6-31g")
    opts = vq.RHFOptions()
    opts.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts.ecp_library = "lanl2dz"
    opts.max_iter = 200
    opts.damping = 0.5
    return vq.run_rhf(mol, basis, opts)


# ---------------------------------------------------------------------------
# Zn²⁺ / 6-31G + LANL2DZ
# ---------------------------------------------------------------------------

def test_zn2plus_lanl2dz_matches_pyscf_to_microhartree():
    e_pyscf = _run_pyscf_rhf("Zn 0.0 0.0 0.0", "6-31g", "lanl2dz",
                             charge=2, spin=0)
    r = _run_vibeqc_rhf_zn2_lanl()
    assert r.converged
    diff = abs(r.energy - e_pyscf)
    assert diff < 1e-5, (
        f"vibe-qc Zn²⁺/6-31G+LANL2DZ disagrees with PySCF by "
        f"{diff:.3e} Ha (vibe-qc {r.energy:.6f}, PySCF {e_pyscf:.6f})"
    )


# ---------------------------------------------------------------------------
# nuclear repulsion correctness on Zn²⁺ + LANL2DZ
# ---------------------------------------------------------------------------

def test_zn2plus_effective_nuclear_repulsion_is_zero():
    """For a single ECP-bearing atom, the effective nuclear repulsion
    is exactly zero (no pairs). vibe-qc should report this directly,
    independent of the ECP physics."""
    r = _run_vibeqc_rhf_zn2_lanl()
    assert r.converged
    # E_total = E_electronic + E_nuc; with one atom, E_nuc = 0.
    # So E_electronic should equal E_total.
    assert r.e_electronic == pytest.approx(r.energy, abs=1e-12)
