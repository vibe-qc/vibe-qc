"""Open-shell UHF-CCM — unrestricted HF on a Cyclic Cluster Model reference.

UHF-CCM solves the CCM-weighted integrals with two spin densities, reaching
odd-electron / spin-polarised clusters. These tests pin (i) the molecular limit
— an isolated open-shell cluster must reproduce vibe-qc's own ``run_uhf``;
(ii) closed-shell consistency — UHF-CCM must collapse to ``run_ccm_rhf`` when
there is no spin polarisation; and (iii) an open-shell *periodic* cluster
(odd-electron H chain doublet) converges with a clean spin density.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550 (the CCM); UHF as in Szabo & Ostlund, Modern Quantum
Chemistry, §3.8.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, UHFOptions, run_uhf
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.uhf import run_ccm_uhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _cell(lattice, atoms, mult=None):
    ne = sum(a.Z for a in atoms)
    m = mult if mult is not None else (1 if ne % 2 == 0 else 2)
    return PeriodicSystem(3, np.asarray(lattice, float), atoms, charge=0, multiplicity=m)


def test_ccm_uhf_isolated_equals_molecular():
    """Isolated Li atom (3e doublet): UHF-CCM total energy == molecular run_uhf.

    In the molecular limit the WSSC-weighted ERI is the molecular ERI, so the
    open-shell CCM energy must reproduce vibe-qc's ``run_uhf`` (kernel + spin
    bookkeeping correctness), with the exact doublet ⟨S²⟩ = 0.75.
    """
    ccm = CCMSystem(_cell(np.diag([80.0, 80.0, 80.0]), [Atom(3, [0, 0, 0])]), (1, 1, 1), "sto-3g")
    assert ccm.supercell.multiplicity == 2          # auto doublet for odd electrons
    res = run_ccm_uhf(ccm)

    mol = ccm.supercell
    ref = run_uhf(mol, BasisSet(mol, "sto-3g"), UHFOptions())

    assert res.converged
    assert (res.n_alpha, res.n_beta) == (2, 1)
    assert res.energy == pytest.approx(ref.energy, abs=1e-8)
    assert res.s_squared == pytest.approx(0.75, abs=1e-6)
    assert res.idempotency_error < 1e-8
    # IID 344: the result identifies its executing route and positively
    # reports the parity-hold state (this route cannot hit the GDF hold).
    assert res.backend == "ccm-fourcenter-dense-union12-uhf"
    assert res.parity_held is False


def test_ccm_uhf_closed_shell_equals_rhf():
    """Closed-shell H₄ chain: UHF-CCM collapses to RHF-CCM (Pα = Pβ).

    From the core-Hamiltonian guess (identical for both spins) a closed shell has
    no spin polarisation, so UHF must return exactly the RHF energy with ⟨S²⟩ = 0.
    """
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    unit = _cell(np.diag([4.0 * BOHR, 40.0, 40.0]), [Atom(1, p) for p in pos])
    ccm = CCMSystem(unit, (6, 1, 1), "sto-3g")

    # Reuse the same effective ERI so the two SCFs are bit-for-bit comparable.
    from vibeqc.periodic.ccm.padded import ccm_eri
    eri = ccm_eri(ccm)
    ru = run_ccm_uhf(ccm, eri=eri)
    rr = run_ccm_rhf(ccm, eri=eri)

    assert ru.converged and rr.converged
    assert ru.energy == pytest.approx(rr.energy, abs=1e-8)
    assert abs(ru.s_squared) < 1e-8


def test_ccm_supercell_propagates_explicit_high_spin_multiplicity():
    """An explicit high-spin unit cell reaches the supercell spin counts.

    ``CCMSystem._build_supercell`` used to flatten the unit cell's multiplicity
    to the parity floor (even electrons -> singlet, odd -> doublet), so an
    even-electron triplet cell ran every spin-count consumer (``run_ccm_uhf``,
    ``run_ccm_uks``, the direct-torus drivers, the shell-aware route dispatch)
    as if it were closed shell -- the silent-wrong-number trap (CLAUDE.md § 7).
    An explicit multiplicity above the parity floor now replicates into every
    image, spins aligned (the Γ-point identical-cells convention):
    ``(multiplicity - 1) * N_c`` supercell unpaired electrons.
    """
    trip = _cell(np.diag([6.0, 6.0, 6.0]),
                 [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], mult=3)
    # (1,1,1): the supercell IS the unit cell -- the multiplicity carries over.
    assert CCMSystem(trip, (1, 1, 1), "sto-3g").supercell.multiplicity == 3
    # (2,1,1): two aligned triplet cells -> 4 unpaired -> quintet.
    assert CCMSystem(trip, (2, 1, 1), "sto-3g").supercell.multiplicity == 5

    # Parity behaviour without explicit high spin is unchanged.
    li = _cell(np.diag([8.0, 8.0, 8.0]), [Atom(3, [4.0, 4.0, 4.0])])
    assert CCMSystem(li, (1, 1, 1), "sto-3g").supercell.multiplicity == 2
    assert CCMSystem(li, (2, 1, 1), "sto-3g").supercell.multiplicity == 1

    # An inconsistent explicit spin fails at construction, not mid-SCF.
    bad = _cell(np.diag([6.0, 6.0, 6.0]),
                [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], mult=4)
    with pytest.raises(ValueError, match="incompatible"):
        CCMSystem(bad, (1, 1, 1), "sto-3g")


def test_ccm_uhf_even_electron_triplet_reaches_open_shell_counts():
    """UHF-CCM on a triplet H₂ cell fills (n_alpha, n_beta) = (2, 0) -- the
    even-electron open-shell case the parity flattening used to swallow."""
    trip = _cell(np.diag([6.0, 6.0, 6.0]),
                 [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], mult=3)
    res = run_ccm_uhf(CCMSystem(trip, (1, 1, 1), "sto-3g"))
    assert res.converged
    assert (res.n_alpha, res.n_beta) == (2, 0)
    # Two aligned spins: <S^2> = S(S+1) = 2 exactly at n_beta = 0.
    assert res.s_squared == pytest.approx(2.0, abs=1e-8)


def test_ccm_uhf_conv_tol_grad_tightens_exit_gate():
    """``conv_tol_grad`` is an explicit DIIS-residual exit criterion (default
    1e-6 = the historical hardcoded gate; same knob as
    ``run_ccm_uhf_direct``): the energy is stationary at convergence, so an
    energy-only gate can report converged=True with a loose minority-spin
    density. Tightening the bound must reach the same fixed point with at
    least as many iterations."""
    unit = _cell(np.diag([2.0, 30.0, 30.0]), [Atom(1, [0, 0, 0])])
    ccm = CCMSystem(unit, (5, 1, 1), "sto-3g")
    base = run_ccm_uhf(ccm)
    tight = run_ccm_uhf(ccm, conv_tol=1e-12, conv_tol_grad=1e-9)
    assert base.converged and tight.converged
    assert tight.n_iter >= base.n_iter
    assert tight.energy == pytest.approx(base.energy, abs=1e-8)
    assert tight.idempotency_error < 1e-8


def test_ccm_uhf_open_shell_periodic_doublet():
    """Open-shell periodic cluster (odd-electron H chain) converges cleanly.

    A half-filled H chain (1 H/cell) on an odd cluster is an open-shell doublet;
    UHF-CCM must converge to a properly idempotent spin density with ⟨S²⟩ near
    the doublet value (small spin contamination from the correlated chain is
    expected and physical).
    """
    unit = _cell(np.diag([2.0, 30.0, 30.0]), [Atom(1, [0, 0, 0])])
    res = run_ccm_uhf(CCMSystem(unit, (5, 1, 1), "sto-3g"))

    assert res.converged
    assert (res.n_alpha, res.n_beta) == (3, 2)
    assert res.idempotency_error < 1e-8
    # Doublet floor 0.75, mild contamination from the correlated half-filled chain.
    assert 0.74 < res.s_squared < 1.0
