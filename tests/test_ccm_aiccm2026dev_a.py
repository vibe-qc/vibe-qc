"""The ``aiccm2026dev-a`` symmetric four-center — this dev line's CCM method of record.

``run_ccm_rhf(ccm, method="aiccm2026dev-a")`` builds the symmetric Born–von Kármán-torus
effective ERI (:func:`vibeqc.periodic.ccm.padded.ccm_eri_symmetric`,
``AICCM_ALGORITHM.md`` §13): the historical eq-18 weight
``ω_μν·½(ω_μρ+ω_νρ)·ω_ρσ`` is replaced by a symmetric bridge
``¼(ω_μρ+ω_νρ+ω_μσ+ω_νσ)`` with ρ and σ folded *independently* to the home bra's
WSC. The result is **exactly 8-fold permutationally symmetric for any lattice**,
where eq 18 (``method="union12"``) breaks ρ↔σ / μ↔ν symmetry in ≥2-D.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550; AICCM_ALGORITHM.md §13.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    KPoints,
    PeriodicSystem,
    RHFOptions,
    UHFOptions,
    run_mp2,
    run_rhf,
    run_uhf,
)
from vibeqc.periodic.ccm import CCMSystem, ccm_band_structure, ccm_mayer_bond_orders
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.ump2 import run_ccm_ump2
from vibeqc.periodic.ccm.padded import (
    build_padded_cluster,
    ccm_eri,
    ccm_eri_symmetric,
    eri_cells,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable
from vibeqc.periodic.ccm.uhf import run_ccm_uhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _max_asym(eff):
    """Max relative violation of the generating ERI permutation symmetries."""
    nrm = np.linalg.norm(eff)
    return max(
        np.linalg.norm(eff - eff.transpose(1, 0, 2, 3)) / nrm,   # μ↔ν
        np.linalg.norm(eff - eff.transpose(0, 1, 3, 2)) / nrm,   # ρ↔σ
        np.linalg.norm(eff - eff.transpose(2, 3, 0, 1)) / nrm,   # bra↔ket
    )


def _h4_unit():
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    return PeriodicSystem(3, np.diag([4.0 * BOHR, 40.0, 40.0]),
                          [Atom(1, p) for p in pos], charge=0, multiplicity=1)


def _hex2d(a=4.0):
    lat = np.array([[a, 0, 0], [a * 0.5, a * np.sqrt(3) / 2, 0], [0, 0, 40.0]])
    return PeriodicSystem(3, lat, [Atom(1, [0, 0, 0])], charge=0, multiplicity=2)


def _oblique2d(a=4.0):
    lat = np.array([[a, 0, 0], [a * 0.4, a * 0.9, 0], [0, 0, 40.0]])
    return PeriodicSystem(3, lat, [Atom(1, [0, 0, 0])], charge=0, multiplicity=2)


def test_aiccm2026dev_a_keyword_runs():
    """The method is reachable via the keyword and gives a converged result."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    res = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    assert res.converged
    assert np.isfinite(res.energy)


def test_unknown_method_raises():
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="unknown CCM four-center method"):
        run_ccm_rhf(ccm, method="not_a_method")


def test_aiccmdev_alias_back_compat():
    """The deprecated "aiccmdev" keyword is a back-compat alias of
    "aiccm2026dev-a" (the competing line's harness still calls the old name)."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    e_new = run_ccm_rhf(ccm, method="aiccm2026dev-a").energy_per_atom
    e_alias = run_ccm_rhf(ccm, method="aiccmdev").energy_per_atom
    assert e_alias == pytest.approx(e_new, abs=1e-12)


@pytest.mark.parametrize("name,unit,nrep", [
    ("h4-1d", _h4_unit(), (4, 1, 1)),
    ("hex-2d", _hex2d(), (2, 2, 1)),
    ("oblique-2d", _oblique2d(), (2, 2, 1)),
])
def test_aiccm2026dev_a_tensor_is_eightfold_symmetric(name, unit, nrep):
    """The aiccm2026dev-a tensor is exactly 8-fold symmetric for every lattice;
    union12 is not (it degrades sharply going 1-D → 2-D)."""
    ccm = CCMSystem(unit, nrep, "sto-3g")
    pad = build_padded_cluster(ccm, eri_cells(ccm))
    assert _max_asym(ccm_eri_symmetric(ccm, pad)) < 1e-12
    if name != "h4-1d":                      # union12 is only ~symmetric in 1-D
        assert _max_asym(ccm_eri(ccm, pad)) > 1e-2


def test_aiccm2026dev_a_isolated_equals_molecular():
    """Isolated (huge-cell) cluster: aiccm2026dev-a CCM == molecular RHF."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    res = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    mol = ccm.supercell
    ref = run_rhf(mol, BasisSet(mol, "sto-3g"), RHFOptions())
    assert res.converged
    assert res.energy == pytest.approx(ref.energy, abs=1e-7)
    bonds = ccm_mayer_bond_orders(res, ccm, threshold=1.0e-3)
    assert bonds.bonds
    assert bonds.translational_spread < 1.0e-8
    kpath = KPoints.band_path(
        cell,
        scheme="manual",
        segments=[([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.0], "X")],
        points_per_segment=3,
    ).to_kpath()
    bands = ccm_band_structure(res, ccm, kpath)
    assert bands.energies.shape == (kpath.n_points, ccm.basis.nbasis)
    assert np.all(np.isfinite(bands.energies))


def test_aiccm2026dev_a_matches_union12_in_1d():
    """In 1-D both methods give the same energy (eq 18's asymmetry is invisible),
    converging toward the gold -0.542875 Ha/atom."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    e_dev = run_ccm_rhf(ccm, method="aiccm2026dev-a").energy_per_atom
    e_u12 = run_ccm_rhf(ccm, method="union12").energy_per_atom
    assert e_dev == pytest.approx(e_u12, abs=1e-6)
    assert e_dev == pytest.approx(-0.542875, abs=5e-4)


def test_aiccm2026dev_a_diverges_from_union12_in_2d():
    """In 2-D the symmetric method departs from eq 18 (the asymmetry now bites)."""
    ccm = CCMSystem(_hex2d(), (2, 2, 1), "sto-3g")
    e_dev = run_ccm_rhf(ccm, method="aiccm2026dev-a").energy_per_atom
    e_u12 = run_ccm_rhf(ccm, method="union12").energy_per_atom
    assert abs(e_dev - e_u12) > 1e-4


# -- end-to-end: UHF + MP2 reach aiccm2026dev-a via the same keyword ---------------

def test_aiccm2026dev_a_uhf_isolated_equals_molecular():
    """Isolated Li atom (3e doublet): UHF-CCM aiccm2026dev-a == molecular run_uhf."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]), [Atom(3, [0, 0, 0])], 0, 2)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    res = run_ccm_uhf(ccm, method="aiccm2026dev-a")
    mol = ccm.supercell
    ref = run_uhf(mol, BasisSet(mol, "sto-3g"), UHFOptions())
    assert res.converged
    assert res.energy == pytest.approx(ref.energy, abs=1e-7)
    assert res.s_squared == pytest.approx(0.75, abs=1e-6)


def test_aiccm2026dev_a_uhf_closed_shell_matches_rhf():
    """Closed-shell isolated cluster: UHF aiccm2026dev-a == RHF aiccm2026dev-a."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    ru = run_ccm_uhf(ccm, method="aiccm2026dev-a")
    rr = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    assert ru.energy == pytest.approx(rr.energy, abs=1e-8)


def test_aiccm2026dev_a_mp2_isolated_equals_molecular():
    """Isolated (huge-cell) cluster: MP2-CCM aiccm2026dev-a correlation == molecular RMP2."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    res = run_ccm_mp2(ccm, method="aiccm2026dev-a")
    mol = ccm.supercell
    basis = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, basis, RHFOptions())
    mp2 = run_mp2(mol, basis, rhf)
    assert res.e_correlation == pytest.approx(mp2.e_correlation, abs=1e-8)


@pytest.mark.parametrize("name,unit,nrep", [
    ("h4-1d", _h4_unit(), (4, 1, 1)),
    ("hex-2d", _hex2d(), (2, 2, 1)),
])
def test_aiccm2026dev_a_scalable_matches_padded(name, unit, nrep):
    """The scalable C++ aiccm2026dev-a builder reproduces the Python padded
    ccm_eri_symmetric energy to µHa and converges — in 1-D and 2-D."""
    ccm = CCMSystem(unit, nrep, "sto-3g")
    pad = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    cxx = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")
    assert cxx.converged
    assert cxx.energy_per_atom == pytest.approx(pad.energy_per_atom, abs=5e-6)


def test_aiccm2026dev_a_scalable_unknown_method_raises():
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="unknown CCM four-center method"):
        run_ccm_rhf_scalable(ccm, method="not_a_method")


def test_aiccm2026dev_a_ump2_isolated_equals_molecular():
    """Isolated Li atom (3e doublet): UMP2-CCM aiccm2026dev-a == molecular run_ump2."""
    from vibeqc import UMP2Options, run_ump2
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]), [Atom(3, [0, 0, 0])], 0, 2)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    res = run_ccm_ump2(ccm, method="aiccm2026dev-a")
    mol = ccm.supercell
    basis = BasisSet(mol, "sto-3g")
    ref = run_ump2(mol, basis, run_uhf(mol, basis, UHFOptions()), UMP2Options())
    assert res.e_correlation == pytest.approx(ref.e_correlation, abs=1e-7)


def test_aiccm2026dev_a_ump2_closed_shell_matches_rmp2():
    """Closed-shell isolated H2: UMP2-CCM == RMP2-CCM (aiccm2026dev-a)."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    u = run_ccm_ump2(ccm, method="aiccm2026dev-a").e_correlation
    r = run_ccm_mp2(ccm, method="aiccm2026dev-a").e_correlation
    assert u == pytest.approx(r, abs=1e-9)


def test_aiccm2026dev_a_ccsd_h2_matches_molecular():
    """Isolated H2/cc-pVDZ: CCSD-CCM (exact integrals) matches molecular DF-CCSD
    to the density-fitting gap (the CCM uses exact ERIs, the reference is DF)."""
    from vibeqc import CCSDOptions, run_ccsd
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "cc-pvdz")
    r = run_ccm_ccsd(ccm, method="aiccm2026dev-a", compute_triples=False)
    mol = ccm.supercell
    basis = BasisSet(mol, "cc-pvdz")
    opts = CCSDOptions()
    opts.compute_triples = False
    ref = run_ccsd(mol, basis, run_rhf(mol, basis, RHFOptions()), opts)
    assert r.converged
    assert r.e_ccsd_correlation == pytest.approx(ref.e_ccsd_correlation, abs=2e-4)


def test_aiccm2026dev_a_ccsd_more_bound_than_mp2():
    """Fast sanity (H2/sto-3g): CCSD-CCM converges, is bound, and captures more
    correlation than MP2 (CCSD = FCI here, so it is the more negative of the two)
    while staying the same order of magnitude."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    cc = run_ccm_ccsd(ccm, method="aiccm2026dev-a", compute_triples=False)
    mp2 = run_ccm_mp2(ccm, method="aiccm2026dev-a")
    assert cc.converged
    assert cc.e_ccsd_correlation < mp2.e_correlation < 0          # CCSD more bound
    assert cc.e_ccsd_correlation == pytest.approx(mp2.e_correlation, abs=0.02)


@pytest.mark.slow
def test_aiccm2026dev_a_3d_nonorthorhombic_divergence():
    """The symmetric effect is genuinely 3-D: via the scalable C++ route on a
    dense BCC crystal (non-orthorhombic), aiccm2026dev-a departs from union12 (the
    orthorhombic cubic control shows no divergence), and both converge."""
    a = 4.5
    bcc = a * np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]])
    unit = PeriodicSystem(3, bcc, [Atom(1, [0, 0, 0])], 0, 2)
    ccm = CCMSystem(unit, (2, 2, 2), "sto-3g")
    eu = run_ccm_rhf_scalable(ccm, method="union12")
    ed = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")
    assert eu.converged and ed.converged
    assert abs(ed.energy_per_atom - eu.energy_per_atom) > 1e-3   # ~5 mHa/atom in BCC
    # Orthorhombic control: simple cubic shows NO divergence (WSC symmetry).
    cub = PeriodicSystem(3, np.diag([5.0, 5.0, 5.0]), [Atom(1, [0, 0, 0])], 0, 2)
    cc = CCMSystem(cub, (2, 2, 2), "sto-3g")
    cu = run_ccm_rhf_scalable(cc, method="union12").energy_per_atom
    cd = run_ccm_rhf_scalable(cc, method="aiccm2026dev-a").energy_per_atom
    assert cd == pytest.approx(cu, abs=1e-7)


@pytest.mark.slow
def test_aiccm2026dev_a_ccsd_t_matches_molecular():
    """LiH/cc-pVDZ: CCSD-CCM correlation + (T) match molecular DF-CCSD(T); the
    (T) correction matches to ~1e-6 (it depends on converged T1/T2 amplitudes)."""
    from vibeqc import CCSDOptions, run_ccsd
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(3, [0, 0, 0]), Atom(1, [0, 0, 3.0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "cc-pvdz")
    r = run_ccm_ccsd(ccm, method="aiccm2026dev-a", compute_triples=True)
    mol = ccm.supercell
    basis = BasisSet(mol, "cc-pvdz")
    opts = CCSDOptions()
    opts.compute_triples = True
    ref = run_ccsd(mol, basis, run_rhf(mol, basis, RHFOptions()), opts)
    assert r.converged
    assert r.e_ccsd_correlation == pytest.approx(ref.e_ccsd_correlation, abs=2e-4)
    assert r.e_t == pytest.approx(ref.e_t, abs=1e-5)


def test_aiccm2026dev_a_mp2_close_to_union12_in_1d():
    """1-D H₄: aiccm2026dev-a and union12 give *close* MP2 correlation.

    Unlike the HF energy (identical in 1-D — the total-energy contraction is
    blind to eq 18's residual asymmetry), MP2 probes individual (ia|jb) elements
    that differ slightly between the asymmetric eq-18 tensor and the symmetric
    aiccm2026dev-a tensor, so the correlation energies are close but not identical."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    c_dev = run_ccm_mp2(ccm, method="aiccm2026dev-a").e_correlation_per_atom
    c_u12 = run_ccm_mp2(ccm, method="union12").e_correlation_per_atom
    assert c_dev == pytest.approx(c_u12, abs=5e-5)


def test_run_ccm_rhf_emits_experimental_warning():
    """Every public Γ-CCM SCF driver fires AICCM2026DevAExperimentalWarning
    (the -a mirror of AICCM2026DevBExperimentalWarning; correlation and
    gradient drivers inherit it through the SCF they run)."""
    from vibeqc.periodic.ccm import AICCM2026DevAExperimentalWarning
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    with pytest.warns(AICCM2026DevAExperimentalWarning):
        run_ccm_rhf(ccm, method="aiccm2026dev-a")


def test_experimental_warning_exported_from_top_level():
    """Users filter the warning via the top-level name, so it must stay in
    vibeqc.__all__ (the same contract the -b class satisfies)."""
    import vibeqc
    from vibeqc.periodic.ccm import AICCM2026DevAExperimentalWarning
    assert vibeqc.AICCM2026DevAExperimentalWarning is AICCM2026DevAExperimentalWarning
    assert "AICCM2026DevAExperimentalWarning" in vibeqc.__all__
    assert issubclass(AICCM2026DevAExperimentalWarning, UserWarning)
