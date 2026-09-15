"""MP2-CCM — second-order correlation on a Cyclic Cluster Model reference
(AICCM roadmap Phase 3 / milestone M4).

The CCM reaches periodic correlation through the *molecular* post-HF path: the
WSSC-weighted AO ERIs and the HF-CCM canonical MOs are fed to the closed-shell
RMP2 energy expression (:func:`vibeqc.periodic.ccm.mp2.run_ccm_mp2`). These
tests pin (i) the molecular limit — an isolated cluster must reproduce vibe-qc's
own ``run_mp2`` exactly — and (ii) cluster-size convergence of the correlation
energy per atom on the 1-D H₄ chain.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, RHFOptions, run_mp2, run_rhf
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _cell(lattice, atoms):
    mult = 1 if sum(a.Z for a in atoms) % 2 == 0 else 2
    return PeriodicSystem(3, np.asarray(lattice, float), atoms, charge=0, multiplicity=mult)



def _assert_guess(result, ccm, requested="AUTO"):
    from vibeqc.guess import InitialGuess
    selection = result.guess_selection
    assert selection.requested == getattr(InitialGuess, requested)
    assert selection.effective == InitialGuess.HCORE
    assert selection.transport == InitialGuess.HCORE
    ref = getattr(result, "scf", getattr(result, "uhf", result))
    if hasattr(ref, "density_alpha"):
        for spin in ("alpha", "beta"):
            count = np.trace(getattr(ref, "density_"+spin) @ ref.overlap)
            assert count == pytest.approx(getattr(ref, "n_"+spin), abs=2e-12)
    else:
        assert np.trace(ref.density @ ref.overlap) == pytest.approx(
            ccm.supercell.n_electrons(), abs=2e-12)


@pytest.mark.parametrize("selector", ["SAD", "SAP", "PATOM", "MINAO", "HUECKEL", "READ", "FRAGMO", "typo", 123])
def test_ccm_reference_entrypoints_validate_before_integrals(selector, monkeypatch):
    import inspect
    from vibeqc.periodic.ccm import scf, uhf, dft, ri, direct, mp2, ump2, ccsd, uccsd, neutral
    ccm = CCMSystem(_cell(np.diag([8., 8., 8.]),
                    [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    def forbidden(*args, **kwargs):
        pytest.fail("integral construction preceded guess validation")
    monkeypatch.setattr(scf, "_ccm_eri_for_method", forbidden)
    monkeypatch.setattr(neutral, "ccm_neutral_cderi_fold", forbidden)
    checked = 0
    for module in (scf, uhf, dft, ri, direct, mp2, ump2, ccsd, uccsd):
        for name, fn in vars(module).items():
            if not name.startswith("run_ccm_") or not inspect.isfunction(fn):
                continue
            if fn.__module__ != module.__name__ or "initial_guess" not in inspect.signature(fn).parameters:
                continue
            extra = {"functional": "pbe"} if "functional" in inspect.signature(fn).parameters else {}
            with pytest.raises((ValueError, TypeError, NotImplementedError), match="guess|selector|InitialGuess"):
                fn(ccm, initial_guess=selector, **extra)
            checked += 1
    assert checked >= 20


def test_ccm_hcore_builder_and_initial_metric_count(monkeypatch):
    from vibeqc.periodic.ccm import scf
    ccm = CCMSystem(_cell(np.diag([80., 80., 80.]),
                    [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    real_einsum = np.einsum
    seeds = []
    def capture(expression, *args, **kwargs):
        if expression == "mnrs,rs->mn" and not seeds:
            seeds.append(np.array(args[1], copy=True))
        return real_einsum(expression, *args, **kwargs)
    monkeypatch.setattr(np, "einsum", capture)
    result = scf.run_ccm_rhf(ccm, initial_guess="CORE")
    _assert_guess(result, ccm, "HCORE")
    assert len(seeds) == 1
    x = scf._orthonormaliser(result.overlap, 1e-7)
    _, cp = np.linalg.eigh(x.T @ result.hcore @ x)
    occupied = (x @ cp)[:, :ccm.supercell.n_electrons()//2]
    np.testing.assert_allclose(seeds[0], 2*occupied @ occupied.T, atol=2e-13)
    assert np.trace(seeds[0] @ result.overlap) == pytest.approx(2., abs=2e-13)
    with pytest.raises(ValueError, match="relabel"):
        run_ccm_mp2(ccm, result, initial_guess="SAD", eri=np.empty(0))


def test_ccm_mp2_isolated_equals_molecular():
    """Isolated (huge-cell) cluster: MP2-CCM correlation == molecular RMP2.

    In the molecular limit the WSSC-weighted ERI is the molecular ERI, so the
    CCM correlation energy must reproduce vibe-qc's ``run_mp2`` to ~nHa (the
    kernel + AO→MO transform correctness check).
    """
    ccm = CCMSystem(_cell(np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    res = run_ccm_mp2(ccm)
    _assert_guess(res, ccm)

    mol = ccm.supercell
    basis = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, basis, RHFOptions())
    mp2 = run_mp2(mol, basis, rhf)

    assert res.e_correlation == pytest.approx(mp2.e_correlation, abs=1e-8)
    assert res.e_os == pytest.approx(mp2.e_os, abs=1e-8)
    assert res.e_ss == pytest.approx(mp2.e_ss, abs=1e-8)
    # Closed-shell MP2 correlation is bound and opposite-spin-dominated.
    assert res.e_correlation < 0.0
    assert abs(res.e_os) > abs(res.e_ss)


def test_ccm_mp2_h4_chain_converges():
    """H₄ alternating chain: MP2-CCM correlation/atom converges with cluster size.

    Energy/cell of a CCM grows toward the periodic limit as the cluster grows;
    the correlation per atom changes by a shrinking amount (no oscillation —
    CLAUDE.md §7) and stays physical (negative, opposite-spin-dominated).
    """
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    unit = _cell(np.diag([4.0 * BOHR, 40.0, 40.0]), [Atom(1, p) for p in pos])
    ec = {n: run_ccm_mp2(CCMSystem(unit, (n, 1, 1), "sto-3g")).e_correlation_per_atom
          for n in (4, 6, 8)}

    # Physical and bounded.
    assert all(v < 0.0 for v in ec.values())
    # Converging: the 6→8 step is smaller than the 4→6 step.
    d46 = abs(ec[6] - ec[4])
    d68 = abs(ec[8] - ec[6])
    assert d68 < d46
    # Correlation per atom is in the expected small range for H/STO-3G.
    assert -0.02 < ec[8] < -0.004


def test_ri_mp2_equals_dense_on_neutral_kernel():
    """RI-MP2 (cderi=L) reproduces the dense MP2 on the same neutral four-center
    to machine ε, without ever forming the n_ref_ao**4 AO tensor — the lever that
    lets canonical correlation reach 3-D (scalability hand-off, suggestion 2)."""
    _KE = 40.0
    h2 = PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])],
                        charge=0, multiplicity=1)
    for nrep in [(2, 1, 1), (4, 1, 1)]:
        ccm = CCMSystem(h2, nrep, "sto-3g")
        g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
        L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
        scf = run_ccm_rhf(ccm, eri=g)
        dense = run_ccm_mp2(ccm, scf, eri=g)
        ri = run_ccm_mp2(ccm, scf, cderi=L)
        assert ri.e_correlation == pytest.approx(dense.e_correlation, abs=1e-12)
        assert ri.e_os == pytest.approx(dense.e_os, abs=1e-12)
        assert ri.e_ss == pytest.approx(dense.e_ss, abs=1e-12)


def test_ri_mp2_requires_scf_reference():
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], charge=0, multiplicity=1),
                    (2, 1, 1), "sto-3g")
    L = ccm_neutral_cderi(ccm, ke_cutoff=40.0)
    with pytest.raises(ValueError):
        run_ccm_mp2(ccm, cderi=L)


def test_ri_mp2_wrapper_matches_manual_recipe_and_dispatches():
    """run_ccm_ri_mp2 == the manual (build L → lean neutral SCF → RI-MP2) recipe,
    dispatching closed→RMP2 / open→UMP2 by shell (the safe one-call entry that
    can't mismatch the SCF reference and L)."""
    from vibeqc.periodic.ccm.mp2 import run_ccm_ri_mp2
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_ri_neutral
    from vibeqc.periodic.ccm.uhf import run_ccm_uhf
    from vibeqc.periodic.ccm.ump2 import CCMUMP2Result, run_ccm_ump2
    _KE = 40.0
    # closed shell
    cs = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                   [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), (2, 1, 1), "sto-3g")
    L = ccm_neutral_cderi(cs, ke_cutoff=_KE)
    man = run_ccm_mp2(cs, run_ccm_rhf_ri_neutral(cs, cderi=L), cderi=L)
    w = run_ccm_ri_mp2(cs, ke_cutoff=_KE, reference="neutral")
    _assert_guess(w, cs)
    assert w.e_total == pytest.approx(man.e_total, abs=1e-12)
    # open shell → UMP2
    osys = CCMSystem(PeriodicSystem(3, np.diag([2.5, 15.0, 15.0]),
                     [Atom(1, [0, 0, 0])], charge=0, multiplicity=2), (3, 1, 1), "sto-3g")
    Lo = ccm_neutral_cderi(osys, ke_cutoff=_KE)
    mano = run_ccm_ump2(osys, run_ccm_uhf(osys, cderi=Lo), cderi=Lo)
    wo = run_ccm_ri_mp2(osys, ke_cutoff=_KE, reference="neutral")
    _assert_guess(wo, osys)
    assert isinstance(wo, CCMUMP2Result)
    assert wo.e_total == pytest.approx(mano.e_total, abs=1e-10)


def test_ri_ccsd_wrapper_matches_manual_recipe():
    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd, run_ccm_ri_ccsd
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_ri_neutral
    _KE = 40.0
    cs = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                   [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), (2, 1, 1), "sto-3g")
    L = ccm_neutral_cderi(cs, ke_cutoff=_KE)
    man = run_ccm_ccsd(cs, run_ccm_rhf_ri_neutral(cs, cderi=L), cderi=L, compute_triples=True)
    w = run_ccm_ri_ccsd(cs, ke_cutoff=_KE, compute_triples=True,
                        reference="neutral")
    _assert_guess(w, cs)
    assert w.e_total == pytest.approx(man.e_total, abs=1e-12)


def test_ri_wrappers_reference_direct_opt_in():
    """reference='direct' (the default since D-5, 2026-09-05) == the manual
    (build L → run_ccm_rhf_direct → RI correlation) recipe on ONE shared L.

    A historical D2 KMP2 diagnostic gave residuals of 3e-6 Ha/cell for a (1,1,1)
    c-diamond neutral-control diagnostic and 0.24 mHa/atom for LiH (2,2,2)
    against external PySCF KMP2@exxdiv='ewald'. Those provenance-incomplete
    values are not reportable benchmarks and are not a Γ-CCM/χ-CCM comparison.
    Neutral and direct references also differ in S, h, and E_nn, so their
    ordering cannot be assigned to the seam. The observed correlation ordering
    below is retained only as a route-regression check.
    Open-shell + invalid values fail loudly; default composition unchanged."""
    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd, run_ccm_ri_ccsd
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
    from vibeqc.periodic.ccm.mp2 import run_ccm_ri_mp2
    from vibeqc.periodic.exchange_convention import BVK_EWALD
    _KE = 40.0
    cs = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                   [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), (2, 1, 1), "sto-3g")
    L = cs_L = ccm_neutral_cderi(cs, ke_cutoff=_KE)
    scf_dir = run_ccm_rhf_direct(cs, cderi=cs_L)
    man = run_ccm_mp2(cs, scf_dir, cderi=L)
    w = run_ccm_ri_mp2(cs, ke_cutoff=_KE, reference="direct")
    _assert_guess(w, cs)
    assert w.e_total == pytest.approx(man.e_total, abs=1e-12)
    assert w.scf.exchange_q0 == BVK_EWALD
    # Observed complete-route ordering; this is not a seam-isolation theorem.
    neu = run_ccm_ri_mp2(cs, ke_cutoff=_KE, reference="neutral")
    assert abs(w.e_total - w.e_hf) < abs(neu.e_total - neu.e_hf)
    # CCSD sibling: same contract (triples off — the reference is the point).
    man_cc = run_ccm_ccsd(cs, scf_dir, cderi=L, compute_triples=False)
    w_cc = run_ccm_ri_ccsd(cs, ke_cutoff=_KE, compute_triples=False,
                           reference="direct")
    assert w_cc.e_total == pytest.approx(man_cc.e_total, abs=1e-12)
    # Invalid reference fails loudly.
    with pytest.raises(ValueError, match="reference"):
        run_ccm_ri_mp2(cs, ke_cutoff=_KE, reference="bogus")
    with pytest.raises(ValueError, match="reference"):
        run_ccm_ri_ccsd(cs, ke_cutoff=_KE, reference="bogus")
    # symmetry= threads to the cderi build (2026-07-14). The symmetry path
    # reconstructs the fit in the canonical aux frame (max|dL| ~ 0.6 vs the
    # un-reduced compact frame — an arbitrary rotation), but that frame rotation
    # cancels in every physical contraction (J/K and the MP2 (ia|jb), which
    # contract L with itself), so the correlation energy is invariant to the
    # documented sto-3g cell-list reconstruction floor (measured 5.8e-9 here at
    # ke=40; test_ccm_symmetry.py pins < 1e-10 at production defaults). The point
    # is that the kwarg reaches ccm_neutral_cderi and does not perturb the energy.
    w_sym = run_ccm_ri_mp2(cs, ke_cutoff=_KE, symmetry=True,
                           reference="neutral")
    assert w_sym.e_total == pytest.approx(neu.e_total, abs=1e-7)


def test_ri_mp2_reference_direct_open_shell():
    """Open-shell reference='direct' (2026-07-14): run_ccm_ri_mp2 on a doublet
    now composes run_ccm_uhf_direct (BvK-ewald seam per spin) + run_ccm_ump2 on
    one shared L, == the manual recipe. This closes the gap the opt-in landed
    with — previously open-shell direct raised NotImplementedError, so half the
    benchmark set (open-shell ionics) could not use the corrected reference."""
    from vibeqc.periodic.ccm.direct import run_ccm_uhf_direct
    from vibeqc.periodic.ccm.mp2 import run_ccm_ri_mp2
    from vibeqc.periodic.ccm.ump2 import CCMUMP2Result, run_ccm_ump2
    from vibeqc.periodic.exchange_convention import BVK_EWALD
    _KE = 40.0
    osys = CCMSystem(PeriodicSystem(3, np.diag([2.5, 15.0, 15.0]),
                     [Atom(1, [0, 0, 0])], charge=0, multiplicity=2), (3, 1, 1), "sto-3g")
    Lo = ccm_neutral_cderi(osys, ke_cutoff=_KE)
    uhf_dir = run_ccm_uhf_direct(osys, cderi=Lo)
    man = run_ccm_ump2(osys, uhf_dir, cderi=Lo)
    w = run_ccm_ri_mp2(osys, ke_cutoff=_KE, reference="direct")
    assert isinstance(w, CCMUMP2Result)
    assert uhf_dir.exchange_q0 == BVK_EWALD
    assert w.e_total == pytest.approx(man.e_total, abs=1e-10)
    # Distinct from the strict-zero route (the reference actually moved).
    neu = run_ccm_ri_mp2(osys, ke_cutoff=_KE, reference="neutral")
    assert abs(w.e_total - neu.e_total) > 1e-3


def test_ri_wrappers_default_reference_is_direct():
    """The A-line RI default is ``reference="direct"`` (D-5, 2026-09-05).

    ``HANDOVER_D2_EXXDIV.md`` measured that the strict-zero neutral reference
    error grows linearly in the cluster size, so ``exxdiv="ewald"`` is the
    correct convention and a standard method cannot ship correlation on the
    neutral reference. Reproduced on this cell while making the flip, both legs
    riding the same ``L`` so only the SCF reference route differs:
    ``E_hf(direct) - E_hf(neutral)`` is +0.956443 Ha at ``nrep=(2,1,1)``,
    +2.449583 at (3,1,1) and +4.070138 at (4,1,1) -- about 1.5 Ha per added
    cell, per-cell still climbing (0.478, 0.817, 1.018 Ha/cell), while the
    correlation itself moves only 6.1 to 7.8 mHa over the same range.

    This pins the *default*, not the routes: both remain selectable, and the
    tests above pin each one explicitly. Without this, a future edit could flip
    the default back and every test here would still pass, because they all
    name their reference.
    """
    from vibeqc.periodic.ccm.ccsd import run_ccm_ri_ccsd
    from vibeqc.periodic.ccm.mp2 import run_ccm_ri_mp2

    _KE = 40.0
    cs = CCMSystem(
        PeriodicSystem(
            3,
            np.diag([6.0, 15.0, 15.0]),
            [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])],
            0,
            1,
        ),
        (2, 1, 1),
        "sto-3g",
    )
    defaulted = run_ccm_ri_mp2(cs, ke_cutoff=_KE)
    direct = run_ccm_ri_mp2(cs, ke_cutoff=_KE, reference="direct")
    neutral = run_ccm_ri_mp2(cs, ke_cutoff=_KE, reference="neutral")
    assert defaulted.e_total == pytest.approx(direct.e_total, abs=1e-12)
    # The two routes are genuinely different, so the assertion above has force.
    assert abs(direct.e_total - neutral.e_total) > 0.5

    cc_defaulted = run_ccm_ri_ccsd(cs, ke_cutoff=_KE, compute_triples=False)
    cc_direct = run_ccm_ri_ccsd(
        cs, ke_cutoff=_KE, compute_triples=False, reference="direct"
    )
    assert cc_defaulted.e_total == pytest.approx(cc_direct.e_total, abs=1e-12)


def test_ccm_open_shell_initial_spin_counts(monkeypatch):
    from vibeqc.periodic.ccm.uhf import run_ccm_uhf
    ccm = CCMSystem(_cell(np.diag([80., 80., 80.]),
                    [Atom(3, [0, 0, 0])]), (1, 1, 1), "sto-3g")
    real_einsum = np.einsum
    seeds = []
    def capture(expression, *args, **kwargs):
        if expression == "msrn,rs->mn" and len(seeds) < 2:
            seeds.append(np.array(args[1], copy=True))
        return real_einsum(expression, *args, **kwargs)
    monkeypatch.setattr(np, "einsum", capture)
    result = run_ccm_uhf(ccm)
    _assert_guess(result, ccm)
    assert len(seeds) == 2
    for density, count in zip(seeds, (2, 1)):
        np.testing.assert_allclose(density, density.T, atol=1e-14)
        assert np.trace(density @ result.overlap) == pytest.approx(count, abs=2e-12)
