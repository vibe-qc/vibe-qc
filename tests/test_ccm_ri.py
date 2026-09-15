"""≤3-center Coulomb approximations for the CCM (RI, RIJCOSX) — experimental.

Two routes (:mod:`vibeqc.periodic.ccm.ri`):

* **RI via GDF** (``run_ccm_rhf_gdf`` / ``run_ccm_rks_gdf``) — a separately
  declared neutral fitted-torus control. Its real-Γ and character-mesh forms
  represent the same specified block-circulant control Hamiltonian; neither is
  the union-and-weight Γ-CCM construction.
* **WSSC RI-J / RIJCOSX** (``run_ccm_rhf_rij`` / ``run_ccm_rhf_rijcosx``) — the
  research route with the explicit eq-13 union 3-center weighting. Exact in the
  isolated limit (RIJCOSX == molecular RIJCOSX there); ~few-% periodically.

Reference: Peintinger & Bredow 2014; Neese et al., Chem. Phys. 356, 98 (2009).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, RHFOptions, run_rhf
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.dft import run_ccm_rks
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.ri import (
    run_ccm_rhf_gdf,
    run_ccm_rhf_ri_neutral,
    run_ccm_rhf_rij,
    run_ccm_rhf_rijcosx,
    run_ccm_rks_gdf,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _h2(a=80.0):
    return PeriodicSystem(3, np.diag([a, a, a]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)


@pytest.mark.slow
def test_ri_gdf_hf_matches_four_center():
    """The isolated H2 control shares the molecular limit within RI error.

    This numerical anchor compares two declared constructions; it is not a
    global Γ-CCM/GDF equivalence proof.
    """
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    r4 = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    g = run_ccm_rhf_gdf(ccm)
    assert g.converged
    assert g.energy_per_atom == pytest.approx(r4.energy_per_atom, abs=2e-4)


@pytest.mark.slow
def test_ri_gdf_ks_runs():
    """RI (GDF) KS-DFT runs (pure + hybrid)."""
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    for func in ("pbe", "pbe0"):
        r = run_ccm_rks_gdf(ccm, func)
        assert r.converged and np.isfinite(r.energy)


@pytest.mark.parametrize("route_name", ["run_ccm_rks_gdf", "run_ccm_uks_gdf"])
def test_neutral_bloch_external_xc_fails_before_gdf_setup(
    monkeypatch,
    route_name,
):
    import vibeqc as vq
    from vibeqc.periodic.ccm import direct as direct_module
    from vibeqc.periodic.ccm import ri as ri_module

    name = f"test-neutral-bloch-external-{route_name}"
    vq.define_external_functional(name, lambda _features: {})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external-XC guard must precede GDF setup")

    monkeypatch.setattr(ri_module, "_ccm_gdf", forbidden)
    monkeypatch.setattr(
        direct_module,
        "_reject_vacuum_padded_direct",
        forbidden,
    )

    with pytest.raises(
        NotImplementedError,
        match="external XC.*neutral-Bloch",
    ):
        getattr(ri_module, route_name)(None, name)


def test_gdf_result_carries_inner_backend():
    """CCMGDFResult.backend mirrors the inner driver verbatim (IID 344): a
    consumer can tell a held number from a converged one without re-reading
    the .err sidecar. Because the wrapper copies the string, a +PARITY_HELD
    suffix set by the inner driver (pinned at the pbc_gdf layer in
    test_pbc_gdf_compcell / test_pbc_gdf_mdf) can never be dropped on its
    way out of the CCM wrapper."""
    cell = PeriodicSystem(3, np.diag([5.0, 5.0, 5.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 2), "sto-3g")
    r = run_ccm_rhf_gdf(ccm)
    inner = str(getattr(r.raw, "backend", "") or "")
    assert r.backend == inner
    assert r.backend  # non-empty on the wired GDF path
    assert "+PARITY_HELD" not in r.backend  # this fixture is not the hold class
    # Structured negative control (IID 344): an unheld run must POSITIVELY
    # report held=False -- absence of the flag is not a verdict.
    assert r.parity_held is False


def test_gdf_result_reports_structured_parity_hold(monkeypatch):
    """IID 344: a parity hold must be a structured field on the CCM result
    (``parity_held``), not only a ``+PARITY_HELD`` substring a consumer has
    to know to grep for. The inner multi-k driver is faked with a held
    backend string (the hold class itself is pinned for real at the pbc_gdf
    layer in test_pbc_gdf_compcell / test_pbc_gdf_mdf); the wrapper must
    carry the string verbatim AND derive the boolean from it."""
    import vibeqc

    cell = PeriodicSystem(3, np.diag([5.0, 5.0, 5.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 2), "sto-3g")

    class _HeldInner:
        converged = True
        energy = -1.0
        n_iter = 7
        e_hf_exchange = -0.5
        backend = "native-multi-k-gdf-gdf-rhf+PARITY_HELD"

    monkeypatch.setattr(
        vibeqc, "run_krhf_periodic_gdf", lambda *a, **k: _HeldInner()
    )
    r = run_ccm_rhf_gdf(ccm)
    assert r.backend == "native-multi-k-gdf-gdf-rhf+PARITY_HELD"
    assert r.parity_held is True


def test_neutral_ri_result_carries_backend_identity():
    """IID 344: the neutral supercell-Γ RI route identifies its executing
    operator on the result -- a record consumer must be able to tell WHICH
    route produced a number (and that no hold fired) without out-of-band
    knowledge. Routes whose result carried no identity were all demoted
    together in the aic7 wave."""
    cell = PeriodicSystem(3, np.diag([5.0, 5.0, 5.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 2), "sto-3g")
    L = ccm_neutral_cderi(ccm)
    r = run_ccm_rhf_ri_neutral(ccm, cderi=L)
    assert r.converged
    assert r.backend == "ccm-neutral-ri-rhf"
    assert r.parity_held is False


def test_ri_neutral_prebuilt_cderi_rejects_aux_basis():
    """IID 344 sibling: the lean supercell-Γ neutral route takes the same
    fail-closed stance -- aux_basis beside a prebuilt cderi raises instead
    of being silently dropped."""
    cell = PeriodicSystem(3, np.diag([5.0, 5.0, 5.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 2), "sto-3g")
    L = ccm_neutral_cderi(ccm)
    with pytest.raises(ValueError, match="aux_basis is ignored"):
        run_ccm_rhf_ri_neutral(ccm, cderi=L, aux_basis="def2-svp-jk")


@pytest.mark.slow
def test_ri_gdf_rijcosx_matches_exact_exx():
    """RIJCOSX on the neutral fitted-torus GDF control route.

    RI-J plus seminumerical COSX-K via ``k_exchange="cosx"`` reproduces exact
    GDF exchange to the COSX grid floor for both HF and a hybrid. This is a
    same-control backend gate, not Γ-CCM construction evidence.
    """
    cell = PeriodicSystem(3, np.diag([5.0, 5.0, 5.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 2), "sto-3g")
    exx = run_ccm_rhf_gdf(ccm)
    cosx = run_ccm_rhf_gdf(ccm, k_exchange="cosx", use_compcell=True)
    assert cosx.converged
    assert cosx.energy_per_atom == pytest.approx(exx.energy_per_atom, abs=1e-3)  # ~1.5e-4
    # Hybrid + RIJCOSX also runs on the same neutral control.
    h = run_ccm_rks_gdf(ccm, "pbe0", k_exchange="cosx", use_compcell=True)
    assert h.converged and np.isfinite(h.energy)


@pytest.mark.slow
def test_ri_gdf_3d_matches_four_center():
    """The compact 3-D fixture agrees within its declared control tolerance.

    This is a numerical cross-check, not evidence that GDF constructs Γ-CCM.
    """
    cell = PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (2, 2, 2), "sto-3g")
    k4 = run_ccm_rks(ccm, "pbe", method="aiccm2026dev-a")
    kg = run_ccm_rks_gdf(ccm, "pbe")
    assert kg.converged
    assert kg.energy_per_atom == pytest.approx(k4.energy_per_atom, abs=2e-4)


def test_wssc_rij_isolated_matches_four_center():
    """WSSC RI-J: isolated limit reproduces the four-center within RI fitting error."""
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    r4 = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    rij = run_ccm_rhf_rij(ccm, method="aiccm2026dev-a")
    assert rij.converged
    assert rij.energy == pytest.approx(r4.energy, abs=5e-4)


@pytest.mark.slow
def test_rijcosx_isolated_matches_molecular():
    """RIJCOSX-CCM (WSSC RI-J + COSX-K): isolated limit == molecular RIJCOSX."""
    ccm = CCMSystem(_h2(), (1, 1, 1), "cc-pvdz")
    rc = run_ccm_rhf_rijcosx(ccm, aux_basis="cc-pvdz-jkfit")
    mol = ccm.supercell
    opts = RHFOptions()
    opts.cosx = True
    opts.density_fit = True
    opts.aux_basis = "cc-pvdz-jkfit"
    ref = run_rhf(mol, BasisSet(mol, "cc-pvdz"), opts)
    assert rc.converged
    assert rc.energy == pytest.approx(ref.energy, abs=1e-3)


def _h2_chain_cell():
    return PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)


def test_ri_neutral_lean_scf_matches_dense_g():
    """run_ccm_rhf_ri_neutral assembles J/K from the cderi L (the dense n_ref**4
    neutral g is never formed) and reproduces run_ccm_rhf(eri=ccm_eri_neutral) to
    machine ε -- the lean supercell-Γ neutral reference that pairs with the RI
    correlation routes (run_ccm_mp2/ccsd(cderi=L))."""
    for nrep in [(2, 1, 1), (4, 1, 1)]:
        ccm = CCMSystem(_h2_chain_cell(), nrep, "sto-3g")
        g = ccm_eri_neutral(ccm, ke_cutoff=40.0)
        L = ccm_neutral_cderi(ccm, ke_cutoff=40.0)
        dense = run_ccm_rhf(ccm, eri=g)
        lean = run_ccm_rhf_ri_neutral(ccm, cderi=L)
        assert lean.converged
        assert lean.energy == pytest.approx(dense.energy, abs=1e-10)
        # MOs live in the same supercell-Γ AO basis as L (square, S-orthonormal).
        assert np.asarray(lean.mo_coeffs).shape == (ccm.nbf, ccm.nbf)


def test_ri_neutral_lean_scf_builds_own_cderi():
    """With cderi=None the lean driver builds L itself; same energy as the dense g."""
    ccm = CCMSystem(_h2_chain_cell(), (2, 1, 1), "sto-3g")
    dense = run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm, ke_cutoff=40.0))
    lean = run_ccm_rhf_ri_neutral(ccm, ke_cutoff=40.0)
    assert lean.energy == pytest.approx(dense.energy, abs=1e-10)


def test_gdf_route_records_exchange_q0_field():
    """The GDF route records exchange_q0 / applicability as result FIELDS
    (theory-chat ask 2026-07-16) rather than requiring the caller to derive them.

    Keyed on the driver's HF-exchange energy: RHF (exact exchange) applies the
    ewald K-shift -> BvK-ewald / active; a pure-DFT KS run has no exact exchange
    -> inactive. Mirrors CCMDirectResult.exchange_q0.
    """
    from vibeqc import Atom as _Atom, PeriodicSystem as _PS, PeriodicRHFOptions
    from vibeqc.periodic.exchange_convention import BVK_EWALD

    cell = _PS(3, np.diag([6.0, 6.0, 6.0]),
               [_Atom(1, [0, 0, 0]), _Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(cell, (2, 1, 1), "sto-3g")
    o = PeriodicRHFOptions()
    o.conv_tol_energy = 1e-9
    o.max_iter = 128
    o.damping = 0.0

    r = run_ccm_rhf_gdf(ccm, options=o)          # RHF: exact exchange present
    assert r.exchange_q0 == BVK_EWALD
    assert r.exchange_q0_applicability == "active"

    k = run_ccm_rks_gdf(ccm, "pbe", options=o)   # pure DFT: no exact exchange
    assert k.exchange_q0_applicability == "inactive"
    assert k.exchange_q0 == ""
