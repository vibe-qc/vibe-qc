"""State selection on the real-Gamma torus (#519, RG-9).

An explicit SAD seed on the direct drivers and the runner arm (AUTO stays
HCORE), the RHF stability check (Seeger-Pople singlet and triplet sectors of
the real orbital Hessian on the operator the loop ran on), the following of a
translation-invariant singlet instability, and the orbitals of the evaluated
state that both rely on.

Rows: the Davidson eigenvalues equal the explicit Hessian's and the energy
curvature along the mode is four times the eigenvalue; the restriction to
translation-invariant rotations; the verdict fields reach the molecular
verdict object; the returned orbitals are the canonical orbitals of the
returned Fock after an oscillating start (the loop used to return those of a
DIIS trial, 0.0095 and 0.0298 Ha against the physical -0.516 and 0.430 Ha);
SAD is explicit only and reaches the same states as HCORE on these cells; the
refusals. The slow rows run SAD on the UHF and RKS drivers and follow the
bond-alternation instability of an equidistant carbon chain, whose uniform
RHF state is a translation-invariant singlet saddle on the five-cell torus.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, InitialGuess, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

from vibeqc.guess import periodic_guess_capabilities  # noqa: E402
from vibeqc.output.verdict import verdict_from_result  # noqa: E402
from vibeqc.periodic.ccm.direct import (  # noqa: E402
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
)
from vibeqc.periodic.ccm.real_gamma_runner import run_real_gamma_scf  # noqa: E402
from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral  # noqa: E402
from vibeqc.periodic.ccm.stability import (  # noqa: E402
    STABILITY_TOL,
    _response,
    ccm_rhf_stability,
    circulant_projection,
    semicanonical_orbitals,
)

TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=200)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _h2_cube():
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _h_chain(d):
    """Two H atoms per cell, equidistant along z (spacing ``d`` bohr)."""
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 2.0 * d]),
        [Atom(1, [3.0, 3.0, 0.5 * d]), Atom(1, [3.0, 3.0, 1.5 * d])], 0, 1)


def _li_doublet():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _energy(L, S, h, xi, e_nn):
    def energy(D):
        J = ccm_ri_j_neutral(L, D)
        K = ccm_ri_k_neutral(L, D) + xi * (S @ D @ S)
        return float(np.sum(D * h) + 0.5 * np.sum(D * J)
                     - 0.25 * np.sum(D * K) + e_nn)
    return energy


def test_davidson_is_the_hessian_and_the_curvature_of_the_energy():
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    res = _quiet(run_ccm_rhf_direct, ccm, **TIGHT)
    st = ccm_rhf_stability(res, ccm)
    assert st.converged and not st.internal_instability
    assert st.orbital_gradient < 1e-8
    L, S = np.asarray(res.cderi), res.overlap
    xi = ccm_exchange_q0_madelung(ccm)
    Co, Cv = st.orbitals
    n_occ = Co.shape[1]
    eo = np.diag(Co.T @ res.fock @ Co)
    ev = np.diag(Cv.T @ res.fock @ Cv)
    de = (ev[:, None] - eo[None, :]).ravel()
    apply = _response(L, S, xi)
    nv = Cv.shape[1]
    for name in ("singlet", "triplet"):
        H = np.empty((nv * n_occ, nv * n_occ))
        for j in range(nv * n_occ):
            e = np.zeros(nv * n_occ)
            e[j] = 1.0
            _, J, K = apply(Cv @ e.reshape(nv, n_occ), Co,
                            coulomb=name == "singlet")
            G = (2.0 * J - K) if name == "singlet" else -K
            H[:, j] = de * e + (Cv.T @ G @ Co).ravel()
        assert np.max(np.abs(H - H.T)) < 1e-12
        lowest = np.linalg.eigvalsh(0.5 * (H + H.T))[0]
        assert st.sector(name).lowest_eigenvalue == pytest.approx(lowest, abs=1e-10)

    # d^2E/dtheta^2 = 4 lambda along the unit singlet mode.
    energy = _energy(L, S, res.hcore, xi, res.e_nuclear)
    e0 = energy(res.density)
    assert e0 == pytest.approx(res.energy, abs=1e-12)
    u, s, vt = np.linalg.svd(st.sector("singlet").mode, full_matrices=False)
    theta = 1e-3
    side = []
    for sign in (1.0, -1.0):
        a = sign * theta * s
        occ = Co + (Co @ vt.T * (np.cos(a) - 1.0)) @ vt + (Cv @ u * np.sin(a)) @ vt
        side.append(energy(2.0 * occ @ occ.T))
    curvature = (side[0] + side[1] - 2.0 * e0) / theta ** 2
    assert curvature / 4.0 == pytest.approx(
        st.sector("singlet").lowest_eigenvalue, rel=1e-4)


def test_translation_invariant_rotations():
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    res = _quiet(run_ccm_rhf_direct, ccm, **TIGHT)
    full = ccm_rhf_stability(res, ccm, sectors=("singlet",))
    ti = ccm_rhf_stability(res, ccm, sectors=("singlet",),
                           translation_invariant=True)
    # The lowest mode of the H2 cube breaks the torus translation; the
    # translation-invariant one lies above it.
    assert full.sector("singlet").translation_invariant_fraction < 1e-12
    assert ti.sector("singlet").translation_invariant_fraction == pytest.approx(1.0, abs=1e-12)
    assert ti.sector("singlet").restricted_to_translation_invariant
    assert ti.sector("singlet").lowest_eigenvalue > full.sector("singlet").lowest_eigenvalue
    P = circulant_projection(ccm.nrep, res.density)
    assert np.max(np.abs(P - res.density)) < 1e-12
    M = np.random.default_rng(3).standard_normal(res.density.shape)
    PM = circulant_projection(ccm.nrep, M)
    np.testing.assert_allclose(circulant_projection(ccm.nrep, PM), PM, atol=1e-14)


def test_the_verdict_reaches_the_molecular_vocabulary():
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    res = _quiet(run_ccm_rhf_direct, ccm, stability_check=True, **TIGHT)
    assert res.stability_checked and res.stability_analysis_converged
    assert not res.internal_instability
    assert res.stability_eigenvalue == pytest.approx(
        res.stability.lowest_eigenvalue, abs=0.0)
    verdict = verdict_from_result(res)
    assert verdict.stability == "verified_stable" and verdict.usable
    assert res.instability_follow == ()

    # A stretched equidistant H chain: RHF -> UHF (triplet) unstable, the
    # singlet sector stable, so the union verdict is unstable and nothing is
    # followed.
    chain = CCMSystem(_h_chain(4.0), (1, 1, 1), "sto-3g")
    r = _quiet(run_ccm_rhf_direct, chain, follow_instability=True,
               allow_vacuum_padding=True, ke_cutoff=80.0, **TIGHT)
    assert r.stability.sector("triplet").lowest_eigenvalue < -STABILITY_TOL
    assert r.stability.sector("singlet").lowest_eigenvalue > 0.0
    assert r.internal_instability and r.instability_follow == ()
    assert verdict_from_result(r).stability == "unstable"


def test_the_result_orbitals_belong_to_the_returned_fock():
    """An oscillating start used to leak the DIIS trial's spectrum.

    From the doubly occupied antibonding orbital of a stretched H chain the
    loop flips to the bonding state; the iterates oscillate and the DIIS
    trial Fock shared the physical eigenvectors but not its eigenvalues
    (0.0095 and 0.0298 Ha against the physical -0.516 and 0.430 Ha). The
    returned orbitals now span the returned density and diagonalise the
    returned Fock within its occupied and virtual spaces."""
    from scipy.linalg import eigh

    chain = CCMSystem(_h_chain(3.0), (1, 1, 1), "sto-3g")
    kw = dict(allow_vacuum_padding=True, ke_cutoff=80.0, **TIGHT)
    g = _quiet(run_ccm_rhf_direct, chain, **kw)
    Du = 2.0 * np.outer(g.mo_coeffs[:, 1], g.mo_coeffs[:, 1])
    u = _quiet(run_ccm_rhf_direct, chain, initial_guess="READ",
               initial_density=Du, cderi=g.cderi, stability_check=True, **kw)
    assert u.converged and u.energy == pytest.approx(g.energy, abs=1e-10)
    for r in (g, u):
        np.testing.assert_allclose(r.mo_energies, eigh(r.fock, r.overlap)[0],
                                   atol=1e-9)
        residual = r.fock @ r.mo_coeffs - (r.overlap @ r.mo_coeffs) * r.mo_energies[None, :]
        assert np.max(np.abs(residual)) < 1e-8
        Co = r.mo_coeffs[:, :1]
        np.testing.assert_allclose(r.density, 2.0 * Co @ Co.T, atol=1e-12)
    np.testing.assert_allclose(u.mo_energies, g.mo_energies, atol=1e-8)
    st = ccm_rhf_stability(g, chain)
    assert u.stability.sector("singlet").lowest_eigenvalue == pytest.approx(
        st.sector("singlet").lowest_eigenvalue, abs=1e-8)
    Co, eo, Cv, ev = semicanonical_orbitals(u.fock, u.density, u.overlap, 1)
    np.testing.assert_allclose(2.0 * Co @ Co.T, u.density, atol=1e-10)


def test_sad_is_explicit_only():
    assert InitialGuess.SAD not in periodic_guess_capabilities("aiccm-real-gamma")
    assert InitialGuess.SAD not in periodic_guess_capabilities(
        "aiccm-real-gamma", requested="AUTO")
    assert InitialGuess.SAD in periodic_guess_capabilities(
        "aiccm-real-gamma", requested="SAD")
    # Other routes are unchanged by the request.
    assert periodic_guess_capabilities("aiccm-hcore", requested="SAD") == {InitialGuess.HCORE}
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    hc = _quiet(run_ccm_rhf_direct, ccm, **TIGHT)
    assert hc.guess_selection.requested == InitialGuess.AUTO
    assert hc.guess_selection.transport == InitialGuess.HCORE
    sad = _quiet(run_ccm_rhf_direct, ccm, initial_guess="SAD", cderi=hc.cderi, **TIGHT)
    assert sad.guess_selection.transport == InitialGuess.SAD
    assert sad.energy == pytest.approx(hc.energy, abs=1e-10)
    # The first trace row of a seeded loop carries no HCORE frontier gap.
    assert np.isnan(sad.scf_trace[0].homo_lumo)
    arm = _quiet(run_real_gamma_scf, _h2_cube(), "sto-3g", "RHF", (2, 1, 1),
                 initial_guess="SAD")
    assert arm.ccm_result.guess_selection.transport == InitialGuess.SAD
    # The arm runs at its own (looser) default tolerances.
    assert arm.energy == pytest.approx(hc.energy / ccm.n_cells, abs=1e-7)
    assert arm.real_gamma.frontier_gap == pytest.approx(
        hc.mo_energies[2] - hc.mo_energies[1], abs=1e-5)


@pytest.mark.slow
def test_sad_on_the_open_shell_driver():
    li = CCMSystem(_li_doublet(), (1, 1, 1), "sto-3g")
    # A coarse fit keeps the row fast; both seeds share it.
    hc = _quiet(run_ccm_uhf_direct, li, ke_cutoff=80.0, **TIGHT)
    sad = _quiet(run_ccm_uhf_direct, li, initial_guess="SAD", cderi=hc.cderi, **TIGHT)
    assert sad.guess_selection.transport == InitialGuess.SAD
    assert sad.energy == pytest.approx(hc.energy, abs=1e-9)
    # <S^2> of the accepted densities.
    S = sad.overlap
    sz = 0.5 * (sad.n_alpha - sad.n_beta)
    expected = sz * (sz + 1.0) + sad.n_beta - np.trace(
        sad.density_alpha @ S @ sad.density_beta @ S)
    assert sad.s_squared == pytest.approx(expected, abs=1e-12)


@pytest.mark.slow
def test_sad_on_the_ks_driver():
    h2 = CCMSystem(_h2_cube(), (1, 1, 1), "sto-3g")
    k_hc = _quiet(run_ccm_rks_direct, h2, "pbe", **TIGHT)
    k_sad = _quiet(run_ccm_rks_direct, h2, "pbe", initial_guess="SAD",
                   cderi=k_hc.cderi, **TIGHT)
    assert k_sad.guess_selection.transport == InitialGuess.SAD
    assert k_sad.energy == pytest.approx(k_hc.energy, abs=1e-9)


def test_refusals():
    ccm = CCMSystem(_h2_cube(), (1, 1, 1), "sto-3g")
    capped = _quiet(run_ccm_rhf_direct, ccm, max_iter=1, stability_check=True)
    assert not capped.converged and capped.stability_checked is False
    assert capped.stability is None
    with pytest.raises(ValueError, match="stability_tol"):
        run_ccm_rhf_direct(ccm, stability_check=True, stability_tol=-1.0)
    with pytest.raises(ValueError, match="max_instability_follow"):
        run_ccm_rhf_direct(ccm, follow_instability=True,
                           max_instability_follow=-1)
    with pytest.raises(ValueError, match="did not converge"):
        ccm_rhf_stability(capped, ccm)
    with pytest.raises(NotImplementedError, match="stability check covers RHF"):
        _quiet(run_real_gamma_scf, _h2_cube(), "sto-3g", "RKS", (1, 1, 1),
               stability_check=True)
    with pytest.raises(NotImplementedError, match="initial_guess"):
        _quiet(run_real_gamma_scf, _h2_cube(), "sto-3g", "RHF", (1, 1, 1),
               initial_guess="SAP")


@pytest.mark.slow
def test_following_the_bond_alternation_of_a_carbon_chain():
    """Equidistant linear carbon chain, two atoms per cell, five cells.

    The HCORE solution is uniform and a translation-invariant singlet saddle
    (lowest eigenvalue measured -0.0809 Ha); one step along its mode and a
    re-converged SCF lower the energy by 7.8 mHa per cell to a singlet-stable
    state. The triplet sector stays unstable, so the union verdict does too.
    """
    d, lateral = 2.42, 7.0
    cell = PeriodicSystem(
        3, np.diag([2.0 * d, lateral, lateral]),
        [Atom(6, [0.5 * d, 0.5 * lateral, 0.5 * lateral]),
         Atom(6, [1.5 * d, 0.5 * lateral, 0.5 * lateral])], 0, 1)
    ccm = CCMSystem(cell, (5, 1, 1), "sto-3g")
    kw = dict(conv_tol=1e-11, conv_tol_grad=1e-8, max_iter=300, ke_cutoff=80.0,
              allow_vacuum_padding=True)
    uniform = _quiet(run_ccm_rhf_direct, ccm, stability_check=True, **kw)
    singlet = uniform.stability.sector("singlet")
    assert singlet.lowest_eigenvalue < -0.05
    assert singlet.translation_invariant_fraction == pytest.approx(1.0, abs=1e-8)
    followed = _quiet(run_ccm_rhf_direct, ccm, cderi=uniform.cderi,
                      follow_instability=True, **kw)
    assert followed.converged
    drop = (followed.energy - uniform.energy) / ccm.n_cells
    assert drop < -5e-3
    assert followed.stability.sector("singlet").lowest_eigenvalue > -STABILITY_TOL
    (step,) = followed.instability_follow
    assert step["outcome"] == "restarted"
    assert step["energy_rotated"] < step["energy_before"]
    assert step["energy_after"] == pytest.approx(followed.energy, abs=0.0)
    # The followed state keeps the torus translation symmetry.
    P = circulant_projection(ccm.nrep, followed.density)
    assert np.max(np.abs(P - followed.density)) < 1e-8
