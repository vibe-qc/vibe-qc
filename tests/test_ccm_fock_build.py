"""The evaluated state of the real-Gamma producer (#472, #519 RG-3).

The closed-shell real-Gamma drivers freeze one :class:`~vibeqc.fock_state.FockBuild`
per density and report the terminal one: the physical Fock, the density it was
built from, the energy components, the neutral-source and frame identities.
These rows are the acceptance the #472 contract asks of a migrated producer:
the components sum to the energy the SCF reports; the frozen Fock is the
derivative of an independently assembled energy functional at that density;
the build owns its arrays (mutating the result does not change it); a run
stopped at the iteration cap still reports an evaluated state, not a DIIS
trial; and the runner manifest carries the identities.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.fock_state import FockBuild
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]


def _h2_cube(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module")
def rhf():
    ccm = _h2_cube()
    return ccm, _quiet(run_ccm_rhf_direct, ccm, conv_tol=1e-11, conv_tol_grad=1e-8)


def _rhf_exchange_with_seam(ccm, res):
    """``K(D) + xi_N S D S`` through the unfactored contraction, so it accepts
    the indefinite trial densities of a finite-difference derivative."""
    L, S = res.cderi, res.overlap
    xi = ccm_exchange_q0_madelung(ccm)

    def exchange(D):
        return ccm_ri_k_neutral(L, D) + xi * (S @ D @ S)
    return exchange


def _rhf_energy_functional(ccm, res):
    """E[D] of the neutral torus, assembled independently of the builder."""
    h, e_nn = res.hcore, res.e_nuclear
    exchange = _rhf_exchange_with_seam(ccm, res)

    def energy(D):
        J = ccm_ri_j_neutral(res.cderi, D)
        return (float(np.sum(D * h)) + 0.5 * float(np.sum(D * J))
                - 0.25 * float(np.sum(D * exchange(D))) + e_nn)
    return energy


def test_terminal_build_is_the_reported_state(rhf):
    ccm, res = rhf
    build = res.fock_build
    assert isinstance(build, FockBuild) and res.converged
    assert build.method == "ccm-neutral-real-gamma-rhf"
    assert build.generation == res.n_iter - 1
    assert build.energy.total == pytest.approx(res.energy, abs=1e-12)
    assert build.energy.nuclear == pytest.approx(res.e_nuclear, abs=0.0)
    parts = dict(build.energy.components)
    assert set(parts) == {"hcore", "coulomb", "exchange"}
    assert sum(parts.values()) == pytest.approx(res.e_electronic, abs=1e-10)
    assert build.source.identity == res.neutral_source.identity
    (channel,) = build.channels
    np.testing.assert_array_equal(channel.density, res.density)
    np.testing.assert_array_equal(channel.operator.matrix, res.fock)
    assert channel.operator.metadata.role == "physical"
    assert channel.operator.metadata.kind == "fock"
    assert channel.operator.frame.retained_rank <= channel.operator.n
    np.testing.assert_allclose(channel.operator.overlap, res.overlap, atol=0.0)
    assert build.metric_electron_counts[0] == pytest.approx(
        ccm.supercell.n_electrons(), abs=1e-9)
    assert build.commutator_norm < 1e-6


def test_frozen_fock_is_the_energy_derivative(rhf):
    ccm, res = rhf
    energy = _rhf_energy_functional(ccm, res)
    (channel,) = res.fock_build.channels
    D, F = channel.density, channel.operator.matrix
    assert energy(D) == pytest.approx(res.fock_build.energy.total, abs=1e-10)
    rng = np.random.default_rng(519)
    delta = rng.normal(size=D.shape)
    delta = 0.05 * (delta + delta.T)
    step = 1e-5
    numerical = (energy(D + step * delta) - energy(D - step * delta)) / (2.0 * step)
    assert numerical == pytest.approx(float(np.sum(F * delta)), abs=2e-9)


def test_build_owns_its_arrays(rhf):
    ccm, res = rhf
    (channel,) = res.fock_build.channels
    density_before = channel.density.copy()
    fock_before = channel.operator.matrix.copy()
    assert not channel.density.flags.writeable
    assert not channel.operator.matrix.flags.writeable
    res.density[0, 0] += 1.0
    res.fock[0, 0] += 1.0
    np.testing.assert_array_equal(channel.density, density_before)
    np.testing.assert_array_equal(channel.operator.matrix, fock_before)
    res.density[0, 0] -= 1.0
    res.fock[0, 0] -= 1.0


def test_iteration_cap_reports_an_evaluated_state(rhf):
    """A run that stops at max_iter reports the last state it evaluated:
    F equals the Fock rebuilt from the reported density, and the energy is
    that state's, never a DIIS trial with the density it would produce next.

    In a minimal basis the H2 occupied orbital is fixed by symmetry, so the
    HCORE seed is already the solution; the run starts from the converged
    orbitals rotated into the virtual space instead, at tolerances no SCF
    meets in three cycles."""
    ccm, converged = rhf
    n_occ = ccm.supercell.n_electrons() // 2
    C = np.array(converged.mo_coeffs)
    theta = 0.3
    rotation = np.eye(C.shape[1])
    rotation[0, 0] = rotation[n_occ, n_occ] = np.cos(theta)
    rotation[0, n_occ] = -np.sin(theta)
    rotation[n_occ, 0] = np.sin(theta)
    C_rot = C @ rotation
    D_rot = 2.0 * C_rot[:, :n_occ] @ C_rot[:, :n_occ].T
    res = _quiet(run_ccm_rhf_direct, ccm, max_iter=3, conv_tol=1e-15,
                 conv_tol_grad=1e-13, initial_guess="READ",
                 initial_density=D_rot)
    assert not res.converged and res.n_iter == 3
    build = res.fock_build
    assert build.generation == 2
    assert build.energy.total == pytest.approx(res.energy, abs=1e-12)
    assert res.energy > converged.energy - 1e-12
    D = res.density
    F = (res.hcore + ccm_ri_j_neutral(res.cderi, D)
         - 0.5 * _rhf_exchange_with_seam(ccm, res)(D))
    np.testing.assert_allclose(res.fock, 0.5 * (F + F.T), atol=1e-11)
    C_last = res.mo_coeffs[:, :n_occ]
    np.testing.assert_allclose(D, 2.0 * C_last @ C_last.T, atol=1e-12)


def test_ks_build_carries_the_xc_component():
    ccm = _h2_cube()
    res = _quiet(run_ccm_rks_direct, ccm, "pbe")
    build = res.fock_build
    assert res.converged and build.method == "ccm-neutral-real-gamma-rks"
    (channel,) = build.channels
    assert channel.operator.metadata.kind == "ks"
    parts = dict(build.energy.components)
    assert set(parts) == {"hcore", "coulomb", "xc"}
    assert parts["xc"] == pytest.approx(res.e_xc, abs=0.0)
    assert parts["coulomb"] == pytest.approx(res.e_coulomb, abs=0.0)
    assert build.energy.total == pytest.approx(res.energy, abs=1e-12)
    assert build.source.identity == res.neutral_source.identity
    np.testing.assert_array_equal(channel.operator.matrix, res.fock)
    assert build.commutator_norm < 1e-5


def test_runner_manifest_carries_the_build_identities(tmp_path):
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = _h2_cube().unit_system
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    rg = _quiet(run_periodic_job, cell, basis, method="aiccm",
                variant="real-gamma", aiccm_lattice_extension=(2, 1, 1),
                initial_guess="HCORE", output=str(tmp_path / "rg"),
                progress=False)
    assert rg.converged
    build = rg.fock_build
    assert isinstance(build, FockBuild)
    text = (tmp_path / "rg.system").read_text(encoding="utf-8")
    for key, value in (("fock_build_id", build.identity),
                       ("evaluated_density_id", build.density_id),
                       ("fock_source_id", build.source.identity)):
        assert re.search(rf'{key}\s*=\s*"{value}"', text), key
    assert 'fock_operator_role = "physical"' in text
    assert "fock_build_generation" in text
    assert "evaluated_metric_electrons" in text


# ---------------------------------------------------------------------------
# Open shell (#519, RG-4): two channels, alpha and beta, one build
# ---------------------------------------------------------------------------

def _li_doublet(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)
    return CCMSystem(cell, nrep, "sto-3g")


@pytest.fixture(scope="module")
def uhf():
    ccm = _li_doublet()
    return ccm, _quiet(run_ccm_uhf_direct, ccm, conv_tol=1e-12,
                       conv_tol_grad=1e-9, max_iter=300)


def _uhf_energy_functional(ccm, res):
    """E[Da, Db] of the neutral torus with the unfactored per-spin exchange.

    The UHF result carries no ``hcore``; the one-electron fold is rebuilt."""
    from vibeqc.periodic.ccm.direct import ccm_direct_oneelectron

    L, S, e_nn = res.cderi, res.overlap, res.e_nuclear
    _, h, _ = ccm_direct_oneelectron(ccm)
    xi = ccm_exchange_q0_madelung(ccm)

    def exchange(D):
        return ccm_ri_k_neutral(L, D) + xi * (S @ D @ S)

    def energy(Da, Db):
        dt = Da + Db
        J = ccm_ri_j_neutral(L, dt)
        return (float(np.sum(dt * h)) + 0.5 * float(np.sum(dt * J))
                - 0.5 * (float(np.sum(Da * exchange(Da)))
                         + float(np.sum(Db * exchange(Db)))) + e_nn)
    return energy


def test_uhf_terminal_build_is_the_reported_state(uhf):
    ccm, res = uhf
    build = res.fock_build
    assert isinstance(build, FockBuild) and res.converged
    assert build.method == "ccm-neutral-real-gamma-uhf"
    assert build.energy.total == pytest.approx(res.energy, abs=1e-12)
    parts = dict(build.energy.components)
    assert set(parts) == {"hcore", "coulomb", "exchange"}
    assert sum(parts.values()) == pytest.approx(res.e_electronic, abs=1e-10)
    assert build.source.identity == res.neutral_source.identity
    a, b = build.channels
    assert (a.operator.metadata.spin, b.operator.metadata.spin) == ("alpha", "beta")
    np.testing.assert_array_equal(a.density, res.density_alpha)
    np.testing.assert_array_equal(b.density, res.density_beta)
    assert a.operator.frame is b.operator.frame
    assert build.metric_electron_counts == pytest.approx(
        (res.n_alpha, res.n_beta), abs=1e-9)
    assert build.commutator_norm < 1e-7
    # The reported orbitals define the reported densities (no DIIS step ahead).
    ca, cb = res.mo_coeffs_alpha[:, :res.n_alpha], res.mo_coeffs_beta[:, :res.n_beta]
    np.testing.assert_allclose(res.density_alpha, ca @ ca.T, atol=1e-12)
    np.testing.assert_allclose(res.density_beta, cb @ cb.T, atol=1e-12)


def test_uhf_frozen_focks_are_the_spin_derivatives(uhf):
    ccm, res = uhf
    energy = _uhf_energy_functional(ccm, res)
    a, b = res.fock_build.channels
    Da, Db = a.density, b.density
    assert energy(Da, Db) == pytest.approx(res.fock_build.energy.total, abs=1e-10)
    rng = np.random.default_rng(4)
    deltas = []
    for _ in range(2):
        d = rng.normal(size=Da.shape)
        deltas.append(0.05 * (d + d.T))
    step = 1e-5
    numerical = (energy(Da + step * deltas[0], Db + step * deltas[1])
                 - energy(Da - step * deltas[0], Db - step * deltas[1])) / (2.0 * step)
    expected = (float(np.sum(a.operator.matrix * deltas[0]))
                + float(np.sum(b.operator.matrix * deltas[1])))
    assert numerical == pytest.approx(expected, abs=2e-9)


def test_uhf_iteration_cap_reports_an_evaluated_state():
    ccm = _li_doublet()
    res = _quiet(run_ccm_uhf_direct, ccm, max_iter=3, conv_tol=1e-15,
                 conv_tol_grad=1e-13)
    assert not res.converged and res.n_iter == 3
    build = res.fock_build
    assert build.generation == 2
    assert build.energy.total == pytest.approx(res.energy, abs=1e-12)
    a, b = build.channels
    np.testing.assert_array_equal(a.density, res.density_alpha)
    np.testing.assert_array_equal(b.density, res.density_beta)
    ca = res.mo_coeffs_alpha[:, :res.n_alpha]
    np.testing.assert_allclose(res.density_alpha, ca @ ca.T, atol=1e-12)


def test_uks_build_carries_the_xc_component():
    ccm = _li_doublet()
    res = _quiet(run_ccm_uks_direct, ccm, "pbe0", conv_tol=1e-11,
                 conv_tol_grad=1e-8, max_iter=300)
    build = res.fock_build
    assert res.converged and build.method == "ccm-neutral-real-gamma-uks"
    parts = dict(build.energy.components)
    assert set(parts) == {"hcore", "coulomb", "exchange", "xc"}
    assert parts["xc"] == pytest.approx(res.e_xc, abs=0.0)
    assert parts["exchange"] == pytest.approx(res.e_hf_exchange, abs=0.0)
    assert build.energy.total == pytest.approx(res.energy, abs=1e-12)
    a, b = build.channels
    assert a.operator.metadata.kind == "ks"
    np.testing.assert_array_equal(a.operator.matrix, res.fock)
    np.testing.assert_array_equal(b.operator.matrix, res.fock_beta)
    assert build.source.identity == res.neutral_source.identity


def test_runner_manifest_carries_the_open_shell_build(tmp_path):
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = _li_doublet().unit_system
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    rg = _quiet(run_periodic_job, cell, basis, method="aiccm",
                variant="real-gamma", aiccm_lattice_extension=(2, 1, 1),
                scf_reference="uhf", initial_guess="HCORE",
                output=str(tmp_path / "rg"), progress=False)
    assert rg.converged
    build = rg.fock_build
    assert isinstance(build, FockBuild) and len(build.channels) == 2
    text = (tmp_path / "rg.system").read_text(encoding="utf-8")
    assert re.search(rf'fock_build_id\s*=\s*"{build.identity}"', text)
    assert "evaluated_metric_electrons" in text
