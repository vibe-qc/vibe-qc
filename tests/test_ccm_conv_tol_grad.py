"""IIDs 295/360: CCM SCF commutator tolerances are exposed and validated.

``conv_tol`` gates only the energy half of the CCM exit condition; the DIIS
commutator residual bound used to be a hard-coded ``1e-6`` literal that no
argument could tighten (``scf.run_ccm_rhf``, ``ri._rhf_loop`` / ``ri._rks_loop``,
and their wrappers). These tests pin that the gate honors a caller-supplied
``conv_tol_grad`` and that the literal no longer appears in the closed-shell
exit conditions.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc.periodic.ccm.dft import run_ccm_rks, run_ccm_uks
from vibeqc.periodic.ccm.direct import (
    run_ccm_rhf_direct,
    run_ccm_rhf_direct_rijcosx,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.lowd_scf import run_ccm_rhf_wire
from vibeqc.periodic.ccm.ri import (
    _rhf_loop,
    _rks_loop,
    run_ccm_rhf_gdf,
    run_ccm_rhf_ri_neutral,
    run_ccm_rhf_rij,
    run_ccm_rhf_rijcosx,
    run_ccm_rks_gdf,
    run_ccm_uhf_gdf,
    run_ccm_uks_gdf,
)
from vibeqc.periodic.ccm.scf import (
    _commutator_exit_ok,
    run_ccm_rhf,
    run_ccm_rhf_scalable,
)
from vibeqc.periodic.ccm.uhf import run_ccm_uhf

pytestmark = pytest.mark.experimental  # Γ-CCM research lane

_CCM_ROOT = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "vibeqc"
    / "periodic"
    / "ccm"
)

# The two files that used to carry the literal gate in their closed-shell
# loops.  A future edit that reintroduces the literal must consciously touch
# this pin.
_LITERAL_GATE_FILES = (
    _CCM_ROOT / "scf.py",
    _CCM_ROOT / "ri.py",
)

_PUBLIC_ROUTES = (
    run_ccm_rhf,
    run_ccm_rhf_scalable,
    run_ccm_rhf_ri_neutral,
    run_ccm_rhf_rij,
    run_ccm_rhf_rijcosx,
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_rhf_direct_rijcosx,
    run_ccm_rks,
    run_ccm_uks,
    run_ccm_rhf_wire,
    run_ccm_uhf,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
    run_ccm_rhf_gdf,
    run_ccm_rks_gdf,
    run_ccm_uhf_gdf,
    run_ccm_uks_gdf,
)

_GDF_ROUTES = (
    run_ccm_rhf_gdf,
    run_ccm_rks_gdf,
    run_ccm_uhf_gdf,
    run_ccm_uks_gdf,
)

_INVALID_TOLERANCES = (-1.0, 0.0, float("nan"), float("inf"))


def _fake_ccm(n_atoms: int = 2, n_electrons: int = 2, charge: int = 0):
    """Minimal stand-in for the shared CCM SCF-loop contract."""
    ccm = SimpleNamespace()
    ccm.n_atoms = n_atoms
    ccm.unit_system = SimpleNamespace(charge=charge)
    ccm.supercell = SimpleNamespace(
        n_electrons=lambda: n_electrons,
        multiplicity=1,
    )
    return ccm


def _fake_gdf_ccm():
    """Compact 3-D CCM stand-in that truthfully passes the IID 291 audit."""
    molecule = SimpleNamespace(
        atoms=(SimpleNamespace(xyz=np.zeros(3)),),
    )
    unit = SimpleNamespace(
        charge=0,
        dim=3,
        lattice=6.0 * np.eye(3),
        unit_cell_molecule=lambda: molecule,
    )
    return SimpleNamespace(
        unit_system=unit,
        basis_name="sto-3g",
        nrep=(1, 1, 1),
        n_atoms=1,
        n_cells=1,
    )


def test_exit_predicate_passes_when_both_criteria_hold():
    assert _commutator_exit_ok(1e-10, np.array([1e-7]), conv_tol=1e-9,
                               conv_tol_grad=1e-6)


def test_exit_predicate_rejects_energy_above_conv_tol():
    assert not _commutator_exit_ok(1e-8, np.array([1e-10]), conv_tol=1e-9,
                                   conv_tol_grad=1e-6)


def test_exit_predicate_rejects_residual_above_conv_tol_grad():
    # This is the case the old literal gate could not express: the caller
    # wants a tight density criterion and the residual violates it, even
    # though the energy criterion passes.
    assert not _commutator_exit_ok(1e-12, np.array([1e-8]), conv_tol=1e-9,
                                   conv_tol_grad=1e-10)


def test_exit_predicate_is_strict_at_the_bound():
    # Strict inequalities: a residual exactly at the bound must not pass.
    assert not _commutator_exit_ok(0.0, np.array([1e-6]), conv_tol=1e-9,
                                   conv_tol_grad=1e-6)


def test_rhf_loop_accepts_conv_tol_grad_and_converges_on_fixed_point():
    """The loop accepts ``conv_tol_grad`` and reaches a true fixed point.

    ``j = k = 0`` makes ``F = h`` constant, so the first diagonalisation is
    already the fixed point and the commutator residual is exactly zero.
    A residual of zero passes any bound, including a much tighter one than
    the historical 1e-6.
    """
    ccm = _fake_ccm()
    S = np.eye(2)
    h = np.diag([-1.0, 0.5])
    e_nn = 0.0
    res = _rhf_loop(
        ccm, S, h, e_nn,
        lambda D: np.zeros((2, 2)),
        lambda D: np.zeros((2, 2)),
        max_iter=25, conv_tol=1e-12, conv_tol_grad=1e-14, diis_dim=8,
        lindep_tol=1e-8,
    )
    assert res.converged
    assert res.n_iter == 2


def test_rhf_loop_neutral_positive_energy_guard_remains_active():
    """The minimal loop contract must not bypass the IID 291 sanity guard."""
    ccm = _fake_ccm()
    with pytest.raises(ValueError, match="positive total energy"):
        _rhf_loop(
            ccm,
            np.eye(2),
            np.diag([0.25, 0.5]),
            0.0,
            lambda density: np.zeros((2, 2)),
            lambda density: np.zeros((2, 2)),
            max_iter=25,
            conv_tol=1e-12,
            conv_tol_grad=1e-14,
            diis_dim=8,
            lindep_tol=1e-8,
        )


def test_rks_loop_terminal_result_is_one_physical_density_generation():
    """A max-iteration exit must not return the next DIIS trial density.

    The physical Fock and every energy component are evaluated on the density
    accepted at the start of an iteration.  Even when that Fock rotates the
    orbitals strongly, ``max_iter=1`` must retain that accepted density while
    reporting canonical eigenvectors of the corresponding physical Fock.
    """
    ccm = _fake_ccm()
    overlap = np.eye(2)
    hcore = np.diag([-1.0, 0.5])
    initial_density = np.diag([2.0, 0.0])

    def j_of_d(density):
        off_diagonal = 0.10 * density[0, 0] + 0.05 * density[0, 1]
        return np.array(
            [
                [0.20 * density[0, 0], off_diagonal],
                [off_diagonal, 0.10 * density[1, 1]],
            ]
        )

    def k_of_d(density):
        return 0.12 * density

    def xc_of_d(density):
        return 0.03 * float(np.sum(density**2)), 0.06 * density

    alpha = 0.25
    e_nuclear = 0.2
    result = _rks_loop(
        ccm,
        overlap,
        hcore,
        e_nuclear,
        j_of_d,
        k_of_d,
        xc_of_d,
        alpha,
        "synthetic",
        max_iter=1,
        conv_tol=1e-12,
        conv_tol_grad=1e-12,
        diis_dim=8,
        lindep_tol=1e-12,
    )

    np.testing.assert_allclose(result.density, initial_density, atol=1e-14)
    J = j_of_d(result.density)
    K = k_of_d(result.density)
    e_xc, V_xc = xc_of_d(result.density)
    physical_fock = hcore + J + V_xc - 0.5 * alpha * K
    physical_fock = 0.5 * (physical_fock + physical_fock.T)
    e_coulomb = 0.5 * float(np.sum(result.density * J))
    e_exchange = -0.25 * alpha * float(np.sum(result.density * K))
    energy = (
        float(np.sum(result.density * hcore))
        + e_coulomb
        + e_exchange
        + e_xc
        + e_nuclear
    )
    np.testing.assert_allclose(result.fock, physical_fock, atol=1e-14)
    assert result.e_coulomb == pytest.approx(e_coulomb, abs=1e-14)
    assert result.e_hf_exchange == pytest.approx(e_exchange, abs=1e-14)
    assert result.e_xc == pytest.approx(e_xc, abs=1e-14)
    assert result.energy == pytest.approx(energy, abs=1e-14)
    residual = (
        result.fock @ result.mo_coeffs
        - (result.overlap @ result.mo_coeffs) * result.mo_energies[None, :]
    )
    assert np.max(np.abs(residual)) < 1e-12


def _run_contracting_rhf(*, conv_tol_grad=None, max_iter=60):
    """Two-orbital fixed-point iteration with a slowly shrinking residual."""
    ccm = _fake_ccm()
    overlap = np.eye(2)
    hcore = np.diag([-1.0, 0.5])

    def j_of_d(density):
        off_diagonal = 0.2 + 0.5 * density[0, 1]
        return np.array([[0.0, off_diagonal], [off_diagonal, 0.0]])

    kwargs = {
        "max_iter": max_iter,
        "conv_tol": 1.0,
        "diis_dim": 1,
        "lindep_tol": 1e-12,
    }
    if conv_tol_grad is not None:
        kwargs["conv_tol_grad"] = conv_tol_grad
    return _rhf_loop(
        ccm,
        overlap,
        hcore,
        0.0,
        j_of_d,
        lambda density: np.zeros((2, 2)),
        **kwargs,
    )


def test_tightened_threshold_changes_numerical_stopping_semantics():
    implicit_default = _run_contracting_rhf()
    explicit_default = _run_contracting_rhf(conv_tol_grad=1e-6)
    tightened = _run_contracting_rhf(conv_tol_grad=1e-10)

    assert implicit_default.converged
    assert explicit_default.converged
    assert tightened.converged
    assert implicit_default.n_iter == explicit_default.n_iter == 31
    assert tightened.n_iter == 51
    assert implicit_default.energy == explicit_default.energy

    assert _run_contracting_rhf(conv_tol_grad=1e-6, max_iter=40).converged
    assert not _run_contracting_rhf(
        conv_tol_grad=1e-10, max_iter=40
    ).converged


def test_uhf_default_and_explicit_positive_tolerance_match(monkeypatch):
    """Validation preserves the historical default and positive numerics."""
    from vibeqc.periodic.ccm import uhf as uhf_module

    ccm = _fake_ccm()
    overlap = np.eye(2)
    hcore = np.diag([-1.0, 0.5])
    eri = np.zeros((2, 2, 2, 2))
    monkeypatch.setattr(uhf_module, "ccm_overlap", lambda _: overlap)
    monkeypatch.setattr(
        uhf_module,
        "ccm_hcore",
        lambda _: (hcore, np.zeros_like(hcore), np.zeros_like(hcore)),
    )
    monkeypatch.setattr(uhf_module, "ccm_nuclear_repulsion", lambda _: 0.0)
    monkeypatch.setattr(uhf_module, "_warn_experimental", lambda: None)

    implicit = run_ccm_uhf(ccm, eri=eri, max_iter=4, diis_dim=1)
    explicit = run_ccm_uhf(
        ccm, eri=eri, max_iter=4, diis_dim=1, conv_tol_grad=1e-6
    )
    tightened = run_ccm_uhf(
        ccm, eri=eri, max_iter=4, diis_dim=1, conv_tol_grad=1e-10
    )

    assert implicit.converged and explicit.converged and tightened.converged
    assert implicit.n_iter == explicit.n_iter == tightened.n_iter == 2
    assert implicit.energy == explicit.energy == tightened.energy


@pytest.mark.parametrize("route", _PUBLIC_ROUTES, ids=lambda route: route.__name__)
@pytest.mark.parametrize(
    "conv_tol_grad",
    _INVALID_TOLERANCES,
    ids=("negative", "zero", "nan", "positive-infinity"),
)
def test_public_routes_reject_invalid_conv_tol_grad(route, conv_tol_grad):
    with pytest.raises(
        ValueError,
        match=rf"{route.__name__}: conv_tol_grad must be finite and positive\.",
    ):
        route(None, conv_tol_grad=conv_tol_grad)


@pytest.mark.parametrize("route", _GDF_ROUTES, ids=lambda route: route.__name__)
@pytest.mark.parametrize(
    "conv_tol_grad",
    _INVALID_TOLERANCES,
    ids=("negative", "zero", "nan", "positive-infinity"),
)
def test_gdf_options_reject_invalid_conv_tol_grad(route, conv_tol_grad):
    options = SimpleNamespace(conv_tol_grad=conv_tol_grad)
    with pytest.raises(
        ValueError,
        match=rf"{route.__name__}: conv_tol_grad must be finite and positive\.",
    ):
        route(None, options=options)


@pytest.mark.parametrize(
    ("route", "driver_name"),
    (
        (run_ccm_rhf_gdf, "run_krhf_periodic_gdf"),
        (run_ccm_rks_gdf, "run_krks_periodic_gdf"),
        (run_ccm_uhf_gdf, "run_kuhf_periodic_gdf"),
        (run_ccm_uks_gdf, "run_kuks_periodic_gdf"),
    ),
    ids=lambda value: value.__name__ if callable(value) else value,
)
def test_gdf_explicit_positive_tolerance_remains_unsupported(
    monkeypatch, route, driver_name
):
    """Validation does not expand the downstream GDF keyword API."""
    import vibeqc
    from vibeqc.periodic.ccm import ri as ri_module

    def downstream_driver(system, basis, kmesh, options=None, *, functional=None):
        raise AssertionError("the unsupported keyword should fail at binding")

    monkeypatch.setattr(vibeqc, "make_basis", lambda *_: object())
    monkeypatch.setattr(vibeqc, "monkhorst_pack", lambda *_: (1, 1, 1))
    monkeypatch.setattr(vibeqc, driver_name, downstream_driver)
    monkeypatch.setattr(ri_module, "_warn_experimental", lambda: None)
    ccm = _fake_gdf_ccm()

    with pytest.raises(TypeError, match="unexpected keyword argument 'conv_tol_grad'"):
        route(ccm, conv_tol_grad=1e-6)


@pytest.mark.parametrize(
    "conv_tol_grad",
    _INVALID_TOLERANCES,
    ids=("negative", "zero", "nan", "positive-infinity"),
)
def test_shared_rhf_loop_rejects_invalid_conv_tol_grad(conv_tol_grad):
    with pytest.raises(
        ValueError,
        match=r"_rhf_loop: conv_tol_grad must be finite and positive\.",
    ):
        _rhf_loop(
            None,
            None,
            None,
            0.0,
            None,
            None,
            max_iter=1,
            conv_tol=1e-9,
            conv_tol_grad=conv_tol_grad,
            diis_dim=1,
            lindep_tol=1e-7,
        )


@pytest.mark.parametrize(
    "conv_tol_grad",
    _INVALID_TOLERANCES,
    ids=("negative", "zero", "nan", "positive-infinity"),
)
def test_shared_rks_loop_rejects_invalid_conv_tol_grad(conv_tol_grad):
    with pytest.raises(
        ValueError,
        match=r"_rks_loop: conv_tol_grad must be finite and positive\.",
    ):
        _rks_loop(
            None,
            None,
            None,
            0.0,
            None,
            None,
            None,
            0.0,
            "pbe",
            max_iter=1,
            conv_tol=1e-9,
            conv_tol_grad=conv_tol_grad,
            diis_dim=1,
            lindep_tol=1e-7,
        )


def test_no_hardcoded_commutator_literal_in_closed_shell_gates():
    """The historical ``1e-6`` literal must not reappear in the exit gates."""
    for path in _LITERAL_GATE_FILES:
        source = path.read_text(encoding="utf-8")
        assert "np.max(np.abs(err)) < 1e-6" not in source, path
