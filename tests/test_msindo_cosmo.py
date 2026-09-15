"""MSINDO implicit solvation (COSMO) — the multipole SolutePotentialProvider.

MSINDO couples the solute to the cavity through a distributed-multipole B-matrix
(two-centre Slater penetration integrals), not an ESP-on-grid.  These tests pin:

* the multipole coupling is correct — the seam energy identity
  ``Tr(P V_q) = q·V_elec`` holds to machine precision, and the solvation energy
  reproduces the reference-MSINDO oracle to within the cavity-convention
  difference (vibe-qc reuses its Lebedev-on-Bondi cavity, not MSINDO's GEPOL);
* it is *not* the monopole shortcut (atomic point charges over-screen ~17×);
* the physically required behaviour — E_solv < 0, gas limit at ε→1, dielectric
  monotonicity, charge conservation, and the d-block (H₂S) path.

Oracle reference values are from a reference MSINDO 2025e build, run
out-of-process (``examples/regression/msindo/runner_msindo.py``,
``CARTES RHF COSMO DIELEC 78.39``); vibe-qc imports no MSINDO code (CLAUDE.md
§10).  vibe-qc's own COSMO totals differ from the oracle by the cavity
convention only — see ``docs/user_guide/msindo.md`` § COSMO.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.semiempirical.methods import msindo_cosmo as _cosmo
from vibeqc.semiempirical.methods import msindo_gepol as _gepol
from vibeqc.semiempirical.methods import msindo as _m
from vibeqc.semiempirical.methods.msindo_cosmo import (
    MSINDOMultipoleProvider,
    _core_potential_at_cavity,
    msindo_cosmo,
)
from vibeqc.solvation.cavity import build_cavity
from vibeqc.solvation.screening import ScreeningModel
from vibeqc.solvation.cpcm import build_A_matrix, solve_apparent_charges
from vibeqc.solvation.provider import SolutePotentialProvider

# H₂O, the COSMO validation geometry (Å) — O at origin (HANDOVER § 4b).
_H2O_Z = [8, 1, 1]
_H2O_XYZ = [(0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)]

# Reference MSINDO oracle (CARTES RHF COSMO DIELEC 78.39):
_ORACLE_GAS = -17.0182088897
_ORACLE_COSMO = -17.0264718414  # solvation = -8.263 mHa

# vibe-qc COSMO total on the default GEPOL cavity — reproduces the oracle to
# ~2e-9 Ha (vibe-qc builds each atom's cavity independently = the oracle's NOSYM
# path; the oracle's default symmetry path can differ by ~10 µHa).
_VIBE_GEPOL_TOTAL = -17.0264718433

# vibe-qc COSMO total on the Lebedev-on-Bondi cavity (cavity="lebedev",
# 302 pts/sphere) — a deterministic regression anchor; the ~1.4 mHa gap from the
# oracle is the cavity convention (GEPOL/COSMOR vs Lebedev/Bondi), not the
# coupling (the Gaussian provider gives −6.93 mHa on the same cavity).
_VIBE_LEBEDEV_TOTAL = -17.025042859087634


def _h2o():
    """Default-cavity (GEPOL) H₂O COSMO run."""
    return msindo_cosmo(_H2O_Z, _H2O_XYZ, epsilon=78.39)


def _cpp_cosmo_b_kernel_or_skip():
    kernel = _cosmo._cpp_cosmo_b_kernel()
    if kernel is None:
        pytest.skip("native MSINDO COSMO B-matrix binding is unavailable")
    return kernel


def _cpp_cosmo_ops_kernel_or_skip():
    kernel = _cosmo._cpp_cosmo_ops_kernel()
    if kernel is None:
        pytest.skip("native MSINDO COSMO vector-op bindings are unavailable")
    return kernel


def _cpp_cosmo_reaction_kernel_or_skip():
    kernel = _cosmo._cpp_cosmo_reaction_kernel()
    if kernel is None:
        pytest.skip("native MSINDO COSMO reaction-field binding is unavailable")
    return kernel


def _core_potential_reference(C, cz, points):
    C = np.asarray(C, dtype=np.float64)
    cz = np.asarray(cz, dtype=np.float64)
    S = np.asarray(points, dtype=np.float64)
    diff = C[:, None, :] - S[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (cz[:, None] / dist).sum(axis=0)


# --------------------------------------------------------------------------- #
# The seam / provider
# --------------------------------------------------------------------------- #


def test_provider_satisfies_protocol():
    """MSINDOMultipoleProvider is a structural SolutePotentialProvider."""
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, _ = _m._atom_blocks(_H2O_Z)
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=110)
    prov = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    assert isinstance(prov, SolutePotentialProvider)


def test_cosmo_b_matrix_cpp_matches_python_reference(monkeypatch):
    """Native COSMO B-matrix construction matches the Python reference loop."""
    build_b, params = _cpp_cosmo_b_kernel_or_skip()
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, _ = _m._atom_blocks(_H2O_Z)
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)

    B_cpp, pairs_cpp, nsto_cpp = build_b(_H2O_Z, C, cav.points, params)
    monkeypatch.setattr(_cosmo, "_cpp_cosmo_b_kernel", lambda: None)
    ref = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)

    assert int(nsto_cpp) == ref.nsto
    assert [tuple(pair) for pair in pairs_cpp] == ref._pairs
    np.testing.assert_allclose(np.asarray(B_cpp), ref._B, atol=1e-12, rtol=0.0)


def test_public_cosmo_provider_uses_cpp_b_matrix(monkeypatch):
    """The public provider routes B-matrix construction to C++ when available."""
    build_b, params = _cpp_cosmo_b_kernel_or_skip()

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python COSMO penetration loop was used")

    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, _ = _m._atom_blocks(_H2O_Z)
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)
    monkeypatch.setattr(_cosmo, "_v2sas_global", _forbidden)

    prov = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    B_cpp, pairs_cpp, nsto_cpp = build_b(_H2O_Z, C, cav.points, params)

    assert prov.nsto == int(nsto_cpp)
    assert prov._pairs == [tuple(pair) for pair in pairs_cpp]
    np.testing.assert_allclose(prov._B, np.asarray(B_cpp), atol=1e-12, rtol=0.0)


def test_cosmo_vector_ops_cpp_match_python_reference(monkeypatch):
    """Native per-SCF COSMO vector operations match the Python reference path."""
    charge_vector, esp_at_cavity, fock_contribution, core_potential = (
        _cpp_cosmo_ops_kernel_or_skip()
    )
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(_H2O_Z)
    cz = [_m.eff_core_charge(z) for z in _H2O_Z]
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)
    native = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    monkeypatch.setattr(_cosmo, "_cpp_cosmo_ops_kernel", lambda: None)
    ref = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)

    P = np.arange(nsto * nsto, dtype=np.float64).reshape(nsto, nsto)
    P = 0.01 * (P + P.T)
    q = np.linspace(-0.02, 0.03, cav.points.shape[0])

    np.testing.assert_allclose(
        np.asarray(charge_vector(P, native.nsto, native._pairs)),
        ref.charge_vector(P),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(esp_at_cavity(native._B, P, native.nsto, native._pairs)),
        ref.esp_at_cavity(P),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(fock_contribution(native._B, q, native.nsto, native._pairs)),
        ref.fock_contribution(q),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(core_potential(C, cz, cav.points)),
        _core_potential_reference(C, cz, cav.points),
        atol=1e-12,
        rtol=0.0,
    )


def test_public_cosmo_provider_uses_cpp_vector_ops(monkeypatch):
    """The public provider uses native charge, ESP, Fock, and core-potential ops."""
    charge_vector, esp_at_cavity, fock_contribution, core_potential = (
        _cpp_cosmo_ops_kernel_or_skip()
    )
    called = {"charge": False, "esp": False, "fock": False, "core": False}

    def spy_charge(*args):
        called["charge"] = True
        return charge_vector(*args)

    def spy_esp(*args):
        called["esp"] = True
        return esp_at_cavity(*args)

    def spy_fock(*args):
        called["fock"] = True
        return fock_contribution(*args)

    def spy_core(*args):
        called["core"] = True
        return core_potential(*args)

    monkeypatch.setattr(
        _cosmo,
        "_cpp_cosmo_ops_kernel",
        lambda: (spy_charge, spy_esp, spy_fock, spy_core),
    )

    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(_H2O_Z)
    cz = [_m.eff_core_charge(z) for z in _H2O_Z]
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)
    prov = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    P = np.eye(nsto, dtype=np.float64)
    q = np.linspace(-0.01, 0.02, cav.points.shape[0])

    prov.charge_vector(P)
    prov.esp_at_cavity(P)
    prov.fock_contribution(q)
    _core_potential_at_cavity(C, cz, cav.points)

    assert called == {"charge": True, "esp": True, "fock": True, "core": True}


def test_cosmo_reaction_field_cpp_matches_python_reference(monkeypatch):
    """Native one-call COSMO reaction field matches the Python reference path."""
    reaction = _cpp_cosmo_reaction_kernel_or_skip()
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(_H2O_Z)
    cz = [_m.eff_core_charge(z) for z in _H2O_Z]
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)
    A = build_A_matrix(cav.points, cav.weights)
    native = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    V_core = _core_potential_at_cavity(C, cz, cav.points)
    P = np.arange(nsto * nsto, dtype=np.float64).reshape(nsto, nsto)
    P = 0.01 * (P + P.T)

    res = reaction(native._B, A, V_core, P, native.nsto, native._pairs, 78.39)

    monkeypatch.setattr(_cosmo, "_cpp_cosmo_b_kernel", lambda: None)
    monkeypatch.setattr(_cosmo, "_cpp_cosmo_ops_kernel", lambda: None)
    monkeypatch.setattr(_cosmo, "_cpp_cosmo_reaction_kernel", lambda: None)
    ref_provider = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    ref = _cosmo._cosmo_reaction_field(
        ref_provider, A, V_core, P,
        screening=ScreeningModel.from_variant(78.39, "cosmo"),
    )
    F_ref, e_add_ref, q_ref, e_pol_ref = (
        ref.fock, ref.e_core_share, ref.q, ref.e_pol
    )
    # The fused native path and the shared generic step must also agree on
    # the decomposition, not merely on the totals.
    np.testing.assert_allclose(ref.V_core, V_core, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(
        ref.V_elec + ref.V_core, ref.V_total, atol=1e-13, rtol=0.0
    )

    np.testing.assert_allclose(np.asarray(res.fock), F_ref, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(res.q), q_ref, atol=1e-12, rtol=0.0)
    assert float(res.e_add) == pytest.approx(e_add_ref, abs=1e-12)
    assert float(res.e_pol) == pytest.approx(e_pol_ref, abs=1e-12)


def test_public_cosmo_reaction_field_uses_cpp(monkeypatch):
    """The COSMO reaction-field seam routes the full per-density step to C++."""
    reaction = _cpp_cosmo_reaction_kernel_or_skip()
    called = {"reaction": False}

    def spy(*args):
        called["reaction"] = True
        return reaction(*args)

    monkeypatch.setattr(_cosmo, "_cpp_cosmo_reaction_kernel", lambda: spy)
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(_H2O_Z)
    cz = [_m.eff_core_charge(z) for z in _H2O_Z]
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)
    A = build_A_matrix(cav.points, cav.weights)
    provider = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    V_core = _core_potential_at_cavity(C, cz, cav.points)
    P = np.eye(nsto, dtype=np.float64)

    field = _cosmo._cosmo_reaction_field(
        provider, A, V_core, P,
        screening=ScreeningModel.from_variant(78.39, "cosmo"),
    )
    F_add, e_add, q, e_pol = (
        field.fock, field.e_core_share, field.q, field.e_pol
    )

    assert called["reaction"]
    assert F_add.shape == (nsto, nsto)
    assert q.shape == (cav.points.shape[0],)
    assert e_add < 0.0
    assert e_pol < 0.0


def test_seam_energy_identity():
    """Tr(P·V_q) == q·V_elec — the two arrows are sign/scale consistent, the
    invariant the reaction-field energy decomposition relies on."""
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, _ = _m._atom_blocks(_H2O_Z)
    cz = [_m.eff_core_charge(z) for z in _H2O_Z]
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=110)
    A = build_A_matrix(cav.points, cav.weights)
    prov = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    V_core = _core_potential_at_cavity(C, cz, cav.points)

    P = _h2o().density
    V_tot = prov.esp_at_cavity(P) + V_core
    cpcm = solve_apparent_charges(A, V_tot, epsilon=78.39, variant="cosmo")
    V_q = prov.fock_contribution(cpcm.q)

    lhs = float(np.sum(P * V_q))
    rhs = float(np.dot(cpcm.q, prov.esp_at_cavity(P)))
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_charge_vector_pairs_carry_factor_two():
    """Diagonal components are P_μμ; same-atom off-diagonal pairs are 2·P_μν
    (cosmo_cbi.f).  O carries the three p-p pairs; H atoms carry none."""
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(_H2O_Z)
    cav = build_cavity(C, _H2O_Z, n_points_per_sphere=50)
    prov = MSINDOMultipoleProvider(_H2O_Z, C, blocks, cav.points)
    # O has s+p (4 AOs) → 3 p-p pairs; 2×H (1 AO each) → 0.  nsto = 4+1+1 = 6.
    assert nsto == 6
    assert prov.n_comp == nsto + 3
    P = _h2o().density
    Q = prov.charge_vector(P)
    np.testing.assert_allclose(Q[:nsto], np.diag(P))
    for cmp, mu, nu in prov._pairs:
        assert Q[cmp] == pytest.approx(2.0 * P[mu, nu])


# --------------------------------------------------------------------------- #
# Parity with the oracle
# --------------------------------------------------------------------------- #


def test_gas_matches_oracle():
    """The COSMO driver's gas-phase reference reproduces oracle MSINDO."""
    r = _h2o()
    assert r.e_gas == pytest.approx(_ORACLE_GAS, abs=1e-7)


def test_gepol_cavity_reaches_exact_oracle_parity():
    """The default GEPOL cavity reproduces the reference-MSINDO total to ~1e-8 Ha
    — full parity (the residual is the oracle's symmetry-vs-NOSYM cavity
    difference; vibe-qc's independent per-atom build matches the NOSYM path)."""
    r = _h2o()
    assert r.total_energy == pytest.approx(_ORACLE_COSMO, abs=1e-8)
    assert r.total_energy == pytest.approx(_VIBE_GEPOL_TOTAL, abs=1e-8)  # regression


def test_lebedev_cavity_within_cavity_convention():
    """The Lebedev cavity reproduces the oracle solvation energy only to the
    cavity-convention difference (~1.4 mHa) — but the coupling is the same
    physics, nowhere near the monopole shortcut's ~17×-too-large −143 mHa."""
    r = msindo_cosmo(_H2O_Z, _H2O_XYZ, epsilon=78.39, cavity="lebedev")
    assert r.total_energy == pytest.approx(_VIBE_LEBEDEV_TOTAL, abs=1e-8)  # regression
    oracle_e_solv = _ORACLE_COSMO - _ORACLE_GAS  # -8.263 mHa
    assert r.e_solv == pytest.approx(oracle_e_solv, abs=2.0e-3)
    assert -9.0e-3 < r.e_solv < -5.0e-3


def test_gepol_segment_counts():
    """GEPOL tessellation matches the oracle segment counts: isolated atom = 60
    pentakisdodecahedron faces; H₂O = 114 after neighbour culling."""
    from vibeqc.semiempirical.methods.msindo_gepol import build_gepol_cavity

    he = build_gepol_cavity([2], np.array([[0.0, 0.0, 0.0]]))
    assert he.n_points == 60
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    h2o = build_gepol_cavity(_H2O_Z, C)
    assert h2o.n_points == 114


def _native_gepol_cavity_or_skip():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        pytest.skip("native MSINDO GEPOL cavity binding is unavailable")
    if not hasattr(_indo, "gepol_build_cavity"):
        pytest.skip("native MSINDO GEPOL cavity binding is unavailable")
    return _indo


def _assert_gepol_cavity_matches(native, ref):
    np.testing.assert_array_equal(np.asarray(native.seg_atom, dtype=int), ref.seg_atom)
    np.testing.assert_allclose(
        np.asarray(native.points, dtype=np.float64),
        ref.points,
        atol=1e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(native.a_diag, dtype=np.float64),
        ref.a_diag,
        atol=1e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(native.area, dtype=np.float64),
        ref.area,
        atol=1e-13,
        rtol=0.0,
    )


def test_gepol_cavity_cpp_matches_python_reference(monkeypatch):
    """Native GEPOL cavity generation matches the Python reference path."""
    core = _native_gepol_cavity_or_skip()
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    native = core.gepol_build_cavity(_H2O_Z, C)

    monkeypatch.setattr(_gepol, "_cpp_gepol_kernel", lambda: None)
    ref = _gepol.build_gepol_cavity(_H2O_Z, C)

    assert int(np.asarray(native.points).shape[0]) == 114
    _assert_gepol_cavity_matches(native, ref)


def test_gepol_probe_radius_uses_reference_angstrom_unit(monkeypatch):
    """A nonzero RSOLVE follows ``cosmo_sas.f``'s Angstrom input contract."""
    core = _native_gepol_cavity_or_skip()
    C = np.zeros((1, 3))
    rsolve_angstrom = 1.4
    rscale = 0.1
    native = core.gepol_build_cavity(
        [2], C, 0, 3, rsolve_angstrom, rscale
    )

    monkeypatch.setattr(_gepol, "_cpp_gepol_kernel", lambda: None)
    ref = _gepol.build_gepol_cavity(
        [2], C, rsolve=rsolve_angstrom, rscale=rscale
    )
    _assert_gepol_cavity_matches(native, ref)

    expected_radius = (
        _gepol.cosmor_bohr(2)
        + rscale * rsolve_angstrom / _m.MSINDO_BOHR_ANGSTROM
    )
    np.testing.assert_allclose(
        np.linalg.norm(ref.points, axis=1),
        expected_radius,
        atol=1e-13,
        rtol=0.0,
    )


def test_public_gepol_cavity_uses_cpp(monkeypatch):
    """The public GEPOL cavity helper routes to C++ when available."""
    core = _native_gepol_cavity_or_skip()
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    called = {"gepol": False}
    real = core.gepol_build_cavity

    def spy(*args, **kwargs):
        called["gepol"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "gepol_build_cavity", spy)
    cavity = _gepol.build_gepol_cavity(_H2O_Z, C)

    assert called["gepol"]
    assert cavity.n_points == 114
    _assert_gepol_cavity_matches(real(_H2O_Z, C), cavity)


def _gepol_A_reference(cavity):
    pts = cavity.points
    adiag = cavity.a_diag
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    with np.errstate(divide="ignore"):
        A = 1.0 / dist
    cap = 0.5 * np.minimum(adiag[:, None], adiag[None, :])
    np.fill_diagonal(cap, np.inf)
    A = np.minimum(A, cap)
    np.fill_diagonal(A, adiag)
    return A


def _native_capped_A_or_skip():
    try:
        from vibeqc import _vibeqc_core as _core
    except ImportError:
        pytest.skip("native capped CPCM A-matrix binding is unavailable")
    if not hasattr(_core, "cpcm_build_capped_A_matrix"):
        pytest.skip("native capped CPCM A-matrix binding is unavailable")
    return _core


def test_gepol_A_matrix_cpp_matches_python_reference():
    """Native capped GEPOL A-matrix assembly matches the Python reference."""
    core = _native_capped_A_or_skip()
    from vibeqc.semiempirical.methods.msindo_gepol import build_gepol_cavity

    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    cavity = build_gepol_cavity(_H2O_Z, C)
    native = core.cpcm_build_capped_A_matrix(cavity.points, cavity.a_diag)
    np.testing.assert_allclose(
        np.asarray(native),
        _gepol_A_reference(cavity),
        atol=1e-14,
        rtol=0.0,
    )


def test_public_gepol_A_matrix_uses_cpp(monkeypatch):
    """The public GEPOL A-matrix helper routes to C++ when available."""
    core = _native_capped_A_or_skip()
    from vibeqc.semiempirical.methods.msindo_gepol import (
        build_gepol_A_matrix,
        build_gepol_cavity,
    )

    called = {"capped_A": False}
    real = core.cpcm_build_capped_A_matrix

    def spy(points, a_diag):
        called["capped_A"] = True
        return real(points, a_diag)

    monkeypatch.setattr(core, "cpcm_build_capped_A_matrix", spy)
    C = np.asarray(_H2O_XYZ) * _m.ANGSTROM_TO_BOHR
    cavity = build_gepol_cavity(_H2O_Z, C)
    A = build_gepol_A_matrix(cavity)

    assert called["capped_A"]
    np.testing.assert_allclose(A, _gepol_A_reference(cavity), atol=1e-14, rtol=0.0)


def test_solvation_stabilises():
    """COSMO lowers the energy: E_solv < 0 and total < gas."""
    r = _h2o()
    assert r.e_solv < 0.0
    assert r.total_energy < r.e_gas
    assert r.converged


def test_density_conserves_electrons():
    """The in-solvent density still integrates to N_elec (Σ P_μμ = 8 for H₂O)."""
    r = _h2o()
    assert float(np.trace(r.density)) == pytest.approx(8.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# Physical behaviour
# --------------------------------------------------------------------------- #


def test_dielectric_monotonic_and_gas_limit():
    """|E_solv| grows with ε and vanishes as ε→1 (the gas limit)."""
    e = {
        eps: msindo_cosmo(_H2O_Z, _H2O_XYZ, epsilon=eps).e_solv
        for eps in (1.0001, 2.0, 20.0, 78.39)
    }
    assert e[1.0001] == pytest.approx(0.0, abs=1e-5)  # ε→1 ⇒ no screening
    assert e[78.39] < e[20.0] < e[2.0] < e[1.0001]  # more polar ⇒ more stable


def test_epsilon_must_exceed_one():
    with pytest.raises(ValueError):
        msindo_cosmo(_H2O_Z, _H2O_XYZ, epsilon=1.0)


def test_nonpolar_methane_is_weakly_solvated():
    """CH₄ is near-nonpolar → only a small solvation energy (oracle −0.74 mHa)."""
    ch4_z = [6, 1, 1, 1, 1]
    ch4 = [
        (0, 0, 0),
        (0.629, 0.629, 0.629),
        (-0.629, -0.629, 0.629),
        (0.629, -0.629, -0.629),
        (-0.629, 0.629, -0.629),
    ]
    r = msindo_cosmo(ch4_z, ch4, epsilon=78.39)
    assert r.converged
    assert -2.0e-3 < r.e_solv < 0.0


def test_emits_citations(tmp_path):
    """msindo_cosmo(output=...) writes the MSINDO method papers AND the COSMO
    bundle into the .bibtex/.references siblings; the default GEPOL cavity fires
    the GEPOL paper (Silla 1991), not the Lebedev/SWIG one; libint is suppressed
    (INDO uses no Gaussian integrals)."""
    stem = tmp_path / "h2o_cosmo"
    msindo_cosmo(_H2O_Z, _H2O_XYZ, epsilon=78.39, output=stem)  # default gepol
    bib = (tmp_path / "h2o_cosmo.bibtex").read_text()
    assert "ahlswede_jug_msindo_1_1999" in bib  # MSINDO itself (§8)
    assert "klamt_schuurmann_cosmo_1993" in bib  # COSMO conductor model
    assert "cossi_cpcm_2003" in bib  # CPCM cavity/solver
    assert "silla_gepol_1991" in bib  # GEPOL cavity (default)
    assert "scalmani_frisch_csc_2010" not in bib  # Lebedev-only, not fired
    assert "libint" not in bib.lower()  # INDO: no Gaussian ints
    assert (tmp_path / "h2o_cosmo.references").exists()

    # The Lebedev cavity fires the Lebedev/SWIG papers, not GEPOL.
    leb = tmp_path / "h2o_leb"
    msindo_cosmo(_H2O_Z, _H2O_XYZ, epsilon=78.39, cavity="lebedev", output=leb)
    bib_leb = (tmp_path / "h2o_leb.bibtex").read_text()
    assert "scalmani_frisch_csc_2010" in bib_leb
    assert "silla_gepol_1991" not in bib_leb


def _h2o_molecule(mult=1):
    import vibeqc as vq

    return vq.Molecule(
        [
            vq.Atom(z, [c * _m.ANGSTROM_TO_BOHR for c in xyz])
            for z, xyz in zip(_H2O_Z, _H2O_XYZ)
        ],
        0,
        mult,
    )


def test_run_job_solvent_runs_cosmo(tmp_path):
    """run_job(method='msindo', solvent=...) runs the GEPOL COSMO driver: the
    returned energy is the in-solvent total (to the bohr↔Å round-trip), e_solv is
    exposed, and the GEPOL citation (not the Lebedev/SWIG one) fires."""
    import vibeqc as vq

    r = vq.run_job(
        _h2o_molecule(),
        method="msindo",
        solvent="water",
        output=str(tmp_path / "water"),
        verbose=0,
    )
    assert float(r.energy) == pytest.approx(_ORACLE_COSMO, abs=1e-7)
    assert float(r.e_solv) == pytest.approx(_ORACLE_COSMO - _ORACLE_GAS, abs=1e-7)


def test_run_job_solvent_emits_gepol_citation(tmp_path):
    """The run_job COSMO path writes the GEPOL bundle (Silla 1991) to .out/.bibtex
    and surfaces a solvation block; the Lebedev/SWIG + libint papers stay off."""
    import vibeqc as vq

    stem = tmp_path / "h2o"
    result = vq.run_job(
        _h2o_molecule(), method="msindo", solvent="water", output=str(stem), verbose=0
    )
    assert result.solvent_variant == "cosmo"
    bib = (tmp_path / "h2o.bibtex").read_text()
    assert "silla_gepol_1991" in bib
    assert "klamt_schuurmann_cosmo_1993" in bib
    assert "scalmani_frisch_csc_2010" not in bib  # Lebedev-only
    assert "libint" not in bib.lower()  # INDO
    out_text = (tmp_path / "h2o.out").read_text()
    assert "Implicit solvation (COSMO)" in out_text
    assert "Gas-phase total" in out_text
    assert "In-solvent total" in out_text


def test_run_job_solvent_vacuum_is_gas(tmp_path):
    """solvent='vacuum' (ε ≤ 1) short-circuits to the gas-phase MSINDO run."""
    import vibeqc as vq

    r = vq.run_job(
        _h2o_molecule(),
        method="msindo",
        solvent="vacuum",
        output=str(tmp_path / "vacuum"),
        verbose=0,
    )
    assert float(r.energy) == pytest.approx(_ORACLE_GAS, abs=1e-7)


def test_run_job_open_shell_solvent_converges(tmp_path):
    """Open-shell MSINDO + solvent (COSMO UHF) converges."""
    import vibeqc as vq

    r = vq.run_job(
        _h2o_molecule(mult=3),
        method="msindo",
        solvent="water",
        output=str(tmp_path / "open_shell_water"),
        verbose=0,
    )
    assert float(r.e_solv) < 0.0  # solvation energy negative
    assert r.converged


def test_d_block_solute_h2s():
    """A d-bearing solute (H₂S) exercises the d-d pair components and solvates."""
    h2s_z = [16, 1, 1]
    h2s = [(0, 0, 0), (0.9627, 0, 0.9264), (-0.9627, 0, 0.9264)]
    r = msindo_cosmo(h2s_z, h2s, epsilon=78.39)
    assert r.converged
    assert r.e_solv < 0.0
    # S carries a d shell (9 AOs) → 10 d-d pair components are present.
    C = np.asarray(h2s) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(h2s_z)
    cav = build_cavity(C, h2s_z, n_points_per_sphere=50)
    prov = MSINDOMultipoleProvider(h2s_z, C, blocks, cav.points)
    # S: 3 p-p + 10 d-d = 13 same-atom pairs; H atoms add none.
    assert prov.n_comp == nsto + 13
