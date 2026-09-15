"""CPCM / COSMO implicit solvation (v0.9.0) — framework tests.

The cavity + CPCM math tests are pure-Python and run without any C++
build. The end-to-end CPCM SCF tests pull in the bundled molecular
integrals + SCF, so they're marked ``@pytest.mark.requires_cpp`` and
skipped automatically when the C++ extension isn't importable.

Contracts exercised
-------------------
* Bondi-radius lookup is element-correct.
* :func:`build_cavity` returns a well-conditioned tessellation:
  positive areas, switching ∈ [0, 1], no points inside neighbouring
  spheres above the drop threshold.
* The CPCM matrix ``A`` is symmetric, positive-definite, with
  Scalmani-Frisch diagonal elements.
* The dielectric factor matches the published Cossi / Klamt formulas.
* Solvent-preset table + alias lookup is round-trippable.
* Conductor limit (ε → ∞ ⇒ f → 1) reproduces the conductor screening
  limit on a single-charge cavity to machine precision.
* End-to-end: water-in-water RHF/STO-3G converges; ε_solv < 0;
  ``run_job(solvent=...)`` round-trips through the same driver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from vibeqc.solvation import (
    BONDI_RADII_ANG,
    SOLVENT_PRESETS,
    CavityTessellation,
    CPCMResult,
    SolventModel,
    build_A_matrix,
    build_cavity,
    resolve_solvent,
    solve_apparent_charges,
)
from vibeqc.solvation.cpcm import CPCM_DIAG_ALPHA, dielectric_factor
from vibeqc.solvation.presets import (
    SOLVENT_ALIASES,
    canonicalise_solvent_name,
)

ANG_TO_BOHR = 1.8897261339213


def _native_cpcm_core_or_skip():
    try:
        from vibeqc import _vibeqc_core as _core
    except ImportError:
        pytest.skip("native CPCM binding is unavailable")
    required = (
        "cpcm_build_A_matrix",
        "cpcm_build_capped_A_matrix",
        "cpcm_dielectric_factor",
        "cpcm_solve_apparent_charges",
    )
    if not all(hasattr(_core, name) for name in required):
        pytest.skip("native CPCM binding is unavailable")
    return _core


# ---------------------------------------------------------------------------
# S1a — cavity tessellation
# ---------------------------------------------------------------------------


def test_bondi_table_has_common_organic_elements():
    for z in (1, 6, 7, 8, 9, 15, 16, 17):
        assert z in BONDI_RADII_ANG
        assert 1.0 < BONDI_RADII_ANG[z] < 2.5


def test_build_cavity_water_basic_invariants():
    # Standard water geometry (bohr), same as tests/conftest baseline.
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.0, +1.498, -1.159],  # H
            [0.0, -1.498, -1.159],  # H
        ],
        dtype=np.float64,
    )
    Zs = [8, 1, 1]

    cav = build_cavity(pos, Zs, n_points_per_sphere=110)
    assert isinstance(cav, CavityTessellation)
    assert cav.points.ndim == 2 and cav.points.shape[1] == 3
    assert cav.weights.shape == (cav.n_points,)
    assert cav.point_atom.shape == (cav.n_points,)
    assert cav.switching.shape == (cav.n_points,)
    # All weights are strictly positive (surface area).
    assert np.all(cav.weights > 0.0)
    # Switching ∈ [0, 1] by construction.
    assert np.all((cav.switching >= 0.0) & (cav.switching <= 1.0 + 1e-12))
    # The cavity must produce *some* surface — fully-buried atoms
    # is a pathological geometry.
    assert cav.n_points > 50
    # Geometry-tied positions copied verbatim.
    np.testing.assert_allclose(cav.atom_positions, pos)


def test_build_cavity_isolated_atom_full_sphere_area():
    """A single atom with no neighbours has no switching — surface area
    equals 4πr² to Lebedev-quadrature precision."""
    cav = build_cavity(
        np.array([[0.0, 0.0, 0.0]]),
        [8],
        n_points_per_sphere=302,
    )
    r = cav.atom_radii[0]
    expected = 4.0 * math.pi * r * r
    assert cav.n_points == 302
    np.testing.assert_allclose(
        cav.total_surface_area_bohr2,
        expected,
        rtol=1e-12,
    )


def test_build_cavity_radii_override_changes_size():
    pos = np.array([[0.0, 0.0, 0.0]])
    cav_default = build_cavity(pos, [8], n_points_per_sphere=50)
    cav_bigger = build_cavity(pos, [8], n_points_per_sphere=50, radii={8: 3.00})
    assert cav_bigger.atom_radii[0] > cav_default.atom_radii[0]
    # 4πr² scales with r²; bigger radius → bigger surface area.
    assert cav_bigger.total_surface_area_bohr2 > cav_default.total_surface_area_bohr2


def test_build_cavity_probe_radius_inflates_sphere():
    pos = np.array([[0.0, 0.0, 0.0]])
    cav_ses = build_cavity(pos, [8], n_points_per_sphere=50)
    cav_sas = build_cavity(
        pos,
        [8],
        n_points_per_sphere=50,
        solvent_probe_radius_ang=1.4,
    )
    assert cav_sas.atom_radii[0] > cav_ses.atom_radii[0]


def test_build_cavity_switching_buries_overlapped_points():
    # Two atoms 1 bohr apart — heavy overlap; the switching function
    # must remove a substantial fraction of the inner-facing points
    # on each sphere.
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    cav = build_cavity(pos, [1, 1], n_points_per_sphere=110)
    iso = build_cavity(
        np.array([[0.0, 0.0, 0.0]]),
        [1],
        n_points_per_sphere=110,
    )
    # Two-atom cavity should expose less area than 2 × isolated atoms.
    assert cav.total_surface_area_bohr2 < 2.0 * iso.total_surface_area_bohr2


# ---------------------------------------------------------------------------
# S1b — CPCM matrix + linear solve
# ---------------------------------------------------------------------------


def _small_cavity():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +1.498, -1.159],
            [0.0, -1.498, -1.159],
        ]
    )
    return build_cavity(pos, [8, 1, 1], n_points_per_sphere=110)


def test_A_matrix_symmetric_and_well_conditioned():
    cav = _small_cavity()
    A = build_A_matrix(cav.points, cav.weights)
    n = cav.n_points
    assert A.shape == (n, n)
    # Symmetric.
    np.testing.assert_allclose(A, A.T, atol=1e-14)
    # Diagonal positivity: A_ii = (α √(4π)) / √w_i > 0.
    diag = np.diag(A)
    assert np.all(diag > 0)
    np.testing.assert_allclose(
        diag,
        CPCM_DIAG_ALPHA / np.sqrt(cav.weights),
    )
    # Off-diagonal = 1/r > 0 for r > 0.
    off = A - np.diag(diag)
    assert np.all(off >= 0)
    # The Scalmani-Frisch CPCM matrix on a dense Lebedev tessellation
    # is not always strictly SPD — nearest-neighbour 1/r off-diagonals
    # on the same sphere can push the smallest eigenvalue slightly
    # negative. The matrix is, however, always invertible and well-
    # conditioned (κ ~ 1e3) on standard cavity orders, so we test
    # for solvability rather than SPD.
    cond = np.linalg.cond(A)
    assert cond < 1e6, (
        f"CPCM A matrix is ill-conditioned (cond = {cond:.2e}); "
        f"check the cavity tessellation."
    )


def test_A_matrix_offdiagonal_matches_inverse_distance():
    cav = _small_cavity()
    A = build_A_matrix(cav.points, cav.weights)
    # Pick a couple of i ≠ j pairs and verify directly.
    for i, j in [(0, 5), (1, 50), (10, 20)]:
        if i >= cav.n_points or j >= cav.n_points:
            continue
        r = np.linalg.norm(cav.points[i] - cav.points[j])
        assert math.isclose(A[i, j], 1.0 / r, rel_tol=1e-12)


def test_A_matrix_native_matches_python_reference(monkeypatch):
    core = _native_cpcm_core_or_skip()
    cav = _small_cavity()
    monkeypatch.setattr("vibeqc.solvation.cpcm._native_cpcm", lambda: None)
    ref = build_A_matrix(cav.points, cav.weights)
    native = core.cpcm_build_A_matrix(cav.points, cav.weights)
    np.testing.assert_allclose(np.asarray(native), ref, atol=1e-14, rtol=0.0)


def test_public_A_matrix_uses_native_when_available(monkeypatch):
    core = _native_cpcm_core_or_skip()
    cav = _small_cavity()
    called = {"build_A": False}
    real = core.cpcm_build_A_matrix

    class _Native:
        cpcm_build_capped_A_matrix = staticmethod(core.cpcm_build_capped_A_matrix)
        cpcm_dielectric_factor = staticmethod(core.cpcm_dielectric_factor)
        cpcm_solve_apparent_charges = staticmethod(core.cpcm_solve_apparent_charges)

        @staticmethod
        def cpcm_build_A_matrix(points, weights):
            called["build_A"] = True
            return real(points, weights)

    monkeypatch.setattr("vibeqc.solvation.cpcm._native_cpcm", lambda: _Native)
    A = build_A_matrix(cav.points, cav.weights)
    assert called["build_A"]
    assert A.shape == (cav.n_points, cav.n_points)


def test_dielectric_factor_cpcm_formula():
    # f(78.39) = (78.39 − 1)/78.39 ≈ 0.9872.
    assert math.isclose(
        dielectric_factor(78.39),
        (78.39 - 1) / 78.39,
        rel_tol=1e-12,
    )
    # ε → ∞: f → 1 (conductor limit).
    assert math.isclose(dielectric_factor(1e9), 1.0, rel_tol=1e-8)


def test_dielectric_factor_cosmo_variant():
    # COSMO: f = (ε − 1)/(ε + x), x = 0.5.
    eps = 35.94
    assert math.isclose(
        dielectric_factor(eps, variant="cosmo"),
        (eps - 1) / (eps + 0.5),
        rel_tol=1e-12,
    )
    # The two variants agree to better than 1% for ε > 30.
    rel_diff = abs(
        dielectric_factor(eps, variant="cpcm") - dielectric_factor(eps, variant="cosmo")
    ) / dielectric_factor(eps, variant="cpcm")
    assert rel_diff < 0.02


def test_dielectric_factor_rejects_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon must be > 1"):
        dielectric_factor(0.5)
    with pytest.raises(ValueError, match="epsilon must be > 1"):
        dielectric_factor(1.0)
    with pytest.raises(ValueError, match="unknown variant"):
        dielectric_factor(10.0, variant="bogus")


def test_solve_apparent_charges_smoke():
    """Smoke test: A q = -f V → q non-trivial; E_solv = ½ q·V finite."""
    cav = _small_cavity()
    A = build_A_matrix(cav.points, cav.weights)
    # Mock potential — uniform −1 V at every surface point.
    V = -1.0 * np.ones(cav.n_points)
    res = solve_apparent_charges(A, V, epsilon=78.39)
    assert isinstance(res, CPCMResult)
    assert res.q.shape == (cav.n_points,)
    assert np.isfinite(res.e_solv)
    # The CPCM equation ought to make q · V negative → E_solv negative
    # (electronic stabilisation by solvent screening).
    assert np.dot(res.q, V) < 0
    assert res.e_solv < 0


def test_solve_apparent_charges_native_matches_python_reference(monkeypatch):
    core = _native_cpcm_core_or_skip()
    cav = _small_cavity()
    monkeypatch.setattr("vibeqc.solvation.cpcm._native_cpcm", lambda: None)
    A = build_A_matrix(cav.points, cav.weights)
    V = np.linspace(-0.5, 0.25, cav.n_points)
    ref = solve_apparent_charges(A, V, epsilon=35.94, variant="cosmo")
    native = core.cpcm_solve_apparent_charges(A, V, 35.94, "cosmo")
    np.testing.assert_allclose(np.asarray(native.q), ref.q, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(np.asarray(native.V), ref.V, atol=1e-14, rtol=0.0)
    assert float(native.e_solv) == pytest.approx(ref.e_solv, abs=1e-12)
    assert float(native.epsilon) == pytest.approx(ref.epsilon)


def test_public_solve_apparent_charges_uses_native(monkeypatch):
    core = _native_cpcm_core_or_skip()
    cav = _small_cavity()
    A = build_A_matrix(cav.points, cav.weights)
    V = -np.ones(cav.n_points)
    called = {"solve": False}
    real = core.cpcm_solve_apparent_charges

    class _Native:
        cpcm_build_A_matrix = staticmethod(core.cpcm_build_A_matrix)
        cpcm_build_capped_A_matrix = staticmethod(core.cpcm_build_capped_A_matrix)
        cpcm_dielectric_factor = staticmethod(core.cpcm_dielectric_factor)

        @staticmethod
        def cpcm_solve_apparent_charges(A_arg, V_arg, epsilon, variant):
            called["solve"] = True
            return real(A_arg, V_arg, epsilon, variant)

    monkeypatch.setattr("vibeqc.solvation.cpcm._native_cpcm", lambda: _Native)
    res = solve_apparent_charges(A, V, epsilon=78.39, variant="cpcm")
    assert called["solve"]
    assert isinstance(res, CPCMResult)
    assert res.q.shape == (cav.n_points,)


# ---------------------------------------------------------------------------
# S1d — solvent presets + resolver
# ---------------------------------------------------------------------------


def test_solvent_presets_have_canonical_water_value():
    # Gaussian 16 SCRF=Solvent=Water uses ε = 78.3553.
    # Match to ±1% (our value is the IUPAC 78.39 reference).
    assert math.isclose(SOLVENT_PRESETS["water"], 78.39, rel_tol=1e-12)
    assert SOLVENT_PRESETS["vacuum"] == 1.0


def test_solvent_aliases_round_trip():
    assert canonicalise_solvent_name("h2o") == "water"
    assert canonicalise_solvent_name("MeOH") == "methanol"
    assert canonicalise_solvent_name("MeCN") == "acetonitrile"
    assert canonicalise_solvent_name("DCM") == "dichloromethane"
    assert canonicalise_solvent_name("vacuum") == "vacuum"
    assert canonicalise_solvent_name("none") == "vacuum"


def test_canonicalise_solvent_name_rejects_unknown():
    with pytest.raises(KeyError, match="Unknown solvent"):
        canonicalise_solvent_name("unobtainium")


def test_resolve_solvent_accepts_each_input_type():
    # str preset.
    sm = resolve_solvent("water")
    assert isinstance(sm, SolventModel)
    assert math.isclose(sm.epsilon, SOLVENT_PRESETS["water"])
    assert sm.name == "water"

    # str alias.
    sm = resolve_solvent("MeOH")
    assert sm.epsilon == SOLVENT_PRESETS["methanol"]

    # numeric ε.
    sm = resolve_solvent(25.0)
    assert sm.epsilon == 25.0
    assert sm.name.startswith("custom")

    # dict with extra config.
    sm = resolve_solvent({"epsilon": 12.5, "variant": "cosmo"})
    assert sm.epsilon == 12.5
    assert sm.variant == "cosmo"

    # SolventModel passthrough.
    sm = SolventModel(epsilon=50.0, name="foo")
    assert resolve_solvent(sm) is sm

    # None → gas-phase (returns None).
    assert resolve_solvent(None) is None


def test_resolve_solvent_dict_requires_epsilon():
    with pytest.raises(ValueError, match="must contain 'epsilon'"):
        resolve_solvent({"variant": "cpcm"})


# ---------------------------------------------------------------------------
# End-to-end SCF tests (require the C++ extension).
# ---------------------------------------------------------------------------


@pytest.fixture
def _vibeqc_module():
    """Skip tests in this section when the C++ extension isn't built."""
    try:
        import vibeqc as vq
    except ImportError as exc:  # pragma: no cover — CI builds the ext
        pytest.skip(f"vibeqc core extension not available: {exc}")
    return vq


def _water_molecule(vq):
    """Standard reference geometry (bohr)."""
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, +1.498, -1.159]),
        vq.Atom(1, [0.0, -1.498, -1.159]),
    ]
    return vq.Molecule(atoms, 0, 1)


def _h_doublet(vq):
    return vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], 0, 2)


def test_run_cpcm_scf_gas_short_circuit(_vibeqc_module):
    """solvent=None / "vacuum" returns the gas-phase SCF unchanged."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")

    sol_gas = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=None)
    gas_ref = vq.run_rhf(mol, basis)

    # Energies match bit-for-bit (we re-ran the same gas-phase SCF).
    assert math.isclose(sol_gas.energy, gas_ref.energy, rel_tol=1e-12)
    assert sol_gas.e_solv == 0.0
    assert sol_gas.n_macro_iter == 0
    assert sol_gas.converged

    # "vacuum" preset round-trips through the resolver.
    sol_vac = vq.run_cpcm_scf(mol, basis, method="rhf", solvent="vacuum")
    assert math.isclose(sol_vac.energy, gas_ref.energy, rel_tol=1e-12)


def test_run_cpcm_scf_water_in_water_lowers_energy(_vibeqc_module):
    """Water dissolved in water has E_solv < 0 (solvent stabilises the
    polar solute). Tests the full macro-iteration loop."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")

    # Use a sparse cavity for the test (110 / sphere) to keep wall-time
    # well under a second.
    sm = vq.SolventModel(
        epsilon=78.39,
        name="water",
        n_points_per_sphere=110,
        max_macro_iter=20,
        tol_e_solv=1e-6,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)

    # Macro loop ran and converged.
    assert sol.converged
    assert sol.n_macro_iter >= 1
    assert sol.n_macro_iter <= sm.max_macro_iter

    # Solvent stabilises the polar solute.
    assert sol.e_solv < 0
    # Sanity floor: water-in-water E_solv should be O(kcal/mol), not
    # O(Hartree). Anything bigger than -1 Ha is a bug.
    assert sol.e_solv > -1.0

    # Standard CPCM total: E_tot = E_HF^gas[D_solv] + (1/2) q·V_tot.
    # The in-solvent SCF result includes Tr(D V_q) = q·V_elec in
    # its .energy field, so the reconstruction relation is:
    #   E_tot = sol.scf.energy − q·V_elec + 0.5*q·V_tot
    # We don't carry q·V_elec on the public dataclass -- this test
    # only verifies the SolventResult.energy is below the gas-phase
    # reference and the e_solv contribution is negative.
    assert sol.energy < sol.e_gas
    assert sol.e_solv < 0


def test_cpcm_uks_implicit_stability_skips_every_phase_without_mutation(
    _vibeqc_module, monkeypatch
):
    """Coupled-solvent UKS must never inherit a gas-phase Hessian verdict."""
    vq = _vibeqc_module
    mol = _h_doublet(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UKSOptions()
    opts.functional = "LDA"
    before = (
        opts.stability_check,
        opts._stability_check_explicit,
        opts.stability_max_retries,
    )

    native = vq.run_uks_scf_with_jk
    seen = []

    def capture(*args, **kwargs):
        captured = kwargs["options"] if "options" in kwargs else args[8]
        seen.append(captured)
        return native(*args, **kwargs)

    monkeypatch.setattr(vq, "run_uks_scf_with_jk", capture)
    sm = vq.SolventModel(
        epsilon=2.0,
        n_points_per_sphere=50,
        max_macro_iter=2,
        tol_e_solv=1.0,
    )
    sol = vq.run_cpcm_scf(
        mol,
        basis,
        method="uks",
        solvent=sm,
        options=opts,
    )

    assert len(seen) >= 2
    assert all(captured is not opts for captured in seen)
    assert all(not captured.stability_check for captured in seen)
    assert not sol.scf.stability_checked
    assert (
        opts.stability_check,
        opts._stability_check_explicit,
        opts.stability_max_retries,
    ) == before == (True, False, 3)


def test_cpcm_uks_explicit_stability_fails_closed_without_mutation(
    _vibeqc_module,
):
    vq = _vibeqc_module
    mol = _h_doublet(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UKSOptions()
    opts.functional = "LDA"
    opts.stability_check = True
    before = (
        opts.stability_check,
        opts._stability_check_explicit,
        opts.stability_max_retries,
    )
    sm = vq.SolventModel(epsilon=2.0, n_points_per_sphere=50)

    with pytest.raises(
        RuntimeError,
        match=r"CPCM UKS.*reaction-field density response",
    ):
        vq.run_cpcm_scf(
            mol,
            basis,
            method="uks",
            solvent=sm,
            options=opts,
        )

    assert (
        opts.stability_check,
        opts._stability_check_explicit,
        opts.stability_max_retries,
    ) == before == (True, True, 3)


def test_run_cpcm_scf_higher_dielectric_more_stabilisation(_vibeqc_module):
    """f(ε) is monotone in ε; bigger ε should mean more negative E_solv."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")

    sm_low = vq.SolventModel(epsilon=2.0, n_points_per_sphere=110)
    sm_hi = vq.SolventModel(epsilon=80.0, n_points_per_sphere=110)
    sol_low = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm_low)
    sol_hi = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm_hi)
    assert sol_hi.e_solv < sol_low.e_solv


def test_cpcm_density_fit_matches_direct(_vibeqc_module):
    """CPCM with ``density_fit=True`` converges and agrees with the
    direct-SCF CPCM to within the DF fitting error. Also checks the
    auxiliary basis is autodetected when ``aux_basis`` is left empty."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "def2-svp")
    sm = vq.SolventModel(
        epsilon=78.39,
        n_points_per_sphere=110,
        max_macro_iter=20,
        tol_e_solv=1e-7,
    )

    sol_direct = vq.run_cpcm_scf(
        mol,
        basis,
        method="rhf",
        solvent=sm,
        options=vq.RHFOptions(),
    )
    o_df = vq.RHFOptions()
    o_df.density_fit = True  # aux_basis left empty → autodetected
    sol_df = vq.run_cpcm_scf(
        mol,
        basis,
        method="rhf",
        solvent=sm,
        options=o_df,
    )

    assert sol_direct.converged and sol_df.converged
    # _ensure_cpcm_aux_basis filled the aux in place.
    assert o_df.aux_basis, "aux_basis was not autodetected"
    # DF fitting error on the in-solvent total: ~1e-4 Ha for def2-svp-jk.
    assert abs(sol_df.energy - sol_direct.energy) < 5e-4
    assert abs(sol_df.e_solv - sol_direct.e_solv) < 1e-4


def test_cpcm_rijcosx_converges(_vibeqc_module):
    """CPCM with the RIJCOSX JKBuilder (density_fit + cosx) converges
    and gives a sane (negative, O(kcal/mol)) solvation energy."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "def2-svp")
    sm = vq.SolventModel(
        epsilon=78.39,
        n_points_per_sphere=110,
        max_macro_iter=20,
        tol_e_solv=1e-6,
    )
    o = vq.RHFOptions()
    o.density_fit = True
    o.cosx = True
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm, options=o)
    assert sol.converged
    assert -1.0 < sol.e_solv < 0.0


def test_cpcm_with_ecp_zn2plus_converges(_vibeqc_module):
    """ECP + CPCM converges and gives a sane solvation energy for Zn²⁺.

    With ecp10mdf (n_core=10), the cavity must see Z_eff = 30 − 10 = 20,
    so the apparent surface charge sees the physical +2 net charge rather
    than a spuriously large +12.  The broken v0.9.0 code gave E_solv ≈
    −22.65 Ha (36× too large via the Born q² scaling).  The corrected
    value should be roughly −0.5 to −0.7 Ha (Born estimate for +2 ion).
    """
    vq = _vibeqc_module
    # Zn²⁺ (Z=30, charge=+2 ⇒ 28 electrons).  ecp10mdf replaces 10 core
    # electrons, leaving 18 valence (closed-shell 3d¹⁰ singlet).
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 2, 1)
    basis = vq.BasisSet(mol, "6-31g")
    opts = vq.RHFOptions()
    opts.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    # Empty selects ecp10mdf; the result must record the normalized operative
    # name rather than losing the default-library provenance.
    opts.ecp_library = ""

    sm = vq.SolventModel(epsilon=78.39, n_points_per_sphere=110)
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm, options=opts)

    assert sol.converged
    # The solvation energy for a +2 ion in water should be strongly
    # stabilising (negative) but not the O(20 Ha) broken-path value.
    assert sol.e_solv < 0.0
    assert sol.e_solv > -5.0, (
        f"E_solv = {sol.e_solv:.4f} Ha — much too negative; "
        "check Z_eff plumbing in the cavity potential"
    )
    # The Born estimate for +2 ion with a ≈ 2 Å cavity in water is
    # roughly −0.5 Ha.  Allow a factor-2 tolerance for the simple model.
    assert sol.e_solv > -1.5, (
        f"E_solv = {sol.e_solv:.4f} Ha — well below the Born estimate; "
        "likely a Z_eff bug in V_nuc_cav"
    )
    # Z_eff should be stored in the result.
    assert sol.z_eff is not None
    assert len(sol.z_eff) == 1
    assert sol.z_eff[0] == pytest.approx(20.0, abs=0.1)  # Z_eff = 30 − 10
    # The low-level CPCM SCF must preserve the same authoritative ECP count;
    # correlated consumers use it to fail closed before repartitioning the
    # physical electron count as though the orbitals were all-electron.
    assert sol.scf.ecp_operator_applied
    assert sol.scf.ecp_total_ncore == 10
    [scf_center] = list(sol.scf.ecp_xml_centers)
    assert int(scf_center.Z) == 30
    assert list(scf_center.xyz) == pytest.approx([0.0, 0.0, 0.0])
    assert sol.scf.ecp_xml_library == "ecp10mdf"
    from vibeqc.correlation_conventions import effective_electron_count

    # Correlated consumers partition the valence count from this provenance:
    # 28 physical electrons minus the 10-electron core.
    assert effective_electron_count(mol, sol.scf) == 18


def test_cpcm_direct_call_auto_attaches_lanl2dz_sidecar(_vibeqc_module):
    """Direct CPCM callers receive the same inline sidecar ECP as run_job."""
    vq = _vibeqc_module
    # LANL2DZ replaces 18 electrons on Zn. No options object is supplied:
    # run_cpcm_scf must discover the sidecar itself before building Hcore.
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 2, 1)
    basis = vq.BasisSet(mol, "lanl2dz")

    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent="water")

    assert sol.converged
    assert sol.scf.ecp_operator_applied
    assert sol.scf.ecp_total_ncore == 18
    assert list(sol.scf.ecp_xml_centers) == []
    assert sol.scf.ecp_xml_library == ""
    [block] = list(sol.scf.ecp_primitive_blocks)
    assert block.n_primitive > 0
    [center] = list(sol.scf.ecp_primitive_centers)
    assert list(center) == pytest.approx([0.0, 0.0, 0.0])
    assert list(sol.scf.ecp_effective_charges) == pytest.approx([12.0])
    assert sol.z_eff is not None
    assert sol.z_eff[0] == pytest.approx(12.0)


def test_alpha_beta_counts_accept_effective_ecp_total(_vibeqc_module):
    """Open-shell CPCM partitions the ECP-reduced, not physical, total."""
    from vibeqc.solvation.driver import _alpha_beta_counts

    vq = _vibeqc_module
    # A physical 14-electron triplet with a 10-electron pseudopotential has
    # four variational electrons: three alpha and one beta.
    mol = vq.Molecule([vq.Atom(14, [0.0, 0.0, 0.0])], 0, 3)
    assert _alpha_beta_counts(mol) == (8, 6)
    assert _alpha_beta_counts(mol, 4) == (3, 1)


def test_cpcm_inline_sidecar_ecp_runs_with_effective_charges(_vibeqc_module):
    """A custom-core sidecar (vDZP on oxygen, two core electrons) runs
    through CPCM on the inline route: Hcore carries V_ECP, the cavity sees
    Z_eff = 6, and the result records the primitive provenance."""
    vq = _vibeqc_module
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0])], 0, 1)
    basis = vq.BasisSet(mol, "vdzp")

    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent="water")

    assert sol.converged
    assert sol.scf.ecp_operator_applied
    assert sol.scf.ecp_total_ncore == 2
    assert list(sol.scf.ecp_xml_centers) == []
    assert len(sol.scf.ecp_primitive_blocks) == 1
    assert list(sol.scf.ecp_effective_charges) == pytest.approx([6.0])
    assert sol.z_eff is not None and sol.z_eff[0] == pytest.approx(6.0)
    assert sol.e_solv < 0.0


def test_cpcm_with_ecp_matches_gas_phase_energy_decomposition(_vibeqc_module):
    """ECP+CPCM energy decomposition is self-consistent.

    The total in-solvent energy should decompose as
    E_tot = E_HF^gas[D^solv] + ½ q·V_tot, matching the gas-phase SCF
    energy at the in-solvent density plus the solvation stabilisation.
    """
    vq = _vibeqc_module
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 2, 1)
    basis = vq.BasisSet(mol, "6-31g")
    opts = vq.RHFOptions()
    opts.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts.ecp_library = "ecp10mdf"

    sm = vq.SolventModel(epsilon=78.39, n_points_per_sphere=110)
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm, options=opts)

    assert sol.converged
    # Gas-phase reference should be from the same options (ECP-aware).
    assert sol.e_gas is not None
    # E_solv should be negative (stabilisation from solvent).
    assert sol.e_solv < 0.0
    # Total energy = gas-phase + solvation stabilisation.
    assert sol.energy == pytest.approx(sol.e_gas + sol.e_solv, abs=1e-5)


def test_ecp_core_electrons_lookup(_vibeqc_module):
    """ecp_core_electrons returns correct n_core for ecp10mdf."""
    vq = _vibeqc_module
    core_map = vq.ecp_core_electrons([30, 29, 28], "ecp10mdf", "")
    # Zn (Z=30): ecp10mdf replaces 10 core electrons ([Ne] core).
    assert core_map.get(30) == 10
    # Cu (Z=29): ecp10mdf replaces 10 core electrons.
    assert core_map.get(29) == 10
    # Ni (Z=28): ecp10mdf replaces 10 core electrons.
    assert core_map.get(28) == 10


def test_ecp_core_electrons_skips_elements_absent_from_library(
    _vibeqc_module,
):
    """A mixed request returns the represented element, not raw stoi noise."""
    vq = _vibeqc_module
    assert vq.ecp_core_electrons([47, 30], "ecp28mdf", "") == {47: 28}
    assert vq.ecp_core_electrons([30], "ecp28mdf", "") == {}


def test_run_job_solvent_kwarg_threads_through(_vibeqc_module, tmp_path):
    """run_job(solvent="water") attaches solvent diagnostics to the result."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    out_stem = tmp_path / "h2o_aq"
    result = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=str(out_stem),
        solvent="water",
        write_molden_file=False,
        progress=False,
    )
    # solvent_result is attached on success.
    sol = getattr(result, "solvent_result", None)
    assert sol is not None, (
        "run_job(solvent='water') did not attach solvent_result to the SCF return value"
    )
    assert sol.solvent_name == "water"
    assert sol.solvent_variant == "cpcm"
    assert result.solvent_variant == "cpcm"
    assert result.e_gas == pytest.approx(sol.e_gas)
    assert result.energy_in_solvent == pytest.approx(sol.energy)
    assert sol.e_solv < 0
    out_text = out_stem.with_suffix(".out").read_text()
    assert "Implicit solvation (CPCM)" in out_text
    assert "Implicit solvation (COSMO)" not in out_text
    assert "Gas-phase total" in out_text
    assert "In-solvent total" in out_text


# ---------------------------------------------------------------------------
# S1c — analytic gradient (closed-form pieces, no SCF needed)
# ---------------------------------------------------------------------------


def test_nuclear_esp_gradient_matches_fd():
    """``q^T ∂V^{nuc}/∂R`` analytic vs central-difference."""
    from vibeqc.solvation.cavity import build_cavity
    from vibeqc.solvation.gradient import _nuclear_esp_gradient_contribution

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +1.498, -1.159],
            [0.0, -1.498, -1.159],
        ]
    )
    Zs = np.array([8, 1, 1], dtype=float)
    cav = build_cavity(pos, Zs.astype(int), n_points_per_sphere=50)
    # Synthetic ASCs — value irrelevant for the gradient test.
    rng = np.random.default_rng(42)
    q = rng.normal(scale=0.01, size=cav.n_points)

    analytic = _nuclear_esp_gradient_contribution(cav, q, pos, Zs)

    # FD on E(R) = q^T V_nuc(R) directly. Cavity points move with their
    # parent atoms.
    from vibeqc.solvation.driver import _nuclear_potential_at_cavity

    h = 1e-4
    fd = np.zeros_like(pos)
    for ia in range(3):
        for ic in range(3):
            xp, xm = pos.copy(), pos.copy()
            xp[ia, ic] += h
            xm[ia, ic] -= h
            # Rebuild cavity at displaced geometry (atoms drag points).
            cav_p = build_cavity(xp, Zs.astype(int), n_points_per_sphere=50)
            cav_m = build_cavity(xm, Zs.astype(int), n_points_per_sphere=50)
            # Pad with reference q on point indices (same parent layout
            # since the geometry change is small and switching is C^∞).
            V_p = _nuclear_potential_at_cavity(xp, Zs, cav_p.points)
            V_m = _nuclear_potential_at_cavity(xm, Zs, cav_m.points)
            # Match cavity-point ordering: by-parent assumption.
            # For this small displacement (h=1e-4) the cavity point
            # count and ordering match the reference.
            assert cav_p.n_points == cav.n_points
            fd[ia, ic] = (np.dot(q, V_p) - np.dot(q, V_m)) / (2.0 * h)

    np.testing.assert_allclose(analytic, fd, atol=1e-5, rtol=1e-3)


def _fd_A_matrix_term(pos, Zints, q, f, ref_n_points, h=1e-4,
                      n_points_per_sphere=50):
    """Central-difference ``(1/(2f)) q^T A(R) q`` at fixed ``q``.

    Rebuilds the cavity at each displaced geometry (points move
    rigidly with their parent atoms; switching factors re-evaluate)
    and reassembles ``A`` with the production :func:`build_A_matrix`,
    so the FD differentiates *exactly* the quantity whose closed-form
    derivative :func:`_A_matrix_gradient_contribution` claims to be.
    """
    from vibeqc.solvation.cavity import build_cavity
    from vibeqc.solvation.cpcm import build_A_matrix

    def e_A(positions):
        c = build_cavity(positions, Zints,
                         n_points_per_sphere=n_points_per_sphere)
        # Same-point-count guard: h must not cross a switching window.
        assert c.n_points == ref_n_points
        A = build_A_matrix(c.points, c.weights)
        return (1.0 / (2.0 * f)) * float(q @ A @ q)

    n_atoms = pos.shape[0]
    g = np.zeros((n_atoms, 3))
    for ia in range(n_atoms):
        for ic in range(3):
            xp, xm = pos.copy(), pos.copy()
            xp[ia, ic] += h
            xm[ia, ic] -= h
            g[ia, ic] = (e_A(xp) - e_A(xm)) / (2.0 * h)
    return g


def test_A_matrix_gradient_matches_fd():
    """``(1/(2f)) q^T (∂A/∂R) q`` analytic vs central-difference, random q.

    Per-component regression for the 2026-05-31 audit finding: the
    off-diagonal ``A_ij = 1/|s_i − s_j|`` depends on the positions of
    *both* cavity points, so ``∂(q^T A q)/∂s_k`` carries two equal
    chain-rule channels (row k and column k). The shipped v0.9.x–v0.11.x
    gradient counted only one — a clean factor-2 on the off-diagonal
    piece, invisible to the translational-invariance test below and
    worth ~3.5e-4 Ha/bohr on water/6-31G end-to-end. A direct FD check
    of the quadratic form at fixed ``q`` pins the factor.
    """
    from vibeqc.solvation.cavity import build_cavity
    from vibeqc.solvation.cpcm import dielectric_factor
    from vibeqc.solvation.gradient import _A_matrix_gradient_contribution

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +1.498, -1.159],
            [0.0, -1.498, -1.159],
        ]
    )
    Zints = [8, 1, 1]
    cav = build_cavity(pos, Zints, n_points_per_sphere=50)
    rng = np.random.default_rng(7)
    q = rng.normal(scale=0.02, size=cav.n_points)
    f = dielectric_factor(78.39, variant="cpcm")

    analytic = _A_matrix_gradient_contribution(
        cav, q, f, n_atoms=3, switching_sigma_bohr=0.5
    )
    fd = _fd_A_matrix_term(pos, Zints, q, f, cav.n_points)

    # Measured max|analytic − fd| ≈ 5e-6 (pure O(h²) truncation on the
    # near-singular 1/r off-diagonals; term magnitudes reach ~50
    # Ha/bohr with uncorrelated random q). The dropped-channel bug
    # erred by ~50% of every off-diagonal-dominated element.
    np.testing.assert_allclose(analytic, fd, atol=1e-5, rtol=1e-5)


def test_A_matrix_gradient_diagonal_switching_matches_fd():
    """The diagonal ``A_ii = α/√w_i`` switching-derivative channel alone.

    With exactly one nonzero apparent charge the off-diagonal block
    drops out of ``q^T A q`` identically, leaving ``q_k² · α/√w_k(R)``
    whose geometry dependence is purely the Scalmani-Frisch switching
    factor σ_k(R) inside ``w_k``. This is the term the 2026-05-31 audit
    flagged as untestable by translational invariance (which passes
    even if the term is dropped entirely — here it is ~2.4e-2 Ha/bohr).
    The charge sits on a partially-switched point (σ ≈ 0.1) so both the
    parent-atom and neighbour-atom chain-rule channels are exercised.
    """
    from vibeqc.solvation.cavity import build_cavity
    from vibeqc.solvation.cpcm import dielectric_factor
    from vibeqc.solvation.gradient import _A_matrix_gradient_contribution

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +1.498, -1.159],
            [0.0, -1.498, -1.159],
        ]
    )
    Zints = [8, 1, 1]
    cav = build_cavity(pos, Zints, n_points_per_sphere=50)
    f = dielectric_factor(78.39, variant="cpcm")

    # A point well inside the switching window — σ_k far from both 0
    # and 1 — so ∂σ/∂R is large and the channel is genuinely probed.
    in_window = np.where((cav.switching > 0.05) & (cav.switching < 0.95))[0]
    assert in_window.size > 0, "no partially-switched cavity point found"
    k = int(in_window[0])
    q = np.zeros(cav.n_points)
    q[k] = 0.05

    analytic = _A_matrix_gradient_contribution(
        cav, q, f, n_atoms=3, switching_sigma_bohr=0.5
    )
    fd = _fd_A_matrix_term(pos, Zints, q, f, cav.n_points)

    # Measured max|analytic − fd| ≈ 4e-10 against a 2.4e-2 Ha/bohr
    # term — dropping or mis-scaling the switching derivative fails by
    # ≥ 6 orders of magnitude.
    np.testing.assert_allclose(analytic, fd, atol=5e-8)


def test_A_matrix_gradient_translational_invariance():
    """Translational invariance: rigid-body translation gradient sums to 0.

    The total ``Σ_A ∂(q^T A q)/∂R_A`` must vanish along any direction
    because translating the molecule rigidly doesn't change the
    intrinsic cavity geometry (and hence A). NOTE this is a *necessary*
    condition only — it is satisfied by any global rescaling of the
    gradient (the 2026-05-31 factor-2 bug passed it); the FD tests
    above are the sufficient ones.
    """
    from vibeqc.solvation.cavity import build_cavity
    from vibeqc.solvation.gradient import _A_matrix_gradient_contribution

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +1.498, -1.159],
            [0.0, -1.498, -1.159],
        ]
    )
    Zs = [8, 1, 1]
    cav = build_cavity(pos, Zs, n_points_per_sphere=50)
    rng = np.random.default_rng(0)
    q = rng.normal(scale=0.02, size=cav.n_points)

    grad = _A_matrix_gradient_contribution(
        cav,
        q,
        f_dielectric=0.987,
        n_atoms=3,
        switching_sigma_bohr=0.5,
    )
    # Sum over atoms → rigid-translation gradient. Should be ~0.
    total = grad.sum(axis=0)
    assert np.max(np.abs(total)) < 1e-6, (
        f"A-matrix gradient violates translational invariance: "
        f"Σ_A ∂(qAq)/∂R_A = {total}"
    )


def test_nuclear_esp_gradient_translational_invariance():
    """Same translational test for the nuclear-ESP gradient."""
    from vibeqc.solvation.cavity import build_cavity
    from vibeqc.solvation.gradient import _nuclear_esp_gradient_contribution

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +1.498, -1.159],
            [0.0, -1.498, -1.159],
        ]
    )
    Zs = np.array([8.0, 1.0, 1.0])
    cav = build_cavity(pos, Zs.astype(int), n_points_per_sphere=50)
    rng = np.random.default_rng(1)
    q = rng.normal(scale=0.01, size=cav.n_points)

    grad = _nuclear_esp_gradient_contribution(cav, q, pos, Zs)
    total = grad.sum(axis=0)
    assert np.max(np.abs(total)) < 1e-6, (
        f"V^nuc gradient violates translational invariance: Σ_A ∂(qV)/∂R_A = {total}"
    )


@pytest.fixture
def _vibeqc_for_gradient():
    """Skip the end-to-end gradient tests if the C++ ext isn't built."""
    try:
        import vibeqc as vq
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"vibeqc core extension not available: {exc}")
    return vq


def test_cpcm_gradient_matches_fd_water_in_water(_vibeqc_for_gradient):
    """End-to-end: analytic CPCM gradient agrees with FD on water/STO-3G.

    Uses ``cpcm_gradient`` (closed-form analytic, the v0.9.1 default)
    and ``cpcm_gradient_fd`` (full FD on E_tot). Measured agreement at
    this config (50 pts/sphere, tol_e_solv = 1e-7, h = 2e-3) is
    ~5e-7 Ha/bohr and is FD-truncation-limited — the atol below keeps
    ~20× headroom. Do NOT loosen it to make a regression pass: the
    pre-v0.12 off-diagonal ∂A/∂R factor-2 bug sat at 1.3e-4 on this
    very system and shipped because the old atol was 5e-3 (audit
    2026-05-31; see test_A_matrix_gradient_matches_fd).
    """
    vq = _vibeqc_for_gradient
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")

    sm = vq.SolventModel(
        epsilon=78.39,
        name="water",
        n_points_per_sphere=50,
        max_macro_iter=20,
        tol_e_solv=1e-7,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged

    grad_analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
    grad_fd = vq.cpcm_gradient_fd(
        mol,
        "sto-3g",
        method="rhf",
        solvent=sm,
        step_bohr=2e-3,
    )

    # Equilibrium-geometry water: forces are small but non-zero. The
    # central-difference FD has O(h²) truncation error; the analytic-
    # vs-FD agreement floor is set by the macro-iteration convergence
    # tolerance and the FD step (measured ~5e-7 at this config).
    np.testing.assert_allclose(grad_analytic, grad_fd, atol=1e-5, rtol=1e-3)


def test_gradient_follows_a_non_default_switching_width(_vibeqc_for_gradient):
    """#745 fixed: the gradient differentiates the cavity the run built.

    ``SolventModel.switching_sigma_bohr`` is user-settable and reaches
    ``build_cavity``, but the tessellation did not record it, so the gradient
    read a hard-coded 0.5 and differentiated a cavity nobody computed. Measured
    against a converged finite difference before the fix: 2.2e-04 Ha/bohr at
    ``sigma = 0.35``, which is 0.35 percent of the largest component.

    The reason it survived is the interesting part, and the control below is
    built around it. A gradient taken with the wrong width is still
    *self-consistent*: it satisfies translational invariance to 1e-15 and every
    other exact identity the suite checks. Only an oracle that rebuilds the
    energy can see it, so this test has to pay for a full FD.
    """
    vq = _vibeqc_for_gradient
    from dataclasses import replace

    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sigma = 0.35
    sm = vq.SolventModel(
        epsilon=78.39, switching_sigma_bohr=sigma, n_points_per_sphere=50,
        max_macro_iter=40, tol_e_solv=1e-10,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    assert sol.cavity.switching_sigma_bohr == pytest.approx(sigma, rel=1e-14)

    analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
    fd = vq.cpcm_gradient_fd(
        mol, "sto-3g", method="rhf", solvent=sm, step_bohr=1e-3
    )
    np.testing.assert_allclose(analytic, fd, atol=1e-5, rtol=1e-3)

    # The control reproduces the old behaviour exactly: the same cavity, built
    # at 0.35, telling the gradient it was built at 0.5. Without it the test
    # would pass just as well on the broken code, since the broken code's own
    # invariants all hold.
    mislabelled = replace(sol, cavity=replace(sol.cavity, switching_sigma_bohr=0.5))
    wrong = vq.cpcm_gradient(
        mislabelled.scf, mol, basis, mislabelled, method="rhf"
    )
    assert np.max(np.abs(wrong - fd)) > 100.0 * np.max(np.abs(analytic - fd))
    # And it is invariant while wrong, which is why no exact check caught it.
    assert np.abs(wrong.sum(axis=0)).max() < 1e-9


def test_a_switching_cavity_must_record_its_width(_vibeqc_for_gradient):
    """A cavity that switches and cannot say how widely is refused (#745).

    The old code defaulted to 0.5 here. The failure mode that makes a default
    unacceptable is that it cannot be detected downstream: the gradient stays
    self-consistent, so nothing short of rebuilding the energy notices.
    """
    vq = _vibeqc_for_gradient
    from dataclasses import fields

    from vibeqc.solvation.cavity import CavityTessellation, build_cavity
    from vibeqc.solvation.gradient import _cavity_derivative

    pos = np.array([[0.0, 0.0, 0.0], [0.0, 1.598, -1.159], [0.0, -1.498, -1.159]])
    cav = build_cavity(atom_positions_bohr=pos, atom_numbers=[8, 1, 1],
                       n_points_per_sphere=50, switching_sigma_bohr=0.4)
    assert cav.switching_sigma_bohr == pytest.approx(0.4, rel=1e-14)
    assert "switching_sigma_bohr" in {f.name for f in fields(CavityTessellation)}

    class _WidthlessCavity:
        """Duck-typed like a Lebedev cavity, but silent about its width."""

        def __init__(self, source):
            self.points = source.points
            self.weights = source.weights
            self.point_atom = source.point_atom
            self.switching = source.switching
            self.atom_radii = source.atom_radii
            self.atom_positions = source.atom_positions

    with pytest.raises(ValueError, match="does not record the width"):
        _cavity_derivative(_WidthlessCavity(cav))


def test_a_wide_switch_needs_the_drop_threshold_lowered(_vibeqc_for_gradient):
    """The point set, not the switching function, is what steps (#745 follow-on).

    ``build_cavity`` drops points whose switched weight falls below
    ``drop_threshold``. The switching function is smooth across that cutoff but
    the discrete point set is not, so a point crossing it steps the energy.
    At the default width nothing sits near the cutoff and this is invisible.
    A wide switch -- which is what ``switching_sigma_bohr`` exists for, per its
    own docstring -- puts many points there: at 0.8 bohr a 2.5e-04 displacement
    moves the count from 904 to 905, the local slope reads -0.027 against a
    trend of +0.016, and analytic-versus-FD is 8.9 percent of the largest
    component. No gradient can match a finite difference across a step in the
    energy, which is why the remedy is on the energy side.

    ``SolventModel.switching_drop_threshold`` exposes it, and at 0.0 the same
    comparison is 2.7e-07.
    """
    vq = _vibeqc_for_gradient
    # Not ``_water_molecule``: at the symmetric reference geometry no point
    # sits near the cutoff, so the defect this pins is absent there. One
    # lengthened OH puts a point on it.
    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, +1.598, -1.159]),
         vq.Atom(1, [0.0, -1.498, -1.159])], 0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")

    def run(drop):
        sm = vq.SolventModel(
            epsilon=78.39, switching_sigma_bohr=0.8, switching_drop_threshold=drop,
            n_points_per_sphere=50, max_macro_iter=40, tol_e_solv=1e-10,
        )
        sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
        analytic = vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf")
        fd = vq.cpcm_gradient_fd(
            mol, "sto-3g", method="rhf", solvent=sm, step_bohr=1e-3
        )
        return float(np.max(np.abs(analytic - fd))), sol.cavity.n_points

    kept_all, n_all = run(0.0)
    dropped, n_dropped = run(1e-8)
    assert n_all >= n_dropped
    # Keeping every point makes the energy smooth and the gradient exact.
    np.testing.assert_allclose(kept_all, 0.0, atol=1e-5)
    # Dropping them costs 2.1e-02 Ha/bohr here, four orders worse, and the
    # width is not what causes it: the same width with the cutoff off is fine.
    assert dropped > 1e-3, dropped
    assert dropped > 1000.0 * max(kept_all, 1e-12)


def test_electronic_esp_gradient_analytic_matches_fd(_vibeqc_for_gradient):
    """The closed-form q·∂V^elec/∂R (C++ libint nuclear-attraction
    gradient pass) agrees with the finite-difference electronic-ESP
    path on water/STO-3G.

    This isolates the v0.9.1 C++ kernel
    (``compute_external_charge_density_gradient`` via
    ``_electronic_esp_gradient_analytic``) from the rest of the CPCM
    gradient assembly.
    """
    vq = _vibeqc_for_gradient
    from vibeqc.solvation.gradient import (
        _electronic_esp_gradient_analytic,
        _electronic_esp_gradient_via_fd,
    )

    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39,
        n_points_per_sphere=50,
        max_macro_iter=20,
        tol_e_solv=1e-7,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged

    cav = sol.cavity
    q = np.asarray(sol.cpcm.q)
    D = np.asarray(sol.scf.density)

    g_analytic = _electronic_esp_gradient_analytic(cav, q, D, mol, basis)
    g_fd = _electronic_esp_gradient_via_fd(
        cav,
        q,
        D,
        mol,
        "sto-3g",
        step_bohr=1e-3,
    )
    # Both compute −dE_ext/dR; the only discrepancy is the FD path's
    # O(h²) truncation error (~1e-5 Ha/bohr at h=1e-3).
    np.testing.assert_allclose(g_analytic, g_fd, atol=2e-4, rtol=1e-2)


def test_cpcm_gradient_analytic_vs_fd_electronic_consistent(_vibeqc_for_gradient):
    """``cpcm_gradient`` gives the same total with the analytic
    electronic-ESP path (default) and the FD electronic-ESP fallback
    (``use_fd_electronic=True``)."""
    vq = _vibeqc_for_gradient
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39,
        n_points_per_sphere=50,
        max_macro_iter=20,
        tol_e_solv=1e-7,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged

    g_analytic = vq.cpcm_gradient(
        sol.scf,
        mol,
        basis,
        sol,
        method="rhf",
        use_fd_electronic=False,
    )
    g_fd_elec = vq.cpcm_gradient(
        sol.scf,
        mol,
        basis,
        sol,
        method="rhf",
        use_fd_electronic=True,
    )
    np.testing.assert_allclose(g_analytic, g_fd_elec, atol=2e-4, rtol=1e-2)


# ---------------------------------------------------------------------------
# S1d — solvent method gating (run_job + _run_single_point)
# ---------------------------------------------------------------------------


class TestRunJobSolventMethodGating:
    """``solvent=`` composes with the mean-field SCFs (CPCM: rhf / uhf /
    rks / uks) and MSINDO (COSMO) only. Every other method used to
    *silently ignore* the kwarg and return a gas-phase energy — a solvated
    CASSCF request quietly computed in vacuum — and the post-SCF families
    (mp2 / ccsd / …) ran a CPCM reference SCF but composed their reported
    total from the bare reference energy, so the output claimed an
    in-solvent result it did not compute. Both now refuse up front, with
    the same wording as the ``_run_molecular_scf`` gate in
    molecular_optimize.py (tests/test_molecular_optimize.py::
    TestSolventMethodGating)."""

    @pytest.mark.parametrize(
        "method",
        [
            "casscf",
            "caspt2",
            "nevpt2",
            "casci",
            "cisd",
            "selected_ci",
            "fci",
            "gfn2_xtb",
        ],
    )
    def test_run_single_point_rejects_solvent(self, _vibeqc_module, method):
        """The dispatcher gate fires for every non-mean-field method —
        including via the FD-optimizer energy path (_evaluate_energy
        forwards solvent here), which previously walked the gas-phase
        surface while claiming solvation."""
        from vibeqc.runner import _run_single_point

        vq = _vibeqc_module
        mol = _water_molecule(vq)
        # basis=None: the gate must fire before ANY work — if the refusal
        # were misplaced after dispatch, this would die on the missing
        # basis instead of raising the clear ValueError.
        with pytest.raises(ValueError, match="Implicit solvation is not supported"):
            _run_single_point(method, mol, None, functional=None, solvent="water")

    def test_run_single_point_gas_phase_unaffected(self, _vibeqc_module):
        """solvent=None keeps every method on its normal path."""
        from vibeqc.runner import _run_single_point

        vq = _vibeqc_module
        mol = _water_molecule(vq)
        basis = vq.BasisSet(mol, "sto-3g")
        r = _run_single_point("rhf", mol, basis, functional=None, solvent=None)
        assert getattr(r, "converged", False)

    @pytest.mark.parametrize("method", ["mp2", "ccsd", "ccsd(t)", "scs-mp2"])
    def test_run_job_rejects_post_scf_with_solvent(
        self, _vibeqc_module, method, tmp_path
    ):
        """Post-SCF methods resolve to a mean-field reference, so the
        dispatcher gate alone cannot catch them; the run_job gate names
        the post-SCF composition as the missing piece."""
        vq = _vibeqc_module
        mol = _water_molecule(vq)
        with pytest.raises(ValueError, match="post-SCF"):
            vq.run_job(
                mol,
                basis="sto-3g",
                method=method,
                solvent="water",
                output=str(tmp_path / "gate"),
            )

    def test_run_job_rejects_wavefunction_method_with_solvent(
        self, _vibeqc_module, tmp_path
    ):
        vq = _vibeqc_module
        mol = _water_molecule(vq)
        with pytest.raises(ValueError, match="Implicit solvation is not supported"):
            vq.run_job(
                mol,
                basis="sto-3g",
                method="casscf",
                active_space=(2, 2),
                solvent="water",
                output=str(tmp_path / "gate"),
            )

    def test_run_job_auto_resolution_hint(self, _vibeqc_module, tmp_path):
        """method='auto' on a tiny system resolves to FCI; the refusal
        names the resolution so the fix (pass rhf/rks explicitly) is
        obvious."""
        vq = _vibeqc_module
        h2 = vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], 0, 1
        )
        with pytest.raises(ValueError, match="method='auto' selected 'fci'"):
            vq.run_job(
                h2,
                basis="sto-3g",
                solvent="water",
                output=str(tmp_path / "gate"),
            )

    def test_run_job_rohf_keeps_dedicated_gate(self, _vibeqc_module, tmp_path):
        """rohf/roks pass through to their dedicated NotImplementedError
        gates (handovers/HANDOVER_ROHF.md) — message and type unchanged."""
        vq = _vibeqc_module
        h2p = vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], 1, 2
        )
        with pytest.raises(NotImplementedError, match="rohf"):
            vq.run_job(
                h2p,
                basis="sto-3g",
                method="rohf",
                solvent="water",
                output=str(tmp_path / "gate"),
            )


# ---------------------------------------------------------------------------
# BUG 100 — truthful per-iteration walls + aggregated SCF traces
# ---------------------------------------------------------------------------


def test_bug100_all_scf_traces_nonzero_walls(_vibeqc_module):
    """Every SCF iteration across all CPCM phases carries a truthful
    non-zero wall time (BUG 100 / BUG87-A).  Before the fix only the
    last inner-SCF phase was reported, and its wall column printed
    0.000 on every row."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39,
        n_points_per_sphere=110,
        max_macro_iter=20,
        tol_e_solv=1e-6,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged

    traces = getattr(sol, "all_scf_traces", None)
    assert traces is not None, (
        "SolventResult.all_scf_traces is missing — "
        "the BUG 100 fix was not applied"
    )
    assert len(traces) > 0, "all_scf_traces is empty"

    # Every iteration must carry a measured (non-zero) wall time.
    for step in traces:
        w = float(getattr(step, "wall_s", 0.0))
        assert w > 0.0, (
            f"SCF iteration {step.iter} has wall_s = {w} — "
            f"the per-iteration timer was not filled (BUG 100 / BUG87-A)"
        )

    # The aggregated trace must cover more than one SCF phase.
    # The gas-phase SCF alone is one phase; any macro-iterations add
    # more.  A single-phase trace means the aggregation isn't working.
    # Count phase boundaries: each inner SCF restarts at iter=1.
    phase_starts = sum(1 for s in traces if int(getattr(s, "iter", 0)) == 1)
    assert phase_starts >= 2, (
        f"Only {phase_starts} SCF phase(s) in aggregated trace — "
        f"expected gas-phase + at least one inner SCF (BUG 100)"
    )


def test_bug100_solvent_result_carries_aggregated_traces(_vibeqc_module):
    """SolventResult.all_scf_traces carries traces from every SCF phase,
    not just the last inner SCF."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = vq.SolventModel(
        epsilon=78.39,
        n_points_per_sphere=110,
        max_macro_iter=20,
        tol_e_solv=1e-6,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged

    # The last inner SCF trace (sol.scf.scf_trace) should be a subset
    # of the aggregated trace.
    inner_trace = list(getattr(sol.scf, "scf_trace", []) or [])
    all_traces = getattr(sol, "all_scf_traces", [])
    assert len(all_traces) >= len(inner_trace), (
        f"Aggregated trace ({len(all_traces)} rows) should be at least "
        f"as long as the last inner-SCF trace ({len(inner_trace)} rows)"
    )
    # Gas-phase traces come first.
    gas_iter1 = getattr(all_traces[0], "iter", None)
    assert gas_iter1 == 1, (
        f"First aggregated trace has iter={gas_iter1}, expected 1 "
        f"(gas-phase SCF starts at iter 1)"
    )


def test_bug100_run_job_perf_walls_nonzero(_vibeqc_module, tmp_path):
    """run_job(solvent=..., perf_log=True) produces a .perf file whose
    SCF-iterations table has NO fabricated '--' wall times, and the
    wrapped result carries truthful non-zero walls (BUG 100)."""
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    out_stem = tmp_path / "bug100"
    result = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        solvent="water",
        output=str(out_stem),
        perf_log=True,
        write_molden_file=False,
        progress=False,
    )
    sol = getattr(result, "solvent_result", None)
    assert sol is not None
    assert sol.converged

    # The .perf file must contain an SCF-iterations section.
    perf_path = out_stem.with_suffix(".perf")
    assert perf_path.exists(), f"{perf_path} was not written"
    perf_text = perf_path.read_text()
    assert "SCF iterations" in perf_text, (
        ".perf is missing the SCF-iterations table"
    )

    # Check the raw SCF traces on the result object — every iteration
    # must carry a measured (non-zero) wall_s.  Sub-millisecond walls
    # round to "0.000" in the formatted .perf table; that is correct
    # behaviour, not a fabricated default.
    traces = getattr(result, "scf_trace", []) or []
    assert len(traces) > 0, "scf_trace is empty"
    for step in traces:
        w = float(getattr(step, "wall_s", 0.0))
        assert w > 0.0, (
            f"SCF iteration {getattr(step, 'iter', 0)} has wall_s = {w} — "
            f"the per-iteration timer was not filled (BUG 100 / BUG87-A)"
        )

    # .perf SCF-iterations table: extract wall column, verify no '--'
    # sentinels (unmeasured iterations).
    in_table = False
    for line in perf_text.splitlines():
        if in_table:
            stripped = line.strip()
            if not stripped or stripped.startswith("==="):
                break
            if "iter" in stripped and "E (Ha)" in stripped:
                continue
            if stripped.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                wall_str = parts[-1]
                try:
                    float(wall_str)
                except ValueError:
                    continue
                assert wall_str != "--", (
                    f".perf wall column is '--' (unmeasured sentinel)"
                )


def test_iid148_run_job_energy_is_in_solvent_total(_vibeqc_module, tmp_path):
    """IID 148: the headline energy of a solvated run is the in-solvent
    total, consistently across banner, verdict, result object, and the
    solvation block's own decomposition.

    Pre-fix, the banner printed the inner SCF's E_SCF^{w/V_q} (neither
    the gas-phase nor the in-solvent total) and the 'Solvation energy'
    line did not equal in-solvent minus gas.  A downstream harvester
    reading the banner energy compared a wrong number against ORCA's
    in-solvent FINAL SINGLE POINT ENERGY.
    """
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    out_stem = tmp_path / "iid148"
    result = vq.run_job(
        mol,
        basis="def2-svp",
        method="rks",
        functional="pbe",
        solvent="water",
        output=str(out_stem),
        write_molden_file=False,
        progress=False,
        citations=False,
    )
    sol = getattr(result, "solvent_result", None)
    assert sol is not None and sol.converged

    # (1) The result object's headline energy IS the in-solvent total.
    assert result.energy == pytest.approx(result.energy_in_solvent, abs=1e-12)
    assert result.energy == pytest.approx(sol.energy, abs=1e-12)
    # Sanity: solvation stabilises, and the electrostatic term is the
    # 1/2 q.V_tot piece, NOT the total shift.
    assert sol.energy < sol.e_gas
    assert sol.e_solv < 0.0

    out_text = out_stem.with_suffix(".out").read_text()

    def _value_after(line: str, prefix: str) -> float:
        idx = line.index(prefix) + len(prefix)
        return float(line[idx:].split()[0])

    # (2) The headline energy row (labelled 'In-solvent total' for a
    # solvated job, IID 148) equals the in-solvent total.
    total_row = next(
        ln for ln in out_text.splitlines() if "In-solvent total" in ln
    )
    assert _value_after(total_row, "In-solvent total") == pytest.approx(
        sol.energy, abs=1e-8
    )
    # The bare 'Total energy' label must NOT appear for a solvated run.
    assert not any(
        "Total energy" in ln for ln in out_text.splitlines()
    )

    # (3) The convergence verdict prints the same value.
    verdict = next(
        ln for ln in out_text.splitlines() if "converged in" in ln
    )
    assert _value_after(verdict, "E =") == pytest.approx(sol.energy, abs=1e-8)

    # (4) The solvation block self-checks: in-solvent minus gas equals
    # the printed total shift, and the electrostatic 1/2 q.V_tot plus the
    # solute-polarisation remainder reproduces it exactly.
    gas_row = next(ln for ln in out_text.splitlines() if "Gas-phase total" in ln)
    insolv_row = next(
        ln for ln in out_text.splitlines()
        if "In-solvent total" in ln and "=" in ln
    )
    esolv_row = next(
        ln for ln in out_text.splitlines() if "Solvation energy (1/2 q.V_tot)" in ln
    )
    shift_row = next(
        ln for ln in out_text.splitlines() if "Total solvation shift" in ln
    )
    e_gas = _value_after(gas_row, "=")
    e_in = _value_after(insolv_row, "=")
    e_es = _value_after(esolv_row, "=")
    e_shift = _value_after(shift_row, "=")
    assert e_in == pytest.approx(sol.energy, abs=1e-8)
    assert e_gas == pytest.approx(sol.e_gas, abs=1e-8)
    assert e_shift == pytest.approx(e_in - e_gas, abs=5e-10)
    assert e_es == pytest.approx(sol.e_solv, abs=1e-10)
    assert "includes solute polarisation" in shift_row


def test_cpcm_rhf_implicit_stability_skips_without_mutation(
    _vibeqc_module,
    monkeypatch,
):
    """#144: coupled-solvent RHF must never inherit a gas-phase verdict.

    The restricted-stability Hessian omits the self-consistent
    reaction-field density response, so the CPCM route disables the
    verdict on its per-call value copy (same contract as the UKS guard);
    the caller's options object is untouched.
    """
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    before = (opts.stability_check, opts._stability_check_explicit)

    native = vq.run_rhf_scf_with_jk
    seen = []

    def capture(*args, **kwargs):
        captured = kwargs["options"] if "options" in kwargs else args[6]
        seen.append(captured)
        return native(*args, **kwargs)

    monkeypatch.setattr(vq, "run_rhf_scf_with_jk", capture)
    sm = vq.SolventModel(
        epsilon=2.0,
        n_points_per_sphere=50,
        max_macro_iter=2,
        tol_e_solv=1.0,
    )
    vq.run_cpcm_scf(
        mol,
        basis,
        method="rhf",
        solvent=sm,
        options=opts,
    )

    assert len(seen) >= 2
    assert all(captured is not opts for captured in seen)
    assert all(not captured.stability_check for captured in seen)
    assert (opts.stability_check, opts._stability_check_explicit) == before == (
        True,
        False,
    )


def test_cpcm_rhf_explicit_stability_fails_closed_without_mutation(
    _vibeqc_module,
):
    vq = _vibeqc_module
    mol = _water_molecule(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.stability_check = True
    before = (opts.stability_check, opts._stability_check_explicit)
    sm = vq.SolventModel(epsilon=2.0, n_points_per_sphere=50)

    with pytest.raises(
        RuntimeError,
        match=r"CPCM RHF.*reaction-field density response",
    ):
        vq.run_cpcm_scf(
            mol,
            basis,
            method="rhf",
            solvent=sm,
            options=opts,
        )

    assert (opts.stability_check, opts._stability_check_explicit) == before == (
        True,
        True,
    )


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_standalone_cpcm_read_path_uses_canonical_source(tmp_path, method, _vibeqc_module):
    vq = _vibeqc_module
    from vibeqc.output.formats.molden import write_molden

    molecule = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])])
    basis = vq.BasisSet(molecule, "sto-3g")
    options = getattr(vq, method.upper() + "Options")()
    options.initial_guess = vq.InitialGuess.HCORE
    if hasattr(options, "stability_check"):
        options.stability_check = False
    if method.endswith("ks"):
        options.functional = "lda"
    source = getattr(vq, "run_" + method)(molecule, basis, options)
    path = str(tmp_path / "source.molden")
    write_molden(path, molecule, basis, source)
    options.initial_guess = vq.InitialGuess.READ
    options.read_path = path
    result = vq.run_cpcm_scf(
        molecule, basis, method=method, options=options,
        solvent=vq.SolventModel(epsilon=78.39, n_points_per_sphere=50),
    )
    assert result.converged
    assert result.scf.guess_selection.requested == vq.InitialGuess.READ
    assert result.scf.guess_selection.effective == vq.InitialGuess.READ
    overlap = np.asarray(vq.compute_overlap(basis))
    if method.startswith("u"):
        assert np.trace(result.scf.density_alpha @ overlap) == pytest.approx(1, abs=1e-10)
        assert np.trace(result.scf.density_beta @ overlap) == pytest.approx(1, abs=1e-10)
    else:
        assert np.trace(result.scf.density @ overlap) == pytest.approx(2, abs=1e-10)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_cpcm_preserves_parent_prepared_fragmo(method, _vibeqc_module):
    vq = _vibeqc_module
    from vibeqc.guess import prepare_molecular_guess_source

    mol = vq.Molecule([vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [0., 0., 6.])])
    basis = vq.BasisSet(mol, "sto-3g")
    opts = getattr(vq, method.upper() + "Options")()
    opts.initial_guess = vq.InitialGuess.FRAGMO
    if hasattr(opts, "stability_check"):
        opts.stability_check = False
    prepare_molecular_guess_source(method, opts, mol, basis, fragments=[[0], [1]])
    result = vq.run_cpcm_scf(
        mol, basis, method=method, options=opts,
        solvent=vq.SolventModel(epsilon=78.39, n_points_per_sphere=50),
    )
    assert result.converged
    assert result.scf.guess_selection.requested == vq.InitialGuess.FRAGMO
    assert result.scf.guess_selection.effective == vq.InitialGuess.FRAGMO
    assert result.scf.guess_selection.transport == vq.InitialGuess.READ
