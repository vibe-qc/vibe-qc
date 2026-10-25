"""Phase G1d — open-shell (UKS) periodic atomic gradient.

Pinned contracts:

1. **Public API** — ``vq.compute_gradient_periodic_uks_multi_k`` is
   exposed at the top level. Returns ``(n_atoms, 3)`` Ha/bohr.

2. **Closed-shell limit (multiplicity=1) ≡ RKS multi-k**. Running
   UKS on a closed-shell system should reproduce the RKS result to
   machine precision.

3. **Hybrid functionals refused** — α_HF ≠ 0 raises
   ``NotImplementedError`` (per-spin periodic K piece deferred to
   v0.6.x).

4. **Refusal on un-converged SCF**.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


def _h2_closed_shell_box(box_ang: float = 20.0):
    R = 1.0 * ANGSTROM_TO_BOHR
    big_box = box_ang * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0, 0, 0]),
        vq.Atom(1, [0, 0, R]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box]*3), atoms,
                              charge=0, multiplicity=1)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.max_iter = 100
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    return sys, basis, opts, kmesh


# ---------------------------------------------------------------------------
# 1. Public API
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_uks_multi_k_exposed():
    assert hasattr(vq, "compute_gradient_periodic_uks_multi_k")


# ---------------------------------------------------------------------------
# 2. Closed-shell limit: UKS LDA matches RKS LDA bit-for-bit
# ---------------------------------------------------------------------------

def test_uks_closed_shell_matches_rks():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    r_uks = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    r_rks = vq.run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r_uks.converged
    assert r_rks.converged
    assert r_uks.ecp_operator is False
    assert r_uks.ecp_total_ncore == 0

    g_uks = vq.compute_gradient_periodic_uks_multi_k(
        sys, basis, r_uks, kmesh, lattice_opts=opts.lattice_opts)
    g_rks = vq.compute_gradient_periodic_rks_multi_k(
        sys, basis, r_rks, kmesh, lattice_opts=opts.lattice_opts)
    np.testing.assert_allclose(g_uks, g_rks, atol=1e-9,
        err_msg="UKS gradient on closed-shell system diverges from RKS")


# ---------------------------------------------------------------------------
# 2b. GGA closed-shell limit: UKS PBE matches RKS PBE
#
#     Guards the UKS multi-k XC Pulay reroute to the open-shell lattice
#     primitive ``xc_lattice_gradient_contribution_uks`` (full GGA σ-piece).
#     On a closed-shell system the spin-polarized GGA XC Pulay must reduce
#     to the closed-shell one — exercising the σ-coupled path for both.
# ---------------------------------------------------------------------------

def test_uks_pbe_closed_shell_matches_rks():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    opts.functional = "pbe"
    r_uks = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    r_rks = vq.run_rks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r_uks.converged
    assert r_rks.converged

    g_uks = np.asarray(vq.compute_gradient_periodic_uks_multi_k(
        sys, basis, r_uks, kmesh, lattice_opts=opts.lattice_opts))
    g_rks = np.asarray(vq.compute_gradient_periodic_rks_multi_k(
        sys, basis, r_rks, kmesh, lattice_opts=opts.lattice_opts))
    np.testing.assert_allclose(g_uks, g_rks, atol=1e-9,
        err_msg="UKS PBE gradient on closed-shell system diverges from RKS "
                "PBE — open-shell GGA σ XC Pulay inconsistent with closed-shell")


# ---------------------------------------------------------------------------
# 3. Hybrid functionals raise NotImplementedError
# ---------------------------------------------------------------------------

def test_hybrid_functional_raises():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    opts.functional = "b3lyp"
    r = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    if not r.converged:
        pytest.skip("B3LYP UKS didn't converge in test fixture; "
                    "structural test only.")
    with pytest.raises(NotImplementedError, match="hybrid"):
        vq.compute_gradient_periodic_uks_multi_k(
            sys, basis, r, kmesh, lattice_opts=opts.lattice_opts)


# ---------------------------------------------------------------------------
# 4. Un-converged input rejected
# ---------------------------------------------------------------------------

def test_unconverged_rejected():
    sys, basis, opts, kmesh = _h2_closed_shell_box()
    opts.max_iter = 1
    opts.conv_tol_energy = 1e-30
    r = vq.run_uks_periodic_multi_k_ewald3d(sys, basis, kmesh, opts)
    assert r.converged is False
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_gradient_periodic_uks_multi_k(
            sys, basis, r, kmesh, lattice_opts=opts.lattice_opts)


@pytest.mark.parametrize("metadata", [
    {"ecp_operator": True, "ecp_total_ncore": 0},  # zero-core model potential
    {"ecp_operator": True, "ecp_total_ncore": 10},
    {"ecp_total_ncore": 10},  # older result adapter
])
def test_ecp_result_refused_before_gradient_assembly(monkeypatch, metadata):
    """Manual ECPs need result provenance even on an all-electron basis."""
    import vibeqc.periodic_gradient_multi_k as gradients

    system, basis, opts, kmesh = _h2_closed_shell_box()
    result = SimpleNamespace(converged=True, functional="pbe", **metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("ECP result reached the all-electron gradient assembly")

    monkeypatch.setattr(gradients, "_resolve_gradient_lattice_opts", forbidden)
    with pytest.raises(NotImplementedError, match="ECP.*analytic gradient"):
        vq.compute_gradient_periodic_uks_multi_k(system, basis, result, kmesh)


@pytest.mark.parametrize("metadata", [{}, {"ecp_operator": None}])
def test_legacy_ecp_basis_refused_before_gradient_assembly(monkeypatch, metadata):
    import vibeqc.periodic_gradient_multi_k as gradients

    system = vq.PeriodicSystem(
        3, np.eye(3) * 20.0, [vq.Atom(17, [0, 0, 0])], 0, 2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    result = SimpleNamespace(converged=True, functional="pbe", **metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("legacy ECP result reached all-electron gradient assembly")

    monkeypatch.setattr(gradients, "_resolve_gradient_lattice_opts", forbidden)
    with pytest.raises(NotImplementedError, match="ECP.*analytic gradient"):
        vq.compute_gradient_periodic_uks_multi_k(
            system, basis, result, vq.monkhorst_pack(system, [1, 1, 1]),
        )


def test_legacy_light_atom_in_ecp_family_keeps_all_electron_gradient(monkeypatch):
    import vibeqc.periodic_gradient_multi_k as gradients

    system, _, _, kmesh = _h2_closed_shell_box()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    result = SimpleNamespace(converged=True, functional="pbe")

    class ReachedAssembly(Exception):
        pass

    def reached(*args, **kwargs):
        raise ReachedAssembly

    monkeypatch.setattr(gradients, "_resolve_gradient_lattice_opts", reached)
    with pytest.raises(ReachedAssembly):
        vq.compute_gradient_periodic_uks_multi_k(system, basis, result, kmesh)


def _ecp_valence_case(*, mesh=3, multiplicity=1, zero_core=False,
                      functional="pbe", custom_grid=False, physical=False,
                      relabeled=False, displacement=None):
    """Two atom-following local/nonlocal projectors with real image overlap."""
    core = vq._vibeqc_core
    xyz = np.array([[4., 4., 4.], [5.6, 4.3, 4.2]])
    if displacement is not None:
        atom, axis, step = displacement
        xyz[atom, axis] += step
    if relabeled:
        xyz[1, 0] += 7.
    z = 1 if zero_core else 3
    system = vq.PeriodicSystem(
        3, np.eye(3) * 7., [vq.Atom(z, list(p)) for p in xyz], 0, multiplicity,
    )
    basis = core.BasisSet(
        system.unit_cell_molecule(),
        [core.ShellInfo(a, 0, True, [.8], [1.], list(p)) for a, p in enumerate(xyz)],
        "explicit", coefficients_pre_normalized=False,
    )
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.lattice_opts.cutoff_bohr = 8.
    opts.lattice_opts.nuclear_cutoff_bohr = 18.
    opts.lattice_opts.pair_complete_1e = physical
    if custom_grid:
        opts.grid.n_radial = 35
        opts.becke_image_radius_bohr = 8.5
    opts.ecp_effective_charges = [1., 1.]
    opts.ecp_total_ncore = 0 if zero_core else 4
    opts.ecp_home_centers = xyz.tolist()
    blocks = []
    for _ in xyz:
        block = core.ECPPrimitiveBlock()
        block.n_primitive = 2
        block.exponents = [1.25, .8]
        block.coefficients = [-.3, .1]
        block.ams = [1, 0]
        block.ns = [2, 2]
        blocks.append(block)
    opts.ecp_primitive_blocks = blocks
    kmesh = vq.monkhorst_pack(system, [mesh, 1, 1])
    return system, basis, opts, kmesh


def _run_ecp_valence_case(**kwargs):
    system, basis, opts, kmesh = _ecp_valence_case(**kwargs)
    result = vq.run_uks_periodic_multi_k_ewald3d(
        system, basis, kmesh, opts, progress=False, verbose=0,
        auto_optimize_truncation=False,
    )
    assert result.converged
    return system, basis, opts, kmesh, result


@pytest.mark.parametrize("case", [
    dict(mesh=1),
    dict(mesh=3, multiplicity=3),
    dict(mesh=3, zero_core=True, functional="lda"),
    dict(mesh=3, functional="r2scan"),
    dict(mesh=3, custom_grid=True),
    dict(mesh=1, physical=True),
    dict(mesh=3, multiplicity=3, physical=True),
    dict(mesh=3, zero_core=True, functional="lda", physical=True),
    dict(mesh=3, functional="r2scan", physical=True),
], ids=["gamma-pbe", "multik-triplet-pbe", "multik-zero-core-lda", "multik-r2scan",
        "multik-custom-grid-pbe", "physical-gamma-pbe", "physical-multik-triplet-pbe",
        "physical-multik-zero-core-lda", "physical-multik-r2scan"])
def test_ecp_uks_gradient_matches_all_component_scf_differences(case, record_property):
    """Test the valence Pulay, Z_eff, projectors and moving XC together."""
    system, basis, opts, kmesh, result = _run_ecp_valence_case(**case)
    gradient = np.asarray(vq.compute_gradient_periodic_uks_multi_k(
        system, basis, result, kmesh,
    ))
    finite = []
    for h in (1e-3, 5e-4):
        fd = np.zeros_like(gradient)
        for atom in range(2):
            for axis in range(3):
                energies = [
                    _run_ecp_valence_case(**case, displacement=(atom, axis, sign*h))[-1].energy
                    for sign in (1., -1.)
                ]
                fd[atom, axis] = (energies[0] - energies[1]) / (2*h)
        finite.append(fd)
    extrapolated = (4*finite[1] - finite[0])/3
    record_property("gradient", gradient.tolist())
    record_property("scf_fd_extrapolated", extrapolated.tolist())
    record_property("max_gradient_error", float(np.max(np.abs(gradient - extrapolated))))
    np.testing.assert_allclose(finite[0], finite[1], atol=2e-6, rtol=0)
    np.testing.assert_allclose(gradient, extrapolated, atol=2e-6, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-7, rtol=0)
    for da, db, overlap in zip(result.density_alpha_k, result.density_beta_k, result.overlap):
        na = np.trace(np.asarray(da) @ np.asarray(overlap)).real
        nb = np.trace(np.asarray(db) @ np.asarray(overlap)).real
        np.testing.assert_allclose([na, nb], [2., 0.] if case.get("multiplicity") == 3 else [1., 1.],
                                   atol=1e-9, rtol=0)
    if case["mesh"] > 1:
        # Not just duplicated Gamma data in a vacuum box.
        assert np.max(np.abs(np.asarray(result.overlap[0]) - result.overlap[1])) > 1e-7


@pytest.fixture(scope="module")
def ecp_uks_source_case():
    return _run_ecp_valence_case()


def test_physical_ecp_uks_gradient_is_invariant_under_atom_relabeling():
    observations = []
    for relabeled in (False, True):
        system, basis, _, mesh, result = _run_ecp_valence_case(
            physical=True, relabeled=relabeled,
        )
        gradient = vq.compute_gradient_periodic_uks_multi_k(system, basis, result, mesh)
        observations.append((result.energy, np.asarray(gradient)))
    np.testing.assert_allclose(observations[0][0], observations[1][0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(observations[0][1], observations[1][1], atol=1e-8, rtol=0)


def test_ecp_uks_source_owns_mutable_operator_and_options(ecp_uks_source_case):
    from dataclasses import replace
    from vibeqc.periodic_ecp_source import (
        capture_ewald_uks_ecp_source, resolve_ewald_uks_ecp_source,
    )
    from vibeqc.periodic_k_gdf import _periodic_ecp_context

    system, basis, _, mesh, result = ecp_uks_source_case
    # Mutate independent caller inputs, never the shared fixture's options.
    _, _, opts, _ = _ecp_valence_case()
    source = capture_ewald_uks_ecp_source(
        system, basis, mesh, opts, result.effective_lattice_opts,
        _periodic_ecp_context(opts, system, "test"), result.omega,
    )
    receipt = source.identity
    opts.ecp_primitive_blocks[0].coefficients = [10., 20.]
    opts.ecp_home_centers = [[0., 0., 0.], [1., 1., 1.]]
    opts.ecp_effective_charges = [2., 2.]
    opts.lattice_opts.cutoff_bohr = 3.
    opts.use_periodic_becke = False
    context = resolve_ewald_uks_ecp_source(
        system, basis, replace(result, ecp_derivative_source=source), mesh,
        result.effective_lattice_opts, None,
    )
    assert source.identity == receipt
    assert list(context[1][0].coefficients) == [-.3, .1]
    assert context[2] == [[4., 4., 4.], [5.6, 4.3, 4.2]]
    assert context[3].cutoff_bohr == 8.
    assert context[5] is True


@pytest.mark.parametrize("change", ["geometry", "basis", "mesh", "lattice", "grid",
                                   "core", "occupations", "source", "center", "backend"])
def test_ecp_uks_gradient_rejects_changed_source(ecp_uks_source_case, change):
    from dataclasses import replace
    from vibeqc.bipole_domain import GRID_FIELDS, OptionSnapshot
    from vibeqc.fock_state import SourceProvenance
    from vibeqc.periodic_ecp_source import resolve_ewald_uks_ecp_source

    system, basis, opts, mesh, result = ecp_uks_source_case
    lat = result.effective_lattice_opts
    grid = None
    if change == "geometry":
        system = _ecp_valence_case(displacement=(0, 0, .01))[0]
    elif change == "basis":
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    elif change == "mesh":
        mesh = vq.monkhorst_pack(system, [1, 1, 1])
    elif change == "lattice":
        lat = _ecp_valence_case()[2].lattice_opts
        lat.cutoff_bohr = 3.
    elif change == "grid":
        grid = OptionSnapshot.capture(opts.grid, GRID_FIELDS).restore(vq.GridOptions)
        grid.n_radial += 1
    elif change == "core":
        result = replace(result, ecp_total_ncore=0)
    elif change == "occupations":
        result = replace(result, occupations_alpha=[np.array([.9, .1])]*3)
    elif change == "source":
        result = replace(result, ecp_derivative_source=None)
    else:
        source = result.ecp_derivative_source
        details = source.as_dict()["details"]
        if change == "center":
            details["centers"][0][0] += .01
        else:
            details["hartree_ke"] = 100.
        result = replace(result, ecp_derivative_source=SourceProvenance.capture(source.provider, **details))
    with pytest.raises((ValueError, NotImplementedError)):
        resolve_ewald_uks_ecp_source(system, basis, result, mesh, lat, grid)


@pytest.mark.parametrize("functional", ["pbe0", "hse06", "cam-b3lyp"])
def test_ecp_uks_gradient_keeps_hybrid_refusal(ecp_uks_source_case, functional):
    from dataclasses import replace

    system, basis, _, mesh, result = ecp_uks_source_case
    with pytest.raises(NotImplementedError, match="hybrid"):
        vq.compute_gradient_periodic_uks_multi_k(
            system, basis, replace(result, functional=functional), mesh,
        )
