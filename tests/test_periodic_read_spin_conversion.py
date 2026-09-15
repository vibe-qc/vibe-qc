"""READ transitions between restricted and spin-resolved Bloch references."""
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.guess_read import (
    resolve_periodic_read_densities_k_open,
    resolve_periodic_read_density_k_closed,
)


@pytest.fixture
def restart_case():
    system = vq.PeriodicSystem(3, 12 * np.eye(3), [
        vq.Atom(1, [6., 6., 5.3]), vq.Atom(1, [6., 6., 6.7]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    points = np.asarray(vq.monkhorst_pack(system, [3, 1, 1]).kpoints)
    weights = np.array([.2, .3, .5])
    source = SimpleNamespace(restart_basis=basis, restart_lattice=system.lattice,
                             restart_kpoints=points, restart_weights=weights)
    order = [2, 0, 1]
    mesh = SimpleNamespace(kpoints=points[order], weights=weights[order])
    a = [np.array([[.6 + k / 10, .1j], [-.1j, .2]]) for k in range(3)]
    b = [np.array([[.1, -.05j], [.05j, .4 + k / 10]]) for k in range(3)]
    return source, dict(basis=basis, system=system, kmesh=mesh), a, b, order


@pytest.mark.parametrize("representation", ["density", "orbitals"])
@pytest.mark.parametrize("to_open", [False, True])
def test_multik_spin_conversion_preserves_phases_and_populations(restart_case, representation, to_open):
    source, target, a, b, order = restart_case
    expected = [x + y for x, y in zip(a, b)]
    if to_open:
        channels = [("", expected)]
    else:
        channels = [("_alpha", a), ("_beta", b)]
        # A redundant total or later diagonalization must not override spin densities.
        source.density_k = [np.eye(2) * 9] * 3
    originals = [np.array(blocks, copy=True) for _, blocks in channels]
    for spin, blocks in channels:
        if representation == "density":
            setattr(source, f"density{spin}_k", blocks)
            setattr(source, f"mo_coeffs{spin}_k", [np.eye(2)] * 3)
            setattr(source, f"occupations{spin}_k", [np.zeros(2)] * 3)
        else:
            eigenpairs = [np.linalg.eigh(d) for d in blocks]
            setattr(source, f"mo_coeffs{spin}_k", [v for _, v in eigenpairs])
            setattr(source, f"occupations{spin}_k", [w for w, _ in eigenpairs])
    if to_open:
        alpha, beta = resolve_periodic_read_densities_k_open(read_from=source, **target)
        for actual in (alpha, beta):
            np.testing.assert_allclose(actual, np.asarray(expected)[order] / 2, atol=1e-14)
        alpha[0][0, 0] += 1
        np.testing.assert_allclose(beta, np.asarray(expected)[order] / 2, atol=1e-14)
    else:
        actual = resolve_periodic_read_density_k_closed(read_from=source, **target)
        np.testing.assert_allclose(actual, np.asarray(expected)[order], atol=1e-14)
        actual[0][0, 0] += 1
    if representation == "density":
        for (spin, _), original in zip(channels, originals):
            np.testing.assert_array_equal(getattr(source, f"density{spin}_k"), original)


@pytest.mark.parametrize("to_open", [False, True])
@pytest.mark.parametrize("fault", ["missing_alpha", "missing_beta", "empty", "short", "shape",
                                    "nonhermitian", "negative", "nonfinite", "bad_occupations"])
def test_multik_spin_conversion_rejects_incomplete_or_corrupt_channels(restart_case, to_open, fault):
    source, target, a, b, _ = restart_case
    source.density_alpha_k, source.density_beta_k = a, b
    source.density_k = [np.eye(2)] * 3  # never a fallback for a broken spin source
    if fault == "missing_alpha":
        del source.density_alpha_k
    elif fault == "missing_beta":
        del source.density_beta_k
    elif fault == "empty":
        source.density_beta_k = []
    elif fault == "short":
        source.density_beta_k = b[:2]
    elif fault == "shape":
        source.density_beta_k = [np.eye(1)] * 3
    elif fault == "nonhermitian":
        a[0][0, 1] += 1
        b[0][0, 1] -= 1  # corruption cancels in total
    elif fault == "negative":
        a[0][0, 0], b[0][0, 0] = -1., 2.
    elif fault == "nonfinite":
        b[0][0, 0] = np.nan
    else:
        del source.density_beta_k
        source.mo_coeffs_beta_k = [np.eye(2)] * 3
        source.occupations_beta_k = [np.array([2., 0.])] * 3
    resolver = resolve_periodic_read_densities_k_open if to_open else resolve_periodic_read_density_k_closed
    with pytest.raises(ValueError):
        resolver(read_from=source, **target)


def test_restricted_multik_qvf_can_seed_spin_resolved_read(tmp_path, restart_case):
    from vibeqc.output.formats.qvf import qvf_bloch_wf_data, write_qvf
    from vibeqc.output.plan import OutputPlan

    source, target, a, b, order = restart_case
    total = [x + y for x, y in zip(a, b)]
    eigenpairs = [np.linalg.eigh(d) for d in total]
    source.mo_coeffs = [v for _, v in eigenpairs]
    source.occupations = [w for w, _ in eigenpairs]
    source.mo_energies = [np.array([-.5, .1])] * 3
    system, basis = target["system"], target["basis"]
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path / "source", method="rhf",
                                          basis="sto-3g", functional=None, job_kind="periodic_scf")
    path = write_qvf(
        tmp_path / "source", plan, system=system,
        result=SimpleNamespace(converged=True, energy=-1., n_iter=1), method="rhf", basis="sto-3g",
        bloch_wf_data=qvf_bloch_wf_data(
            source, basis, system.unit_cell_molecule(),
            k_points=source.restart_kpoints @ np.asarray(system.lattice) / (2 * np.pi),
            k_weights=source.restart_weights,
        ),
    )
    pair = resolve_periodic_read_densities_k_open(read_path=str(path), **target)
    for blocks in pair:
        np.testing.assert_allclose(blocks, np.asarray(total)[order] / 2, atol=1e-12)


@pytest.mark.parametrize("source_method,target_method,route", [
    ("RHF", "UHF", "gdf"), ("UHF", "RHF", "gdf"),
    ("RHF", "ROHF", "gdf"), ("ROHF", "RHF", "gdf"),
    ("RKS", "UKS", "gdf"), ("UKS", "RKS", "gdf"),
    ("RHF", "UHF", "bipole"), ("UHF", "RHF", "bipole"),
    ("RKS", "ROKS", "bipole"), ("ROKS", "RKS", "bipole"),
    ("RKS", "UKS", "gpw"), ("UKS", "RKS", "gpw"),
    ("RKS", "ROKS", "gpw"), ("ROKS", "RKS", "gpw"),
])
@pytest.mark.parametrize("nk", [1, 3])
def test_public_periodic_read_switches_reference_family(tmp_path, source_method, target_method, route, nk):
    atoms = [vq.Atom(1, [6., 6., 5.3]), vq.Atom(1, [6., 6., 6.7])]
    # Open references carry two alpha electrons, so both conversion directions
    # exercise target population changes as well as reference-family dispatch.
    def context(method):
        system = vq.PeriodicSystem(3, 12 * np.eye(3), atoms,
                                   multiplicity=1 if method in ("RHF", "RKS") else 3)
        return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kwargs = dict(jk_method=route, kpoints=[nk, 1, 1], max_iter=40, convergence="off",
                  output_qvf=False, write_molden_file=False, write_density=False,
                  write_xyz_file=False, write_cif_file=False, write_xsf_structure_file=False,
                  write_population_file=False, citations=False, progress=False)
    if source_method.endswith("KS"):
        kwargs["functional"] = "lda"
    if route == "gpw":
        kwargs["cutoff_ha"] = 8.
    elif route == "gdf":
        kwargs.update(rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.)
    source_system, source_basis = context(source_method)
    system, basis = context(target_method)
    source = vq.run_periodic_job(source_system, source_basis, method=source_method,
                                 initial_guess="HCORE", output=tmp_path / "source", **kwargs)
    result = vq.run_periodic_job(system, basis, method=target_method, initial_guess="READ",
                                 read_from=source, output=tmp_path / "restart", **kwargs)
    reference = vq.run_periodic_job(system, basis, method=target_method, initial_guess="HCORE",
                                    output=tmp_path / "reference", **kwargs)
    assert source.converged and result.converged and reference.converged
    assert result.guess_selection.effective == vq.InitialGuess.READ
    assert result.energy == pytest.approx(reference.energy, abs=1e-8)


@pytest.mark.parametrize("functional", [None, "lda"])
@pytest.mark.parametrize("fail", [False, True])
def test_gamma_gdf_read_retains_route_and_caller_options(monkeypatch, restart_case, functional, fail):
    import vibeqc.periodic_k_gdf as gdf
    import vibeqc.pbc_gdf as gamma_gdf

    _, target, _, _, _ = restart_case
    opts = vq.PeriodicKSOptions() if functional else vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.READ
    original = np.diag([.3, .7])
    opts.read_density = original
    prepared = np.diag([1.2, .8])
    reached = []
    def gamma_driver(system, basis, options, **kwargs):
        assert options.initial_guess == vq.InitialGuess.READ
        np.testing.assert_array_equal(options.read_density, prepared)
        assert kwargs["rsgdf_tail_ke_cutoff"] == 0.
        reached.append(True)
        if fail:
            raise RuntimeError("SCF failure witness")
        return SimpleNamespace(guess_selection=None)
    monkeypatch.setattr(gamma_gdf, "run_pbc_gdf_rhf", gamma_driver)
    monkeypatch.setattr(gdf, "_wrap_gamma_gdf_result", lambda result, *args, **kwargs: result)
    def run():
        # SR/LR now uses the general engine even at Gamma; MDF still delegates.
        return gdf.run_krhf_periodic_gdf(
            target["system"], target["basis"], [1, 1, 1], opts, functional=functional,
            initial_density_k=[prepared], gdf_method="mdf",
            rsgdf_tail_ke_cutoff=0., progress=False,
        )
    if fail:
        with pytest.raises(RuntimeError, match="SCF failure witness"):
            run()
    else:
        assert run().guess_selection.effective == vq.InitialGuess.READ
    assert reached == [True]
    assert opts.initial_guess == vq.InitialGuess.READ
    np.testing.assert_array_equal(opts.read_density, original)


@pytest.mark.parametrize("functional", [None, "lda"])
@pytest.mark.parametrize("backend", ["rsgdf", "mdf"])
def test_gamma_gdf_read_preserves_requested_lattice_density(restart_case, functional, backend):
    """Both the general SR/LR engine and MDF's Gamma adapter retain READ state."""
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    _, target, _, _, _ = restart_case
    options = vq.PeriodicKSOptions() if functional else vq.PeriodicRHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.max_iter = 60
    options.conv_tol_energy = 1e-10
    options.lattice_opts.cutoff_bohr = 5.
    options.lattice_opts.nuclear_cutoff_bohr = 5.
    kwargs = dict(functional=functional, gdf_method=backend,
                  rsgdf_ke_cutoff=12., mdf_ke_cutoff=12., progress=False,
                  return_lattice_density=True)
    source = run_krhf_periodic_gdf(target["system"], target["basis"], [1, 1, 1],
                                   options, **kwargs)
    assert source.converged
    options.initial_guess = vq.InitialGuess.READ
    saved = np.diag([.3, .7])
    options.read_density = saved
    restarted = run_krhf_periodic_gdf(
        target["system"], target["basis"], [1, 1, 1], options,
        initial_density_k=source.density, **kwargs)
    assert restarted.converged
    assert restarted.energy == pytest.approx(source.energy, abs=1e-9)
    assert restarted.guess_selection.effective == vq.InitialGuess.READ
    assert restarted.density_lattice is not None
    np.testing.assert_allclose(vq.bloch_sum(restarted.density_lattice, np.zeros(3)),
                               restarted.density[0], atol=1e-11)
    np.testing.assert_array_equal(options.read_density, saved)
    assert options.initial_guess == vq.InitialGuess.READ


@pytest.mark.parametrize("fault", ["count", "complex", "negative", "shape"])
def test_gamma_gdf_rejects_invalid_restart_before_fast_path(monkeypatch, restart_case, fault):
    import vibeqc.periodic_k_gdf as gdf
    import vibeqc.pbc_gdf as gamma_gdf

    _, target, _, _, _ = restart_case
    blocks = [np.eye(2, dtype=complex)]
    if fault == "count":
        blocks *= 2
    elif fault == "complex":
        blocks[0][0, 1], blocks[0][1, 0] = .1j, -.1j
    elif fault == "negative":
        blocks[0][0, 0] = -1
    else:
        blocks = [np.eye(1)]
    monkeypatch.setattr(gamma_gdf, "run_pbc_gdf_rhf", lambda *a, **k: pytest.fail("invalid READ reached SCF"))
    opts = vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.READ
    with pytest.raises(ValueError):
        gdf.run_krhf_periodic_gdf(target["system"], target["basis"], [1, 1, 1], opts,
                                  initial_density_k=blocks, gdf_method="mdf", progress=False)


def test_multik_spin_mos_allow_different_orbital_ranks(restart_case):
    source, target, _, _, order = restart_case
    source.mo_coeffs_alpha_k = [np.array([[1.], [1j]]) / np.sqrt(2)] * 3
    source.mo_coeffs_beta_k = [np.eye(2, dtype=complex)] * 3
    source.occupations_alpha_k = [np.array([.8])] * 3
    source.occupations_beta_k = [np.array([.1, .4])] * 3
    expected = np.array([[.5, -.4j], [.4j, .8]])
    actual = resolve_periodic_read_density_k_closed(read_from=source, **target)
    np.testing.assert_allclose(actual, [expected] * len(order), atol=1e-14)


@pytest.mark.parametrize("source_method,target_method", [
    ("rhf", "uhf"), ("uhf", "rhf"), ("rks", "roks"), ("roks", "rks"),
])
def test_ewald_multik_read_switches_reference_family(source_method, target_method):
    import importlib

    def setup(method):
        opened = method not in ("rhf", "rks")
        system = vq.PeriodicSystem(3, np.eye(3) * 10., [
            vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]),
        ], multiplicity=3 if opened else 1)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        mesh = vq.monkhorst_pack(system, [3, 1, 1])
        opts = vq.PeriodicKSOptions() if method.endswith("ks") else vq.PeriodicRHFOptions()
        opts.initial_guess = vq.InitialGuess.HCORE
        opts.max_iter = 30
        opts.lattice_opts.cutoff_bohr = 12.
        opts.lattice_opts.nuclear_cutoff_bohr = 12.
        if method.endswith("ks"):
            opts.functional = "lda"
        module = importlib.import_module(f"vibeqc.periodic_{method}_multi_k_ewald")
        driver = getattr(module, f"run_{method}_periodic_multi_k_ewald3d")
        return driver, (system, basis, mesh, opts)
    kwargs = dict(grid_shape=8, progress=False, auto_optimize_truncation=False, verbose=0)
    driver, args = setup(source_method)
    source = driver(*args, **kwargs)
    driver, args = setup(target_method)
    reference = driver(*args, **kwargs)
    system, basis, mesh, opts = args
    resolver = (resolve_periodic_read_density_k_closed if target_method in ("rhf", "rks")
                else resolve_periodic_read_densities_k_open)
    density = resolver(read_from=source, basis=basis, system=system, kmesh=mesh)
    opts.initial_guess = vq.InitialGuess.READ
    result = driver(*args, initial_density_k=density, **kwargs)
    assert source.converged and reference.converged and result.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-8)
    assert result.guess_selection.effective == vq.InitialGuess.READ


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("source_spin", [False, True])
@pytest.mark.parametrize("point", [[0., 0., 0.], [.25, 0., 0.]])
def test_one_point_read_uses_physical_density_payload(restart_case, opened, source_spin, point):
    source, target, a, b, _ = restart_case
    fractional = np.asarray([point])
    source.restart_kpoints = fractional @ (2 * np.pi * np.linalg.inv(source.restart_lattice))
    source.restart_weights = np.ones(1)
    target["kmesh"] = SimpleNamespace(kpoints=source.restart_kpoints, weights=np.ones(1))
    # Gamma densities are real; shifted one-point states retain complex phases.
    if point[0] == 0:
        a, b = [a[0].real], [b[0].real]
    else:
        a, b = a[:1], b[:1]
    if source_spin:
        source.density_alpha_k, source.density_beta_k = a, b
    else:
        source.density_k = [a[0] + b[0]]
    resolver = resolve_periodic_read_densities_k_open if opened else resolve_periodic_read_density_k_closed
    result = resolver(read_from=source, **target)
    if opened:
        expected = (a, b) if source_spin else ([source.density_k[0] / 2],) * 2
    else:
        expected = [a[0] + b[0]]
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("complex_coefficients", [False, True])
@pytest.mark.parametrize("selected_gamma", [False, True])
@pytest.mark.parametrize("point", [[0., 0., 0.], [.25, 0., 0.]])
def test_one_point_bloch_qvf_read_uses_all_k_payload(tmp_path, restart_case, opened, complex_coefficients, selected_gamma, point):
    from vibeqc.output.formats.qvf import qvf_bloch_wf_data, qvf_wf_data, write_qvf
    from vibeqc.output.plan import OutputPlan

    source, target, _, _, _ = restart_case
    source.mo_coeffs = [np.array([[1., 0.], [0., 1.]]) if point[0] == 0 or not complex_coefficients else
                        np.array([[1., 1.], [1j, -1j]]) / np.sqrt(2)]
    source.occupations = [np.array([1.5, .5])]
    source.mo_energies = [np.array([-.5, .1])]
    target["kmesh"] = SimpleNamespace(
        kpoints=np.asarray([point]) @ (2 * np.pi * np.linalg.inv(source.restart_lattice)),
        weights=np.ones(1),
    )
    system, basis = target["system"], target["basis"]
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path / "one", method="rhf",
                                          basis="sto-3g", functional=None, job_kind="periodic_scf")
    path = write_qvf(tmp_path / "one", plan, system=system,
        result=SimpleNamespace(converged=True, energy=-1., n_iter=1), method="rhf", basis="sto-3g",
        bloch_wf_data=qvf_bloch_wf_data(source, basis, system.unit_cell_molecule(),
                                       k_points=[point], k_weights=[1.]),
        wf_data=(qvf_wf_data(SimpleNamespace(mo_coeffs=np.eye(2), occupations=np.array([2., 0.]),
                    mo_energies=np.array([-.5, .1])), basis, system.unit_cell_molecule(),
                    k_point=[0., 0., 0.]) if selected_gamma else None),
    )
    resolver = resolve_periodic_read_densities_k_open if opened else resolve_periodic_read_density_k_closed
    actual = resolver(read_path=str(path), **target)
    c = source.mo_coeffs[0]
    total = (c * source.occupations[0]) @ c.conj().T
    expected = ([total / 2], [total / 2]) if opened else [total]
    np.testing.assert_allclose(actual, expected, atol=1e-12)

    from vibeqc.guess_read import resolve_periodic_read_density_closed
    if point[0] != 0:
        with pytest.raises(NotImplementedError, match="non-Gamma"):
            resolve_periodic_read_density_closed(basis, read_from=path)
    else:
        np.testing.assert_allclose(resolve_periodic_read_density_closed(basis, read_from=path),
                                   total, atol=1e-12)


@pytest.mark.parametrize("method", ["RHF", "UHF", "RKS", "UKS"])
@pytest.mark.parametrize("file_source", [False, True])
def test_public_shifted_one_point_read(tmp_path, restart_case, method, file_source):
    if file_source and method in ("UHF", "UKS"):
        # Use a restricted archive as the spin seed; spin-source export is a
        # separate capability. This also checks the reference-family switch.
        source_method = "RHF" if method == "UHF" else "RKS"
    else:
        source_method = method
    _, target, _, _, _ = restart_case
    system, basis = target["system"], target["basis"]
    mesh = (vq.KPoints.monkhorst_pack(system, [1, 1, 1], shift=[1, 0, 0])
            if method.endswith("HF") else
            vq.KPoints.from_list(system, [[.25, 0., 0.]], weights=[1.]))
    kwargs = dict(jk_method="gdf", kpoints=mesh, max_iter=40, convergence="off",
                  output_qvf=False, write_molden_file=False, write_density=False,
                  write_xyz_file=False, write_cif_file=False, write_xsf_structure_file=False,
                  write_population_file=False, citations=False, progress=False,
                  rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.)
    if method.endswith("KS"):
        kwargs["functional"] = "lda"
    source = vq.run_periodic_job(system, basis, method=source_method, initial_guess="HCORE",
                                 output=tmp_path / "source", **kwargs)
    if file_source:
        from vibeqc.output.formats.qvf import qvf_bloch_wf_data, write_qvf
        from vibeqc.output.plan import OutputPlan
        plan = OutputPlan.from_run_job_kwargs(output=tmp_path / "state", method=source_method,
            basis="sto-3g", functional=kwargs.get("functional"), job_kind="periodic_scf")
        source = write_qvf(tmp_path / "state", plan, system=system, result=source,
            method=source_method, basis="sto-3g", bloch_wf_data=qvf_bloch_wf_data(
                source, basis, system.unit_cell_molecule(), k_points=mesh.kpoints_frac, k_weights=[1.]))
    result = vq.run_periodic_job(system, basis, method=method, initial_guess="READ",
                                 read_from=source, output=tmp_path / "restart", **kwargs)
    reference = vq.run_periodic_job(system, basis, method=method, initial_guess="HCORE",
                                    output=tmp_path / "reference", **kwargs)
    assert result.converged and reference.converged
    assert result.guess_selection.effective == vq.InitialGuess.READ
    assert result.energy == pytest.approx(reference.energy, abs=1e-8)


@pytest.mark.parametrize("kind", ["native", "kpoints", "cartesian_only"])
@pytest.mark.parametrize("shifted", [False, True])
def test_runner_one_point_classification_uses_coordinates(restart_case, kind, shifted):
    from vibeqc.periodic_runner import _is_multik_kpoints
    system = restart_case[1]["system"]
    mesh = vq.KPoints.monkhorst_pack(system, [1, 1, 1], shift=[int(shifted), 0, 0])
    if kind == "native":
        mesh = mesh.to_bloch_kmesh()
    elif kind == "cartesian_only":
        mesh = SimpleNamespace(kpoints_cart=mesh.kpoints_cart, n_kpoints=1)
    assert _is_multik_kpoints(mesh) is shifted


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("point", [[0., 0., 0.], [1., 0., 0.], [.25, 0., 0.]])
@pytest.mark.parametrize("per_k", [False, True])
def test_gamma_memory_read_checks_source_point_even_for_real_density(restart_case, opened, point, per_k):
    from vibeqc.guess_read import resolve_periodic_read_density_closed
    source, target, _, _, _ = restart_case
    source.restart_kpoints = np.asarray([point]) @ (2 * np.pi * np.linalg.inv(source.restart_lattice))
    source.restart_weights = np.ones(1)
    total = np.diag([1.5, .5])
    if opened:
        source.density_alpha = [total / 2] if per_k else total / 2
        source.density_beta = [total / 2] if per_k else total / 2
    else:
        source.density = [total] if per_k else total
    if point[0] == .25:
        with pytest.raises(ValueError, match="non-Gamma"):
            resolve_periodic_read_density_closed(target["basis"], read_from=source)
    else:
        actual = resolve_periodic_read_density_closed(target["basis"], read_from=source)
        np.testing.assert_allclose(actual, total, atol=1e-12)


@pytest.mark.parametrize("point", [[0., 0., 0.], [.25, 0., 0.]])
def test_legacy_selected_qvf_checks_gamma_coordinates(tmp_path, restart_case, point):
    from vibeqc.guess_read import resolve_periodic_read_density_closed
    from vibeqc.output.formats.qvf import qvf_wf_data, write_qvf
    from vibeqc.output.plan import OutputPlan
    _, target, _, _, _ = restart_case
    system, basis = target["system"], target["basis"]
    state = SimpleNamespace(mo_coeffs=np.eye(2), occupations=np.array([1.5, .5]),
                            mo_energies=np.array([-.5, .1]), converged=True, energy=-1., n_iter=1)
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path / "legacy", method="rhf",
        basis="sto-3g", functional=None, job_kind="periodic_scf")
    path = write_qvf(tmp_path / "legacy", plan, system=system, result=state,
        method="rhf", basis="sto-3g", wf_data=qvf_wf_data(
            state, basis, system.unit_cell_molecule(), k_point=point))
    if point[0] != 0:
        with pytest.raises(NotImplementedError, match="non-Gamma"):
            resolve_periodic_read_density_closed(basis, read_from=path)
    else:
        np.testing.assert_allclose(resolve_periodic_read_density_closed(basis, read_from=path),
                                   np.diag([1.5, .5]), atol=1e-12)


def _write_spin_restart(tmp_path, source, target):
    from vibeqc.output.formats.qvf import qvf_bloch_wf_data, write_qvf, validate_qvf
    from vibeqc.output.plan import OutputPlan
    system, basis = target["system"], target["basis"]
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path / "spin", method="uhf",
        basis="sto-3g", functional=None, job_kind="periodic_scf")
    payload = qvf_bloch_wf_data(source, basis, system.unit_cell_molecule(),
        k_points=source.restart_kpoints @ np.asarray(system.lattice) / (2 * np.pi),
        k_weights=source.restart_weights)
    assert payload["mo_metadata"]["restart_kind"] == "spin_density_bloch_kpoints"
    path = write_qvf(tmp_path / "spin", plan, system=system,
        result=SimpleNamespace(converged=True, energy=-1., n_iter=1), method="uhf",
        basis="sto-3g", bloch_wf_data=payload)
    report = validate_qvf(path)
    assert report["valid"], report["errors"]
    return path


def test_spin_qvf_roundtrip_preserves_physical_state(tmp_path, restart_case):
    source, target, a, b, order = restart_case
    source.density_alpha_k, source.density_beta_k = a, b
    # These are deliberately unrelated orbitals from a later diagonalization.
    source.mo_coeffs_alpha_k = source.mo_coeffs_beta_k = [np.eye(2)] * 3
    source.occupations_alpha_k = source.occupations_beta_k = [np.zeros(2)] * 3
    # Restricted-open results can also expose a common orbital set.
    source.mo_coeffs, source.occupations = [np.eye(2)] * 3, [np.array([2., 0.])] * 3
    path = _write_spin_restart(tmp_path, source, target)
    pair = resolve_periodic_read_densities_k_open(read_path=str(path), **target)
    np.testing.assert_array_equal(pair[0], np.asarray(a)[order])
    np.testing.assert_array_equal(pair[1], np.asarray(b)[order])
    total = resolve_periodic_read_density_k_closed(read_from=path, **target)
    np.testing.assert_array_equal(total, (np.asarray(a) + b)[order])


@pytest.mark.parametrize("fault", ["missing_beta", "short", "negative", "nonhermitian", "weights"])
def test_spin_qvf_writer_rejects_incomplete_state(tmp_path, restart_case, fault):
    source, target, a, b, _ = restart_case
    source.density_alpha_k, source.density_beta_k = a, b
    if fault == "missing_beta":
        del source.density_beta_k
    elif fault == "short":
        source.density_beta_k = b[:2]
    elif fault == "negative":
        b[0][0, 0] = -1
    elif fault == "nonhermitian":
        b[0][0, 1] += 1
    else:
        source.restart_weights = None
    with pytest.raises(ValueError):
        _write_spin_restart(tmp_path, source, target)
    assert not (tmp_path / "spin.qvf").exists()


@pytest.mark.parametrize("fault", ["missing_member", "alias", "short", "weights", "layout", "components", "metadata_checksum", "checksum", "negative", "nonhermitian"])
def test_spin_qvf_reader_rejects_corrupt_state(tmp_path, restart_case, fault):
    import hashlib
    import json
    import zipfile
    source, target, a, b, _ = restart_case
    source.density_alpha_k, source.density_beta_k = a, b
    path = _write_spin_restart(tmp_path, source, target)
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["manifest.json"])
    section = next(s for s in manifest["sections"] if s["kind"] == "x_vibeqc.bloch_wavefunction")
    members = section["members"]
    meta_path = members["mo_metadata"]["path"]
    meta = json.loads(contents[meta_path])
    if fault == "missing_member":
        del members[meta["blocks"][0]["density_beta"]]
    elif fault == "alias":
        meta["blocks"][0]["density_beta"] = meta["blocks"][0]["density_alpha"]
    elif fault == "short":
        meta["blocks"].pop()
    elif fault == "weights":
        meta["blocks"][0]["k_weight"] = -1
    elif fault == "layout":
        meta["density_layout"] = "transposed"
    elif fault == "components":
        meta["density_components"] = ["imag", "real"]
    elif fault == "metadata_checksum":
        meta["blocks"][0]["k_weight"], meta["blocks"][1]["k_weight"] = .3, .2
    else:
        member = members[meta["blocks"][0]["density_beta"]]
        data = np.frombuffer(contents[member["path"]], dtype=np.float64).reshape(member["shape"]).copy()
        if fault == "negative":
            data[0, 0, 0] = -1
        elif fault == "nonhermitian":
            data[0, 1, 0] += 1
        else:
            data *= 2
        contents[member["path"]] = data.tobytes()
        if fault != "checksum":
            member["sha256"] = hashlib.sha256(contents[member["path"]]).hexdigest()
    contents[meta_path] = json.dumps(meta).encode()
    if fault != "metadata_checksum":
        members["mo_metadata"]["sha256"] = hashlib.sha256(contents[meta_path]).hexdigest()
    contents["manifest.json"] = json.dumps(manifest).encode()
    corrupt = tmp_path / "corrupt.qvf"
    with zipfile.ZipFile(corrupt, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    for resolver in (resolve_periodic_read_density_k_closed, resolve_periodic_read_densities_k_open):
        with pytest.raises(ValueError):
            resolver(read_from=corrupt, **target)


@pytest.mark.parametrize("method,route", [("UHF", "gdf"), ("UKS", "gdf"), ("ROHF", "gdf"),
    ("UHF", "bipole"), ("ROKS", "bipole"), ("UKS", "gpw"), ("ROKS", "gpw")])
@pytest.mark.parametrize("nk", [1, 3])
def test_public_spin_qvf_restart(tmp_path, restart_case, method, route, nk):
    from vibeqc.guess_read import resolve_periodic_read_densities_open
    _, target, _, _, _ = restart_case
    system, basis = target["system"], target["basis"]
    source_system = vq.PeriodicSystem(3, system.lattice, system.unit_cell, multiplicity=3)
    kwargs = dict(method=method, jk_method=route, kpoints=[nk, 1, 1], max_iter=40,
        convergence="off", output_qvf=False, write_molden_file=False, write_density=False,
        write_xyz_file=False, write_cif_file=False, write_xsf_structure_file=False,
        write_population_file=False, citations=False, progress=False)
    if method.endswith("KS"):
        kwargs["functional"] = "lda"
    if route == "gdf":
        kwargs.update(rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.)
    if route == "gpw":
        kwargs["cutoff_ha"] = 8.
    source = vq.run_periodic_job(source_system, basis, initial_guess="HCORE",
        output=tmp_path / "source", **dict(kwargs, output_qvf=True))
    archive = tmp_path / "source.qvf"
    assert source.converged and archive.exists()
    mesh = vq.monkhorst_pack(system, [nk, 1, 1])
    expected = resolve_periodic_read_densities_k_open(read_from=source, basis=basis, system=system, kmesh=mesh)
    actual = resolve_periodic_read_densities_k_open(read_from=archive, basis=basis, system=system, kmesh=mesh)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    if nk == 1:
        pair = resolve_periodic_read_densities_open(basis, read_from=archive)
        np.testing.assert_allclose(pair, [actual[0][0].real, actual[1][0].real], atol=1e-12)
    restarted = vq.run_periodic_job(system, basis, initial_guess="READ", read_from=archive,
                                   output=tmp_path / "restart", **kwargs)
    reference = vq.run_periodic_job(system, basis, initial_guess="HCORE", output=tmp_path / "reference", **kwargs)
    assert restarted.converged and reference.converged
    assert restarted.guess_selection.effective == vq.InitialGuess.READ
    assert restarted.energy == pytest.approx(reference.energy, abs=1e-8)


@pytest.mark.parametrize("representation", ["ragged", "stacked", "gamma"])
def test_spin_qvf_mo_only_sources(tmp_path, restart_case, representation):
    source, target, _, _, order = restart_case
    ca = [np.array([[1.], [1j]]) / np.sqrt(2)] * 3
    cb = [np.eye(2)] * 3
    oa, ob = [np.array([.8])] * 3, [np.array([.1, .4])] * 3
    if representation == "gamma":
        source.restart_kpoints, source.restart_weights = np.zeros((1, 3)), np.ones(1)
        target["kmesh"] = SimpleNamespace(kpoints=source.restart_kpoints, weights=source.restart_weights)
        ca, cb, oa, ob = np.array([[1.], [0.]]), cb[0], oa[0], ob[0]
    elif representation == "stacked":
        ca, cb, oa, ob = map(np.asarray, (ca, cb, oa, ob))
    source.mo_coeffs_alpha, source.mo_coeffs_beta = ca, cb
    source.occupations_alpha, source.occupations_beta = oa, ob
    path = _write_spin_restart(tmp_path, source, target)
    actual = resolve_periodic_read_densities_k_open(read_from=path, **target)
    expected_a = np.diag([.8, 0.]) if representation == "gamma" else np.array([[.4, -.4j], [.4j, .4]])
    count = 1 if representation == "gamma" else 3
    np.testing.assert_allclose(actual[0], [expected_a] * count, atol=1e-12)
    np.testing.assert_allclose(actual[1], [np.diag([.1, .4])] * count, atol=1e-12)


@pytest.mark.parametrize('family',['rhf','rks'])
def test_native_bloch_qvf_restart_roundtrip(tmp_path,family):
    from vibeqc.output.formats.qvf import qvf_bloch_wf_data,write_qvf,validate_qvf
    from vibeqc.output.plan import OutputPlan
    system=vq.PeriodicSystem(3,16*np.eye(3),[vq.Atom(1,[0,0,0]),vq.Atom(1,[0,0,1.4])])
    basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[1,1,3])
    opts=vq.PeriodicSCFOptions() if family=='rhf' else vq.PeriodicKSOptions()
    opts.max_iter=30; opts.lattice_opts.cutoff_bohr=3; opts.lattice_opts.nuclear_cutoff_bohr=3
    run=getattr(vq,'run_'+family+'_periodic')
    source=run(system,basis,mesh,opts)
    assert source.converged
    payload=qvf_bloch_wf_data(source,basis,system.unit_cell_molecule(),
        k_points=np.asarray(mesh.kpoints)@np.asarray(system.lattice)/(2*np.pi), k_weights=mesh.weights)
    assert len(payload['mo_coefficients'])==3
    plan=OutputPlan.from_run_job_kwargs(output=tmp_path/'native',method=family,basis='sto-3g',
        functional=None if family=='rhf' else 'lda',job_kind='periodic_scf')
    path=write_qvf(tmp_path/'native',plan,system=system,result=source,method=family,
        basis='sto-3g',bloch_wf_data=payload)
    assert validate_qvf(path)['valid']
    opts.initial_guess=vq.InitialGuess.READ
    result=run(system,basis,mesh,opts,read_from=path)
    assert result.converged
    assert result.energy==pytest.approx(source.energy,abs=1e-8)
