"""Bounded admission tests for the maintained public Gamma GDF driver.

Load its actual function body and alias without importing the native extension.
Native boundaries are sentinels: no integrals, grids or SCF are evaluated.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import numpy as np


class _ReachedWork(RuntimeError):
    pass


class _WithoutImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return ast.copy_location(ast.Pass(), node)


def _driver_case(*, dim=3, cells=7, functional="", private_slab=False):
    events = []

    def work(stage):
        def sentinel(*args, **kwargs):
            events.append(stage)
            raise _ReachedWork(stage)
        return sentinel

    def lattice_cells(system, cutoff):
        events.append("cells")
        return range(cells)

    opts = SimpleNamespace(
        initial_guess="hcore", lattice_opts=SimpleNamespace(
            cutoff_bohr=4.0, nuclear_cutoff_bohr=4.0,
            coulomb_method="slab" if private_slab else "ewald"),
        functional=functional, level_shift=0.0, max_iter=4,
    )
    system = SimpleNamespace(
        dim=dim, lattice=np.eye(3), multiplicity=1, symmetry=None,
        unit_cell_molecule=lambda: None, n_electrons=lambda: 2,
    )
    logger = SimpleNamespace(info=lambda *a, **k: None,
                             stage=lambda *a, **k: nullcontext())
    ns = dict(
        np=np, _refuse_ecp_options=lambda *a, **k: None,
        select_initial_guess=lambda *a, **k: SimpleNamespace(effective="hcore"),
        periodic_guess_capabilities=lambda *a, **k: (),
        resolve_progress=lambda *a, **k: logger,
        Functional=lambda *a: SimpleNamespace(hf_exchange_fraction=0.0),
        reject_unscreened_range_separated=lambda *a, **k: None,
        _resolve_fock_mixing=lambda *a: 0.0,
        _resolve_level_shift_warmup_cycles=lambda *a, **k: 0,
        default_aux_for=lambda *a: "example-jk",
        direct_lattice_cells=lattice_cells,
        dump_active_settings=lambda *a: None,
        _density_set_gamma=None, GridOptions=lambda: None,
        build_grid=work("grid"), build_periodic_becke_grid=work("grid"),
        _gauge_lat_opts_for_v_ne_and_e_nuc=work("ewald gauge"),
        compute_overlap_lattice=work("overlap"),
        CoulombMethod=SimpleNamespace(SLAB_EWALD_2D="slab"),
        _WIRE_VACUUM_BOX_HINT="Use a 3D vacuum box.",
    )
    path = Path(__file__).resolve().parents[1] / "python/vibeqc/periodic_rhf_gdf.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "run_rhf_periodic_gamma_gdf"
    ) or (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == "run_rks_periodic_gamma_gdf" for t in node.targets)
    )]
    tree = _WithoutImports().visit(tree)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    basis = SimpleNamespace(name="example", nbasis=2, nshells=2)
    return ns, system, basis, opts, events


def _recommend_case():
    """Load the real classifier/recommend flow, with mesh/native sentinels."""
    ns, system, basis, opts, events = _driver_case()
    root = Path(__file__).resolve().parents[1] / "python/vibeqc"
    path = root / "kpoints.py"
    tree = ast.parse(path.read_text())
    functions = {"_classify_pre_scf", "_classify_kpoint_character",
                 "_delta_k_for_character", "_format_kpoint_rationale"}
    constants = {"GAP_METAL_EPS_EV", "GAP_INSULATOR_EV", "DELTA_K_METAL",
                 "DELTA_K_SMALL_GAP", "DELTA_K_INSULATOR",
                 "TOLERANCE_MEV_PER_ATOM_DEFAULT", "GAMMA_ONLY_CELL_VOLUME_BOHR3"}
    selected = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in functions
    ) or (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KPoints")
    recommend = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "recommend")
    recommend.decorator_list = []
    selected.append(recommend)
    # Use the maintained smearing preset, without importing the native package.
    presets = ast.parse((root / "smearing/resolution.py").read_text())
    selected.extend(n for n in presets.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SMEARING_PRESETS"
                            for t in n.targets))
    tree.body = selected
    tree = _WithoutImports().visit(tree)

    def unexpected_gap(*args, **kwargs):
        raise AssertionError("Rejected SCF must not produce a gap")

    def mesh(system, dk, **kwargs):
        events.append(("mesh spacing", dk))
        return SimpleNamespace(mesh=(2, 2, 2), grid_type="mp",
                               is_symmetry_reduced=False, n_kpoints=8)

    ns.update(PeriodicRHFOptions=lambda: opts,
              _system_n_electrons=lambda s: s.n_electrons(),
              converged_gap_hartree=unexpected_gap, SmearingOptions=SimpleNamespace,
              _system_dim=lambda s: s.dim, _spacegroup_number=lambda s: None,
              _is_hexagonal_like=lambda *a: False,
              _system_cell_volume_bohr3=lambda s: float(np.linalg.det(s.lattice)),
              _mesh_is_gamma_only=lambda *a: False)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    builder = SimpleNamespace(from_kspacing=mesh)
    return ns, system, basis, opts, events, builder


class GammaGDFAdmission(unittest.TestCase):
    def test_public_rhf_and_rks_refuse_image_cells_before_work(self):
        for functional, driver in (("", "run_rhf_periodic_gamma_gdf"),
                                   ("pbe", "run_rks_periodic_gamma_gdf")):
            with self.subTest(functional=functional):
                ns, system, basis, opts, events = _driver_case(functional=functional)
                self.assertIs(ns["run_rhf_periodic_gamma_gdf"], ns["run_rks_periodic_gamma_gdf"])
                try:
                    ns[driver](system, basis, opts)
                except NotImplementedError as exc:
                    self.assertIn("molecular-limit exchange", str(exc))
                    self.assertIn("run_periodic_job", str(exc))
                    self.assertIn("run_pbc_gdf_rhf", str(exc))
                except _ReachedWork as exc:
                    self.fail(f"Unsupported public compact cell was admitted to {exc}")
                else:
                    self.fail("Unsupported public compact cell was silently admitted")
                self.assertNotIn("grid", events)
                self.assertNotIn("overlap", events)
                self.assertNotIn("ewald gauge", events)

    def test_home_cell_and_dim2_admission_is_unchanged(self):
        for dim, cells, slab in ((3, 1, False), (2, 7, False), (2, 7, True)):
            with self.subTest(dim=dim, cells=cells, private_slab=slab):
                ns, system, basis, opts, _ = _driver_case(dim=dim, cells=cells, private_slab=slab)
                with self.assertRaisesRegex(_ReachedWork, "overlap"):
                    ns["run_rhf_periodic_gamma_gdf"](system, basis, opts, _slab_truncated_private=slab)

    def test_private_torus_force_retains_ewald_admission(self):
        for cells in (1, 7):
            with self.subTest(cells=cells):
                ns, system, basis, opts, _ = _driver_case(cells=cells)
                with self.assertRaisesRegex(_ReachedWork, "ewald gauge"):
                    ns["run_rhf_periodic_gamma_gdf"](system, basis, opts, _force_ewald_jk=True)

    def test_invalid_force_dimension_and_public_wire_still_refuse(self):
        for dim in (1, 2):
            with self.subTest(dim=dim):
                ns, system, basis, opts, events = _driver_case(dim=dim)
                with self.assertRaisesRegex(ValueError, "requires a fully 3D-periodic system"):
                    ns["run_rhf_periodic_gamma_gdf"](system, basis, opts, _force_ewald_jk=True)
                self.assertEqual(events, [])
        ns, system, basis, opts, events = _driver_case(dim=1)
        with self.assertRaisesRegex(NotImplementedError, "no consistent Coulomb gauge"):
            ns["run_rhf_periodic_gamma_gdf"](system, basis, opts)
        self.assertEqual(events, [])


    def test_actual_pre_scf_helper_and_recommend_use_safe_fallback(self):
        ns, system, basis, opts, events, builder = _recommend_case()
        self.assertEqual(ns["_classify_pre_scf"](system, basis, warnings), (None, None))
        self.assertEqual(events, ["cells"])
        self.assertEqual((opts.max_iter, opts.conv_tol_energy, opts.use_diis),
                         (3, 1e-4, False))
        events.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = ns["recommend"](builder, system, classifier="pre-scf",
                                   basis=basis, symmetry=False)
        self.assertTrue(any("assuming METALLIC" in str(w.message) for w in caught))
        self.assertTrue(spec.rationale.startswith("metal"))
        self.assertNotIn("character from pre-SCF", spec.rationale)
        self.assertEqual(spec.smearing.temperature, ns["SMEARING_PRESETS"]["metal"])
        self.assertEqual(events, ["cells", ("mesh spacing", ns["DELTA_K_METAL"])])

    def test_known_gap_bypasses_unavailable_pre_scf(self):
        ns, system, basis, _, events, builder = _recommend_case()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = ns["recommend"](builder, system, band_gap=2.0,
                                   classifier="pre-scf", basis=basis, symmetry=False)
        self.assertEqual(caught, [])
        self.assertTrue(spec.rationale.startswith("insulator"))
        self.assertIsNone(spec.smearing)
        self.assertEqual(events, [("mesh spacing", ns["DELTA_K_INSULATOR"])])


@contextmanager
def _real_solver_adapter():
    """Load the maintained Python adapter without importing the native package."""
    import importlib
    import sys
    import types

    name = "_gamma_gdf_admission_solvers"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/vibeqc/solvers")]
    previous = {key: value for key, value in sys.modules.items()
                if key == name or key.startswith(name + ".")}
    sys.modules[name] = package
    try:
        yield importlib.import_module(name + "._scf")
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key)
        sys.modules.update(previous)


def _scalar_driver_case(route, fit, seed, scalar, adapter):
    """Actual driver bodies, with native setup and options represented by stubs."""
    import enum
    import os
    from unittest.mock import patch

    class Guess(enum.Enum):
        HCORE = 1
        SAD = 2
        PATOM = 3
        SAP = 4
        HUECKEL = 5
    class ExxDiv(enum.Enum):
        EWALD = "ewald"
    legacy = route.startswith("legacy")
    method = route.split("_")[-1]
    functional = "pbe" if method in {"rks", "uks"} else ""
    ns, system, basis, opts, events = _driver_case(cells=1, functional=functional)
    system.unit_cell = [SimpleNamespace(Z=2)]
    opts.initial_guess = getattr(Guess, seed)
    opts.use_davidson = scalar != "inactive_guess"
    opts.davidson_min_dim = 0
    opts.conv_tol_energy = 1e-8
    opts.davidson = SimpleNamespace(
        n_eig=2, n_guess=2, max_refine=5, max_subspace=32, conv_tol=1e-8,
        max_iter=100, preshift=0., verbosity=0,
        guess_vectors=np.eye(2), guess_vectors_cplx=np.empty((0, 0), dtype=complex))
    if scalar in {"complex_guess", "inactive_guess"}:
        opts.davidson.guess_vectors_cplx = np.eye(2) * np.exp(.4j)
    x = np.eye(2) * np.exp(.4j) if scalar == "complex_frame" else np.eye(2)
    seen = []
    class StopAtSeed(RuntimeError):
        pass
    class Diagonalizer(adapter.FockDiagonalizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)
        def __call__(self, fock):
            if scalar == "complex_fock":
                fock = np.array([[-1., .2j], [-.2j, 1.]])
            coefficients, energies = super().__call__(fock)
            assert not np.iscomplexobj(coefficients)
            np.testing.assert_allclose(coefficients.T @ coefficients, np.eye(2), atol=1e-12)
            assert np.trace(coefficients[:, :1] @ coefficients[:, :1].T) == 1.
            events.append("seed solve")
            raise StopAtSeed
    def fit_setup(*args, **kwargs):
        events.append("fit setup")
        return SimpleNamespace(S=np.eye(2), Hcore=np.diag([-1., 1.]), X=x,
                               Lpq=np.zeros((1, 2, 2)), aux=None, madelung=0.,
                               e_nuc=0., gauge_lat_opts=opts.lattice_opts, n_kept=2)
    def lpq(*args, **kwargs):
        return fit_setup().Lpq
    def no_jk(*args, **kwargs):
        raise AssertionError("No density-dependent J/K is allowed before seed admission")
    noop = lambda *a, **k: None
    ns.update(
        os=os, warnings=warnings, PBCExxDiv=ExxDiv, InitialGuess=Guess,
        _integer_counts=lambda values, **k: values,
        SpinlockMode=SimpleNamespace(OFF=0, PATTERN_HOLD=1, SPIN_SCHEDULE=2),
        check_spinlock_support=noop, refuse_unimplemented_gamma_controls=noop,
        _coerce_periodic_driver_guess=lambda *a, **k: opts.initial_guess,
        select_initial_guess=lambda *a, **k: SimpleNamespace(effective=opts.initial_guess),
        _resolve_rsgdf_ke_cutoff=lambda v: (v, 40.),
        _warn_gamma_dense_core_gdf_parity_hold=lambda *a, **k: False,
        reject_periodic_gdf_unsupported_functional=noop,
        _pbc_gdf_gamma_setup=fit_setup, FockDiagonalizer=Diagonalizer,
        compute_overlap_lattice=lambda *a: np.eye(2),
        compute_kinetic_lattice=lambda *a: np.diag([-1., 1.]),
        compute_nuclear_lattice_dispatch=lambda *a: np.zeros((2, 2)),
        bloch_sum=lambda matrix, *a: matrix,
        _canonical_orthogonalizer=lambda *a: (x, 2),
        scf_preflight_overlap_check=noop, make_aux_basis_set=lambda *a, **k: basis,
        build_lpq_native=lpq, nuclear_repulsion_per_cell=lambda *a: 0.,
        _build_j_from_lpq=no_jk, _build_k_from_lpq=no_jk,
        build_grid=noop, _SmearingOptions=SimpleNamespace(from_legacy_kwarg=noop),
        _occ_tol_fn=lambda *a: 1e-8,
        initial_density_closed_shell=lambda *a, **k:
            None if seed == "HCORE" else np.diag([2., 0.]),
    )
    # The legacy experimental fit warning is setup, not SCF progress.
    ns["resolve_progress"] = lambda *a, **k: SimpleNamespace(
        info=noop, warn=noop, stage=lambda *a, **k: nullcontext())
    if legacy:
        run = ns["run_" + method + "_periodic_gamma_gdf"]
        kwargs = dict(gdf_algorithm=fit)
    else:
        path = Path(__file__).resolve().parents[1] / "python/vibeqc/pbc_gdf.py"
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"run_pbc_gdf_rhf", "run_pbc_gdf_rks",
                                    "run_pbc_gdf_uhf", "run_pbc_gdf_uks"}]
        tree = _WithoutImports().visit(tree)
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
        run = ns["run_pbc_gdf_" + method]
        kwargs = dict(gdf_method=fit, rcut_strategy=None)
    solve = adapter.solve_operator
    def dense_control(operator, options, *, solver, previous):
        assert solver == ("dense" if scalar == "inactive_guess" else "davidson")
        return solve(operator, adapter.SolverOptions(n_roots=options.n_roots), solver="dense")
    errors = {"complex_guess": "complex Davidson guesses",
              "complex_frame": "real X and overlap", "complex_fock": "complex Fock matrices"}
    with patch.object(adapter, "solve_operator", dense_control):
        error = NotImplementedError if scalar in errors else StopAtSeed
        try:
            run(system, basis, opts, **kwargs)
        except error as exc:
            if scalar in errors:
                assert errors[scalar] in str(exc)
        else:
            raise AssertionError("Expected scalar admission or seed sentinel")
    assert events.count("fit setup") == 1  # This guard is deliberately after fit setup.
    assert events.count("seed solve") == (0 if scalar in errors else 1)
    if scalar in {"real", "inactive_guess"}:
        assert len(seen) == (2 if method in {"uhf", "uks"} else 1)
        # Check both actual spin-channel instances, including beta whose HCORE
        # orbitals initially copy alpha. Later PATOM/SCF beta solves use this guard.
        for diagonalizer in seen:
            try:
                adapter.FockDiagonalizer.__call__(diagonalizer, np.array([[-1., .2j], [-.2j, 1.]]))
            except NotImplementedError as exc:
                assert "complex Fock matrices" in str(exc)
            else:
                raise AssertionError("A real-only channel accepted a complex Fock")


class GammaGDFScalarAdmission(unittest.TestCase):
    def test_real_only_public_driver_seed_admission(self):
        routes = [("legacy_" + method, fit) for method in ("rhf", "rks")
                  for fit in ("bare", "rsgdf")]
        routes += [("pbc_" + method, fit) for method in ("rhf", "rks", "uhf", "uks")
                   for fit in ("rsgdf", "compcell", "mdf")]
        with _real_solver_adapter() as adapter:
            for route, fit in routes:
                for seed in ("HCORE", "SAD", "PATOM"):
                    for scalar in ("real", "complex_guess", "complex_frame", "complex_fock", "inactive_guess"):
                        with self.subTest(route=route, fit=fit, seed=seed, scalar=scalar):
                            _scalar_driver_case(route, fit, seed, scalar, adapter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
