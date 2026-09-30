"""Source/metadata contracts for #316; no native import or QC calculation.

Standalone admission-free command: python tests/test_bipole_domain_contract.py
Native relabelling, raw-energy and force acceptance is deliberately separate.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from enum import Enum
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "vibeqc"
SPEC = importlib.util.spec_from_file_location("_bipole_domain_contract", PACKAGE / "bipole_domain.py")
domain = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = domain
SPEC.loader.exec_module(domain)


class CoulombMethod(Enum):
    DIRECT_TRUNCATED = 0
    EWALD_3D = 1


class Angular(Enum):
    ProductGaussLegendre = 0
    Lebedev = 1


class PeriodicXCDensityDomain(Enum):
    AUTO = 0
    MOLECULAR_HOME = 1
    PERIODIC_LATTICE = 2


def lattice():
    values = dict.fromkeys(domain.LATTICE_FIELDS, 0.0)
    values.update(cutoff_bohr=15.0, nuclear_cutoff_bohr=15.0,
                  schwarz_threshold=1e-12, schwarz_threshold_forces=1e-14,
                  pair_complete_1e=True, sr_range_screening=True,
                  sr_sparse_traversal=True, coulomb_method=CoulombMethod.DIRECT_TRUNCATED)
    return SimpleNamespace(**values)


def grid():
    values = dict.fromkeys(domain.GRID_FIELDS, 1)
    values.update(n_radial=200, angular=Angular.Lebedev, lebedev_order=53,
                  orca_angular_points=[50, 110, 194, 110, 50])
    return SimpleNamespace(**values)


def capture(**changes):
    args = dict(lattice_1e=lattice(), lattice_2e=lattice(),
                ewald_options=SimpleNamespace(real_cutoff_bohr=19.0),
                alpha=0.4, reciprocal_cutoff=3.404, ewald_precision=1e-12,
                sr_image_extent=28.0, use_ewald_j_split=True,
                exchange_ewald_split=True, exchange_exxdiv="ewald", exact_zone=None,
                ks_options=SimpleNamespace(grid=grid(), use_periodic_becke=True,
                                           becke_image_radius_bohr=18.0))
    args.update(changes)
    return domain.capture_bipole_domain(**args)


def function(name, file="bipole_gradient.py", **namespace):
    tree = ast.parse((PACKAGE / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Future annotations avoid loading any numerical/native type annotations.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, file, "exec"), namespace)
    return namespace[name]


class DomainContractTests(unittest.TestCase):
    def test_named_policy_overrides_native_flag_without_mutating_it(self):
        clone = function("copy_bipole_lattice_options", file="pbc_bipole_common.py",
            OptionSnapshot=domain.OptionSnapshot, LATTICE_FIELDS=domain.LATTICE_FIELDS,
            LatticeSumOptions=lattice)
        resolve = function("resolve_bipole_lattice_options", file="pbc_bipole_common.py",
            resolve_bipole_pair_policy=domain.resolve_bipole_pair_policy,
            copy_bipole_lattice_options=clone)
        for native_flag in (False, True):
            original = lattice()
            original.pair_complete_1e = native_flag
            self.assertTrue(resolve(original).pair_complete_1e)
            self.assertFalse(resolve(original, historical_domain=True).pair_complete_1e)
            self.assertEqual(original.pair_complete_1e, native_flag)
        for invalid in (None, 0, 1, "false", "true"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                resolve(original, historical_domain=invalid)
        with self.assertRaisesRegex(ValueError, "requires ab initio BIPOLE"):
            domain.resolve_bipole_pair_policy(True, active=False)
        domain.resolve_bipole_pair_policy(False, active=False)

    def test_requested_options_replay_resolved_policy_and_screening(self):
        original = lattice()
        original.pair_complete_1e = False
        original.sr_range_screening = False
        original.nuclear_cutoff_bohr = 25.0
        record = capture(requested_lattice=original)
        result = SimpleNamespace(resolved_domain=record, **{
            k: getattr(record, k) for k in ("ewald_alpha_bohr_inv", "ewald_precision",
                "sr_image_extent_bohr", "exact_zone_bohr", "exchange_ewald_split")})
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice)
        effective = restore(result, original)
        self.assertTrue(effective.pair_complete_1e)
        self.assertTrue(effective.sr_range_screening)
        self.assertEqual(effective.nuclear_cutoff_bohr, 15.0)
        self.assertFalse(original.pair_complete_1e)
        self.assertFalse(original.sr_range_screening)
        # Already resolved settings are accepted too. Arbitrary control changes
        # match neither record and must fail rather than silently restoring.
        self.assertEqual(restore(result, effective), effective)
        original.cutoff_bohr = 10.0
        with self.assertRaisesRegex(ValueError, "cutoff_bohr"):
            restore(result, original)

    def test_policy_reaches_all_runner_and_spin_schedule_dispatches(self):
        names = {"run_pbc_bipole_" + m for m in ("rhf", "uhf", "rks", "uks")}
        runner = ast.parse((PACKAGE / "periodic_runner.py").read_text())
        calls = [n for n in ast.walk(runner) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in names]
        self.assertEqual(len(calls), 6)
        for call in calls:
            self.assertEqual(ast.unparse(next(k.value for k in call.keywords
                if k.arg == "historical_domain")), "bipole_historical_domain")
        optimization, = [n.value for n in ast.walk(runner) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_bipole_opt_driver_kwargs" for t in n.targets)]
        options = dict(zip((k.value for k in optimization.keys), optimization.values))
        self.assertEqual(ast.unparse(options["historical_domain"]), "bipole_historical_domain")
        for suffix in ("", "_uhf", "_rks", "_uks"):
            tree = ast.parse((PACKAGE / f"pbc_bipole{suffix}.py").read_text())
            entry, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            defaults = dict(zip((a.arg for a in entry.args.kwonlyargs), entry.args.kw_defaults))
            self.assertIs(defaults["historical_domain"].value, False)
            for call in ast.walk(entry):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == entry.name:
                    self.assertEqual(ast.unparse(next(k.value for k in call.keywords
                        if k.arg == "historical_domain")), "historical_domain")
            capture_call, = [n for n in ast.walk(entry) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "capture_bipole_domain"]
            self.assertEqual(ast.unparse(next(k.value for k in capture_call.keywords
                if k.arg == "requested_lattice")), "opts.lattice_opts")

    def test_optimizer_and_fd_forward_policy_without_running_a_calculation(self):
        controls = dict(historical_domain=True, ewald_precision=1e-12,
                        sr_image_precision=None)
        result = SimpleNamespace(converged=True, energy=-1.0)
        drivers = {"run_pbc_bipole_" + m: Mock(return_value=result)
                   for m in ("rhf", "uhf", "rks", "uks")}
        optimize = function("_run_scf", file="bipole_optimize.py", **drivers)
        for method in ("RHF", "UHF", "RKS", "UKS"):
            optimize("system", "basis", "mesh", "options", method, "pbe", **controls)
            driver = drivers["run_pbc_bipole_" + method.lower()]
            for key, value in controls.items():
                self.assertEqual(driver.call_args.kwargs[key], value)
        # Evaluate the actual four FD call expressions with stub drivers. This
        # tests forwarding only; no native imports or displaced geometries.
        tree = ast.parse((PACKAGE / "bipole_gradient.py").read_text())
        fd = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "compute_bipole_gradient_fd")
        calls = [n for n in ast.walk(fd) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in drivers]
        self.assertEqual(len(calls), 4)
        for call in calls:
            eval(compile(ast.Expression(call), "fd-forwarding", "eval"), dict(
                drivers, sys_disp="displaced-system", basis_disp="displaced-basis",
                kmesh="mesh", options="options", functional="pbe", bipole_kwargs=controls))
            received = drivers[call.func.id].call_args
            self.assertEqual(received.args, ("displaced-system", "displaced-basis", "mesh", "options"))
            for key, value in controls.items():
                self.assertEqual(received.kwargs[key], value)

    def test_runner_records_planned_and_executed_domains(self):
        tree = ast.parse((PACKAGE / "periodic_runner.py").read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "_bipole_domain_active"
                 and isinstance(n.body[0], ast.Assign)
                 and isinstance(n.body[0].targets[0], ast.Subscript)]
        self.assertEqual(len(nodes), 3)
        for physical in (False, True):
            lat = lattice()
            lat.pair_complete_1e = physical
            record = capture(lattice_1e=lat, lattice_2e=lat)
            namespace = dict(_bipole_domain_active=True, _bipole_physical_pairs=physical,
                _dry_run_fields={}, _initial_run_fields={}, _manifest_run_fields={},
                result=SimpleNamespace(resolved_domain=record))
            for node in nodes:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "manifest", "exec"), namespace)
            for name in ("_dry_run_fields", "_initial_run_fields", "_manifest_run_fields"):
                self.assertEqual(namespace[name]["bipole_domain_policy"], record.policy)
            self.assertNotIn("bipole_resolved_domain", namespace["_dry_run_fields"])
            manifest = namespace["_manifest_run_fields"]
            self.assertEqual(manifest["bipole_resolved_domain"], record.as_dict())
            json.dumps(manifest, allow_nan=False)

    def test_snapshot_is_detached_including_grid_vectors(self):
        opts = SimpleNamespace(grid=grid(), use_periodic_becke=True, becke_image_radius_bohr=18.0)
        original = lattice()
        record = capture(lattice_2e=original, ks_options=opts)
        digest = record.settings_sha256
        original.cutoff_bohr = 2.0
        opts.grid.orca_angular_points[0] = 999
        opts.becke_image_radius_bohr = 3.0
        self.assertEqual(record.settings_sha256, digest)
        self.assertEqual(record.lattice_2e.restore(lattice).cutoff_bohr, 15.0)
        self.assertEqual(record.xc_grid.restore(grid).orca_angular_points[0], 50)
        self.assertEqual(record.becke_image_radius_bohr, 18.0)
        with self.assertRaises(FrozenInstanceError):
            record.ewald_precision = 1e-4
        json.dumps(record.as_dict(), allow_nan=False)

    def test_historical_policy_is_identified_and_changes_digest(self):
        historical = lattice()
        historical.pair_complete_1e = False
        old = capture(lattice_1e=historical, lattice_2e=historical)
        new = capture()
        self.assertEqual(old.policy, "historical_index_ball")
        self.assertEqual(new.policy, "physical_pairs")
        self.assertNotEqual(old.settings_sha256, new.settings_sha256)

    def test_snapshot_rejects_nonfinite_settings(self):
        opts = lattice()
        opts.cutoff_bohr = float("nan")
        with self.assertRaises(ValueError):
            capture(lattice_2e=opts)

    def test_exact_and_incremental_context_cannot_drift(self):
        record = capture()
        ctx = SimpleNamespace(lat_opts_2e=lattice(), omega_used=.4,
                              ewald_precision=1e-12, sr_image_extent=28.0,
                              use_ewald_j_split=True, exchange_split_active=True,
                              exact_zone_bohr=None, exact_j_for_pure_rks=False,
                              exact_j_ke_cutoff=200.0)
        record.require_fock_context(ctx)
        for field in ("cutoff_bohr", "pair_complete_1e", "schwarz_threshold_forces",
                      "eri_interaction_cutoff_bohr", "becke_image_radius_bohr"):
            saved = getattr(ctx.lat_opts_2e, field)
            setattr(ctx.lat_opts_2e, field, not saved if isinstance(saved, bool) else saved + 1)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                record.require_fock_context(ctx)
            setattr(ctx.lat_opts_2e, field, saved)
        ctx.sr_image_extent = 30.0
        with self.assertRaisesRegex(ValueError, "sr_image_extent"):
            record.require_fock_context(ctx)

    def test_exact_j_diagnostic_cannot_change_during_accumulator_reuse(self):
        record = capture(exact_j_ke_cutoff=160.0)
        ctx = SimpleNamespace(lat_opts_2e=lattice(), omega_used=.4,
                              ewald_precision=1e-12, sr_image_extent=28.0,
                              use_ewald_j_split=True, exchange_split_active=True,
                              exact_zone_bohr=None, exact_j_for_pure_rks=True,
                              exact_j_ke_cutoff=160.0)
        record.require_fock_context(ctx)
        ctx.exact_j_ke_cutoff = 200.0
        with self.assertRaisesRegex(ValueError, "exact_j_ke_cutoff"):
            record.require_fock_context(ctx)
        ctx.exact_j_ke_cutoff = 160.0
        ctx.exact_j_for_pure_rks = False
        with self.assertRaisesRegex(ValueError, "exact_j_ke_cutoff"):
            record.require_fock_context(ctx)
        ctx.exact_j_for_pure_rks = True
        with self.assertRaisesRegex(ValueError, "exact_j_ke_cutoff"):
            capture().require_fock_context(ctx)

    def test_ewald_contract_checks_real_reciprocal_alpha_and_precision(self):
        record = capture()
        options = SimpleNamespace(alpha=.4, real_cutoff_bohr=19.0,
                                  recip_cutoff_bohr_inv=3.404, tolerance=1e-12)
        record.require_ewald_options(options, context="gradient")
        for name in vars(options):
            saved = getattr(options, name)
            setattr(options, name, saved + 1)
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "gradient"):
                record.require_ewald_options(options, context="gradient")
            setattr(options, name, saved)
        with self.assertRaisesRegex(ValueError, "missing resolved Ewald"):
            record.require_ewald_options(None, context="gradient")
        capture(alpha=None, ewald_options=None, reciprocal_cutoff=None).require_ewald_options(
            None, context="non-Ewald")

    def test_gradient_rejects_regenerated_ewald_real_radius_drift_before_kernels(self):
        record = capture()
        result = SimpleNamespace(resolved_domain=record, **{
            k: getattr(record, k) for k in ("ewald_alpha_bohr_inv", "ewald_precision",
                "sr_image_extent_bohr", "exact_zone_bohr", "exchange_ewald_split")})
        package = ModuleType("_domain_test_package")
        native = ModuleType("_domain_test_package.bipole_ext_el_pole")
        native.bipole_ewald_reciprocal_cutoff = Mock(return_value=3.404)
        ewald = SimpleNamespace(alpha=.4, real_cutoff_bohr=19.0,
                               recip_cutoff_bohr_inv=3.404, tolerance=1e-12)
        match = Mock(return_value=ewald)
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice,
            __package__=package.__name__, _matching_ewald_options=match,
            np=SimpleNamespace(asarray=lambda x, **kw: x,
                               linalg=SimpleNamespace(det=lambda x: 125.0)))
        system = SimpleNamespace(dim=3, lattice="cell")
        with patch.dict(sys.modules, {package.__name__: package, native.__name__: native}):
            restored = restore(result, None, system=system)
            match.assert_called_once_with(system, restored, .4,
                reciprocal_cutoff_bohr_inv=3.404, tolerance=1e-12)
            ewald.real_cutoff_bohr = 15.0
            with self.assertRaisesRegex(ValueError, "ewald_real_cutoff_bohr"):
                restore(result, None, system=system)

    def test_gradient_accepts_original_scf_options_only_after_same_preparation(self):
        record = capture(alpha=None, ewald_options=None, reciprocal_cutoff=None)
        result = SimpleNamespace(resolved_domain=record, **{
            k: getattr(record, k) for k in ("ewald_alpha_bohr_inv", "ewald_precision",
                "sr_image_extent_bohr", "exact_zone_bohr", "exchange_ewald_split")})
        package = ModuleType("_domain_test_package")
        common = ModuleType(package.__name__ + ".pbc_bipole_common")
        common.copy_bipole_lattice_options = function("copy_bipole_lattice_options",
            file="pbc_bipole_common.py", OptionSnapshot=domain.OptionSnapshot,
            LATTICE_FIELDS=domain.LATTICE_FIELDS, LatticeSumOptions=lattice)
        common.clamp_bipole_nuclear_cutoff = function("clamp_bipole_nuclear_cutoff",
            file="pbc_bipole_common.py")
        progress = ModuleType(package.__name__ + ".progress")
        progress.resolve_progress = Mock(return_value=Mock())
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice,
                           __package__=package.__name__)
        original = lattice()
        original.nuclear_cutoff_bohr = 25.0
        with patch.dict(sys.modules, {m.__name__: m for m in (package, common, progress)}):
            actual = restore(result, original, system=SimpleNamespace(dim=3))
            self.assertEqual(actual.nuclear_cutoff_bohr, 15.0)
            self.assertEqual(original.nuclear_cutoff_bohr, 25.0)
            original.nuclear_cutoff_bohr = 12.0
            with self.assertRaisesRegex(ValueError, "nuclear_cutoff_bohr"):
                restore(result, original, system=SimpleNamespace(dim=3))
            self.assertEqual(original.nuclear_cutoff_bohr, 12.0)

    def test_each_driver_resolves_a_private_lattice_without_changing_next_run(self):
        clone = function("copy_bipole_lattice_options", file="pbc_bipole_common.py",
            OptionSnapshot=domain.OptionSnapshot, LATTICE_FIELDS=domain.LATTICE_FIELDS,
            LatticeSumOptions=lattice)
        resolve = function("resolve_bipole_lattice_options", file="pbc_bipole_common.py",
            resolve_bipole_pair_policy=domain.resolve_bipole_pair_policy,
            copy_bipole_lattice_options=clone)
        passthrough = function("_lattice_options_passthrough", file="pbc_bipole_common.py",
            LatticeSumOptions=lattice,
            _LATTICE_PASSTHROUGH_FIELDS=tuple(n for n in domain.LATTICE_FIELDS if n != "coulomb_method"))
        clamp = function("clamp_bipole_nuclear_cutoff", file="pbc_bipole_common.py")
        prepare = function("prepare_bipole_lattice_options", file="pbc_bipole_common.py",
            clamp_bipole_nuclear_cutoff=clamp, _lattice_options_passthrough=passthrough,
            CoulombMethod=CoulombMethod)
        for suffix in ("", "_uhf", "_rks", "_uks"):
            tree = ast.parse((PACKAGE / f"pbc_bipole{suffix}.py").read_text())
            assignment, = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                           and isinstance(n.target, ast.Name) and n.target.id == "lat_opts"]
            original = lattice()
            original.nuclear_cutoff_bohr = 25.0
            original.coulomb_method = CoulombMethod.EWALD_3D
            before = domain.OptionSnapshot.capture(original, domain.LATTICE_FIELDS)
            # Evaluate the actual driver assignment, then the actual preparation
            # twice. The unsplit second run must not inherit the first clamp.
            for split, expected in ((True, 15.0), (False, 25.0)):
                private = eval(compile(ast.Expression(assignment.value), "driver-options", "eval"),
                    dict(opts=SimpleNamespace(lattice_opts=original),
                         resolve_bipole_lattice_options=resolve, historical_domain=False))
                _, _, two, one = prepare(SimpleNamespace(dim=3), private, split, Mock())
                with self.subTest(driver=suffix, split=split):
                    self.assertEqual(two.nuclear_cutoff_bohr, expected)
                    self.assertEqual(one.nuclear_cutoff_bohr, expected)
                    self.assertIsNot(private, original)
                    self.assertEqual(before, domain.OptionSnapshot.capture(original, domain.LATTICE_FIELDS))
                    self.assertEqual(private.coulomb_method, original.coulomb_method)

    def test_gradient_uses_recorded_lattice_and_refuses_other_domain(self):
        record = capture()
        result = SimpleNamespace(resolved_domain=record, **{
            k: getattr(record, k) for k in ("ewald_alpha_bohr_inv", "ewald_precision",
                "sr_image_extent_bohr", "exact_zone_bohr", "exchange_ewald_split")})
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice)
        self.assertEqual(restore(result, None).cutoff_bohr, 15.0)
        wrong = lattice()
        wrong.pair_complete_1e = False
        with self.assertRaisesRegex(ValueError, "pair_complete_1e"):
            restore(result, wrong)
        result.ewald_precision = 1e-8
        with self.assertRaisesRegex(ValueError, "ewald_precision"):
            restore(result, None)

    def test_gradient_grid_restores_nondefault_profile_without_native_import(self):
        result = SimpleNamespace(resolved_domain=capture())
        package = ModuleType("_domain_test_package")
        native = ModuleType("_domain_test_package._vibeqc_core")
        native.GridOptions = grid
        restore = function("_gradient_domain_grid", __package__="_domain_test_package")
        with patch.dict(sys.modules, {package.__name__: package, native.__name__: native}):
            actual, periodic, radius = restore(result, None, None, None)
            self.assertEqual((actual.n_radial, actual.lebedev_order, periodic, radius),
                             (200, 53, True, 18.0))
            with self.assertRaisesRegex(ValueError, "becke_image_radius"):
                restore(result, None, None, 10.0)
            changed = grid()
            changed.lebedev_order = 17
            with self.assertRaisesRegex(ValueError, "lebedev_order"):
                restore(result, changed, None, None)
            self.assertEqual(restore(SimpleNamespace(), None, None, None), (None, True, 10.0))

    def test_gradient_density_domain_auto_and_unrecorded_remain_accepted(self):
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice)
        for selected in (None, PeriodicXCDensityDomain.AUTO):
            record = capture(xc_density_domain=selected)
            result = SimpleNamespace(resolved_domain=record, **{
                key: getattr(record, key) for key in (
                    "ewald_alpha_bohr_inv", "ewald_precision", "sr_image_extent_bohr",
                    "exact_zone_bohr", "exchange_ewald_split")})
            with self.subTest(selected=selected):
                self.assertEqual(restore(result, None).cutoff_bohr, 15.0)

    def test_nonauto_density_domain_refused_at_both_ks_gradient_entries(self):
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice)
        for selected in (PeriodicXCDensityDomain.MOLECULAR_HOME,
                         PeriodicXCDensityDomain.PERIODIC_LATTICE):
            # Missing numerical objects/metadata are intentional: the refusal
            # must precede any kernel or geometry-dependent domain resolution.
            result = SimpleNamespace(resolved_domain=capture(xc_density_domain=selected))
            for spin in ("rks", "uks"):
                entry = function(
                    "compute_bipole_gradient_" + spin,
                    _gradient_domain_grid=lambda *args: (None, True, 18.0),
                    _reject_m5_domain_analytic_gradient=lambda *args: None,
                    _warn_research_preview=lambda *args: None,
                    _gradient_domain_options=restore,
                )
                with self.subTest(selected=selected, spin=spin):
                    with self.assertRaisesRegex(NotImplementedError, "non-AUTO XC density domain"):
                        entry(object(), object(), result)

    def test_nonauto_density_domain_cannot_restore_as_auto(self):
        restore = function("_gradient_domain_options", LatticeSumOptions=lattice)
        for selected in (PeriodicXCDensityDomain.MOLECULAR_HOME,
                         PeriodicXCDensityDomain.PERIODIC_LATTICE):
            record = capture(xc_density_domain=selected)
            result = SimpleNamespace(resolved_domain=record, **{
                key: getattr(record, key) for key in (
                    "ewald_alpha_bohr_inv", "ewald_precision", "sr_image_extent_bohr",
                    "exact_zone_bohr", "exchange_ewald_split")})
            with self.subTest(selected=selected):
                with self.assertRaisesRegex(NotImplementedError, "non-AUTO XC density domain"):
                    restore(result, None)

    def test_energy_and_nuclear_pair_policies_must_agree(self):
        other = lattice()
        other.pair_complete_1e = False
        with self.assertRaisesRegex(ValueError, "1e/2e domain mismatch"):
            capture(lattice_1e=other)

    def test_output_discloses_historical_policy_and_all_settings(self):
        render = function("_format_bipole_domain", file="output/formats/scf_log.py")
        opts = lattice()
        opts.pair_complete_1e = False
        record = capture(lattice_1e=opts, lattice_2e=opts)
        text = render(SimpleNamespace(resolved_domain=record))
        self.assertIn("not atom-image invariant", text)
        self.assertIn(record.settings_sha256, text)
        for name in domain.LATTICE_FIELDS + domain.GRID_FIELDS:
            self.assertIn(name + ":", text)
        self.assertEqual(render(SimpleNamespace()), "")

    def test_ewald_resolver_pairs_the_same_alpha_and_reciprocal_envelope(self):
        native = ModuleType("_domain_test_package.bipole_ext_el_pole")
        native.bipole_ewald_reciprocal_cutoff = Mock(return_value=3.404)
        alpha = Mock(return_value=.4)
        widen = Mock(side_effect=lambda system, lat, a, tol, log: setattr(lat, "nuclear_cutoff_bohr", 19.0))
        options = Mock(return_value=SimpleNamespace(real_cutoff_bohr=19.0))
        resolver = function("resolve_bipole_ewald_state", file="pbc_bipole_common.py",
            __package__="_domain_test_package",
            np=SimpleNamespace(asarray=lambda x, **kw: x,
                               linalg=SimpleNamespace(det=lambda x: 125.0)),
            resolve_bipole_ewald_alpha=alpha,
            raise_bipole_ewald_real_cutoff=widen, _crystal_ewald_options=options)
        one, two = lattice(), lattice()
        system, log = SimpleNamespace(dim=3, lattice="lattice"), Mock()
        with patch.dict(sys.modules, {native.__name__: native}):
            result = resolver(system, one, two, ewald_omega=.4,
                              ewald_precision=1e-12, erfc_exchange_arm_active=False, plog=log)
        self.assertEqual(result[1:], (.4, 125.0, 3.404))
        alpha.assert_called_once_with(125.0, .4, two, 1e-12,
                                      erfc_exchange_arm_active=False, plog=log)
        native.bipole_ewald_reciprocal_cutoff.assert_called_once_with(125.0, .4)
        options.assert_called_once_with(one, alpha_bohr_inv=.4,
            tolerance=1e-12, recip_cutoff_bohr_inv=3.404)
        self.assertEqual(one.nuclear_cutoff_bohr, 19.0)
        self.assertEqual(two.nuclear_cutoff_bohr, 15.0)
        self.assertEqual(resolver(SimpleNamespace(dim=2), one, two, ewald_omega=None,
            ewald_precision=1e-12, erfc_exchange_arm_active=False, plog=log), (None,)*4)

    def test_prepared_nuclear_erfc_options_reach_kernel_and_record(self):
        for suffix in ("", "_uhf", "_rks", "_uks"):
            tree = ast.parse((PACKAGE / f"pbc_bipole{suffix}.py").read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            for consumer, keyword in (("capture_bipole_domain", "nuclear_erfc_options"),
                                      ("build_bipole_one_electron_lattice", "resolved_erfc_options")):
                call, = [c for c in calls if c.func.id == consumer]
                self.assertEqual(ast.unparse(next(k.value for k in call.keywords if k.arg == keyword)),
                                 "_v_ne_erfc_options")
        tree = ast.parse((PACKAGE / "pbc_bipole_common.py").read_text())
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_bipole_one_electron_lattice")
        call, = [c for c in ast.walk(builder) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == "_compute_nuclear_lattice_ewald_reciprocal_ft"]
        self.assertEqual(ast.unparse(next(k.value for k in call.keywords if k.arg == "resolved_erfc_options")),
                         "resolved_erfc_options")

    def test_all_four_drivers_share_record_with_context_and_result(self):
        for suffix in ("", "_uhf", "_rks", "_uks"):
            tree = ast.parse((PACKAGE / f"pbc_bipole{suffix}.py").read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            shared = [c for c in calls if c.func.id == "resolve_bipole_ewald_state"]
            self.assertEqual(len(shared), 1)
            self.assertFalse(any(c.func.id in ("resolve_bipole_ewald_alpha", "bipole_ewald_reciprocal_cutoff") for c in calls))
            consumers = [c for c in calls if c.func.id == "BipoleFockContext" or (c.func.id.startswith("PBCBipole") and c.func.id.endswith("Result"))]
            self.assertEqual(len(consumers), 2)
            for consumer in consumers:
                value = next(k.value for k in consumer.keywords if k.arg == "resolved_domain")
                self.assertEqual(ast.unparse(value), "_resolved_domain")

    def test_fock_guards_precede_accumulator_access(self):
        tree = ast.parse((PACKAGE / "pbc_bipole_fock.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("build_bipole_restricted_fock", "build_bipole_unrestricted_fock"):
                self.assertIn("require_fock_context(ctx)", ast.unparse(node.body[1]))

    def test_nondefault_grid_reaches_both_response_builders(self):
        tree = ast.parse((PACKAGE / "bipole_gradient.py").read_text())
        for spin in ("closed", "open"):
            name = f"_multi_k_orbital_relaxation_ks_{spin}_diag"
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            grids = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("_build_ks_grid", "build_periodic_becke_grid")]
            self.assertEqual(len(grids), 1)
            call = grids[0]
            self.assertEqual(call.func.id, "_build_ks_grid")
            received = Mock(return_value="grid")
            expression = ast.Expression(body=call)
            selected_grid = object()
            self.assertEqual(eval(compile(expression, "grid-call", "eval"), dict(
                _build_ks_grid=received, system="cell", grid_options=selected_grid,
                use_periodic_becke=False, becke_image_radius_bohr=18.0)), "grid")
            received.assert_called_once_with("cell", selected_grid, False, 18.0)
            callers = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name]
            self.assertEqual(len(callers), 1)
            for field in ("grid_options", "use_periodic_becke", "becke_image_radius_bohr"):
                self.assertEqual(ast.unparse(next(k.value for k in callers[0].keywords if k.arg == field)), field)

    def test_snapshot_inventory_covers_native_option_fields(self):
        source = (ROOT / "cpp/src/bindings.cpp").read_text()
        for klass, expected in (("GridOptions", domain.GRID_FIELDS), ("LatticeSumOptions", domain.LATTICE_FIELDS)):
            start = source.index(f'py::class_<vibeqc::{klass}>')
            end = source.index('\n    m.def(', start) if klass == "GridOptions" else source.index(';', start)
            # A semicolon in a docstring can terminate the quick bound. Use the
            # next class binding to enclose all lattice fields instead.
            if klass == "LatticeSumOptions":
                end = source.index('py::class_<', start + 15)
            fields = set(re.findall(r'\.def_(?:readwrite|property)\("([^"]+)"', source[start:end]))
            self.assertEqual(fields, set(expected))


if __name__ == "__main__":
    unittest.main()
