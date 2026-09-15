"""PM7 parameter-registry and fail-closed route tests.

The bundled PM7 records remain available for completing the implementation,
but no energy or derivative route may run until Stewart's feathered
electrostatics is implemented across every coupled NDDO term.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _semiempirical
from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
from vibeqc.semiempirical.methods.pm7 import PM7Model, UPM7Model
from vibeqc.semiempirical.methods.pm7_params import (
    load_pm7_mopac_params,
    load_pm7_params,
    load_pm7_params_auto,
)
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan
from vibeqc.semiempirical.status import (
    BACKEND_GATED_EXPERIMENTAL,
    semiempirical_route_runtime_available,
    semiempirical_route_status,
)


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.1173]),
            Atom(1, [0.0, 0.7572, -0.4692]),
            Atom(1, [0.0, -0.7572, -0.4692]),
        ],
        charge=0,
        multiplicity=1,
    )


def _oh_radical() -> Molecule:
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.107]), Atom(1, [0.0, 0.0, -0.857])],
        charge=0,
        multiplicity=2,
    )


class TestPM7ParameterLoading:
    def test_inline_loads_hcnof_with_exact_identity(self):
        params = load_pm7_params()
        assert params.method_name() == "pm7"
        assert params.metadata().method_name == "pm7"
        assert params.metadata().version.startswith("pm7")
        for atomic_number in (1, 6, 7, 8, 9):
            assert params.has_element(atomic_number)
        assert params.n_elements() == 5

    def test_auto_loader_uses_full_cache_for_sulfur(self):
        params = load_pm7_params_auto([16])
        assert params.has_element(16)
        assert params.n_elements() == 75

    def test_full_toml_excludes_mopac_pseudo_atoms(self):
        params = load_pm7_mopac_params()
        assert params.n_elements() == 75
        assert params.has_element(1)
        assert params.has_element(82)
        assert params.has_element(98)
        assert not params.has_element(100)

    def test_parser_preserves_poc_core_radius_overrides(self):
        """IID 224: the poc_7 core radii must survive parse and write.

        The array is named ``poc_7`` in the MOPAC source, so the parser
        registers it under the key ``poc_``.  ``name.rstrip("7")`` cannot
        normalize that -- the string ends in an underscore, not a 7 -- so the
        value was stored as ``poc_`` while the writer only ever emitted
        ``poc``.  Every fitted core radius was silently dropped.  The PM6
        parser already special-cases the same array; PM7 did not.
        """
        script = Path(__file__).parents[1] / "scripts" / "parse_mopac_pm7_params.py"
        spec = importlib.util.spec_from_file_location("parse_pm7_poc", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Synthetic MOPAC-format fixture: the real parameter file is not
        # redistributed here, so the round trip is pinned on a snippet in the
        # same Fortran D-format, using the Sc value the issue names.
        source = """
      data     poc_7( 21)/        1.070880D0/
      data      uss7( 21)/      -15.000000D0/
"""
        parsed = module.parse_f90(source)

        assert 21 in parsed["elements"]
        element = parsed["elements"][21]
        # The normalized key is what the writer emits.
        assert "poc" in element, (
            f"poc_7 was not normalized to 'poc': keys={sorted(element)}"
        )
        assert "poc_" not in element
        assert element["poc"] == pytest.approx(1.070880, abs=0.0)

        # ... and it must survive the writer, not just the parse.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pm7.toml"
            module.write_toml(parsed, out)
            assert "poc = 1.070880" in out.read_text()

    def test_parser_excludes_pseudo_elements_and_pairs(self):
        script = Path(__file__).parents[1] / "scripts" / "parse_mopac_pm7_params.py"
        spec = importlib.util.spec_from_file_location("parse_pm7_under_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = """
      data     upp7(  8)/      -20.000000D0/
      data     upp7(100)/       -2.000000D0/
      alpb( 8, 1) = 1.500000d0
      alpb(100, 8) = 2.500000d0
"""
        parsed = module.parse_f90(source)
        assert 8 in parsed["elements"]
        assert 100 not in parsed["elements"]
        assert (1, 8) in parsed["diatomic"]
        assert (8, 100) not in parsed["diatomic"]


class TestPM7Gate:
    @pytest.mark.parametrize(
        ("method", "multiplicity"),
        [("pm7", 1), ("upm7", 2)],
    )
    def test_route_planner_gates_incomplete_hamiltonian(
        self,
        method,
        multiplicity,
    ):
        with pytest.raises(NotImplementedError, match="feathered electrostatics"):
            SemiempiricalRoutePlan.from_request(
                method,
                multiplicity=multiplicity,
            )

    def test_model_wrappers_fail_before_parameter_loading(self, monkeypatch):
        import vibeqc.semiempirical.methods.pm7_params as parameter_module

        def fail_loader(*_args, **_kwargs):
            raise AssertionError("PM7 gate must precede parameter loading")

        monkeypatch.setattr(parameter_module, "load_pm7_params_auto", fail_loader)
        with pytest.raises(NotImplementedError, match="feathered electrostatics"):
            PM7Model(_h2o())
        with pytest.raises(NotImplementedError, match="feathered electrostatics"):
            UPM7Model(_oh_radical())

    def test_native_energy_and_derivative_paths_fail_closed(self):
        params = load_pm7_params()
        restricted_calls = (
            lambda: _nddo.run_pm6(_h2o(), params),
            lambda: _nddo.compute_pm6_gradient_fd(_h2o(), params),
        )
        unrestricted_calls = (
            lambda: _nddo.run_upm6(_oh_radical(), params),
            lambda: _nddo.compute_upm6_gradient_fd(_oh_radical(), params),
        )
        for call in restricted_calls + unrestricted_calls:
            with pytest.raises(ValueError, match="requires PM6 parameters"):
                call()

    def test_unvalidated_analytic_pm6_gradient_is_not_public(self):
        assert not hasattr(_nddo, "compute_pm6_gradient")

    _PM7_GATE_MESSAGE = (
        "PM7 is gated until the published feathered electrostatics and all "
        "coupled NDDO terms are implemented and validated."
    )

    def test_run_job_upm7_without_basis_hits_the_canonical_gate(self):
        """run_job(method='upm7') must reach the PM7 gate, not the generic
        basis requirement (issue #413, path (a))."""
        from vibeqc.runner import run_job

        with pytest.raises(NotImplementedError) as exc_info:
            run_job(_h2o(), method="upm7")
        assert str(exc_info.value) == self._PM7_GATE_MESSAGE

    def test_run_semiempirical_upm7_hits_the_canonical_gate(self):
        """run_semiempirical('upm7') must reach the PM7 gate, not the
        generic dispatch fallback (issue #413, path (b))."""
        from vibeqc.semiempirical.runner import run_semiempirical

        with pytest.raises(NotImplementedError) as exc_info:
            run_semiempirical("upm7", _oh_radical())
        assert str(exc_info.value) == self._PM7_GATE_MESSAGE

    def test_pm7_gate_message_is_single_sourced(self):
        """Every PM7/UPM7 gate path surfaces the one canonical message."""
        from vibeqc.runner import run_job

        for call in (
            lambda: run_job(_h2o(), method="pm7"),
            lambda: run_job(_oh_radical(), method="upm7", basis="sto-3g"),
            lambda: SemiempiricalRoutePlan.from_request("pm7"),
            lambda: PM7Model(_h2o()),
        ):
            with pytest.raises(NotImplementedError) as exc_info:
                call()
            assert str(exc_info.value) == self._PM7_GATE_MESSAGE

    def test_run_job_fails_before_output_planning(self, tmp_path):
        from vibeqc.runner import run_job

        stem = tmp_path / "gated_pm7"
        with pytest.raises(NotImplementedError, match="feathered electrostatics"):
            run_job(
                _h2o(),
                method="pm7",
                output=stem,
                output_qvf=False,
                citations=False,
                record_hostname=False,
            )
        assert not stem.with_suffix(".system").exists()

    def test_status_and_runtime_preflight_are_honest(self):
        for route in ("pm7", "pm7-gradient-fd", "periodic-pm7"):
            status = semiempirical_route_status(route)
            assert status.backend == BACKEND_GATED_EXPERIMENTAL
            assert not status.production
            assert not status.performance_critical
            assert not semiempirical_route_runtime_available(route)

    def test_builtin_registry_does_not_advertise_gated_pm7(self):
        registry_type = _semiempirical.SemiempiricalMethodRegistry
        registry = registry_type.instance()
        # ``register_dftb`` is the non-idempotent process-global initializer
        # for every native builtin, despite its historical singular name.
        # A sibling registry test may already have called it in this pytest
        # process, so only initialize an empty registry.  Re-registering an
        # existing ``dftb0`` raises before this PM7 gate can be inspected.
        if not registry.method_names():
            registry_type.register_dftb()
        pm6 = registry.find_config("pm6")
        assert pm6.supports_periodic
        assert pm6.supports_stress
        assert not pm6.supports_kpoints
        assert "pm7" not in registry.method_names()
        with pytest.raises(RuntimeError, match="Method not found: pm7"):
            registry.find_config("pm7")

        # The native metadata registry is not a dispatch authority.  Even in
        # the same process after all builtins are registered, the public route
        # must still hit the canonical PM7 scientific gate.
        from vibeqc.runner import run_job

        with pytest.raises(NotImplementedError) as exc_info:
            run_job(_h2o(), method="pm7")
        assert str(exc_info.value) == self._PM7_GATE_MESSAGE


def test_pm7_parameter_arrays_are_finite_for_future_implementation():
    params = load_pm7_params()
    for atomic_number in (1, 6, 7, 8, 9):
        element = params.element_data(atomic_number)
        assert element is not None
        values = (
            element.uss,
            element.upp,
            element.betas,
            element.betap,
            element.zs,
            element.zp,
            element.gss,
        )
        assert np.all(np.isfinite(values))
