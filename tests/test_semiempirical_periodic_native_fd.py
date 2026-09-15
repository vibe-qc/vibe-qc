from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
from vibeqc.semiempirical.periodic import (
    finite_difference_gradient,
    finite_difference_stress,
)
from vibeqc.semiempirical.routes import (
    EXECUTION_NATIVE_BATCHED_FD,
    MATURITY_NATIVE_FD,
    plan_periodic_semiempirical_route,
)


def _h2_chain() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([4.0, 20.0, 20.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
        0,
        1,
    )


class TestPeriodicGammaImageCoverage:
    """Origin-only cell lists must be an explicit molecular choice (#316)."""

    cutoff_bohr = 15.0
    # Genuinely isolated at the 15-bohr cutoff: the closest H2 image
    # pair sits at 20.0 - 1.4 = 18.6 bohr > 15.  (The pre-#316-root-fix
    # width of 16.0 put image pairs at 14.6 bohr - INSIDE the declared
    # interaction range - so pair-distance image selection now correctly
    # retains those images instead of raising; see
    # tests/test_semiempirical_pair_image_selection.py.)
    supercell_width_bohr = 20.0
    free_boundary_error = r"no nonzero lattice image.*free-boundary cluster"

    @classmethod
    def _wide_cell(cls, *, open_shell=False):
        atoms = [Atom(1, [0.0, 0.0, 0.0])]
        multiplicity = 2 if open_shell else 1
        if not open_shell:
            atoms.append(Atom(1, [1.4, 0.0, 0.0]))
        return PeriodicSystem(
            1,
            np.diag([cls.supercell_width_bohr, 30.0, 30.0]),
            atoms,
            0,
            multiplicity,
        )

    @pytest.mark.parametrize(
        ("route_name", "open_shell"),
        (
            ("run_dftb0_gamma", False),
            ("run_udftb0_gamma", True),
            ("run_scc_dftb_gamma", False),
            ("run_uscc_dftb_gamma", True),
        ),
    )
    def test_dftb_routes_reject_origin_only_cell_list(
        self,
        route_name,
        open_shell,
    ):
        from vibeqc._vibeqc_core import direct_lattice_cells

        system = self._wide_cell(open_shell=open_shell)
        assert len(direct_lattice_cells(system, self.cutoff_bohr)) == 1

        params = _se.SemiempiricalParameters.dftb0_default()
        if "scc" in route_name:
            options = _se.PeriodicSCCOptions()
        else:
            options = _se.PeriodicDFTB0Options()
        options.cutoff_bohr = self.cutoff_bohr

        route = getattr(_se, route_name)
        with pytest.raises(
            ValueError,
            match=route_name + ".*" + self.free_boundary_error,
        ):
            route(system, params, options)

    @pytest.mark.parametrize(
        "route_name",
        ("run_dftb0_kpoints", "run_scc_dftb_kpoints"),
    )
    def test_dftb_kpoint_routes_reject_origin_only_cell_list(self, route_name):
        from vibeqc._vibeqc_core import monkhorst_pack

        system = self._wide_cell()
        mesh = monkhorst_pack(system, (1, 1, 1))
        params = _se.SemiempiricalParameters.dftb0_default()

        with pytest.raises(
            ValueError,
            match=route_name + ".*" + self.free_boundary_error,
        ):
            if route_name == "run_dftb0_kpoints":
                _se.run_dftb0_kpoints(
                    system,
                    params,
                    mesh,
                    self.cutoff_bohr,
                )
            else:
                _se.run_scc_dftb_kpoints(
                    system,
                    params,
                    mesh,
                    _se.SCCOptions(),
                    self.cutoff_bohr,
                )

    def test_gfn2_rejects_origin_only_cell_list(self):
        from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        options = _xtb.XTBSccOptions()
        with pytest.raises(
            ValueError,
            match="run_gfn2_xtb_gamma.*" + self.free_boundary_error,
        ):
            _xtb.run_gfn2_xtb_gamma(
                self._wide_cell(),
                load_gfn2_params(),
                options,
                self.cutoff_bohr,
            )

    def test_pm6_rejects_origin_only_cell_list(self):
        from vibeqc.semiempirical.methods.pm6_params import (
            load_pm6_mopac_params,
        )

        system = PeriodicSystem(
            1,
            np.diag([self.supercell_width_bohr, 30.0, 30.0]),
            [Atom(2, [0.0, 0.0, 0.0])],
            0,
            1,
        )
        options = _se.nddo.PeriodicPM6Options()
        options.cutoff_bohr = self.cutoff_bohr

        with pytest.raises(
            ValueError,
            match="run_pm6_gamma.*" + self.free_boundary_error,
        ):
            _se.nddo.run_pm6_gamma(
                system,
                load_pm6_mopac_params(),
                options,
            )

    def test_explicit_dftb_gamma_only_zero_mode_remains_available(self):
        options = _se.PeriodicDFTB0Options()
        options.cutoff_bohr = self.cutoff_bohr
        options.gamma_only_0 = True

        result = _se.run_dftb0_gamma(
            self._wide_cell(),
            _se.SemiempiricalParameters.dftb0_default(),
            options,
        )

        assert result.n_cells == 1
        assert result.gamma_only_0 is True

    def test_explicit_pm6_gamma_only_zero_mode_remains_available(self):
        from vibeqc.semiempirical.methods.pm6_params import (
            load_pm6_mopac_params,
        )

        system = PeriodicSystem(
            1,
            np.diag([self.supercell_width_bohr, 30.0, 30.0]),
            [Atom(2, [0.0, 0.0, 0.0])],
            0,
            1,
        )
        options = _se.nddo.PeriodicPM6Options()
        options.cutoff_bohr = self.cutoff_bohr
        options.gamma_only_0 = True

        result = _se.nddo.run_pm6_gamma(
            system,
            load_pm6_mopac_params(),
            options,
        )

        assert result.n_cells == 1
        assert result.converged


def test_periodic_pm6_derivative_plan_is_native_batched_fd():
    plan = plan_periodic_semiempirical_route(
        "pm6",
        _h2_chain(),
        properties=("energy", "gradient", "stress"),
    )

    assert plan.execution == EXECUTION_NATIVE_BATCHED_FD
    assert plan.maturity == MATURITY_NATIVE_FD


def test_pm6_native_batch_matches_python_fd():
    from vibeqc.semiempirical.methods.periodic_pm6 import (
        compute_pm6_gamma_derivatives_fd,
    )
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    system = _h2_chain()
    parameters = load_pm6_params()
    scf_options = _nddo.PeriodicPM6Options()
    scf_options.cutoff_bohr = 5.0

    def energy(candidate):
        return _nddo.run_pm6_gamma(candidate, parameters, scf_options).energy

    expected_gradient = finite_difference_gradient(system, energy)
    expected_stress = finite_difference_stress(
        system,
        energy,
        strain_positions=True,
    )
    result = compute_pm6_gamma_derivatives_fd(
        system,
        parameters,
        cutoff_bohr=5.0,
    )

    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1.0e-12)
    np.testing.assert_allclose(result.stress, expected_stress, atol=1.0e-12)
    assert result.energy_evaluations == 14
    assert result.system_workspace_copies == 1
    assert result.lattice_cell_setups == 3
    assert result.workspace_bytes > 0
    assert not result.memory_counters_complete
    assert result.parameter_sha256 == parameters.content_sha256()
    assert result.parameter_identity == parameters.parameter_identity()


def test_periodic_omx_derivative_routes_fail_closed():
    from vibeqc.semiempirical.methods.omx_params import load_om2_params
    from vibeqc.semiempirical.methods.periodic_omx import (
        compute_omx_gamma_derivatives_fd,
    )

    system = _h2_chain()
    parameters = load_om2_params()
    scf_options = _nddo.PeriodicOMxOptions()
    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        compute_omx_gamma_derivatives_fd(system, "om2", parameters)

    fd_options = _nddo.PeriodicNDDOFDBatchOptions()
    with pytest.raises(RuntimeError, match="derivatives are gated"):
        _nddo.compute_omx_gamma_fd_batch(
            system,
            parameters,
            scf_options,
            fd_options,
        )


def test_periodic_pm7_native_derivative_route_fails_before_work():
    from vibeqc.semiempirical.methods.pm7_params import load_pm7_params

    fd_options = _nddo.PeriodicNDDOFDBatchOptions()
    fd_options.compute_stress = False
    with pytest.raises(ValueError, match="require PM6 parameters"):
        _nddo.compute_pm6_gamma_fd_batch(
            _h2_chain(),
            load_pm7_params(),
            _nddo.PeriodicPM6Options(),
            fd_options,
        )


def test_empty_batch_rejected_before_parameter_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    from vibeqc.semiempirical.methods import periodic_pm6

    def fail_parameters(*_args, **_kwargs):
        raise AssertionError("parameter loading must not run")

    monkeypatch.setattr(periodic_pm6, "_get_params", fail_parameters)
    with pytest.raises(ValueError, match="must request gradient or stress"):
        periodic_pm6.compute_pm6_gamma_derivatives_fd(
            _h2_chain(),
            compute_gradient=False,
            compute_stress=False,
        )


def test_native_batch_rejects_invalid_step_before_energy_dispatch():
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    options = _nddo.PeriodicNDDOFDBatchOptions()
    options.coordinate_step = 0.0
    options.compute_stress = False
    with pytest.raises(ValueError, match="coordinate_step must be > 0"):
        _nddo.compute_pm6_gamma_fd_batch(
            _h2_chain(),
            load_pm6_params(),
            _nddo.PeriodicPM6Options(),
            options,
        )


def test_pm6_native_batch_rejects_nonconverged_displacements():
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    scf_options = _nddo.PeriodicPM6Options()
    scf_options.cutoff_bohr = 5.0
    scf_options.max_iter = 1
    fd_options = _nddo.PeriodicNDDOFDBatchOptions()
    fd_options.compute_stress = False
    with pytest.raises(RuntimeError, match="displaced SCF did not converge"):
        _nddo.compute_pm6_gamma_fd_batch(
            _h2_chain(),
            load_pm6_params(),
            scf_options,
            fd_options,
        )


def test_periodic_pm6_optimizer_uses_combined_native_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    from vibeqc.semiempirical import periodic

    captured = {}

    def fake_optimize(
        system,
        energy_fn,
        gradient_fn,
        **kwargs,
    ):
        captured["energy"] = energy_fn
        captured["gradient"] = gradient_fn
        captured.update(kwargs)
        return system

    monkeypatch.setattr(periodic, "optimize_cell", fake_optimize)
    system = _h2_chain()
    result = periodic.optimize_pm6_cell(system, cutoff_bohr=5.0)

    assert result is system
    assert callable(captured["energy"])
    assert callable(captured["gradient"])
    assert callable(captured["stress_fn"])
    assert callable(captured["derivatives_fn"])


# ---------------------------------------------------------------------------
# Issue #340: periodic semiempirical analytic derivatives must consume the
# lattice-cutoff provenance of the SCF result instead of a hardcoded 15 bohr.
# ---------------------------------------------------------------------------


def _h2_dimer() -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [Atom(2, [0.0, 0.0, 0.0]), Atom(2, [1.4, 0.0, 0.0])],
        0,
        1,
    )


def _open_shell_h2() -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [2.0, 0.7, 0.2])],
        1,
        2,
    )


def _polar_hf_cell() -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.eye(3) * 14.0,
        [Atom(1, [0.5, 0.4, 0.3]), Atom(9, [3.6, 1.7, 1.0])],
        0,
        1,
    )


def _fd_stress(system, energy_fn, step=0.001):
    """Central-difference stress straining lattice and positions (native gate)."""
    h = step
    V = abs(np.linalg.det(np.asarray(system.lattice)))
    L0 = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    stress = np.zeros((3, 3))
    for i in range(system.dim):
        for j in range(system.dim):
            eps = np.zeros((3, 3))
            eps[i, j] = h
            Lp = (np.eye(3) + eps) @ L0
            Lm = (np.eye(3) - eps) @ L0
            atoms_p = [
                Atom(a.Z, list((np.eye(3) + eps) @ np.array(a.xyz))) for a in atoms
            ]
            atoms_m = [
                Atom(a.Z, list((np.eye(3) - eps) @ np.array(a.xyz))) for a in atoms
            ]
            sp = PeriodicSystem(
                system.dim, Lp, atoms_p, system.charge, system.multiplicity
            )
            sm = PeriodicSystem(
                system.dim, Lm, atoms_m, system.charge, system.multiplicity
            )
            stress[i, j] = (energy_fn(sp) - energy_fn(sm)) / (2 * h * V)
    return stress


def test_dftb0_nondefault_cutoff_gradient_matches_fd():
    """Analytic DFTB0 gradient consumes the SCF's nondefault cutoff."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical import SemiempiricalParameters

    system = _h2_dimer()
    params = SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = 18.0

    result = _se.run_dftb0_gamma(system, params, options)
    assert result.cutoff_bohr == pytest.approx(18.0)
    assert result.gamma_only_0 is False

    gradient = np.asarray(
        _se.compute_periodic_dftb0_gradient(system, result, params)
    )

    def energy(candidate):
        return float(_se.run_dftb0_gamma(candidate, params, options).energy)

    np.testing.assert_allclose(
        gradient, finite_difference_gradient(system, energy), atol=1.0e-4
    )


def test_dftb0_gamma_only_0_gradient_matches_fd():
    """Analytic DFTB0 gradient matches the g=0-only molecular-limit SCF."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical import SemiempiricalParameters

    system = _h2_dimer()
    params = SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = 8.0
    options.gamma_only_0 = True

    result = _se.run_dftb0_gamma(system, params, options)
    assert result.gamma_only_0 is True
    assert result.cutoff_bohr == pytest.approx(8.0)

    gradient = np.asarray(
        _se.compute_periodic_dftb0_gradient(system, result, params)
    )

    def energy(candidate):
        return float(_se.run_dftb0_gamma(candidate, params, options).energy)

    np.testing.assert_allclose(
        gradient, finite_difference_gradient(system, energy), atol=1.0e-4
    )


def test_dftb0_nondefault_cutoff_stress_matches_fd():
    """Analytic DFTB0 stress consumes the SCF's nondefault cutoff."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical import SemiempiricalParameters

    system = _h2_dimer()
    params = SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = 18.0

    result = _se.run_dftb0_gamma(system, params, options)
    stress = np.asarray(_se.compute_periodic_dftb0_stress(system, result, params))

    def energy(candidate):
        return float(_se.run_dftb0_gamma(candidate, params, options).energy)

    stress_fd = _fd_stress(system, energy)
    # Same diagonal-only rtol as the default-cutoff periodic stress gate.
    assert np.allclose(
        np.diag(stress), np.diag(stress_fd), rtol=0.1
    ), f"stress diag: {np.diag(stress)} vs FD {np.diag(stress_fd)}"


def test_scc_dftb_nondefault_cutoff_gradient_matches_fd():
    """Analytic SCC-DFTB gradient consumes the SCF's nondefault cutoff."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical import SemiempiricalParameters

    system = _polar_hf_cell()
    params = SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicSCCOptions()
    # Nondefault 12 bohr (default is 15): the converged basin of this
    # polar fixture under pair-distance image selection - see the
    # rationale on _tight_periodic_scc_options in
    # tests/test_semiempirical_benchmarks.py (issue #316).
    options.cutoff_bohr = 12.0
    options.max_iter = 500
    options.conv_tol_charge = 1.0e-11
    options.charge_mixing = 0.1
    # DIIS is off on this fixture since 2026-09-07 (D1): on the Ewald-split
    # Elstner surface, DIIS at charge_mixing 0.1 parks at a saturated
    # q = +-1.00010 state and stops moving -- the same energy at 500 and at
    # 5000 iterations, never meeting the 1e-11 residual. That state is
    # 19 mHa *above* the physical one, so it is a stalled iterate and not a
    # competing minimum: plain damped mixing reaches q = +-0.694,
    # E = -4.903351772 in 34 iterations, and DIIS at charge_mixing 0.3 finds
    # the same point in 28. The subject of this test is the gradient, so it
    # runs on the path that converges; the DIIS behaviour on this surface is
    # a solver defect that needs its own issue.
    options.use_diis = False
    options.diis_subspace = 6

    result = _se.run_scc_dftb_gamma(system, params, options)
    assert result.converged
    assert result.cutoff_bohr == pytest.approx(12.0)

    gradient = np.asarray(
        _se.compute_periodic_scc_dftb_gradient(system, result, params)
    )

    def energy(candidate):
        candidate_result = _se.run_scc_dftb_gamma(candidate, params, options)
        assert candidate_result.converged
        return float(candidate_result.energy)

    np.testing.assert_allclose(
        gradient,
        finite_difference_gradient(system, energy, h=2.0e-4),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_udftb0_nondefault_cutoff_gradient_matches_fd():
    """Analytic UDFTB0 gradient consumes the SCF's nondefault cutoff."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical import SemiempiricalParameters

    system = _open_shell_h2()
    params = SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = 18.0

    result = _se.run_udftb0_gamma(system, params, options)
    assert result.cutoff_bohr == pytest.approx(18.0)

    gradient = np.asarray(
        _se.compute_periodic_udftb0_gradient(system, result, params)
    )

    def energy(candidate):
        return float(_se.run_udftb0_gamma(candidate, params, options).energy)

    np.testing.assert_allclose(
        gradient,
        finite_difference_gradient(system, energy, h=2.0e-4),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_uscc_dftb_nondefault_cutoff_gradient_matches_fd():
    """Analytic USCC-DFTB gradient consumes the SCF's nondefault cutoff."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical import SemiempiricalParameters

    system = _open_shell_h2()
    params = SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicSCCOptions()
    options.cutoff_bohr = 18.0
    options.max_iter = 500
    options.conv_tol_charge = 1.0e-11
    options.charge_mixing = 0.1
    # DIIS is off on this fixture since 2026-09-07 (D1): on the Ewald-split
    # Elstner surface, DIIS at charge_mixing 0.1 parks at a saturated
    # q = +-1.00010 state and stops moving -- the same energy at 500 and at
    # 5000 iterations, never meeting the 1e-11 residual. That state is
    # 19 mHa *above* the physical one, so it is a stalled iterate and not a
    # competing minimum: plain damped mixing reaches q = +-0.694,
    # E = -4.903351772 in 34 iterations, and DIIS at charge_mixing 0.3 finds
    # the same point in 28. The subject of this test is the gradient, so it
    # runs on the path that converges; the DIIS behaviour on this surface is
    # a solver defect that needs its own issue.
    options.use_diis = False
    options.diis_subspace = 6

    result = _se.run_uscc_dftb_gamma(system, params, options)
    assert result.converged
    assert result.cutoff_bohr == pytest.approx(18.0)

    gradient = np.asarray(
        _se.compute_periodic_uscc_dftb_gradient(system, result, params)
    )

    def energy(candidate):
        candidate_result = _se.run_uscc_dftb_gamma(candidate, params, options)
        assert candidate_result.converged
        return float(candidate_result.energy)

    np.testing.assert_allclose(
        gradient,
        finite_difference_gradient(system, energy, h=2.0e-4),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_gfn2_derivatives_consume_nondefault_cutoff():
    """GFN2 result carries the cutoff and its derivatives use it."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    # Deliberately asymmetric dimerized chain (gaps 1.05 / 1.45 bohr):
    # an equidistant chain is inversion symmetric, so its exact forces
    # vanish at EVERY cutoff and cannot discriminate cutoff consumption.
    # (The pre-#316 selection broke that symmetry spuriously, which is
    # what the old equidistant probe was actually measuring.)  The
    # cutoffs 6 and 12 bohr straddle real image interactions, so the
    # pair-selected gradients differ by ~0.3 Ha/bohr.
    system = PeriodicSystem(
        1,
        np.diag([2.5, 20.0, 20.0]),
        [Atom(6, [0.0, 0.0, 0.0]), Atom(6, [1.05, 0.0, 0.0])],
        0,
        1,
    )
    options = _xtb.XTBSccOptions()
    options.electronic_temperature = 0.0

    near = _xtb.run_gfn2_xtb_gamma(system, params, options, cutoff_bohr=6.0)
    far = _xtb.run_gfn2_xtb_gamma(system, params, options, cutoff_bohr=12.0)
    assert near.converged and far.converged
    assert near.parameter_sha256 == params.content_sha256()
    assert far.parameter_sha256 == params.content_sha256()
    assert near.parameter_identity == params.parameter_identity()
    assert far.parameter_identity == params.parameter_identity()
    assert near.cutoff_bohr == pytest.approx(6.0)
    assert far.cutoff_bohr == pytest.approx(12.0)

    gradient_near = np.asarray(
        _se.compute_periodic_gfn2_gradient(system, near, params)
    )
    gradient_far = np.asarray(
        _se.compute_periodic_gfn2_gradient(system, far, params)
    )
    # The derivative is built from the result's own truncated lattice sum, so
    # a wider cutoff must change it (the old code silently reused 15 bohr).
    assert np.max(np.abs(gradient_near - gradient_far)) > 1.0e-8


def test_periodic_gfn2_gradient_matches_finite_difference():
    """IID 338: the analytic periodic GFN2 gradient is dE/dR."""
    pytest.importorskip("vibeqc.semiempirical.methods.gfn2_params")
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb_local
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    options = _xtb_local.XTBSccOptions()
    options.max_iter = 500
    options.electronic_temperature = 0.0

    lattice = np.diag([8.0, 8.5, 9.0])
    base = np.array([[0.1, 0.2, 0.0], [1.45, 0.15, 0.05]])

    def system_at(positions):
        return PeriodicSystem(
            3, lattice, [Atom(1, positions[0]), Atom(1, positions[1])], 0, 1
        )

    def energy_at(positions):
        return _xtb_local.run_gfn2_xtb_gamma(
            system_at(positions), params, options
        ).energy

    result = _xtb_local.run_gfn2_xtb_gamma(system_at(base), params, options)
    assert result.converged
    analytic = np.asarray(
        _se.compute_periodic_gfn2_gradient(system_at(base), result, params)
    )

    step = 1.0e-4
    for atom in range(2):
        for axis in range(3):
            plus = base.copy()
            plus[atom, axis] += step
            minus = base.copy()
            minus[atom, axis] -= step
            fd = (energy_at(plus) - energy_at(minus)) / (2.0 * step)
            assert analytic[atom, axis] == pytest.approx(fd, abs=2.0e-4), (
                f"analytic gradient is not dE/dR at atom {atom} axis {axis}"
            )


def test_periodic_gfn2_polar_gradient_and_stress_match_finite_difference():
    """IID 338: AES response and every strain component share one energy."""
    pytest.importorskip("vibeqc.semiempirical.methods.gfn2_params")
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb_local
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    options = _xtb_local.XTBSccOptions()
    options.max_iter = 1000
    options.conv_tol_charge = 1.0e-9
    options.auto_stabilize = False
    options.electronic_temperature = 0.0
    system = _polar_hf_cell()

    def run(candidate):
        candidate_result = _xtb_local.run_gfn2_xtb_gamma(
            candidate, params, options
        )
        assert candidate_result.converged
        energies = np.asarray(candidate_result.mo_energies, dtype=float)
        n_occ = int(candidate_result.n_occ)
        assert energies[n_occ] - energies[n_occ - 1] > 1.0e-3
        return candidate_result

    result = run(system)
    analytic_gradient = np.asarray(
        _se.compute_periodic_gfn2_gradient(system, result, params)
    )
    analytic_stress = np.asarray(
        _se.compute_periodic_gfn2_stress(system, result, params)
    )

    def energy(candidate):
        return float(run(candidate).energy)

    np.testing.assert_allclose(
        analytic_gradient,
        finite_difference_gradient(system, energy, h=1.0e-4),
        rtol=0.0,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        analytic_stress,
        _fd_stress(system, energy, step=1.0e-4),
        rtol=0.0,
        atol=2.0e-5,
    )


def test_periodic_scc_dftb_gamma_sum_converges_with_cutoff():
    """IID 425, fixed 2026-09-07: the gamma lattice sum converges.

    The route used to sum the Klopman-Ohno form ``1/sqrt(R^2 + eta^2)`` by
    bare real-space truncation.  That is a sum of 1/R, divergent in 3-D:
    charge neutrality should cancel it, but a *pair-distance* cutoff gives
    different pairs different image counts, so each pair keeps a different
    divergent constant and the cancellation is incomplete.

    Elstner et al., Phys. Rev. B 58, 7260 (1998), p. 7263 names this exact
    failure -- the Ohno/Klopman forms "can therefore not be used" for
    periodic systems -- and Eq. 17's ``gamma = 1/R - S(R)`` exists so the
    long-range part can go through Ewald while the exponentially decaying S
    is summed over a few cells.  Both halves are now in place (maintainer
    decision D1, 2026-08-28), and both are needed.  On this fixture:

        cutoff        Elstner    Klopman-Ohno (with Ewald)
            12   -0.442872865   -0.419106186
            20   -0.450119928   -0.499294489
            40   -0.450188859   -0.620488893
            60   -0.450188859   -0.673944650
            80   -0.450188859   -0.720458644
           120   -0.450188859   -0.788949524
           200   -0.450188859   -0.871638200

    Elstner is flat to nine decimals from 40 bohr out; Klopman-Ohno still
    drifts monotonically over a 5x cutoff range even with the Coulomb tail
    Ewald-summed, which is D1's own point that the Ewald half alone does not
    fix this.  The functional form is what supplies the thermodynamic limit.
    """
    params = _se.SemiempiricalParameters.dftb0_default()
    a = 3.8
    system = PeriodicSystem(
        3,
        np.diag([a, a, a]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(3, [a / 2, a / 2, a / 2])],
        0,
        1,
    )

    energies = []
    klopman_ohno = []
    for cutoff in (40.0, 60.0, 80.0):
        options = _se.PeriodicSCCOptions()
        options.cutoff_bohr = cutoff
        options.max_iter = 400
        energies.append(_se.run_scc_dftb_gamma(system, params, options).energy)
        options.gamma_form = _se.ShellGammaForm.KlopmanOhno
        klopman_ohno.append(
            _se.run_scc_dftb_gamma(system, params, options).energy
        )

    # Successive rungs must agree once the sum is converged.
    assert abs(energies[2] - energies[1]) < 1.0e-5, (
        f"gamma lattice sum unconverged in the cutoff: {energies}"
    )
    assert abs(energies[1] - energies[0]) < 1.0e-5
    assert energies[2] == pytest.approx(-0.450188859, abs=1.0e-8)

    # The form is what converges it, not the Ewald split alone: the same
    # Ewald-summed Coulomb tail with the Klopman-Ohno remainder still drifts.
    assert abs(klopman_ohno[2] - klopman_ohno[1]) > 1.0e-2


def test_legacy_result_without_provenance_fails_closed():
    """Default-initialized results carry no cutoff provenance and are rejected."""
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical import SemiempiricalParameters
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    system = _h2_dimer()
    dftb_params = SemiempiricalParameters.dftb0_default()

    with pytest.raises(ValueError, match="lattice-cutoff provenance"):
        _se.compute_periodic_dftb0_gradient(
            system, _se.PeriodicDFTB0Result(), dftb_params
        )
    with pytest.raises(ValueError, match="lattice-cutoff provenance"):
        _se.compute_periodic_dftb0_stress(
            system, _se.PeriodicDFTB0Result(), dftb_params
        )
    with pytest.raises(ValueError, match="lattice-cutoff provenance"):
        _se.compute_periodic_udftb0_gradient(
            system, _se.PeriodicUDFTB0Result(), dftb_params
        )

    gfn2_params = load_gfn2_params()
    with pytest.raises(ValueError, match="lattice-cutoff provenance"):
        _se.compute_periodic_gfn2_gradient(
            system, _xtb.PeriodicGFN2Result(), gfn2_params
        )
    with pytest.raises(ValueError, match="lattice-cutoff provenance"):
        _se.compute_periodic_gfn2_stress(
            system, _xtb.PeriodicGFN2Result(), gfn2_params
        )
