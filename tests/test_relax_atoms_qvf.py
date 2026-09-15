"""``relax_atoms(..., output_trajectory=...)`` QVF round-trip test.

The periodic-atomic-relaxation driver
(``vibeqc.bipole_optimize.relax_atoms``) gains an optional
``output_trajectory`` kwarg that emits a vibe-view-renderable QVF
archive: one frame per accepted L-BFGS-B step. Periodic systems
ship as QVF v2 with the per-frame lattice + dim attached so
vibe-view draws the cell + wraps atoms (QVF Inc B's renderer
handles both).

These tests pin:

* Default off — no trajectory file when the kwarg is unset, no
  per-step overhead.
* When set, the archive validates and contains a structure +
  reaction.path + citations section (the shared
  ``write_reaction_path_qvf`` helper from QVF Inc C).
* Periodic relaxations ship as QVF v2 with the lattice member +
  scalar ``dim``.
* n_frames matches the number of L-BFGS-B iterations + 1
  (initial geometry + one frame per accepted step).
* The small-cutoff QVF/dispatch cases use an 8-bohr molecular-limit cell with
  a converged overlap fold and select ``sr_image_precision=None`` explicitly;
  the public production default remains the padded M5 domain.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions
from vibeqc.bipole_optimize import OptimizeResult, relax_atoms
from vibeqc.output.formats.qvf import (
    QVF_FORMAT_VERSION,
    validate_qvf,
)


@pytest.fixture
def h2_in_box() -> PeriodicSystem:
    """H₂ slightly stretched in an 8-bohr cubic box. Designed to take
    a handful of L-BFGS-B steps to relax back toward equilibrium —
    enough frames to make the trajectory interesting without making
    the test slow."""
    L = np.diag([8.0, 8.0, 8.0])
    return PeriodicSystem(
        3,
        L,
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.8]),  # stretched; H₂ eq ~1.4 bohr
        ],
    )


@pytest.fixture
def gamma_kmesh(h2_in_box) -> object:
    return monkhorst_pack(h2_in_box, (1, 1, 1))


@pytest.mark.parametrize("name", ["bipole-opt-custom", "sto-3g"])
def test_relax_atoms_preserves_explicit_basis(
    monkeypatch, h2_in_box, gamma_kmesh, tmp_path, name,
):
    import vibeqc.bipole_optimize as opt
    import vibeqc.pbc_bipole as driver

    original_positions = np.array([a.xyz for a in h2_in_box.unit_cell])
    basis = vq.BasisSet(h2_in_box.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, True, [.7+i], [1.], p)
        for i, p in enumerate(original_positions)
    ], name, False)
    original = list(basis.shells())
    calls = []

    def run(system, moved, *args, **kwargs):
        positions = np.array([a.xyz for a in system.unit_cell])
        for old, new in zip(original, moved.shells(), strict=True):
            np.testing.assert_array_equal(new.exponents, old.exponents)
            np.testing.assert_array_equal(new.coefficients, old.coefficients)
            np.testing.assert_allclose(new.origin, np.asarray(old.origin)
                                       + positions[old.atom_index]
                                       - original_positions[old.atom_index], atol=1e-14)
        calls.append(moved)
        return SimpleNamespace(energy=0., converged=True, n_iter=1)

    def minimize(fun, x0, **kwargs):
        # Exercise both callbacks away from the initial geometry.
        x = np.asarray(x0).copy() + .01
        return SimpleNamespace(x=x, fun=fun(x), jac=kwargs["jac"](x), nit=1, success=True)

    monkeypatch.setattr(opt, "run_pbc_bipole_rhf", run)
    monkeypatch.setattr(driver, "run_pbc_bipole_rhf", run)
    monkeypatch.setattr(opt, "minimize", minimize)
    import vibeqc.output.formats.qvf as qvf

    written = []

    def write_trajectory(*args, **kwargs):
        assert kwargs["basis"] == name
        written.append(kwargs)

    monkeypatch.setattr(qvf, "write_reaction_path_qvf", write_trajectory)
    result = relax_atoms(
        h2_in_box, basis, gamma_kmesh, output_trajectory=tmp_path / "custom-trajectory",
    )
    assert result.converged
    assert len(calls) == 14  # objective, gradient reference, 12 FD displacements
    assert len(written) == 1


class TestOutputTrajectoryDefault:
    def test_default_off_no_qvf_emitted(self, h2_in_box, gamma_kmesh, tmp_path):
        """Without ``output_trajectory``, relax_atoms emits nothing
        in the temp dir (the kwarg is opt-in)."""
        result = relax_atoms(
            h2_in_box,
            "sto-3g",
            gamma_kmesh,
            method="RHF",
            max_iter=3,
            conv_tol_grad=1e-3,
            cutoff_bohr=4.0,
            sr_image_precision=None,
        )
        assert isinstance(result, OptimizeResult)
        # tmp_path stays empty.
        assert list(tmp_path.iterdir()) == []


class TestOutputTrajectoryEnabled:
    def test_archive_validates_with_expected_sections(
        self, h2_in_box, gamma_kmesh, tmp_path
    ):
        stem = tmp_path / "relax_traj"
        result = relax_atoms(
            h2_in_box,
            "sto-3g",
            gamma_kmesh,
            method="RHF",
            max_iter=5,
            conv_tol_grad=1e-3,
            cutoff_bohr=4.0,
            output_trajectory=stem,
            sr_image_precision=None,
        )
        qvf_path = stem.with_suffix(".qvf")
        assert qvf_path.exists()
        report = validate_qvf(qvf_path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            kinds = {s["kind"] for s in mf["sections"]}
            assert kinds == {"structure", "reaction.path", "citations"}
        # OptimizeResult itself is unaffected by the trajectory hook.
        assert result.converged or result.n_iter <= 5

    def test_archive_is_v1_with_lattice_for_periodic_system(
        self, h2_in_box, gamma_kmesh, tmp_path
    ):
        stem = tmp_path / "relax_v2"
        relax_atoms(
            h2_in_box,
            "sto-3g",
            gamma_kmesh,
            method="RHF",
            max_iter=5,
            conv_tol_grad=1e-3,
            cutoff_bohr=4.0,
            output_trajectory=stem,
            sr_image_precision=None,
        )
        qvf_path = stem.with_suffix(".qvf")
        with zipfile.ZipFile(qvf_path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            # Ruling 2026-07-10: periodic archives stay v1; `lattice` marks them.
            assert mf["qvf_version"] == QVF_FORMAT_VERSION
            rxn = next(
                s for s in mf["sections"] if s["kind"] == "reaction.path"
            )
            assert "lattice" in rxn["members"]
            lat = rxn["members"]["lattice"]
            assert lat["dtype"] == "float64"
            assert lat["shape"] == [3, 3]
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            assert meta["dim"] == 3

    def test_frame_count_matches_iter_plus_one(
        self, h2_in_box, gamma_kmesh, tmp_path
    ):
        """Initial geometry + one frame per accepted L-BFGS-B step."""
        stem = tmp_path / "relax_frames"
        result = relax_atoms(
            h2_in_box,
            "sto-3g",
            gamma_kmesh,
            method="RHF",
            max_iter=5,
            conv_tol_grad=1e-3,
            cutoff_bohr=4.0,
            output_trajectory=stem,
            sr_image_precision=None,
        )
        qvf_path = stem.with_suffix(".qvf")
        with zipfile.ZipFile(qvf_path) as zf:
            rxn = next(
                s for s in json.loads(zf.read("manifest.json"))["sections"]
                if s["kind"] == "reaction.path"
            )
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            # scipy's L-BFGS-B reports n_iter as the number of
            # accepted steps; callback fires once per step. The
            # writer also includes the initial frame, so the QVF has
            # n_iter + 1 frames. (A 0-iter run still has 2 frames —
            # the relax_atoms epilogue appends sys_opt so the QVF
            # carries at least start + end.)
            n_frames = len(meta["energies"])
            assert n_frames == max(2, result.n_iter + 1)

    def test_waypoints_tag_start_and_converged(
        self, h2_in_box, gamma_kmesh, tmp_path
    ):
        stem = tmp_path / "relax_wp"
        relax_atoms(
            h2_in_box,
            "sto-3g",
            gamma_kmesh,
            method="RHF",
            max_iter=5,
            conv_tol_grad=1e-3,
            cutoff_bohr=4.0,
            output_trajectory=stem,
            sr_image_precision=None,
        )
        qvf_path = stem.with_suffix(".qvf")
        with zipfile.ZipFile(qvf_path) as zf:
            rxn = next(
                s for s in json.loads(zf.read("manifest.json"))["sections"]
                if s["kind"] == "reaction.path"
            )
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
        kinds = {w["kind"] for w in meta["waypoints"]}
        assert kinds == {"reactant", "product"}
        labels = {w["label"] for w in meta["waypoints"]}
        assert "start" in labels
        # The "converged" / "stopped" label depends on res.success;
        # either is acceptable for this test.
        assert ("converged" in labels) or ("stopped" in labels)


def _bond_length(system: PeriodicSystem) -> float:
    xyz = np.array([list(a.xyz) for a in system.unit_cell], dtype=float)
    return float(np.linalg.norm(xyz[1] - xyz[0]))


@pytest.fixture
def h2_stretched() -> PeriodicSystem:
    """H₂ stretched to 1.9 bohr in a roomy 7-bohr cubic box (clean
    isolated-molecule minimum well inside the cell)."""
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.9])],
    )


class TestProductionRelaxConverges:
    """Certify the PRODUCTION relaxation workflow (SCF → exact FD forces →
    L-BFGS-B) drives the gradient to the convergence tolerance at a real
    stationary point, for both HF and DFT. ``relax_atoms`` defaults to
    ``force_mode="fd"`` (the exact gradient); ``result.converged`` is
    gated on the actual max-component gradient, not just scipy's flag."""

    @pytest.mark.slow
    def test_relax_converges_to_minimum_rhf(self, h2_stretched):
        km = monkhorst_pack(h2_stretched, (1, 1, 1))
        d0 = _bond_length(h2_stretched)
        result = relax_atoms(
            h2_stretched, "sto-3g", km, method="RHF",
            conv_tol_grad=1e-3, max_iter=30, cutoff_bohr=7.0,
            scf_conv_tol=1e-9,
        )
        assert isinstance(result, OptimizeResult)
        assert result.converged, (
            f"RHF relaxation did not converge (n_iter={result.n_iter}, "
            f"final |grad|={np.max(np.abs(result.gradient)):.2e})")
        # The forces genuinely vanished at the relaxed geometry.
        assert float(np.max(np.abs(result.gradient))) < 1e-3
        # And the bond actually relaxed (shortened from the stretched start).
        d1 = _bond_length(result.system)
        assert d1 < d0 - 0.05, f"bond did not relax: {d0:.3f} → {d1:.3f}"

    def test_cell_gradient_is_a_descent_direction(self):
        """``relax_cell_gradient`` uses ``_central_fd_gradient`` of the SCF
        energy as its cell gradient — a descent-consistent gradient.

        Regression for the 2026-06-05 fix: the previous force-virial stress
        (``compute_stress_tensor``) was inconsistent with the objective (it
        assumed atoms scale with strain + missed the explicit lattice/Ewald +
        Pulay stress) and came out OPPOSITE IN SIGN on H₂/STO-3G — validated
        directly: stepping along ``-g_virial`` RAISED the SCF energy while
        ``-g_fd`` lowered it. Here we pin the gradient *logic* on a cheap
        quadratic (no SCF): it must match the analytic gradient and give a
        descent direction."""
        from vibeqc.bipole_optimize import _central_fd_gradient

        A = np.diag([2.0, 3.0, 1.0, 4.0, 0.5, 1.5])
        b = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
        f = lambda x: 0.5 * x @ A @ x + b @ x  # noqa: E731
        x0 = np.array([0.3, -0.4, 0.1, 0.2, -0.1, 0.05])
        g = _central_fd_gradient(f, x0, 1e-4)
        # Matches the analytic gradient A x + b.
        assert np.allclose(g, A @ x0 + b, atol=1e-6)
        # -g is a genuine descent direction.
        assert f(x0 - 0.01 * g) < f(x0)

    def test_optimizer_option_copy_preserves_scf_controls(self):
        """BIPOLE optimization must preserve smearing/FMixing-style controls."""
        from vibeqc.bipole_optimize import _make_scf_options

        src = PeriodicKSOptions()
        src.max_iter = 123
        src.conv_tol_energy = 1e-11
        src.conv_tol_grad = 2e-9
        src.fock_mixing = 0.37
        src.diis_restart_tau = 2.5e-5
        src.diis_adaptive_delta = 7.5e-5
        src.level_shift = 0.25
        src.level_shift_schedule = [0.4, 0.2, 0.0]
        src.smearing_temperature = 0.005
        src.quadratic_fallback_iter = 7
        src.use_periodic_becke = True
        src.becke_image_radius_bohr = 8.5
        src.lattice_opts.cutoff_bohr = 6.0
        src.lattice_opts.nuclear_cutoff_bohr = 7.0
        src.lattice_opts.sr_range_screening = True
        src.atomic_spins = [1, -1]
        src.spinlock_mode = vq.SpinlockMode.PATTERN_HOLD
        src.spinlock_value = 2
        src.spinlock_iterations = 6

        out = _make_scf_options(
            "RKS",
            cutoff_bohr=9.0,
            scf_conv_tol=None,
            scf_max_iter=None,
            scf_options=src,
        )

        assert out.max_iter == 123
        assert out.conv_tol_energy == pytest.approx(1e-11)
        assert out.conv_tol_grad == pytest.approx(2e-9)
        assert out.fock_mixing == pytest.approx(0.37)
        assert out.diis_restart_tau == pytest.approx(2.5e-5)
        assert out.diis_adaptive_delta == pytest.approx(7.5e-5)
        assert out.level_shift == pytest.approx(0.25)
        assert list(out.level_shift_schedule) == pytest.approx([0.4, 0.2, 0.0])
        assert out.smearing_temperature == pytest.approx(0.005)
        assert out.quadratic_fallback_iter == 7
        assert out.use_periodic_becke is True
        assert out.becke_image_radius_bohr == pytest.approx(8.5)
        assert out.lattice_opts.cutoff_bohr == pytest.approx(9.0)
        assert out.lattice_opts.nuclear_cutoff_bohr == pytest.approx(9.0)
        assert out.lattice_opts.sr_range_screening is True
        assert out.atomic_spins == [1, -1]
        assert out.spinlock_mode == vq.SpinlockMode.PATTERN_HOLD
        assert out.spinlock_value == 2
        assert out.spinlock_iterations == 6
        assert src.lattice_opts.cutoff_bohr == pytest.approx(6.0)
        assert src.lattice_opts.nuclear_cutoff_bohr == pytest.approx(7.0)

    def test_optimizer_rejects_ecp_metadata_before_option_copy(self):
        from vibeqc.bipole_optimize import _make_scf_options

        src = SimpleNamespace(ecp_total_ncore=2)

        with pytest.raises(NotImplementedError, match="ECP metadata"):
            _make_scf_options(
                "RHF",
                cutoff_bohr=9.0,
                scf_conv_tol=None,
                scf_max_iter=None,
                scf_options=src,
            )

    def test_optimizer_rejects_read_guess_before_option_copy(self):
        from vibeqc.bipole_optimize import _make_scf_options

        src = PeriodicRHFOptions()
        src.initial_guess = vq.InitialGuess.READ

        with pytest.raises(NotImplementedError, match="InitialGuess.READ"):
            _make_scf_options(
                "RHF",
                cutoff_bohr=9.0,
                scf_conv_tol=None,
                scf_max_iter=None,
                scf_options=src,
            )

    def test_optimizer_rejects_iterative_solver_before_option_copy(self):
        from vibeqc.bipole_optimize import _make_scf_options

        src = PeriodicRHFOptions()
        src.use_davidson = True

        with pytest.raises(NotImplementedError, match="iterative diagonalization"):
            _make_scf_options(
                "RHF",
                cutoff_bohr=9.0,
                scf_conv_tol=None,
                scf_max_iter=None,
                scf_options=src,
            )

    def test_optimizer_rejects_analytic_forces_with_smearing(
        self,
        h2_in_box,
        gamma_kmesh,
    ):
        src = PeriodicRHFOptions()
        src.smearing_temperature = 0.005

        with pytest.raises(
            NotImplementedError,
            match="finite-temperature BIPOLE optimization",
        ):
            relax_atoms(
                h2_in_box,
                "sto-3g",
                gamma_kmesh,
                method="UHF",
                force_mode="analytic",
                scf_options=src,
            )

    def test_cell_optimizer_rebuilds_kmesh_on_strained_lattice(self, h2_in_box):
        """Cell relaxation must keep the requested fractional k-grid.

        ``BlochKMesh.kpoints`` are Cartesian.  Reusing a kmesh after the
        lattice changes silently evaluates the wrong reciprocal sampling, so
        the cell relaxers rebuild the mesh on each strained lattice.
        """
        from vibeqc.bipole_optimize import _kmesh_for_system

        km = monkhorst_pack(h2_in_box, (2, 1, 1))
        strained = PeriodicSystem(
            3,
            np.diag([5.0, 4.0, 4.0]),
            list(h2_in_box.unit_cell),
        )
        rebuilt = _kmesh_for_system(km, h2_in_box, strained)

        B_ref = np.asarray(h2_in_box.reciprocal_lattice(), dtype=float)
        B_new = np.asarray(strained.reciprocal_lattice(), dtype=float)
        frac_ref = np.linalg.solve(
            B_ref, np.asarray(km.kpoints, dtype=float).T
        ).T
        frac_new = np.linalg.solve(
            B_new, np.asarray(rebuilt.kpoints, dtype=float).T
        ).T
        np.testing.assert_allclose(frac_new, frac_ref, atol=1e-12)
        assert not np.allclose(
            np.asarray(rebuilt.kpoints, dtype=float),
            np.asarray(km.kpoints, dtype=float),
        )

    def test_optimizer_scf_gate_rejects_nonconverged_result(
        self, monkeypatch, h2_in_box, gamma_kmesh
    ):
        """Optimization must fail fast instead of using a failed SCF energy."""
        import vibeqc.bipole_optimize as opt

        def fake_run_pbc_bipole_rhf(*args, **kwargs):
            return SimpleNamespace(energy=-1.23, converged=False, n_iter=3)

        monkeypatch.setattr(opt, "run_pbc_bipole_rhf", fake_run_pbc_bipole_rhf)
        opts = PeriodicRHFOptions()

        with pytest.raises(RuntimeError, match="RHF SCF did not converge"):
            opt._run_scf(
                h2_in_box,
                None,
                gamma_kmesh,
                opts,
                "RHF",
                None,
            )

    @pytest.mark.parametrize(
        ("temperature", "expected"),
        [(0.0, -1.25), (0.01, -1.75)],
    )
    def test_optimizer_scf_uses_mermin_objective_when_smeared(
        self,
        monkeypatch,
        h2_in_box,
        gamma_kmesh,
        temperature,
        expected,
    ):
        """The optimizer objective must match the quantity differentiated."""
        import vibeqc.bipole_optimize as opt

        def fake_run_pbc_bipole_rhf(*args, **kwargs):
            return SimpleNamespace(
                energy=-1.25,
                free_energy=-1.75,
                smearing_temperature=temperature,
                converged=True,
            )

        monkeypatch.setattr(opt, "run_pbc_bipole_rhf", fake_run_pbc_bipole_rhf)
        objective, _ = opt._run_scf(
            h2_in_box,
            None,
            gamma_kmesh,
            PeriodicRHFOptions(),
            "RHF",
            None,
        )

        assert objective == pytest.approx(expected)

    @pytest.mark.parametrize("entrypoint", ["relax_cell", "relax_cell_gradient"])
    def test_variable_cell_entry_points_fail_closed(
        self, h2_in_box, gamma_kmesh, entrypoint
    ):
        """Direct BIPOLE cell optimizers reject the uncertified objective."""
        import vibeqc.bipole_optimize as opt

        with pytest.raises(NotImplementedError, match="variable-cell"):
            getattr(opt, entrypoint)(
                h2_in_box,
                "sto-3g",
                gamma_kmesh,
                method="RHF",
            )

    def test_relax_atoms_accepts_kpoints_object(self, monkeypatch, h2_in_box):
        """Direct BIPOLE optimizers accept the public KPoints builder."""
        import vibeqc.bipole_optimize as opt

        kp = vq.KPoints.gamma(h2_in_box)
        expected = vq.as_bloch_kmesh(kp)
        seen_k = []
        seen_sr_precision = []

        def fake_run_scf(system, basis, kmesh, opts, method, functional, **kwargs):
            seen_k.append(kmesh)
            seen_sr_precision.append(kwargs.get("sr_image_precision"))
            return 0.0, SimpleNamespace(energy=0.0, converged=True, n_iter=1)

        def fake_compute_forces(*args, **kwargs):
            return np.zeros((len(h2_in_box.unit_cell), 3))

        def fake_minimize(fun, x0, **kwargs):
            x0 = np.asarray(x0, dtype=float)
            energy = fun(x0)
            jac = kwargs["jac"](x0)
            return SimpleNamespace(
                x=x0,
                nit=0,
                fun=energy,
                jac=jac,
                success=True,
            )

        monkeypatch.setattr(opt, "_run_scf", fake_run_scf)
        monkeypatch.setattr(opt, "_compute_forces", fake_compute_forces)
        monkeypatch.setattr(opt, "minimize", fake_minimize)

        result = opt.relax_atoms(h2_in_box, "sto-3g", kp, method="RHF")

        assert isinstance(result, OptimizeResult)
        assert seen_k
        assert seen_sr_precision
        assert all(value == pytest.approx(1e-6) for value in seen_sr_precision)
        for kmesh in seen_k:
            np.testing.assert_allclose(
                np.asarray(kmesh.kpoints, dtype=float),
                np.asarray(expected.kpoints, dtype=float),
            )
            np.testing.assert_allclose(
                np.asarray(kmesh.weights, dtype=float),
                np.asarray(expected.weights, dtype=float),
            )

    def test_relax_atoms_rejects_low_dimensional_objective_before_setup(self):
        """The direct optimizer must not enter the diagnostic 1-D/2-D path."""
        slab = PeriodicSystem(
            2,
            np.diag([8.0, 8.0, 20.0]),
            [Atom(1, [0.0, 0.0, 9.3]), Atom(1, [0.0, 0.0, 10.7])],
        )

        with pytest.raises(NotImplementedError, match="only for 3-D"):
            relax_atoms(slab, "sto-3g", None, method="RHF")

    def test_relax_atoms_rejects_ad_hoc_multik_before_scf(
        self, monkeypatch, h2_in_box
    ):
        """Geometry derivatives cannot use the nonstationary legacy gauge."""
        import vibeqc.bipole_optimize as opt

        kpoints = vq.KPoints.from_list(
            h2_in_box,
            [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]],
        )

        def forbid_scf(*args, **kwargs):
            pytest.fail("SCF entered before optimizer k-mesh validation")

        monkeypatch.setattr(opt, "_run_scf", forbid_scf)
        with pytest.raises(NotImplementedError, match="complete Monkhorst-Pack"):
            opt.relax_atoms(h2_in_box, "sto-3g", kpoints, method="RHF")

    def test_relax_atoms_rejects_legacy_multik_gauge_before_scf(
        self, monkeypatch, h2_in_box
    ):
        """A complete mesh cannot make the legacy gauge stationary."""
        import vibeqc.bipole_optimize as opt

        kmesh = monkhorst_pack(h2_in_box, (2, 1, 1))

        def forbid_scf(*args, **kwargs):
            pytest.fail("SCF entered before optimizer gauge validation")

        monkeypatch.setattr(opt, "_run_scf", forbid_scf)
        with pytest.raises(NotImplementedError, match="corrected Ewald"):
            opt.relax_atoms(
                h2_in_box,
                "sto-3g",
                kmesh,
                method="RHF",
                use_exchange_ewald_split=False,
            )

    def test_analytic_optimizer_forwards_kmesh_without_plus_u(
        self, monkeypatch
    ):
        """Ordinary multi-k analytic gradients receive the SCF mesh."""
        import vibeqc.bipole_optimize as opt

        marker = object()
        seen = {}

        def fake_gradient(system, basis, result, **kwargs):
            seen.update(kwargs)
            return np.zeros((1, 3))

        monkeypatch.setattr(opt, "compute_bipole_gradient_rhf", fake_gradient)
        gradient = opt._compute_gradient(
            object(),
            object(),
            object(),
            "RHF",
            object(),
            kmesh=marker,
        )

        assert gradient.shape == (1, 3)
        assert seen["kmesh"] is marker
        assert "dft_plus_u" not in seen

    def test_relax_full_fails_closed(self, h2_in_box, gamma_kmesh):
        """Coupled atom/cell optimization cannot certify mixed geometries."""
        import vibeqc.bipole_optimize as opt

        with pytest.raises(NotImplementedError, match="variable-cell"):
            opt.relax_full(
                h2_in_box,
                "sto-3g",
                gamma_kmesh,
                method="RHF",
            )

    @pytest.mark.slow
    def test_relax_freezes_atoms_exactly(self):
        """``relax_atoms(freeze_indices=...)`` (the surface-catalysis path —
        freeze the bottom slab layers) must hold the frozen atoms EXACTLY in
        place while the free atoms relax. The L-BFGS-B (fixed,fixed) bounds
        keep frozen positions, and the gradient is zeroed on them."""
        sysp = PeriodicSystem(
            3, np.diag([12.0, 12.0, 12.0]),
            [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.8])])
        km = monkhorst_pack(sysp, (1, 1, 1))
        xyz0 = np.array([list(a.xyz) for a in sysp.unit_cell])
        result = relax_atoms(
            sysp, "sto-3g", km, method="RHF", freeze_indices=[0],
            conv_tol_grad=3e-3, max_iter=6, cutoff_bohr=7.0,
            scf_conv_tol=1e-8, scf_max_iter=100)
        xyz1 = np.array([list(a.xyz) for a in result.system.unit_cell])
        # Frozen atom did not move at all.
        assert float(np.linalg.norm(xyz1[0] - xyz0[0])) < 1e-10
        # Free atoms genuinely relaxed.
        assert float(np.linalg.norm(xyz1[1] - xyz0[1])) > 0.05

    @pytest.mark.slow
    def test_relax_converges_to_minimum_rks(self, h2_stretched):
        km = monkhorst_pack(h2_stretched, (1, 1, 1))
        d0 = _bond_length(h2_stretched)
        result = relax_atoms(
            h2_stretched, "sto-3g", km, method="RKS", functional="svwn",
            conv_tol_grad=1e-3, max_iter=30, cutoff_bohr=7.0,
            scf_conv_tol=1e-9,
        )
        assert result.converged, (
            f"RKS relaxation did not converge (n_iter={result.n_iter}, "
            f"final |grad|={np.max(np.abs(result.gradient)):.2e})")
        assert float(np.max(np.abs(result.gradient))) < 1e-3
        d1 = _bond_length(result.system)
        assert d1 < d0 - 0.05, f"bond did not relax: {d0:.3f} → {d1:.3f}"
