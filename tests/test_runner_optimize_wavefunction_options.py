"""Geometry optimization honors the wavefunction solver options.

Pre-2026-06-12 the ASE optimize path dropped solver options that the
final single point honors, so BFGS could walk a different surface than
the reported final energy:

* ``method="casci"`` / ``"casscf"`` were missing from the wavefunction
  routing set in ``_optimize_geometry`` and fell through to the
  mean-field ``VibeQC`` calculator — the optimizer silently walked the
  RHF surface;
* ``_optimize_geometry`` never forwarded ``casscf_options`` /
  ``casci_options`` / ``cas_reference`` to
  ``_make_wavefunction_ase_calculator``, so an SA-CASSCF optimization
  would have stepped on the default single-state surface;
* inside the calculator, the *energy* ``_run_single_point`` call dropped
  ``casscf_options`` while the two FD-force displaced calls carried it —
  energies and forces sampled different surfaces.

The native (``optimizer_backend="native"``) backend had the same bug
one layer down: ``molecular_optimize._evaluate_energy`` (and its caller
``_gradient_via_central_difference``) passed only the mean-field options
to ``_run_single_point`` — no ``active_space``, no wavefunction option
structs — so a ``selected_ci`` / ``casscf`` optimization evaluated every
FD displacement with default options and *no active-space truncation*
(full-space CI per displacement: a different, far more expensive surface
than the final single point). ``run_job``'s native branch likewise never
forwarded those kwargs into ``optimize_molecule``. Fixed 2026-06-12,
same day as the ASE path.

These tests pin both backends end to end: every per-step solver call
receives the user's options (monkeypatch capture), and the trajectory
energies of an SA-CASSCF optimization match a direct
``_run_single_point`` with the same options at the same geometries.
The ASE tests skip without ASE; the native tests run everywhere (the
native backend exists precisely for ASE-less installs).
"""

from __future__ import annotations

import pytest

import vibeqc.runner as runner
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import BasisSet
from vibeqc.runner import run_job
from vibeqc.solvers import CASCIOptions, CASSCFOptions, NEVPT2Options

# H2 at R = 1.4 bohr: CAS(2,2)/STO-3G is the full CI space, so the
# SA(2) average and the single-state ground root are separated by
# ~half the large root gap — an unmissable discriminator for dropped
# casscf_options (or a silent RHF-surface fallback).
H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])

SA2 = dict(nroots=2)
ACTIVE = (2, 2)


def test_optimizer_calculator_receives_cas_options(monkeypatch, tmp_path):
    """Every per-step _run_single_point call (energy + each FD displacement)
    carries the user's casscf_options / casci_options / cas_reference /
    active_space, and method='casscf' routes to the wavefunction
    calculator at all (not the mean-field VibeQC fallback)."""
    pytest.importorskip("ase")
    captured = []

    import numpy as np

    class _Fake:
        energy = -1.0
        gradient = np.zeros((2, 3))

    def _stub(method, mol, basis, **kwargs):
        captured.append((method, kwargs))
        return _Fake()

    monkeypatch.setattr(runner, "_run_single_point", _stub)

    c_opts = CASSCFOptions(nroots=2)
    ci_opts = CASCIOptions(max_det=50_000)
    nev_opts = NEVPT2Options(compute_corr_grad=True)
    runner._optimize_geometry(
        H2,
        "sto-3g",
        functional=None,
        trajectory_path=tmp_path / "h2-capture.traj",
        fmax=0.05,
        max_steps=1,
        method="casscf",
        casscf_options=c_opts,
        casci_options=ci_opts,
        nevpt2_options=nev_opts,
        active_space=ACTIVE,
        cas_reference="rhf",
    )

    # Pre-fix, method="casscf" fell through to the VibeQC mean-field
    # calculator and the stub was never reached.
    assert captured, "casscf did not route to the wavefunction calculator"
    # Analytic path: 1 priming + >=1 scipy energy eval = 2 minimum.
    assert len(captured) >= 2
    for method, kw in captured:
        assert method == "casscf"
        assert kw["casscf_options"] is c_opts
        assert kw["casci_options"] is ci_opts
        assert kw["nevpt2_options"] is nev_opts
        assert kw["active_space"] == ACTIVE
        assert kw["cas_reference"] == "rhf"


def test_sa_casscf_optimization_walks_the_sa_surface(tmp_path):
    """run_job(method='casscf', optimize=True, casscf_options=SA(2)):
    every trajectory frame energy equals a direct _run_single_point with
    the same casscf_options at that frame's geometry."""
    pytest.importorskip("ase")
    from ase.io.trajectory import Trajectory
    from ase.units import Bohr, Hartree

    res = run_job(
        H2,
        basis="sto-3g",
        method="casscf",
        active_space=ACTIVE,
        casscf_options=CASSCFOptions(**SA2),
        optimize=True,
        max_opt_steps=2,
        output=tmp_path / "h2-sa-opt",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert res.converged

    frames = list(Trajectory(str(tmp_path / "h2-sa-opt.traj")))
    assert frames, "optimizer wrote no trajectory frames"

    compared = 0
    for atoms in frames:
        try:
            e_traj = float(atoms.get_potential_energy()) / Hartree
        except Exception:
            continue  # frame without stored energy — nothing to compare
        mol = Molecule(
            [
                Atom(int(z), list(xyz))
                for z, xyz in zip(atoms.numbers, atoms.positions / Bohr)
            ]
        )
        e_direct = float(
            runner._run_single_point(
                "casscf",
                mol,
                BasisSet(mol, "sto-3g"),
                functional=None,
                casscf_options=CASSCFOptions(**SA2),
                active_space=ACTIVE,
            ).energy
        )
        assert abs(e_traj - e_direct) < 1e-8
        compared += 1
    assert compared >= 1

    # Discriminator: at the starting geometry the SA(2) average sits
    # ~(E1-E0)/2 above the single-state ground root, so a per-step
    # calculator running default (single-state) CASSCF — or RHF — could
    # not have produced the frame energies asserted above.
    b0 = BasisSet(H2, "sto-3g")
    e_sa = float(
        runner._run_single_point(
            "casscf",
            H2,
            b0,
            functional=None,
            casscf_options=CASSCFOptions(**SA2),
            active_space=ACTIVE,
        ).energy
    )
    e_single = float(
        runner._run_single_point(
            "casscf",
            H2,
            b0,
            functional=None,
            active_space=ACTIVE,
        ).energy
    )
    assert abs(e_sa - e_single) > 0.05


# ---- native (optimizer_backend="native") backend mirrors ------------------


def test_native_fd_calls_receive_cas_options(monkeypatch):
    """Native mirror of the capture test: every _run_single_point call
    optimize_molecule's analytic path makes carries the user's
    casscf_options / casci_options / cas_reference / active_space.
    Since CASSCF analytic gradients entered via commit c0c88283, the
    method now routes through _run_molecular_scf (not the FD path).
    Pre-fix (2026-06-12), _evaluate_energy forwarded none of them;
    pre-fix (2026-06-22), _run_molecular_scf dropped casci_options
    and cas_reference on the casscf branch."""
    from vibeqc.molecular_optimize import optimize_molecule

    captured = []

    import numpy as np

    class _Fake:
        energy = -1.0
        gradient = np.zeros((2, 3))

    def _stub(method, mol, basis, **kwargs):
        captured.append((method, kwargs))
        return _Fake()

    # _evaluate_energy re-imports _run_single_point from vibeqc.runner on
    # every call, so patching the runner module is seen immediately.
    monkeypatch.setattr(runner, "_run_single_point", _stub)

    c_opts = CASSCFOptions(nroots=2)
    ci_opts = CASCIOptions(max_det=50_000)
    optimize_molecule(
        H2,
        "sto-3g",
        method="casscf",
        casscf_options=c_opts,
        casci_options=ci_opts,
        active_space=ACTIVE,
        cas_reference="rhf",
        max_iter=1,
        record_trajectory=False,
    )

    # Analytic path: 1 priming + >=1 scipy energy eval = 2 minimum.
    assert len(captured) >= 2
    for method, kw in captured:
        assert method == "casscf"
        assert kw["casscf_options"] is c_opts
        assert kw["casci_options"] is ci_opts
        assert kw["active_space"] == ACTIVE
        assert kw["cas_reference"] == "rhf"


def test_native_run_job_forwards_wavefunction_options(monkeypatch, tmp_path):
    """run_job's native branch forwards the wavefunction option structs +
    active_space + cas_reference into optimize_molecule. Pre-fix it
    passed only the mean-field options."""
    import numpy as np

    import vibeqc.molecular_optimize as mo

    captured = {}

    def _capture(mol_arg, basis_name, **kwargs):
        captured.update(kwargs)
        return mo.MolecularOptimizeResult(
            system=mol_arg,
            energy=-1.0,
            gradient=np.zeros((2, 3)),
            n_iter=1,
            converged=True,
        )

    # run_job imports optimize_molecule from the module at call time.
    monkeypatch.setattr(mo, "optimize_molecule", _capture)

    c_opts = CASSCFOptions(nroots=2)
    ci_opts = CASCIOptions(max_det=50_000)
    nev_opts = NEVPT2Options(compute_corr_grad=True)
    run_job(
        H2,
        basis="sto-3g",
        method="casscf",
        active_space=ACTIVE,
        casscf_options=c_opts,
        casci_options=ci_opts,
        nevpt2_options=nev_opts,
        cas_reference="rhf",
        optimize=True,
        optimizer_backend="native",
        max_opt_steps=2,
        output=tmp_path / "h2-native-capture",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )

    assert captured, "run_job never reached the native optimizer"
    assert captured["method"] == "casscf"
    assert captured["casscf_options"] is c_opts
    assert captured["casci_options"] is ci_opts
    assert captured["nevpt2_options"] is nev_opts
    assert captured["active_space"] == ACTIVE
    assert captured["cas_reference"] == "rhf"
    # The remaining wavefunction option kwargs must arrive too (None here,
    # but their absence would mean run_job still drops them).
    for key in (
        "selected_ci_options",
        "dmrg_options",
        "v2rdm_options",
        "transcorrelated_options",
        "caspt2_options",
    ):
        assert key in captured


def test_native_sa_casscf_optimization_walks_the_sa_surface():
    """Native mirror of the SA-surface test: every trajectory energy of
    an optimize_molecule SA-CASSCF(2,2) run equals a direct
    _run_single_point with the same options at that frame's geometry —
    and differs from the single-state surface, so dropped options
    cannot pass."""
    from vibeqc.molecular_optimize import optimize_molecule

    res = optimize_molecule(
        H2,
        "sto-3g",
        method="casscf",
        casscf_options=CASSCFOptions(**SA2),
        active_space=ACTIVE,
        max_iter=2,
        record_trajectory=True,
    )

    assert res.trajectory_frames, "native optimizer recorded no frames"
    assert len(res.trajectory_frames) == len(res.trajectory_energies)
    for mol, e_traj in zip(res.trajectory_frames, res.trajectory_energies):
        e_direct = float(
            runner._run_single_point(
                "casscf",
                mol,
                BasisSet(mol, "sto-3g"),
                functional=None,
                casscf_options=CASSCFOptions(**SA2),
                active_space=ACTIVE,
            ).energy
        )
        assert abs(e_traj - e_direct) < 1e-8

    # Discriminator: frame 0 is the starting geometry, where the SA(2)
    # average sits ~0.48 Ha above the single-state ground root — per-step
    # evaluations that dropped casscf_options (or active_space) could not
    # have produced the frame energies asserted above.
    e_single = float(
        runner._run_single_point(
            "casscf",
            H2,
            BasisSet(H2, "sto-3g"),
            functional=None,
            active_space=ACTIVE,
        ).energy
    )
    assert abs(res.trajectory_energies[0] - e_single) > 0.05
