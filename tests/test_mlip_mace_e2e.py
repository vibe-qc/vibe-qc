"""Real end-to-end MACE runs through ``run_job`` — the M5 validation.

Unlike ``test_mlip_mace.py`` (pure-Python registry/gate logic), these
actually download a foundation model and run a MACE forward pass, so they
require the ``[mace]`` extra (Python <= 3.13). The module
``pytest.importorskip``s ``mace``, so it is **skipped automatically** on
environments without the extra (e.g. the 3.14 dev venv / the collect-only
CI). The supported runtime is <= 3.13; passing this suite on a manually
provisioned newer interpreter is compatibility evidence, not acceptance of
a new supported interpreter range. Marked ``slow`` because first use downloads weights.
"""

import math
import os
import pathlib
import sys

import pytest

# vibe-qc's C++ core (libxc / BLAS) and PyTorch each link their own OpenMP
# runtime; on macOS the double-load aborts (OMP Error #15) unless this is
# set before either loads. Duplicate runtimes can still crash when competing
# worker pools start, so the MACE lane also keeps OpenMP/BLAS pools at one
# thread unless the caller deliberately overrides them. The MACE forward
# pass is verified (M5) to give identical energies with the workaround set.
# Match the platform scope of vibeqc.mlip.mace._maybe_set_openmp_workaround.
# Linux collection must preserve the caller's qualified runtime settings.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for _name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(_name, "1")

pytest.importorskip("mace", reason="requires the [mace] extra (Python <= 3.13)")

from vibeqc import Atom, Molecule  # noqa: E402
from vibeqc.mlip import MLIPOptions  # noqa: E402
from vibeqc.runner import run_job  # noqa: E402

pytestmark = pytest.mark.slow

# H2O geometry in bohr (ASE G2). H + O are covered by both the materials
# (89-element) and organic (10-element) MACE foundation models.
_H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.2253]),
        Atom(1, [0.0, 1.4425, -0.9012]),
        Atom(1, [0.0, -1.4425, -0.9012]),
    ],
    0,
    1,
)

# Published target: H2O / MACE-MPA-0 (the MIT default) = -0.50662347 Ha,
# the reference-shifted DFT-surface energy verified by the standalone
# wrapper in M1 (matches the raw MACE smoke test to 7 figures).
_H2O_MPA0_HA = -0.50662347


def _bibtex_keys(stem):
    text = pathlib.Path(str(stem) + ".bibtex").read_text()
    return {
        line.split("{")[1].rstrip(",")
        for line in text.splitlines()
        if line.startswith("@")
    }


def test_real_mace_mp_materials_singlepoint(tmp_path):
    """The MIT MACE-MPA-0 default runs end-to-end through run_job and
    reproduces the M1 standalone energy; cites the method + MP foundation
    paper, and NOT libint/DIIS (no Gaussian integrals, no SCF)."""
    stem = tmp_path / "h2o_mp"
    res = run_job(_H2O, method="mace", output=str(stem))
    assert math.isclose(float(res.energy), _H2O_MPA0_HA, abs_tol=1e-2)
    keys = _bibtex_keys(stem)
    assert {"batatia_mace_2022", "batatia_mace_mp_2024"} <= keys
    assert "valeev_libint" not in keys
    assert "pulay_diis_1980" not in keys

    # The .out presents MACE honestly — a single pre-trained forward pass,
    # NOT a fabricated SCF: no Fock-commutator / DIIS iteration table, no
    # "converged in N iterations", an MLIP-specific timing label, and the
    # model provenance block. (Regression guard for the output-honesty fix.)
    out = pathlib.Path(str(stem) + ".out").read_text()
    assert "pre-trained MLIP forward pass" in out
    assert "MACE machine-learning interatomic potential" in out
    assert "MLIP energy evaluation" in out          # timings, not "SCF total"
    assert "[F,DS]" not in out                       # no SCF/DIIS iter table
    assert "converged in" not in out                 # no SCF convergence claim
    assert "SCF total" not in out
    assert "Loader:    mace_mp(model='medium-mpa-0')" in out
    assert "Runtime:   device=cpu   dtype=float64" in out
    assert "Weights:   upstream on-demand cache" in out


@pytest.mark.parametrize("model_name,dtype", [
    ("small", "float32"), ("mp-0", "float64"), ("large", "float64"),
])
def test_real_mace_model_selection_survives_qvf_replay(tmp_path, monkeypatch, model_name, dtype):
    """A saved nondefault potential must not replay on the default surface."""
    import json
    import tomllib
    import zipfile
    import numpy as np
    from ase.units import Bohr, Hartree

    from vibeqc.mlip import resolve_model
    from vibeqc.qvf_job import run_container

    stem = tmp_path / "selected"
    result = run_job(
        _H2O, method="mace", output=stem, record_hostname=False,
        mlip_options=MLIPOptions(model=model_name, device="cpu", dtype=dtype),
    )
    replay = run_container(
        stem.with_suffix(".qvf"), output=tmp_path / "replay",
        force=True, update_in_place=False, record_hostname=False,
    )
    assert replay.energy == pytest.approx(result.energy, abs=1e-12, rel=0)
    # Direct upstream float32 evaluations can differ after JIT optimization,
    # even at byte-identical coordinates. Use its existing native fp32 floor
    # (upstream tests/golden/harness.py at f443730), also used below for MPS.
    # The float64 and energy guards remain at their original bounds.
    gradient_atol = 5e-5 * Bohr / Hartree if dtype == "float32" else 1e-12
    np.testing.assert_allclose(replay.gradient(), result.gradient(), atol=gradient_atol, rtol=0)

    info = resolve_model(model_name)
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        spec = json.loads(archive.read("job_spec/spec.json"))
        provenance = json.loads(archive.read("manifest.json"))["provenance"]
    assert spec["options"]["mlip_options"] == {
        "model": info.key, "device": "cpu", "dtype": dtype,
        "weight_sha256": result.weight_identity["sha256"],
    }
    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert manifest["run"]["mlip"] == provenance["mlip"]
    assert provenance["mlip"]["model"] == info.key
    assert provenance["mlip"]["loader_arg"] == info.loader_arg
    assert provenance["mlip"]["license"] == "MIT"
    assert provenance["mlip"]["dtype"] == dtype
    assert provenance["mlip"]["checkpoint"] == result.weight_identity
    assert replay.weight_identity == result.weight_identity
    assert result.weight_identity["size_bytes"] > 0
    assert result.weight_identity["sha256"] in stem.with_suffix(".out").read_text()
    assert provenance["scf_applicable"] is False
    assert "scf_converged" not in provenance
    assert "n_scf_iterations" not in provenance
    assert f"Loader:    mace_mp(model={info.loader_arg!r})" in stem.with_suffix(".out").read_text()
    assert {"batatia_mace_2022", "batatia_mace_mp_2024"} <= _bibtex_keys(stem)
    if info.key == "small":
        from vibeqc.mlip import _mace_checkpoint as checkpoint

        changed = tmp_path / "changed.model"
        changed.write_bytes(b"different cached checkpoint")
        with monkeypatch.context() as patched:
            patched.setattr(checkpoint, "_checkpoint_path", lambda info: changed)
            with pytest.raises(ValueError, match="checkpoint SHA256"):
                run_container(
                    stem.with_suffix(".qvf"), output=tmp_path / "changed-replay",
                    force=True, update_in_place=False, record_hostname=False,
                )


def test_real_mace_off_organic_gated_then_runs(tmp_path):
    """The ASL MACE-OFF23 organic model is gated without acknowledgment
    (fail-fast PermissionError, no output files); with acknowledgment it
    runs and cites the OFF23 paper (not the MP foundation paper)."""
    gated = tmp_path / "gated"
    with pytest.raises(PermissionError, match="Academic Software License"):
        run_job(
            _H2O,
            method="mace",
            mlip_options=MLIPOptions(model="off23-medium"),
            output=str(gated),
        )
    assert not (tmp_path / "gated.out").exists()  # fail-fast

    stem = tmp_path / "h2o_off"
    res = run_job(
        _H2O,
        method="mace",
        mlip_options=MLIPOptions(model="off23-medium", accept_academic_license=True),
        output=str(stem),
    )
    assert math.isfinite(float(res.energy))
    keys = _bibtex_keys(stem)
    assert "kovacs_mace_off_2023" in keys
    assert "batatia_mace_mp_2024" not in keys


@pytest.mark.parametrize("max_steps", [0, 1])
def test_real_mace_cell_optimization_rejects_exhausted_budget(max_steps):
    """A compressed diamond Si cell has zero atomic forces but finite stress."""
    import numpy as np

    from vibeqc import PeriodicSystem
    from vibeqc.mlip.mace import (
        PeriodicMACECellOptimizationError,
        optimize_periodic_mace_cell,
    )

    half_cell = 4.82276969545227  # bohr; compressed diamond primitive cell
    lattice = half_cell * (np.ones((3, 3)) - np.eye(3))
    system = PeriodicSystem(
        3, lattice,
        [Atom(14, [0., 0., 0.]), Atom(14, [half_cell / 2] * 3)],
        charge=0, multiplicity=1,
    )
    with pytest.raises(PeriodicMACECellOptimizationError) as caught:
        optimize_periodic_mace_cell(
            system, MLIPOptions(model="medium-mpa-0", device="cpu", dtype="float64"),
            fmax=0.01, max_steps=max_steps,
        )
    error = caught.value
    assert error.n_steps == max_steps
    assert error.max_steps == max_steps
    assert np.isfinite(error.final_max_force)
    assert error.final_max_force >= error.target_max_force == 0.01
    assert error.system.dim == 3
    assert np.array_equal(system.lattice, lattice)


@pytest.mark.parametrize("max_steps", [0, 1, 200])
def test_real_mace_cell_optimization_matches_upstream(max_steps):
    """#476: compare terminal states against the direct ASE optimizer.

    Use the same compressed Si input, MIT MPA-0 model and fmax=0.01 for
    exhausted and converged controls. The energy/force/stress bounds are
    the existing same-model wrapper canary bounds, not accuracy estimates
    for the learned potential.
    """
    import numpy as np
    from ase import Atoms
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from ase.units import Bohr, Hartree
    from mace.calculators import mace_mp

    from vibeqc import PeriodicSystem
    from vibeqc.mlip.mace import (
        PeriodicMACECellOptimizationError,
        optimize_periodic_mace_cell,
        run_periodic_mace,
    )

    half_cell = 4.82276969545227
    lattice = half_cell * (np.ones((3, 3)) - np.eye(3))
    positions = np.array([[0., 0., 0.], [half_cell / 2] * 3])
    system = PeriodicSystem(3, lattice, [Atom(14, list(r)) for r in positions])
    upstream = Atoms(numbers=[14, 14], positions=positions * Bohr,
                     cell=lattice.T * Bohr, pbc=True)
    upstream.calc = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64")
    cell_filter = FrechetCellFilter(upstream)
    optimizer = BFGS(cell_filter, logfile=None)
    converged = bool(optimizer.run(fmax=0.01, steps=max_steps))
    residual = np.linalg.norm(cell_filter.get_forces(), axis=1).max()
    assert converged == (max_steps == 200)
    if converged:
        final = optimize_periodic_mace_cell(system, fmax=0.01, max_steps=max_steps)
        assert residual < 0.01
    else:
        with pytest.raises(PeriodicMACECellOptimizationError) as caught:
            optimize_periodic_mace_cell(system, fmax=0.01, max_steps=max_steps)
        error = caught.value
        final = error.system
        assert error.n_steps == optimizer.nsteps == max_steps
        assert error.final_max_force == pytest.approx(residual, abs=1e-12, rel=0)
        assert residual >= error.target_max_force == 0.01

    np.testing.assert_allclose(final.lattice, upstream.cell.array.T / Bohr,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose([a.xyz for a in final.unit_cell], upstream.positions / Bohr,
                               atol=1e-12, rtol=0)
    result = run_periodic_mace(final)
    assert result.energy == pytest.approx(upstream.get_potential_energy() / Hartree,
                                          abs=1e-9, rel=0)
    np.testing.assert_allclose(result.gradient(), -upstream.get_forces() * Bohr / Hartree,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.stress(), upstream.get_stress(voigt=False) * Bohr**3 / Hartree,
                               atol=1e-15, rtol=0)
    np.testing.assert_array_equal(system.lattice, lattice)
    np.testing.assert_array_equal([a.xyz for a in system.unit_cell], positions)


def test_real_mace_geometry_optimization(tmp_path, monkeypatch):
    """method="mace" + optimize=True relaxes a distorted geometry via ASE
    BFGS on the MACE calculator, writes a .traj, and lowers the energy
    relative to the distorted single point."""
    import json
    import zipfile
    import numpy as np
    from ase.io.trajectory import Trajectory
    from ase.units import Bohr, Hartree
    from mace.calculators import mace_mp
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    distorted = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.35]),
            Atom(1, [0.0, 1.55, -1.0]),
            Atom(1, [0.0, -1.55, -1.0]),
        ],
        0,
        1,
    )
    sp = run_job(distorted, method="mace", output=str(tmp_path / "sp"))
    loads = []
    original_load = checkpoint._load

    def counted_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        loads.append(loaded._vibeqc_checkpoint_identity)
        return loaded

    monkeypatch.setattr(checkpoint, "_load", counted_load)
    opt = run_job(
        distorted, method="mace", optimize=True, fmax=0.05,
        output=str(tmp_path / "opt"),
    )
    assert (tmp_path / "opt.traj").exists()
    assert math.isfinite(float(opt.energy))
    assert float(opt.energy) <= float(sp.energy) + 1e-6
    assert len(loads) == 1
    assert opt.weight_identity["sha256"] == loads[0].sha256
    with Trajectory(str(tmp_path / "opt.traj")) as trajectory:
        terminal = trajectory[-1]
    assert np.linalg.norm(terminal.get_forces(), axis=1).max() < 0.05
    assert opt.energy == pytest.approx(terminal.get_potential_energy() / Hartree, abs=1e-9, rel=0)
    np.testing.assert_allclose(
        opt.gradient(), -terminal.get_forces() * Bohr / Hartree,
        atol=1e-12, rtol=0,
    )
    # Independent upstream loader at the exact terminal geometry, using the
    # same registered MIT checkpoint and the existing double-precision bounds.
    terminal.calc = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64")
    assert opt.energy == pytest.approx(terminal.get_potential_energy() / Hartree, abs=1e-9, rel=0)
    np.testing.assert_allclose(
        opt.gradient(), -terminal.get_forces() * Bohr / Hartree,
        atol=1e-12, rtol=0,
    )
    with zipfile.ZipFile(tmp_path / "opt.qvf") as archive:
        spec = json.loads(archive.read("job_spec/spec.json"))
        provenance = json.loads(archive.read("manifest.json"))["provenance"]
    assert spec["options"]["mlip_options"]["weight_sha256"] == loads[0].sha256
    assert provenance["mlip"]["checkpoint"] == opt.weight_identity


def test_real_mace_periodic_singlepoint():
    """Periodic MACE on bulk Si: finite energy, ~zero forces (diamond
    symmetry), and a symmetric, isotropic stress tensor (Ha/bohr^3)."""
    import numpy as np
    from ase.build import bulk

    from vibeqc.ase_periodic import atoms_to_periodic_system
    from vibeqc.mlip.mace import run_periodic_mace

    system = atoms_to_periodic_system(bulk("Si", "diamond", a=5.43))
    res = run_periodic_mace(system)
    assert math.isfinite(float(res.energy))
    assert res.gradient().shape == (2, 3)
    assert np.abs(res.gradient()).max() < 1e-6  # zero by symmetry
    s = res.stress()
    assert s.shape == (3, 3)
    assert np.allclose(s, s.T, atol=1e-8)  # symmetric
    assert np.allclose(np.diag(s), s[0, 0], atol=1e-8)  # isotropic (cubic)


def test_real_mace_periodic_cell_relaxation():
    """Variable-cell relaxation drives the stress toward zero and does not
    raise the energy."""
    import numpy as np
    from ase.build import bulk

    from vibeqc.ase_periodic import atoms_to_periodic_system
    from vibeqc.mlip.mace import optimize_periodic_mace_cell, run_periodic_mace

    system = atoms_to_periodic_system(bulk("Si", "diamond", a=5.43))
    before = run_periodic_mace(system)
    relaxed = optimize_periodic_mace_cell(system, fmax=0.01)
    after = run_periodic_mace(relaxed)
    assert np.abs(np.diag(after.stress())).max() < np.abs(np.diag(before.stress())).max()
    assert float(after.energy) <= float(before.energy) + 1e-6


def test_real_mace_true_slab_singlepoint_and_position_relaxation():
    """A genuine dim=2 slab stays non-periodic along its normal, exposes
    energy/forces (not arbitrary-volume stress), and supports a fixed-cell
    relaxation with a frozen bottom layer."""
    import numpy as np
    import vibeqc as vq

    from vibeqc.mlip.mace import (
        optimize_periodic_mace_positions,
        run_periodic_mace,
    )

    slab, info = vq.slab(
        "Cu",
        facet=(1, 1, 1),
        n_layers=2,
        supercell=(1, 1),
    )
    before = run_periodic_mace(slab)
    assert before.dim == 2
    assert before.pbc == (True, True, False)
    assert math.isfinite(float(before.energy))
    assert np.isfinite(before.gradient()).all()
    with pytest.raises(ValueError, match="only for dim=3"):
        before.stress()

    relaxed = optimize_periodic_mace_positions(
        slab,
        fmax=0.1,
        max_steps=5,
        fixed_indices=info.bottom_layer_indices(1),
    )
    assert relaxed.system.dim == 2
    assert relaxed.pbc == (True, True, False)
    assert relaxed.fixed_indices == info.bottom_layer_indices(1)
    assert math.isfinite(float(relaxed.energy))
    fixed = list(relaxed.fixed_indices)
    original = np.asarray([atom.xyz for atom in slab.unit_cell])
    final = np.asarray([atom.xyz for atom in relaxed.system.unit_cell])
    np.testing.assert_allclose(final[fixed], original[fixed], atol=1e-14, rtol=0)
    singlepoint = run_periodic_mace(relaxed.system)
    np.testing.assert_allclose(
        relaxed.gradient(), singlepoint.gradient(), atol=1e-12, rtol=0,
    )


@pytest.mark.parametrize("model_name", ["medium-mpa-0", "small", "medium", "large"])
def test_real_mace_wrapper_parity_with_raw_calculator(model_name):
    """vibe-qc's MACEModel must reproduce the raw ASE MACE calculator
    EXACTLY — only unit conversion (bohr↔Å, eV↔Ha), no silent alteration.
    Same geometry in bohr (vibe-qc) vs Ångström (raw) → identical energy;
    gradient is exactly −forces in Ha/bohr. (Production-readiness audit,
    2026-06-05: this property held to machine precision — Δ_E = 0,
    Δ_grad ~ 1e-18.)"""
    import numpy as np
    from ase import Atoms
    from ase.units import Bohr, Hartree

    from mace.calculators import mace_mp

    from vibeqc.mlip.mace import MACEModel

    pos_A = np.array([[0, 0, 0.119], [0, 0.763, -0.477], [0, -0.763, -0.477]])
    raw = mace_mp(model=model_name, device="cpu", default_dtype="float64")
    a = Atoms(numbers=[8, 1, 1], positions=pos_A)
    a.calc = raw
    E_eV, F_eVA = a.get_potential_energy(), a.get_forces()

    mol = Molecule(
        [Atom(8, list(pos_A[0] / Bohr)), Atom(1, list(pos_A[1] / Bohr)),
         Atom(1, list(pos_A[2] / Bohr))], 0, 1,
    )
    m = MACEModel(mol, MLIPOptions(model=model_name))
    # Every registered materials weight must match its advertised coverage,
    # not only the default model used to establish the registry's element set.
    assert set(raw.z_table.zs) == m.info.elements_z
    assert set(m.calculator.z_table.zs) == m.info.elements_z
    assert len(m.info.elements_z) == m.info.elements == 89
    assert abs(m.energy() - E_eV / Hartree) < 1e-9             # eV→Ha exact
    g_ref = -np.asarray(F_eVA) * Bohr / Hartree                # gradient = −F
    assert np.abs(m.gradient() - g_ref).max() < 1e-12          # Ha/bohr exact


@pytest.mark.parametrize("model_name", ["medium-mpa-0", "small", "medium", "large"])
def test_real_mace_periodic_stress_parity(model_name):
    """Periodic MACE stress reproduces the raw ASE stress exactly
    (eV/Å³ → Ha/bohr³ via ·Bohr³/Hartree) and is symmetric."""
    import numpy as np
    from ase.build import bulk
    from ase.units import Bohr, Hartree

    from mace.calculators import mace_mp

    from vibeqc.ase_periodic import atoms_to_periodic_system
    from vibeqc.mlip.mace import run_periodic_mace

    si = bulk("Si", "diamond", a=5.40)  # off-equilibrium → non-zero stress
    raw = mace_mp(model=model_name, device="cpu", default_dtype="float64")
    a = si.copy()
    a.calc = raw
    s_ref = a.get_stress(voigt=False) * Bohr**3 / Hartree
    s = run_periodic_mace(
        atoms_to_periodic_system(si), MLIPOptions(model=model_name),
    ).stress()
    assert np.abs(s - s_ref).max() < 1e-15
    assert np.abs(s - s.T).max() < 1e-15  # symmetric


@pytest.mark.parametrize("derivative", ["positions", "strain"])
@pytest.mark.parametrize("model_name", ["medium-mpa-0", "small", "medium", "large"])
def test_real_mace_periodic_derivatives_follow_energy(derivative, model_name):
    """Differentiate the public energy in bohr and Cartesian cell strain.

    A distorted nonorthogonal cell exercises force signs, lattice orientation,
    shear components, and the stress volume conversion. Two central-difference
    steps estimate truncation error; the other allowance is floating-point
    cancellation in the energies, not a fitted model-accuracy tolerance.
    """
    import numpy as np
    from ase.build import bulk
    from ase.units import Bohr

    from vibeqc.ase_periodic import atoms_to_periodic_system
    from vibeqc.mlip.mace import PeriodicMACEEvaluator

    atoms = bulk("Si", "diamond", a=5.40)
    atoms.positions[1] += [0.07, -0.04, 0.025]
    evaluator = PeriodicMACEEvaluator(
        MLIPOptions(model=model_name, device="cpu", dtype="float64"),
    )
    base = evaluator.run(atoms_to_periodic_system(atoms))
    if derivative == "positions":
        components = [(i, j) for i in range(len(atoms)) for j in range(3)]
        expected = base.gradient()
    else:
        components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        expected = base.stress() * atoms.get_volume() / Bohr**3

    for i, j in components:
        estimates, scales = [], []
        for step in (2e-4, 1e-4):
            energies = []
            for sign in (-1, 1):
                displaced = atoms.copy()
                if derivative == "positions":
                    displaced.positions[i, j] += sign * step * Bohr
                else:
                    transform = np.eye(3)
                    transform[i, j] += sign * step
                    displaced.set_cell(atoms.cell.array @ transform.T, scale_atoms=True)
                result = evaluator.run(atoms_to_periodic_system(displaced))
                assert result.converged
                energies.append(result.energy)
            estimates.append((energies[1] - energies[0]) / (2 * step))
            scales.append(max(1., *map(abs, energies)))
        truncation = 2 * abs(estimates[1] - estimates[0])
        cancellation = 100 * np.finfo(float).eps * max(scales) / 1e-4
        assert abs(expected[i, j] - estimates[1]) <= truncation + cancellation, (
            derivative, i, j, expected[i, j], estimates, truncation, cancellation,
        )


def test_real_mace_wire_periodic_translation():
    """A one-axis image shift preserves the energy and forces of a wire."""
    import numpy as np
    from ase.units import Bohr

    from vibeqc import PeriodicSystem
    from vibeqc.mlip.mace import PeriodicMACEEvaluator

    lattice = np.diag([4.4, 20., 20.]) / Bohr
    positions = np.array([[0., 0., 0.], [2.2, .15, 0.]]) / Bohr

    def system():
        return PeriodicSystem(
            1, lattice, [Atom(14, list(position)) for position in positions],
            charge=0, multiplicity=1,
        )

    evaluator = PeriodicMACEEvaluator()
    before = evaluator.run(system())
    positions[1] += lattice[:, 0]
    after = evaluator.run(system())
    assert before.dim == after.dim == 1
    assert before.pbc == after.pbc == (True, False, False)
    assert after.energy == pytest.approx(before.energy, abs=1e-12, rel=0)
    np.testing.assert_allclose(after.gradient(), before.gradient(), atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="only for dim=3"):
        after.stress()


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_real_mace_neb_image_matches_periodic_model(dim):
    """A NEB image must use the direct driver's lattice, PBC, and units."""
    import numpy as np
    from ase.units import Bohr

    from vibeqc.mlip.mace import MACEModel
    from vibeqc.neb import _evaluate_image_mace

    lattice = np.array([[4.5, 1.2, .5], [0., 5., 1.1], [0., 0., 5.5]]) / Bohr
    positions = np.array([atom.xyz for atom in _H2O.atoms])
    numbers = np.array([atom.Z for atom in _H2O.atoms])
    pbc = tuple(axis < dim for axis in range(3))
    model = MACEModel(_H2O, cell=lattice, pbc=pbc)
    expected_energy, expected_gradient = model.energy(), model.gradient()
    energy, gradient, state = _evaluate_image_mace(
        positions, calc=model.calculator, numbers=numbers, cell=lattice, pbc=pbc,
    )
    assert state is None
    assert energy == pytest.approx(expected_energy, abs=1e-12, rel=0)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12, rtol=0)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("model_name", ["medium-mpa-0", "small", "medium", "large"])
def test_real_mace_mps_float32_matches_cpu(periodic, model_name):
    """Exercise the advertised device on real weights, including 3D stress.

    MACE 0.3.16's hardcoded double node-energy cast is an active failure here,
    not a skip. Upstream commit 22f0809 corrects that independent dependency bug.
    """
    import numpy as np
    import torch
    from ase.units import Bohr, Hartree
    from vibeqc.mlip.mace import MACEModel

    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device available")
    cell = (np.array([[4.5, 1.2, .5], [0., 5., 1.1], [0., 0., 5.5]]) / Bohr
            if periodic else None)
    cpu = MACEModel(
        _H2O, MLIPOptions(model=model_name, device="cpu", dtype="float32"), cell=cell,
    )
    mps = MACEModel(
        _H2O, MLIPOptions(model=model_name, device="mps", dtype="float32"), cell=cell,
    )
    assert str(mps.calculator.device).split(":", 1)[0] == "mps"
    for model in mps.calculator.models:
        assert next(model.parameters()).device.type == "mps"
        assert next(model.parameters()).dtype == torch.float32
    # Absolute fp32 floor from upstream tests/golden/harness.py at f443730.
    # Apply it in MACE's native units, with no additional relative allowance.
    # This is a backend-parity check, not a force-field accuracy tolerance.
    native_atol = 5e-5
    assert mps.energy() == pytest.approx(cpu.energy(), abs=native_atol / Hartree, rel=0)
    np.testing.assert_allclose(mps.gradient(), cpu.gradient(),
                               atol=native_atol * Bohr / Hartree, rtol=0)
    if periodic:
        np.testing.assert_allclose(mps.stress(), cpu.stress(),
                                   atol=native_atol * Bohr**3 / Hartree, rtol=0)
