"""Real end-to-end MACE runs through ``run_job`` — the M5 validation.

Unlike ``test_mlip_mace.py`` (pure-Python registry/gate logic), these
actually download a foundation model and run a MACE forward pass, so they
require the ``[mace]`` extra (Python <= 3.13). The module
``pytest.importorskip``s ``mace``, so it is **skipped automatically** on
environments without the extra (e.g. the 3.14 dev venv / the collect-only
CI), and runs only on a <= 3.13 MACE environment (the M5 lane / compute-reference /
a local 3.13 build). Marked ``slow`` because first use downloads weights.
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


def test_real_mace_geometry_optimization(tmp_path):
    """method="mace" + optimize=True relaxes a distorted geometry via ASE
    BFGS on the MACE calculator, writes a .traj, and lowers the energy
    relative to the distorted single point."""
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
    opt = run_job(
        distorted, method="mace", optimize=True, fmax=0.05,
        output=str(tmp_path / "opt"),
    )
    assert (tmp_path / "opt.traj").exists()
    assert math.isfinite(float(opt.energy))
    assert float(opt.energy) <= float(sp.energy) + 1e-6


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


def test_real_mace_wrapper_parity_with_raw_calculator():
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
    raw = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64")
    a = Atoms(numbers=[8, 1, 1], positions=pos_A)
    a.calc = raw
    E_eV, F_eVA = a.get_potential_energy(), a.get_forces()

    mol = Molecule(
        [Atom(8, list(pos_A[0] / Bohr)), Atom(1, list(pos_A[1] / Bohr)),
         Atom(1, list(pos_A[2] / Bohr))], 0, 1,
    )
    m = MACEModel(mol, MLIPOptions(model="medium-mpa-0"))
    assert abs(m.energy() - E_eV / Hartree) < 1e-9             # eV→Ha exact
    g_ref = -np.asarray(F_eVA) * Bohr / Hartree                # gradient = −F
    assert np.abs(m.gradient() - g_ref).max() < 1e-12          # Ha/bohr exact


def test_real_mace_periodic_stress_parity():
    """Periodic MACE stress reproduces the raw ASE stress exactly
    (eV/Å³ → Ha/bohr³ via ·Bohr³/Hartree) and is symmetric."""
    import numpy as np
    from ase.build import bulk
    from ase.units import Bohr, Hartree

    from mace.calculators import mace_mp

    from vibeqc.ase_periodic import atoms_to_periodic_system
    from vibeqc.mlip.mace import run_periodic_mace

    si = bulk("Si", "diamond", a=5.40)  # off-equilibrium → non-zero stress
    raw = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64")
    a = si.copy()
    a.calc = raw
    s_ref = a.get_stress(voigt=False) * Bohr**3 / Hartree
    s = run_periodic_mace(atoms_to_periodic_system(si)).stress()
    assert np.abs(s - s_ref).max() < 1e-15
    assert np.abs(s - s.T).max() < 1e-15  # symmetric
