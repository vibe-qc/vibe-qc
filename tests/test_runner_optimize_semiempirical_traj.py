"""Semiempirical optimization trajectories carry per-frame energies.

Pre-2026-06-12, ``_SemiempiricalCalculator.calculate()`` never invoked
the ASE base ``Calculator.calculate``, so ``self.atoms`` was never
recorded and ``check_state()`` reported every property query as
changed: cached results were discarded (each ``get_potential_energy``
/ ``get_forces`` re-ran the semiempirical solve) and ASE's
TrajectoryWriter found no current results to store — ``run_job(
optimize=True)`` trajectories for semiempirical methods lost their
per-frame energies (the QVF reader then logs
``trajectory_frame_energy`` compatibility-fallback warnings).

The sibling ``_WavefunctionCalculator`` got the same one-line fix the
same day (see test_runner_optimize_wavefunction_options.py); this file
pins the semiempirical mirror.
"""

from __future__ import annotations

import json
import tomllib

import pytest

pytest.importorskip("ase")

import numpy as np
import vibeqc.runner as runner
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job
from vibeqc.semiempirical import SCCDFTBModel
from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR

# H2 slightly stretched from the MSINDO equilibrium so BFGS takes real
# steps. MSINDO is the fast-in-CI semiempirical choice: H2 converges in
# <= 5 SCF iterations (tests/test_msindo.py).
H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.6])])

_ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR
TETRAZINE_MSINDO = Molecule(
    [
        Atom(7, [0.0, 2.26767119, 0.0]),
        Atom(7, [2.26767119, 0.0, 0.0]),
        Atom(7, [0.0, -2.26767119, 0.0]),
        Atom(7, [-2.26767119, 0.0, 0.0]),
        Atom(6, [2.26767119, 2.26767119, 0.0]),
        Atom(6, [-2.26767119, 2.26767119, 0.0]),
        Atom(1, [3.91173280, 3.91173280, 0.0]),
        Atom(1, [-3.91173280, 3.91173280, 0.0]),
    ]
)
HCOOH_PM6_ROOT_SWITCH = Molecule(
    [
        Atom(6, [-0.039448903694381504, 0.5815022553595, 0.15977941076165111]),
        Atom(8, [2.40181293489087, 0.3121635953469203, -0.04673295020687813]),
        Atom(8, [-1.329291023520098, 3.032934217009186, 0.27949508748752705]),
        Atom(1, [-1.145066393650231, -0.9523747768866971, -0.09205431187126446]),
        Atom(1, [-0.2711867554426138, 4.903387502993281, -0.30048804313188576]),
    ]
)
S22_FORMAMIDE_DIMER_PM6 = Molecule(
    [
        Atom(6, [-3.38339943, 0.67478903, -0.04887209]),
        Atom(8, [-2.99822792, -1.47461933, -0.34127318]),
        Atom(7, [-5.66741107, 1.48512243, 0.42385042]),
        Atom(1, [-6.33230360, 3.23371234, 0.65850715]),
        Atom(1, [-7.06770937, 0.21439697, 0.50455684]),
        Atom(1, [-1.99098696, 2.05677777, -0.07313240]),
        Atom(6, [3.38339943, -0.67478903, 0.04887209]),
        Atom(8, [2.99822792, 1.47461933, 0.34127318]),
        Atom(7, [5.66741107, -1.48512243, -0.42385042]),
        Atom(1, [6.33230360, -3.23371234, -0.65850715]),
        Atom(1, [7.06770937, -0.21439697, -0.50455684]),
        Atom(1, [1.99098696, -2.05677777, 0.07313240]),
    ]
)
CH3OH_SCC_DFTB_PAPER_CORE = Molecule(
    [
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(8, [0.0, 0.0, 2.68945803]),
        Atom(1, [1.94660674, 0.0, -0.73302471]),
        Atom(1, [-0.97339786, 1.68601353, -0.73302471]),
        Atom(1, [-0.97339786, -1.68601353, -0.73302471]),
        Atom(1, [1.67410825, 0.0, 3.34122452]),
    ]
)
CO2_SCC_DFTB_PAPER_CORE = Molecule(
    [
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(8, [0.0, 0.0, 2.19605057]),
        Atom(8, [0.0, 0.0, -2.19605057]),
    ]
)
HCOOH_SCC_DFTB_PAPER_CORE = Molecule(
    [
        Atom(6, [0.00000000, 0.77686635, 0.00000000]),
        Atom(8, [2.28789125, 0.69541916, 0.00000000]),
        Atom(8, [-1.24986477, 2.94835049, 0.00000000]),
        Atom(1, [-1.17068525, -0.93919382, 0.00000000]),
        Atom(1, [-0.25057767, 4.39606957, 0.00000000]),
    ]
)
HCOOH_SCC_DFTB_LINE_SEARCH_TRIAL = Molecule(
    [
        Atom(6, [0.057045672117, 1.007914686030, 0.000000000000]),
        Atom(8, [1.888638465087, 0.595907462839, 0.000000000000]),
        Atom(8, [-0.911608547622, 2.635713862070, 0.000000000000]),
        Atom(1, [-1.013374733497, -0.622592166823, 0.000000000000]),
        Atom(1, [-0.403937296085, 4.260567905883, 0.000000000000]),
    ]
)
H2CO = Molecule(
    [
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(8, [0.0, 0.0, 1.2086 * _ANGSTROM_TO_BOHR]),
        Atom(
            1,
            [
                0.9436 * _ANGSTROM_TO_BOHR,
                0.0,
                -0.5874 * _ANGSTROM_TO_BOHR,
            ],
        ),
        Atom(
            1,
            [
                -0.9436 * _ANGSTROM_TO_BOHR,
                0.0,
                -0.5874 * _ANGSTROM_TO_BOHR,
            ],
        ),
    ]
)


def _molecule_from_angstrom(rows):
    return Molecule(
        [
            Atom(z, [value * _ANGSTROM_TO_BOHR for value in xyz])
            for z, xyz in rows
        ]
    )


ACETAMIDE = _molecule_from_angstrom(
    [
        (8, (0.424546, 1.327024, 0.008034)),
        (6, (0.077158, 0.149789, -0.004249)),
        (7, (0.985518, -0.878537, -0.048910)),
        (6, (-1.371475, -0.288665, -0.000144)),
        (1, (0.707952, -1.824249, 0.169942)),
        (1, (-1.997229, 0.584922, -0.175477)),
        (1, (-1.560842, -1.039270, -0.771686)),
        (1, (-1.632113, -0.723007, 0.969814)),
        (1, (1.953133, -0.631574, 0.111866)),
    ]
)
ACETONE_THIEL_SCC_DFTB = Molecule(
    [
        Atom(6, [0.00000000, 0.00000000, 0.35148903]),
        Atom(8, [0.00000000, 0.00000000, 2.64561638]),
        Atom(6, [2.43207735, 0.00000000, -1.16974039]),
        Atom(6, [-2.43207735, 0.00000000, -1.16974039]),
        Atom(1, [2.48121022, 1.66862805, -2.39617255]),
        Atom(1, [2.48121022, -1.66862805, -2.39617255]),
        Atom(1, [4.08558759, 0.00000000, 0.06803014]),
        Atom(1, [-2.48121022, 1.66862805, -2.39617255]),
        Atom(1, [-2.48121022, -1.66862805, -2.39617255]),
        Atom(1, [-4.08558759, 0.00000000, 0.06803014]),
    ]
)
CH2O_THIEL_SCC_DFTB = Molecule(
    [
        Atom(6, [0.00000000, 0.00000000, 0.00000000]),
        Atom(8, [0.00000000, 0.00000000, 2.27711982]),
        Atom(1, [0.00000000, 1.77823216, -1.10171025]),
        Atom(1, [0.00000000, -1.77823216, -1.10171025]),
    ]
)
FURAN_THIEL_SCC_DFTB = Molecule(
    [
        Atom(8, [0.00000000, 0.00000000, 2.19775132]),
        Atom(6, [2.07869859, 0.00000000, 0.67085273]),
        Atom(6, [-2.07869859, 0.00000000, 0.67085273]),
        Atom(6, [1.34737463, 0.00000000, -1.78012188]),
        Atom(6, [-1.34737463, 0.00000000, -1.78012188]),
        Atom(1, [3.92307115, 0.00000000, 1.54768558]),
        Atom(1, [-3.92307115, 0.00000000, 1.54768558]),
        Atom(1, [2.63994721, 0.00000000, -3.37883007]),
        Atom(1, [-2.63994721, 0.00000000, -3.37883007]),
    ]
)
PYRIMIDINE_THIEL_SCC_DFTB = Molecule(
    [
        Atom(7, [0.00000000, 2.25822256, 0.00000000]),
        Atom(6, [2.20342050, 0.92974519, 0.00000000]),
        Atom(6, [-2.20342050, 0.92974519, 0.00000000]),
        Atom(7, [0.00000000, -2.25822256, 0.00000000]),
        Atom(6, [2.25822256, -1.69130476, 0.00000000]),
        Atom(6, [-2.25822256, -1.69130476, 0.00000000]),
        Atom(1, [3.97598348, 1.95208695, 0.00000000]),
        Atom(1, [-3.97598348, 1.95208695, 0.00000000]),
        Atom(1, [3.96086567, -2.72120542, 0.00000000]),
        Atom(1, [-3.96086567, -2.72120542, 0.00000000]),
    ]
)
PYRIDINE_THIEL_SCC_DFTB = Molecule(
    [
        Atom(7, [0.00000000, 2.63049858, 0.00000000]),
        Atom(6, [2.26011228, 1.31713901, 0.00000000]),
        Atom(6, [-2.26011228, 1.31713901, 0.00000000]),
        Atom(6, [2.26956091, -1.31902874, 0.00000000]),
        Atom(6, [-2.26956091, -1.31902874, 0.00000000]),
        Atom(6, [0.00000000, -2.62671912, 0.00000000]),
        Atom(1, [4.05346225, 2.33192187, 0.00000000]),
        Atom(1, [-4.05346225, 2.33192187, 0.00000000]),
        Atom(1, [4.08180814, -2.31113488, 0.00000000]),
        Atom(1, [-4.08180814, -2.31113488, 0.00000000]),
        Atom(1, [0.00000000, -4.68652045, 0.00000000]),
    ]
)
TRIAZINE_THIEL_SCC_DFTB = Molecule(
    [
        Atom(7, [0.00000000, 2.49632803, 0.00000000]),
        Atom(6, [2.16184653, 1.24721915, 0.00000000]),
        Atom(6, [-2.16184653, 1.24721915, 0.00000000]),
        Atom(7, [2.16184653, -1.24721915, 0.00000000]),
        Atom(7, [-2.16184653, -1.24721915, 0.00000000]),
        Atom(6, [0.00000000, -2.49632803, 0.00000000]),
        Atom(1, [3.94574786, 2.25822256, 0.00000000]),
        Atom(1, [-3.94574786, 2.25822256, 0.00000000]),
        Atom(1, [0.00000000, -4.54479100, 0.00000000]),
    ]
)
IMIDAZOLE_THIEL_SCC_DFTB = Molecule(
    [
        Atom(7, [0.00000000, 2.13350064, 0.00000000]),
        Atom(6, [2.04657325, 0.62549930, 0.00000000]),
        Atom(6, [-2.04657325, 0.62549930, 0.00000000]),
        Atom(7, [1.29446230, -1.81224722, 0.00000000]),
        Atom(6, [-1.29446230, -1.81224722, 0.00000000]),
        Atom(1, [0.00000000, 4.03834444, 0.00000000]),
        Atom(1, [3.96842458, 1.29824175, 0.00000000]),
        Atom(1, [-3.96842458, 1.29824175, 0.00000000]),
        Atom(1, [-2.15428763, -3.56213349, 0.00000000]),
    ]
)
CYTOSINE_THIEL_SCC_DFTB = Molecule(
    [
        Atom(7, [0.00000000, 2.32058351, 0.00000000]),
        Atom(6, [2.05035270, 0.90517875, 0.00000000]),
        Atom(6, [-2.05035270, 0.90517875, 0.00000000]),
        Atom(7, [1.30958011, -1.58170065, 0.00000000]),
        Atom(6, [-1.30958011, -1.58170065, 0.00000000]),
        Atom(6, [2.22231776, -4.03645471, 0.00000000]),
        Atom(8, [4.20275060, -5.31013003, 0.00000000]),
        Atom(7, [-0.06803014, -5.60303756, 0.00000000]),
        Atom(1, [3.92874033, 1.72154038, 0.00000000]),
        Atom(1, [-0.03968425, 4.21408895, 0.00000000]),
        Atom(1, [-2.86671432, -2.40940064, 0.00000000]),
        Atom(1, [-0.24188493, -7.47953546, 0.00000000]),
        Atom(1, [1.49666298, -7.54567587, 0.00000000]),
    ]
)


def test_pm6_fd_gradient_returns_finite_light_control():
    result = runner._run_semiempirical("pm6", H2)

    gradient = np.asarray(result.gradient(), dtype=float)

    assert result.converged
    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))


def test_msindo_runner_recovers_exact_tetrazine_cycle():
    """BUG67: native and job routes must reach the strict stationary root."""
    from vibeqc.semiempirical.methods.msindo import _cpp_msindo_kernel

    run_msindo_full, _run_msindo_uhf, params = _cpp_msindo_kernel()
    z = [atom.Z for atom in TETRAZINE_MSINDO.atoms]
    xyz_angstrom = [
        [coordinate / _ANGSTROM_TO_BOHR for coordinate in atom.xyz]
        for atom in TETRAZINE_MSINDO.atoms
    ]

    native = run_msindo_full(z, xyz_angstrom, params, max_iter=200)
    result = runner._run_semiempirical("msindo", TETRAZINE_MSINDO)

    assert native.converged
    assert native.n_iter == 270
    assert result.converged
    assert result.n_iter == 270
    assert result.energy == pytest.approx(-50.737177619704, abs=1.0e-10)


def test_pm6_formamide_dimer_reaches_hard_occupation_root_without_cooling():
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

    params = load_pm6_params_auto(
        [atom.Z for atom in S22_FORMAMIDE_DIMER_PM6.atoms]
    )
    cold = run_pm6(S22_FORMAMIDE_DIMER_PM6, params, max_iter=200)
    result = runner._run_semiempirical("pm6", S22_FORMAMIDE_DIMER_PM6)

    assert cold.converged
    assert result.converged
    # #154: the route now always probes for a second zero-temperature basin,
    # so its iteration count carries that cost even when, as here, the probe
    # agrees with the direct SCF and nothing is refused.  What must still
    # match the direct run is the ROOT it reports: the zero temperature, the
    # energy to 1e-12 Ha and the density trace below.
    assert result.n_iter >= cold.n_iter
    assert result.electronic_temperature == 0.0
    assert result.energy == pytest.approx(cold.energy, abs=1.0e-12)
    assert np.trace(np.asarray(result.density)) == pytest.approx(36.0)


def test_pm6_fd_gradient_stays_on_corrected_hard_occupation_root():
    result = runner._run_semiempirical("pm6", HCOOH_PM6_ROOT_SWITCH)

    assert result.converged
    gradient = np.asarray(result.gradient(), dtype=float)

    assert gradient.shape == (5, 3)
    assert np.all(np.isfinite(gradient))
    # Official MOPAC v23.2.5 gives -0.126832104 Ha/bohr.  The residual is
    # below the 1e-5 Ha/bohr cross-code tolerance and includes the bounded
    # finite-difference error of the molecular PM6 gradient implementation.
    assert gradient[0, 0] == pytest.approx(-0.126832104, abs=1.0e-5)
    assert np.max(np.abs(np.sum(gradient, axis=0))) < 5.0e-5


def test_pm6_optimizer_descends_corrected_hard_occupation_root(tmp_path):
    from ase.io.trajectory import Trajectory
    from ase.units import Hartree

    stem = tmp_path / "pm6" / "hcooh-root-switch"

    result = run_job(
        HCOOH_PM6_ROOT_SWITCH,
        method="pm6",
        optimize=True,
        fmax=0.01,
        max_opt_steps=500,
        output=stem,
        output_qvf=False,
        write_molden_file=False,
        write_population_file=False,
        verbose=0,
    )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    trajectory = Trajectory(stem.with_suffix(".traj"))
    energies = [
        float(atoms.get_potential_energy()) / Hartree for atoms in trajectory
    ]

    assert result.converged
    assert manifest["outputs"]["status"] == "complete"
    assert len(energies) > 1
    assert all(
        right <= left + 1.0e-10
        for left, right in zip(energies, energies[1:])
    )
    assert energies[-1] < energies[0] - 0.03
    assert result.energy == pytest.approx(energies[-1], abs=1.0e-10)
    assert result.energy < -24.50


def test_scc_dftb_runner_converges_paper_core_ch3oh_with_native_relaxation():
    """The bounded native mixer converges CH3OH without the runner safety budget."""
    direct = SCCDFTBModel(CH3OH_SCC_DFTB_PAPER_CORE)
    direct_energy = direct.energy()
    assert direct.converged
    assert direct.n_iter < 100

    result = runner._run_semiempirical("scc_dftb", CH3OH_SCC_DFTB_PAPER_CORE)

    assert result.converged
    assert result.n_iter == direct.n_iter
    assert result.energy == pytest.approx(direct_energy, abs=1e-12)
    gradient = np.asarray(result.gradient(), dtype=float)
    assert gradient.shape == (6, 3)
    assert np.all(np.isfinite(gradient))


def test_scc_dftb_paper_core_ch3oh_optimizer_converges(tmp_path):
    """The public optimizer keeps CH3OH on its converged SCC branch."""
    from ase.io.trajectory import Trajectory

    stem = tmp_path / "scc_dftb" / "ch3oh"

    result = run_job(
        CH3OH_SCC_DFTB_PAPER_CORE,
        method="scc_dftb",
        optimize=True,
        fmax=0.01,
        max_opt_steps=80,
        output=stem,
        output_qvf=False,
        write_molden_file=False,
        write_population_file=False,
        verbose=0,
    )

    trajectory = Trajectory(stem.with_suffix(".traj"))
    assert result.converged
    assert len(trajectory) >= 2
    assert np.all(np.isfinite(trajectory[-1].get_positions()))


def test_scc_dftb_paper_core_co2_optimizer_uses_bounded_steps(tmp_path):
    """The SCC-DFTB optimizer should not jump off its converged branch for CO2."""
    from ase.io.trajectory import Trajectory

    stem = tmp_path / "scc_dftb" / "co2"

    result = run_job(
        CO2_SCC_DFTB_PAPER_CORE,
        method="scc_dftb",
        optimize=True,
        fmax=0.01,
        max_opt_steps=80,
        output=stem,
        output_qvf=False,
        write_molden_file=False,
        write_population_file=False,
        verbose=0,
    )

    assert result.converged
    assert len(Trajectory(stem.with_suffix(".traj"))) >= 2


def test_scc_dftb_model_reports_h2co_nonconvergence():
    """ArtVal25 H2CO has no converged integer-occupation SCC solution."""
    model = SCCDFTBModel(H2CO)

    energy = model.energy()

    assert np.isfinite(energy)
    assert not model.converged
    assert model.n_iter == 100


def test_thiel_acetamide_scc_dftb_uses_finite_temperature_retry():
    result = runner._run_semiempirical("scc_dftb", ACETAMIDE)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)
    gradient = np.asarray(result.gradient())
    assert np.all(np.isfinite(gradient))

    step = 1.0e-4
    displaced_energies = []
    for displacement in (-step, step):
        atoms = []
        for index, atom in enumerate(ACETAMIDE.atoms):
            xyz = list(atom.xyz)
            if index == 0:
                xyz[0] += displacement
            atoms.append(Atom(atom.Z, xyz))
        model = SCCDFTBModel(
            Molecule(atoms),
            max_iter=500,
            electronic_temperature=0.001,
        )
        displaced_energies.append(model.energy())
        assert model.converged
    finite_difference = (
        displaced_energies[1] - displaced_energies[0]
    ) / (2.0 * step)
    assert gradient[0, 0] == pytest.approx(finite_difference, abs=2.0e-4)


def test_thiel_acetone_scc_dftb_uses_finite_temperature_retry():
    """BUG67: acetone needs the public Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        ACETONE_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", ACETONE_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-8.7685882829, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_ch2o_scc_dftb_uses_finite_temperature_retry():
    """BUG67: CH2O needs the public Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        CH2O_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", CH2O_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-5.0276818817, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_furan_scc_dftb_uses_finite_temperature_retry():
    """BUG67: furan needs the public Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        FURAN_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", FURAN_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-9.7519736134, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_pyrimidine_scc_dftb_uses_finite_temperature_retry():
    """BUG67: pyrimidine needs the Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        PYRIMIDINE_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", PYRIMIDINE_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-11.1331880662, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_pyridine_scc_dftb_uses_finite_temperature_retry():
    """BUG67: pyridine needs the Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        PYRIDINE_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", PYRIDINE_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-10.4898128335, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_triazine_scc_dftb_uses_finite_temperature_retry():
    """BUG67: triazine needs the Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        TRIAZINE_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", TRIAZINE_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-11.7678399398, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_imidazole_scc_dftb_uses_finite_temperature_retry():
    """BUG67: imidazole needs the Mermin rung, not a larger zero-T cap."""
    zero_temperature = SCCDFTBModel(
        IMIDAZOLE_THIEL_SCC_DFTB,
        max_iter=500,
    )
    zero_temperature.energy()
    assert not zero_temperature.converged

    result = runner._run_semiempirical("scc_dftb", IMIDAZOLE_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 500
    assert result.electronic_temperature == pytest.approx(0.001)
    assert float(result.energy) == pytest.approx(-9.7218320822, abs=5.0e-7)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)


def test_thiel_cytosine_scc_dftb_uses_second_temperature_retry():
    first_retry = SCCDFTBModel(
        CYTOSINE_THIEL_SCC_DFTB,
        max_iter=500,
        electronic_temperature=0.001,
    )
    first_retry.energy()
    assert not first_retry.converged

    result = runner._run_semiempirical("scc_dftb", CYTOSINE_THIEL_SCC_DFTB)

    assert result.converged
    assert result.n_iter > 1000
    assert result.electronic_temperature == pytest.approx(0.0012)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)
    gradient = np.asarray(result.gradient(), dtype=float)
    assert gradient.shape == (13, 3)
    assert np.all(np.isfinite(gradient))


def test_scc_dftb_hcooh_trial_prefers_zero_temperature_relaxation():
    # The bounded Mermin fallback remains available for this frame when called
    # directly, but the public route should now solve the zero-temperature
    # charge fixed point before reaching that retry.
    #
    # 2026-08-12: the ladder rungs were re-pinned to the verified recipe
    # (no DIIS, charge_mixing=0.05; see
    # handovers/HANDOVER_SCC_DFTB_CONVERGENCE.md).  At the historical 0.2
    # mixing the T=0.0012 Ha rung stalls past the 500-iteration budget (619
    # needed); at 0.05 it converges in 332.  The converged fixed point is
    # unchanged -- only the path length differs.
    prior_retry = SCCDFTBModel(
        HCOOH_SCC_DFTB_LINE_SEARCH_TRIAL,
        max_iter=500,
        charge_mixing=0.05,
        electronic_temperature=0.0012,
        use_diis=False,
    )
    prior_retry.energy()
    assert prior_retry.converged

    final_retry = SCCDFTBModel(
        HCOOH_SCC_DFTB_LINE_SEARCH_TRIAL,
        max_iter=500,
        charge_mixing=0.05,
        electronic_temperature=0.0015,
        use_diis=False,
    )
    final_retry.energy()
    assert final_retry.converged

    result = runner._run_semiempirical(
        "scc_dftb",
        HCOOH_SCC_DFTB_LINE_SEARCH_TRIAL,
    )

    assert result.converged
    assert result.n_iter == 163
    assert result.electronic_temperature == pytest.approx(0.0)
    # 2026-08-06: energy pin updated — the SCC path shifted slightly
    # (DIIS subspace default changed to 2, giving a more contractive
    # Aitken-only loop) but the converged root is the same.
    assert result.energy == pytest.approx(-8.2942879453, abs=1.0e-10)
    assert sum(result.mulliken_charges) == pytest.approx(0.0, abs=1.0e-10)
    gradient = np.asarray(result.gradient(), dtype=float)
    assert gradient.shape == (5, 3)
    assert np.all(np.isfinite(gradient))


def test_artval25_h2co_scc_dftb_optimizer_uses_smoothed_occupations(tmp_path):
    """A finite-temperature retry supplies a continuous SCC surface."""
    stem = tmp_path / "scc_dftb" / "h2co"

    result = run_job(
        H2CO,
        method="scc_dftb",
        optimize=True,
        fmax=0.01,
        max_opt_steps=500,
        output=stem,
        output_qvf=False,
        write_molden_file=False,
        write_population_file=False,
        verbose=0,
    )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert result.converged
    assert manifest["outputs"]["status"] == "complete"
    assert stem.with_suffix(".traj").is_file()


def test_scc_dftb_hcooh_optimizer_uses_low_mixing_smoothed_occupations(tmp_path):
    """The HCOOH optimizer can step through slow SCC charge tails."""
    from ase.io.trajectory import Trajectory

    stem = tmp_path / "scc_dftb" / "hcooh"

    result = run_job(
        HCOOH_SCC_DFTB_PAPER_CORE,
        method="scc_dftb",
        optimize=True,
        fmax=0.01,
        max_opt_steps=80,
        output=stem,
        output_qvf=False,
        write_molden_file=False,
        write_population_file=False,
        verbose=0,
    )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert result.converged
    assert np.isfinite(result.energy)
    assert manifest["outputs"]["status"] == "complete"
    assert len(Trajectory(stem.with_suffix(".traj"))) > 5


@pytest.mark.parametrize("method", ["msindo", "scc_dftb"])
def test_capped_ase_optimization_fails_closed(
    monkeypatch,
    tmp_path,
    method,
):
    """BUG 44: 501-frame capped H2CO runs must not finalize complete."""
    import ase.optimize
    from ase.io.trajectory import Trajectory

    class _Result:
        energy = -1.0

        def gradient(self):
            # 0.001 Ha/bohr = 0.0514 eV/A, above the retained input's
            # 0.01 eV/A target.
            return np.full((4, 3), 0.001)

    def _fake_run_semiempirical(method_key, molecule, **kwargs):
        return _Result()

    class _CappedOptimizer:
        """Deterministic ASE stand-in that consumes every allowed step."""

        def __init__(self, atoms, logfile=None, **kwargs):
            self.atoms = atoms
            self.nsteps = 0
            self._observers = []

        def attach(self, function, *args, **kwargs):
            if hasattr(function, "write"):
                function = function.write
            self._observers.append((function, args, kwargs))

        def _notify(self):
            self.atoms.get_potential_energy()
            self.atoms.get_forces()
            for function, args, kwargs in self._observers:
                function(*args, **kwargs)

        def run(self, *, fmax, steps):
            self._notify()  # initial frame
            for _ in range(steps):
                self.atoms.positions[0, 0] += 1.0e-4
                self.nsteps += 1
                self._notify()
            return False

    monkeypatch.setattr(runner, "_run_semiempirical", _fake_run_semiempirical)
    monkeypatch.setattr(ase.optimize, "BFGSLineSearch", _CappedOptimizer)

    stem = tmp_path / method / "h2co"
    with pytest.raises(
        runner._ASEGeometryOptimizationError,
        match="did not converge after 500 of 500 allowed steps",
    ):
        run_job(
            H2CO,
            method=method,
            optimize=True,
            fmax=0.01,
            max_opt_steps=500,
            output=stem,
            structured_log=True,
            output_qvf=False,
            write_molden_file=False,
            write_population_file=False,
            verbose=0,
        )

    assert len(Trajectory(stem.with_suffix(".traj"))) == 501
    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert manifest["outputs"]["status"] == "crashed"
    assert stem.with_suffix(".xyz").is_file()

    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
        if line
    ]
    failure = next(
        row
        for row in records
        if row["event"] == "geometry_optimization_nonconverged"
    )
    assert failure["converged"] is False
    assert failure["n_steps"] == 500
    assert failure["max_steps"] == 500
    assert failure["final_max_force_ev_per_angstrom"] > 0.01
    assert failure["target_max_force_ev_per_angstrom"] == pytest.approx(0.01)
    assert failure["final_max_displacement_angstrom"] > 0.0

    text = stem.with_suffix(".out").read_text()
    assert "FATAL: ASE BFGS geometry optimization did not converge" in text
    assert "Optimized geometry" not in text
    assert "converged in" not in text


def test_semiempirical_optimization_frames_carry_energies(tmp_path):
    """run_job(method='msindo', optimize=True): every .traj frame stores
    the potential energy of that frame's geometry."""
    from ase.io.trajectory import Trajectory
    from ase.units import Bohr, Hartree

    res = run_job(
        H2,
        method="msindo",
        optimize=True,
        max_opt_steps=30,
        output=tmp_path / "h2-msindo-opt",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        verbose=0,
    )
    assert res.converged

    frames = list(Trajectory(str(tmp_path / "h2-msindo-opt.traj")))
    assert frames, "optimizer wrote no trajectory frames"

    for atoms in frames:
        # Pre-fix every frame raised here: TrajectoryWriter stored no
        # energy because check_state() reported the results as stale.
        e_traj = float(atoms.get_potential_energy()) / Hartree
        mol = Molecule(
            [
                Atom(int(z), list(xyz))
                for z, xyz in zip(atoms.numbers, atoms.positions / Bohr)
            ]
        )
        e_direct = float(runner._run_semiempirical("msindo", mol).energy)
        assert e_traj == pytest.approx(e_direct, abs=1e-8)


def test_semiempirical_ase_calculator_uses_unified_gradient(monkeypatch):
    """ASE semiempirical optimization should not FD when the runner has forces."""
    from ase import Atoms
    from ase.units import Bohr, Hartree

    gradient = np.array([[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6]])
    calls = []

    class Result:
        energy = -1.25

        def gradient(self):
            calls.append("gradient")
            return gradient

    def fake_run(method, molecule, **kwargs):
        calls.append((method, len(molecule.atoms), kwargs))
        return Result()

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run)
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4 * Bohr]],
    )
    atoms.calc = runner._make_semiempirical_ase_calculator(mol, "gfn2_xtb")

    forces = atoms.get_forces()

    assert calls == [("gfn2_xtb", 2, {}), "gradient"]
    np.testing.assert_allclose(forces, -gradient * (Hartree / Bohr))


def test_semiempirical_ase_calculator_fd_fallback_for_energy_only(monkeypatch):
    """Energy-only semiempirical results keep the legacy finite-difference path."""
    from ase import Atoms
    from ase.units import Bohr, Hartree

    calls = []

    class Result:
        def __init__(self, energy):
            self.energy = energy

        def gradient(self):
            return None

    def fake_run(method, molecule, **kwargs):
        calls.append((method, len(molecule.atoms), kwargs))
        energy = sum(float(atom.xyz[0]) for atom in molecule.atoms)
        return Result(energy)

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run)
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], 0, 2)
    atoms = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]])
    atoms.calc = runner._make_semiempirical_ase_calculator(mol, "msindo")

    forces = atoms.get_forces()

    assert len(calls) == 7
    np.testing.assert_allclose(
        forces,
        [[-1.0 * (Hartree / Bohr), 0.0, 0.0]],
        atol=1e-10,
    )
