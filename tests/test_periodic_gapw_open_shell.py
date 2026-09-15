"""Tests for the open-shell (UHF / UKS / ROHF / ROKS) GPW route.

The closed-shell GPW SCF (:func:`vibeqc.run_periodic_rhf_gpw`) is the
parity reference: running UHF on an even-electron system with
``n_alpha = n_beta = n_elec // 2`` and the symmetric Hcore guess must
reproduce the RHF energy bit-for-bit. UKS with the same input + the
same functional must match RKS. Open-shell cells (e.g. a single H
atom) test the actual spin-broken path.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core


_ROHF_GPW_TEST_CUTOFF_HA = 50.0
_ROKS_GPW_TEST_CUTOFF_HA = 10.0


def _he_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _h2_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def _h_atom_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    sys.multiplicity = 2
    return sys


def _li_atom_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(3, [L / 2, L / 2, L / 2])]
    sys.multiplicity = 2
    return sys


def test_rohf_gpw_electron_counts_include_periodic_charge():
    """The public route uses PeriodicSystem.n_electrons(), including charge."""
    from vibeqc.periodic_gapw_open_shell import infer_alpha_beta_from_system

    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * 16.0
    system.unit_cell = [core.Atom(4, [8.0, 8.0, 8.0])]
    system.charge = 2
    system.multiplicity = 1

    assert system.n_electrons() == 2
    assert infer_alpha_beta_from_system(system) == (1, 1)
    assert infer_alpha_beta_from_system(
        system, n_alpha=1, n_beta=1,
    ) == (1, 1)
    with pytest.raises(ValueError, match="doesn't match"):
        infer_alpha_beta_from_system(system, n_alpha=2, n_beta=2)

    molecule = vq.Molecule(list(system.unit_cell), 2, 1)
    basis = vq.BasisSet(molecule, "sto-3g")
    result = vq.run_periodic_rohf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA, quiet=True,
    )
    assert result.converged
    assert (result.n_alpha, result.n_beta) == (1, 1)
    assert np.sum(result.mo_occupations) == pytest.approx(2.0, abs=1e-12)


# ============================================================
# 0. ROHF GPW restricted-open-shell route
# ============================================================


def test_rohf_gpw_he_matches_rhf_gpw():
    """The closed-shell limit of the ROHF GPW route is RHF/GPW."""
    class _CaptureLogger:
        def __init__(self):
            self.iters = []

        def iteration(self, n, **fields):
            self.iters.append((n, fields))

    system = _he_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    progress = _CaptureLogger()

    rhf = vq.run_periodic_rhf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA, quiet=True,
    )
    rohf = vq.run_periodic_rohf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA, quiet=True,
        progress=progress,
    )

    assert rohf.converged
    assert rohf.energy == pytest.approx(rhf.energy, abs=1e-9)
    assert rohf.energy == pytest.approx(
        rohf.e_electronic + rohf.e_nuclear, abs=1e-12,
    )
    assert rohf.energy == pytest.approx(rohf.breakdown.e_total, abs=1e-12)
    assert rohf.s_squared == pytest.approx(0.0, abs=1e-12)
    assert np.sum(rohf.mo_occupations) == pytest.approx(2.0, abs=1e-12)
    assert [n for n, _ in progress.iters] == list(range(1, rohf.n_iter + 1))
    assert all(fields["energy"] is not None for _, fields in progress.iters)


def test_rohf_gpw_h_atom_matches_uhf_gpw():
    """With no closed shell, one-electron ROHF and UHF are identical."""
    system = _h_atom_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    uhf = vq.run_periodic_uhf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA,
        max_iter=80, quiet=True,
    )
    rohf = vq.run_periodic_rohf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA,
        max_iter=80, quiet=True,
    )

    assert uhf.converged and rohf.converged
    assert rohf.energy == pytest.approx(uhf.energy, abs=1e-9)
    assert rohf.s_squared == pytest.approx(0.75, abs=1e-12)
    assert np.count_nonzero(rohf.mo_occupations == 1.0) == 1


def test_rohf_gpw_li_is_spin_pure_with_restricted_orbitals():
    """Li exercises one closed plus one open orbital on the GPW route."""
    system = _li_atom_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    rohf = vq.run_periodic_rohf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA,
        max_iter=80, quiet=True,
    )
    uhf = vq.run_periodic_uhf_gpw(
        system, basis, cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA,
        max_iter=80, quiet=True,
    )

    assert rohf.converged and uhf.converged
    assert rohf.energy == pytest.approx(uhf.energy, abs=1e-9)
    assert rohf.energy == pytest.approx(rohf.breakdown.e_total, abs=1e-12)
    assert rohf.method == "rohf"
    assert rohf.s_squared == pytest.approx(0.75, abs=1e-12)
    assert np.count_nonzero(rohf.mo_occupations == 2.0) == 1
    assert np.count_nonzero(rohf.mo_occupations == 1.0) == 1
    assert np.sum(rohf.mo_occupations) == pytest.approx(3.0, abs=1e-12)
    assert np.allclose(rohf.mo_coeffs_alpha, rohf.mo_coeffs_beta)
    assert np.allclose(rohf.mo_energies_alpha, rohf.mo_energies_beta)

    from vibeqc.periodic_gapw_j import _overlap_lattice_gamma

    overlap = _overlap_lattice_gamma(basis, system)
    n_alpha = float(np.einsum("ij,ij->", rohf.density_alpha, overlap))
    n_beta = float(np.einsum("ij,ij->", rohf.density_beta, overlap))
    assert n_alpha == pytest.approx(2.0, abs=1e-8)
    assert n_beta == pytest.approx(1.0, abs=1e-8)


# ============================================================
# 0b. ROKS GPW restricted-open-shell route
# ============================================================


def test_roks_gpw_he_matches_rks_gpw():
    """Closed-shell ROKS reduces exactly to RKS on the same GPW grid."""
    system = _he_system(L=12.0)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    rks = vq.run_periodic_rhf_gpw(
        system,
        basis,
        functional="pbe",
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        quiet=True,
    )
    roks = vq.run_periodic_roks_gpw(
        system,
        basis,
        functional="pbe",
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        quiet=True,
    )

    assert roks.converged
    assert roks.energy == pytest.approx(rks.energy, abs=1e-10)
    assert roks.e_xc == pytest.approx(rks.breakdown.e_xc, abs=1e-12)
    assert roks.breakdown.e_total == pytest.approx(roks.energy, abs=1e-12)
    assert roks.s_squared == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("functional", ["pbe", "b3lyp", "r2scan"])
def test_roks_gpw_h_atom_matches_uks_gpw(functional):
    """One-electron ROKS reduces to UKS for GGA, hybrid, and meta-GGA."""
    system = _h_atom_system(L=12.0)
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    uks = vq.run_periodic_uks_gpw(
        system,
        basis,
        functional=functional,
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        quiet=True,
    )
    roks = vq.run_periodic_roks_gpw(
        system,
        basis,
        functional=functional,
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        quiet=True,
    )

    assert roks.converged and uks.converged
    assert roks.energy == pytest.approx(uks.energy, abs=1e-10)
    assert roks.breakdown.e_hf_exchange == pytest.approx(
        uks.breakdown.e_hf_exchange, abs=1e-12
    )
    assert roks.e_xc == pytest.approx(uks.breakdown.e_xc, abs=1e-12)
    assert roks.s_squared == pytest.approx(0.75, abs=1e-12)
    assert tuple(roks.mo_occupations[:1]) == pytest.approx((1.0,))


def test_roks_gpw_li_is_spin_pure_and_energy_components_close():
    """Li exercises the closed-plus-open ROKS orbital and XC accounting."""
    system = _li_atom_system(L=12.0)
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    roks = vq.run_periodic_roks_gpw(
        system,
        basis,
        functional="pbe",
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        max_iter=80,
        quiet=True,
    )

    assert roks.converged
    assert roks.method == "roks"
    assert roks.energy == pytest.approx(-7.287678457000975, abs=1e-9)
    assert roks.energy == pytest.approx(
        roks.e_electronic + roks.e_nuclear, abs=1e-12
    )
    assert roks.energy == pytest.approx(roks.breakdown.e_total, abs=1e-12)
    assert roks.e_xc == pytest.approx(roks.breakdown.e_xc, abs=1e-12)
    assert roks.s_squared == pytest.approx(0.75, abs=1e-12)
    assert tuple(roks.mo_occupations[:2]) == pytest.approx((2.0, 1.0))
    assert np.allclose(roks.mo_coeffs_alpha, roks.mo_coeffs_beta)


@pytest.mark.parametrize(
    ("functional", "message"),
    [
        ("hse06", "range-separated functionals are not supported"),
        ("b2plyp", "would omit its perturbative correlation term"),
    ],
)
def test_roks_gpw_rejects_incomplete_functional_routes(functional, message):
    """Screened exchange and double-hybrid correlation cannot be omitted."""
    system = _h_atom_system(L=12.0)
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    with pytest.raises(NotImplementedError, match=message):
        vq.run_periodic_roks_gpw(
            system,
            basis,
            functional=functional,
            cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
            quiet=True,
        )


# ============================================================
# 1. He closed-shell: UHF parity with RHF
# ============================================================


def test_uhf_gpw_he_matches_rhf_gpw_to_machine_precision():
    """Closed-shell He forced through UHF (n_alpha = n_beta = 1) must
    converge to the RHF/GPW energy bit-for-bit. The shared Hcore
    initial guess gives identical α / β starting densities, so each
    SCF step is symmetric and the open-shell driver collapses onto
    the closed-shell solution."""
    import warnings
    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    system = _he_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    rhf = vq.run_periodic_rhf_gpw(
        system, basis, cutoff_ha=300.0, quiet=True,
    )
    uhf = vq.run_periodic_uhf_gpw(
        system, basis, n_alpha=1, n_beta=1,
        cutoff_ha=300.0, quiet=True,
    )

    assert uhf.converged
    assert uhf.n_alpha == 1
    assert uhf.n_beta == 1
    assert abs(uhf.energy - rhf.energy) < 1e-10


# ============================================================
# 2. H2 closed-shell: UKS LDA parity with RKS LDA
# ============================================================


def test_uks_gpw_h2_lda_matches_rks_gpw():
    """Closed-shell H2 with UKS/LDA (n_alpha = n_beta = 1) reproduces
    the RKS/LDA energy from the closed-shell GPW driver."""
    import warnings
    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    system = _h2_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    rks = vq.run_periodic_rhf_gpw(
        system, basis, cutoff_ha=300.0, functional="lda",
        max_iter=80, quiet=True,
    )
    uks = vq.run_periodic_uks_gpw(
        system, basis, functional="lda",
        n_alpha=1, n_beta=1,
        cutoff_ha=300.0, max_iter=80, quiet=True,
    )

    assert uks.converged
    assert abs(uks.energy - rks.energy) < 1e-7
    assert uks.breakdown.e_xc != 0.0
    assert uks.breakdown.functional == "lda"


# ============================================================
# 3. H atom UHF: real open-shell case with a sensible energy
# ============================================================


def test_uhf_gpw_h_atom_doublet_gives_sensible_energy():
    """H atom (n_alpha = 1, n_beta = 0) converges and lands near the
    molecular UHF/STO-3G H-atom reference (-0.4665 Ha). The
    periodic GPW result carries the Madelung shift on the charged
    (non-zero net density) cell, so we use a loose tolerance — the
    point is that the open-shell driver produces a physically
    reasonable energy, not bit-for-bit parity with a molecular
    code."""
    import warnings
    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    system = _h_atom_system()
    # vq.Molecule needs (atoms, charge, multiplicity).
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    # Molecular UHF reference at STO-3G — the GPW H atom should land
    # in the same ballpark up to the lattice-Madelung shift.
    uhf_opts = vq.UHFOptions()
    uhf_opts.conv_tol_energy = 1e-10
    e_mol = vq.run_uhf(mol, basis, uhf_opts).energy

    result = vq.run_periodic_uhf_gpw(
        system, basis,
        n_alpha=1, n_beta=0,
        cutoff_ha=300.0, max_iter=80, quiet=True,
    )
    assert result.converged
    assert result.n_alpha == 1
    assert result.n_beta == 0
    # Per-spin density traces with the overlap give the right
    # electron counts (1 alpha, 0 beta).
    from vibeqc.periodic_gapw_j import _overlap_lattice_gamma
    S = _overlap_lattice_gamma(basis, system)
    n_a_trace = float(np.einsum("ij,ij->", result.density_alpha, S))
    n_b_trace = float(np.einsum("ij,ij->", result.density_beta, S))
    assert abs(n_a_trace - 1.0) < 1e-8
    assert abs(n_b_trace) < 1e-8
    # Energy in the right ballpark of molecular UHF (Madelung shift
    # is ≲ 0.6 Ha on a 16-bohr cube for a charged-like density, so
    # we just require it stays within 1 Ha of the molecular value).
    assert abs(result.energy - e_mol) < 1.0


# ============================================================
# 4. Runner dispatch: UHF on He converges
# ============================================================


def test_runner_dispatch_gpw_uhf_he_converges():
    """``run_periodic_job(method='UHF', jk_method='gpw')`` on He
    closed-shell converges to the RHF energy (the runner default
    occupations fall back to n_alpha = n_beta for even-electron
    cells with multiplicity = 1)."""
    import warnings
    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    system = _he_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        rhf_runner = vq.run_periodic_job(
            system=system, basis=basis,
            method="RHF",
            output=str(Path(tmp) / "he_rhf_runner"),
            jk_method="gpw",
        )
        uhf_runner = vq.run_periodic_job(
            system=system, basis=basis,
            method="UHF",
            output=str(Path(tmp) / "he_uhf_runner"),
            jk_method="gpw",
        )
    assert uhf_runner.converged
    assert abs(uhf_runner.energy - rhf_runner.energy) < 1e-9
    # Per-spin fields surfaced on the runner result.
    assert hasattr(uhf_runner, "density_alpha")
    assert hasattr(uhf_runner, "mo_energies_alpha")
    assert hasattr(uhf_runner, "density_beta")


def test_runner_dispatch_gpw_rohf_li_matches_standalone(tmp_path, monkeypatch):
    """The public dispatcher reaches the maintained-preview Gamma GPW ROHF.

    The runner must preserve the restricted 2/1/0 orbital contract, emit the
    ROHF citation, and record the executed GPW backend in the manifest.
    """
    system = _li_atom_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    direct = vq.run_periodic_rohf_gpw(
        system,
        basis,
        cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA,
        max_iter=80,
        initial_guess=core.InitialGuess.SAD,
        quiet=True,
    )
    output_stem = tmp_path / "li_rohf_gpw"
    guess_calls = []
    from vibeqc import guess as guess_module

    original_guess = guess_module.initial_densities_open_shell

    def record_guess(*args, **kwargs):
        guess_calls.append(args[4])
        return original_guess(*args, **kwargs)

    monkeypatch.setattr(
        guess_module,
        "initial_densities_open_shell",
        record_guess,
    )

    result = vq.run_periodic_job(
        system,
        basis,
        method="ROHF",
        jk_method="gpw",
        cutoff_ha=_ROHF_GPW_TEST_CUTOFF_HA,
        max_iter=80,
        output=output_stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=True,
        progress=False,
    )

    assert result.converged
    assert guess_calls == [core.InitialGuess.SAD]
    assert result.backend == "gpw-rohf"
    assert result.energy == pytest.approx(direct.energy, abs=1e-10)
    assert result.s_squared == pytest.approx(0.75, abs=1e-12)
    assert tuple(result.occupations[:2]) == pytest.approx((2.0, 1.0))
    assert np.allclose(result.mo_coeffs_alpha, result.mo_coeffs_beta)
    output_text = output_stem.with_suffix(".out").read_text()
    assert "Job: PERIODIC ROHF" in output_text
    assert "converged in" in output_text
    assert "Roothaan" in output_text
    manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
    assert manifest["run"]["jk_method_requested"] == "gpw"
    assert manifest["run"]["jk_method_resolved"] == "gpw"
    assert manifest["run"]["jk_method_executed"] == "gpw"
    assert manifest["outputs"]["status"] == "complete"
    assert "roothaan_rohf_1960" in output_stem.with_suffix(".bibtex").read_text()


def test_runner_dispatch_gpw_roks_li_matches_standalone(tmp_path):
    """The public dispatcher runs Gamma GPW ROKS end to end."""
    system = _li_atom_system(L=12.0)
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    direct = vq.run_periodic_roks_gpw(
        system,
        basis,
        functional="pbe",
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        max_iter=80,
        conv_tol_energy=1e-9,
        damping=0.0,
        initial_guess=core.InitialGuess.SAD,
        quiet=True,
    )
    output_stem = tmp_path / "li_roks_gpw"

    result = vq.run_periodic_job(
        system,
        basis,
        method="ROKS",
        functional="pbe",
        jk_method="gpw",
        cutoff_ha=_ROKS_GPW_TEST_CUTOFF_HA,
        max_iter=80,
        conv_tol_energy=1e-9,
        damping=0.0,
        output=output_stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=True,
        progress=False,
    )

    assert result.converged
    assert result.backend == "gpw-roks"
    assert result.method == "roks"
    assert result.energy == pytest.approx(direct.energy, abs=1e-10)
    assert result.e_xc == pytest.approx(direct.e_xc, abs=1e-12)
    assert result.s_squared == pytest.approx(0.75, abs=1e-12)
    assert tuple(result.occupations[:2]) == pytest.approx((2.0, 1.0))
    assert np.allclose(result.mo_coeffs_alpha, result.mo_coeffs_beta)
    output_text = output_stem.with_suffix(".out").read_text()
    assert "Job: PERIODIC ROKS" in output_text
    assert "converged in" in output_text
    assert "Roothaan" in output_text
    manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
    assert manifest["run"]["jk_method_requested"] == "gpw"
    assert manifest["run"]["jk_method_resolved"] == "gpw"
    assert manifest["run"]["jk_method_executed"] == "gpw"
    assert manifest["outputs"]["status"] == "complete"
    assert "roothaan_rohf_1960" in output_stem.with_suffix(".bibtex").read_text()


@pytest.mark.parametrize(
    ("method", "system_dim", "kwargs", "message"),
    [
        # GitLab #647: this row used to pin ROHF/GDF as gated. 12fa81c78
        # (2026-08-02, "expose native GDF ROHF dispatch") made ROHF on GDF a
        # public route, so the gate no longer fires there; the row was red on
        # main from that commit on. GAPW is the backend that still fails
        # closed for ROHF, and test_runner_dispatch_rohf_gdf_is_not_gated
        # below pins the lifted gate so the expectation cannot drift back.
        ("ROHF", 3, {"jk_method": "gapw"}, "restricted-open-shell backends remain gated"),
        (
            "ROHF",
            3,
            {"jk_method": "gpw", "kpoints": (2, 1, 1)},
            "Gamma-only with one k point",
        ),
        (
            "ROHF",
            3,
            {"jk_method": "gpw", "smearing_temperature": 0.01},
            "integer 2/1/0 occupations",
        ),
        (
            "ROHF",
            3,
            {"jk_method": "gpw", "optimize": True},
            "gradients, geometry optimization",
        ),
        ("ROHF", 2, {"jk_method": "gpw"}, "bulk .* cannot run a dim=2 slab"),
        (
            "ROKS",
            3,
            {"jk_method": "gdf", "functional": "pbe"},
            "wired for jk_method='gpw'",
        ),
    ],
)
def test_runner_dispatch_gpw_rohf_unsupported_envelopes_fail_closed(
    tmp_path,
    monkeypatch,
    method,
    system_dim,
    kwargs,
    message,
):
    system = _li_atom_system()
    system.dim = system_dim
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    def fail_engine(*_args, **_kwargs):
        raise AssertionError("ROHF engine should not run for an unsupported route")

    monkeypatch.setattr(
        "vibeqc.periodic_gapw_open_shell.run_periodic_rohf_gpw",
        fail_engine,
    )
    monkeypatch.setattr(
        "vibeqc.periodic_gapw_open_shell.run_periodic_roks_gpw",
        fail_engine,
    )
    with pytest.raises(NotImplementedError, match=message):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            output=tmp_path / "unsupported_rohf_gpw",
            output_qvf=False,
            citations=False,
            **kwargs,
        )


class _GdfEngineReached(RuntimeError):
    """Sentinel raised in place of the GDF ROHF engine."""


def test_runner_dispatch_rohf_gdf_is_not_gated(tmp_path, monkeypatch):
    """GitLab #647: ROHF on native GDF is a public route (12fa81c78), so the
    restricted-open-shell backend gate must let it through to the KROHF/GDF
    engine. Pinned by replacing the engine with a sentinel: the dispatch
    reaching it proves no NotImplementedError fired on the way. On the
    fixed tree the real engine converges the Li-atom fixture to
    -7.32227152 Ha with backend 'native-multi-k-gdf-rohf' in ~2 s; the
    sentinel keeps this test SCF-free."""
    import vibeqc.periodic_runner as runner_module

    system = _li_atom_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    def sentinel_engine(*_args, **_kwargs):
        raise _GdfEngineReached("KROHF/GDF engine reached")

    monkeypatch.setattr(runner_module, "run_krohf_periodic_gdf", sentinel_engine)
    with pytest.raises(_GdfEngineReached):
        vq.run_periodic_job(
            system,
            basis,
            method="ROHF",
            jk_method="gdf",
            output=tmp_path / "rohf_gdf_not_gated",
            output_qvf=False,
            citations=False,
        )


# ============================================================
# 5. Runner dispatch: UKS LDA on H2 converges
# ============================================================


def test_runner_dispatch_gpw_uks_lda_h2_converges():
    """``run_periodic_job(method='UKS', jk_method='gpw',
    functional='lda')`` on closed-shell H2 converges to the same
    energy as the RKS dispatch through the closed-shell driver."""
    import warnings
    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    system = _h2_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        rks_runner = vq.run_periodic_job(
            system=system, basis=basis,
            method="RKS", functional="lda",
            output=str(Path(tmp) / "h2_rks_runner"),
            jk_method="gpw",
        )
        uks_runner = vq.run_periodic_job(
            system=system, basis=basis,
            method="UKS", functional="lda",
            output=str(Path(tmp) / "h2_uks_runner"),
            jk_method="gpw",
        )
    assert uks_runner.converged
    assert abs(uks_runner.energy - rks_runner.energy) < 1e-7
    assert uks_runner.e_xc != 0.0


# ============================================================
# 6. UKS requires functional
# ============================================================


def test_uks_gpw_without_functional_raises():
    """run_periodic_uks_gpw without a functional= argument raises
    a clear ValueError — the function name says UKS, the caller
    has to pick the XC."""
    import warnings
    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    system = _he_system()
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with pytest.raises(TypeError):
        # functional is keyword-only; missing it raises TypeError.
        vq.run_periodic_uks_gpw(system, basis, quiet=True)


# ============================================================
# 7. SAP routing (GitLab #667)
# ============================================================


@pytest.mark.parametrize(
    ("driver_name", "functional"),
    [
        ("run_periodic_uhf_gpw", None),
        ("run_periodic_uks_gpw", "lda"),
    ],
    ids=["uhf", "uks"],
)
def test_gamma_open_shell_gpw_sap_reaches_lattice_potential(
    monkeypatch,
    driver_name,
    functional,
):
    """Gamma UHF/UKS GPW execute the shared periodic SAP builder."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(_basis, system, _grid, table, _lattice_opts):
        calls.append((system, table))
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    system = _h_atom_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    kwargs = {"functional": functional} if functional is not None else {}

    with pytest.raises(SapPotentialReached):
        getattr(gpw_open, driver_name)(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            quiet=True,
            **kwargs,
        )

    assert calls == [(system, "sap_helfem_large")]


@pytest.mark.parametrize(
    ("driver_name", "functional"),
    [
        ("run_periodic_rohf_gpw", None),
        ("run_periodic_roks_gpw", "lda"),
    ],
    ids=["rohf", "roks"],
)
def test_gamma_restricted_open_gpw_sap_uses_one_electron_lattice_cutoff(
    monkeypatch,
    driver_name,
    functional,
):
    """ROHF/ROKS SAP uses the same default cutoff as its S/T pencil."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    class SapDensityReached(RuntimeError):
        pass

    cutoffs = []

    def stop_at_sap(*_args, lattice_opts, **_kwargs):
        cutoffs.append(float(lattice_opts.cutoff_bohr))
        raise SapDensityReached

    monkeypatch.setattr(
        guess_module,
        "initial_densities_open_shell",
        stop_at_sap,
    )
    system = _h_atom_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    kwargs = {"functional": functional} if functional is not None else {}

    with pytest.raises(SapDensityReached):
        getattr(gpw_open, driver_name)(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            quiet=True,
            **kwargs,
        )

    assert cutoffs == pytest.approx([core.LatticeSumOptions().cutoff_bohr])


@pytest.mark.parametrize(
    "driver_name",
    ["run_periodic_uks_gpw_multi_k", "run_periodic_roks_gpw_multi_k"],
    ids=["uks", "roks"],
)
def test_multik_open_shell_gpw_sap_reaches_lattice_potential(
    monkeypatch,
    driver_name,
):
    """Both pure-DFT open-shell multi-k GPW adapters build SAP."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(_basis, _system, _grid, table, _lattice_opts):
        calls.append(table)
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    system = _h_atom_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(SapPotentialReached):
        getattr(gpw_open, driver_name)(
            system,
            basis,
            kmesh,
            functional="lda",
            grid=grid,
            max_iter=1,
            initial_guess=core.InitialGuess.SAP,
            quiet=True,
        )

    assert calls == ["sap_helfem_large"]


@pytest.mark.parametrize(
    "selector",
    [
        core.InitialGuess.SAP,
        core.InitialGuess.PATOM,
        core.InitialGuess.FRAGMO,
    ],
)
def test_gamma_open_shell_gpw_restart_density_precedes_selector(
    monkeypatch,
    selector,
):
    """A supplied spin-density pair is the effective READ artifact."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    class RestartDensityReached(RuntimeError):
        pass

    def unexpected_vsap(*_args, **_kwargs):
        pytest.fail("guess construction must not run for an explicit density")

    restart = (np.array([[0.25]]), np.array([[0.125]]))

    def stop_at_first_j(_builder, density):
        overlap = gpw_open._overlap_lattice_gamma(basis, system)
        assert np.trace(density @ overlap) == pytest.approx(1, abs=1e-12)
        raise RestartDensityReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", unexpected_vsap)
    monkeypatch.setattr(gpw_open.GpwJBuilder, "build_J", stop_at_first_j)
    system = _h_atom_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)

    error = NotImplementedError if selector == core.InitialGuess.FRAGMO else RestartDensityReached
    with pytest.raises(error):
        gpw_open.run_periodic_uhf_gpw(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_density=restart,
            initial_guess=selector,
            quiet=True,
        )


def test_gamma_open_shell_gpw_restart_rejects_malformed_selector():
    """A spin restart must not hide an invalid guess spelling."""
    import vibeqc.periodic_gapw_open_shell as gpw_open

    system = _h_atom_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(ValueError, match="unknown initial_guess='bogus'"):
        gpw_open.run_periodic_uhf_gpw(
            system,
            basis,
            initial_density=(np.array([[0.25]]), np.array([[0.125]])),
            initial_guess="bogus",
            quiet=True,
        )


@pytest.mark.parametrize(
    ("selector", "effective"),
    [
        (core.InitialGuess.SAD, core.InitialGuess.SAD),
        ("auto", core.InitialGuess.SAD),
        ("minao", core.InitialGuess.MINAO),
    ],
    ids=["enum-sad", "string-auto", "string-minao"],
)
def test_gamma_open_shell_gpw_density_guesses_use_shared_adapter(
    monkeypatch,
    selector,
    effective,
):
    """Open-shell GPW uses the shared spin-density guess adapter."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    class SharedGuessReached(RuntimeError):
        pass

    seen = []

    def stop_at_shared_guess(_mol, _basis, _na, _nb, guess, **kwargs):
        seen.append((guess, kwargs["periodic_system"]))
        raise SharedGuessReached

    monkeypatch.setattr(
        guess_module,
        "initial_densities_open_shell",
        stop_at_shared_guess,
    )
    system = _h_atom_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * 16.0, 8, 8, 8)

    with pytest.raises(SharedGuessReached):
        gpw_open.run_periodic_uhf_gpw(
            system,
            basis,
            grid=grid,
            max_iter=1,
            initial_guess=selector,
            quiet=True,
        )

    assert seen == [(effective, system)]


@pytest.mark.parametrize(
    ("driver_name", "args", "kwargs"),
    [
        (
            "run_periodic_uks_gpw_multi_k",
            (None, None, None),
            {"functional": "lda"},
        ),
        (
            "run_periodic_roks_gpw_multi_k",
            (None, None, None),
            {"functional": "lda"},
        ),
    ],
    ids=["uks-multik", "roks-multik"],
)
def test_direct_open_shell_gpw_guess_selectors_fail_closed_for_patom(
    driver_name,
    args,
    kwargs,
):
    """Pure-DFT multi-k GPW has no exact exchange for the PATOM step."""
    import vibeqc.periodic_gapw_open_shell as gpw_open

    system = vq.PeriodicSystem(3, 10 * np.eye(3), [vq.Atom(1, [5., 5., 5.])], multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    args = (system, basis, vq.monkhorst_pack(system, [3, 1, 1]))

    with pytest.raises(
        NotImplementedError,
        match="initial_guess=PATOM.*not implemented by this route",
    ):
        getattr(gpw_open, driver_name)(
            *args,
            initial_guess=core.InitialGuess.PATOM,
            **kwargs,
        )


@pytest.mark.parametrize(
    'route,family,system_name',
    [(route, family, 'h2')
     for route in ('gpw', 'gapw')
     for family in ('rhf', 'rks', 'uhf', 'uks', 'rohf', 'roks')
     if route == 'gpw' or family not in ('rohf', 'roks')]
    + [(route, family, 'lih_triplet')
       for route in ('gpw', 'gapw')
       for family in ('uhf', 'uks', 'rohf', 'roks')
       if route == 'gpw' or family not in ('rohf', 'roks')],
)
def test_gamma_patom_routes_execute_normalized_hf_seed(
    monkeypatch, route, family, system_name
):
    """PATOM reaches the physical builder in every implemented Gamma family."""
    import vibeqc as vq
    import vibeqc.guess as guess_module
    from vibeqc import periodic_gapw_augment as gapw
    from vibeqc import periodic_gapw_j as gpw
    from vibeqc import periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    triplet = system_name == 'lih_triplet'
    lattice = np.eye(3) * 16.0
    system = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(3 if triplet else 1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 2.5 if triplet else 1.4])],
        multiplicity=3 if triplet else 1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    grid = PlaneWaveGrid(lattice, 8, 8, 8)
    overlap = gpw._overlap_lattice_gamma(basis, system)
    targets = (3, 1) if triplet else (1, 1)

    def assert_density(density, electrons):
        density = np.asarray(density)
        assert np.all(np.isfinite(density))
        np.testing.assert_allclose(density, density.conj().T, atol=1e-12, rtol=0)
        # One Gamma point has unit Brillouin-zone weight.
        assert np.trace(density @ overlap).real == pytest.approx(electrons, abs=1e-11)
        assert np.linalg.eigvalsh(density).min() >= -1e-12

    seed_pairs = []
    original_step = guess_module.patom_spin_density_step

    def observed_step(*args, **kwargs):
        pair = original_step(*args, **kwargs)
        assert (args[4], args[5]) == targets
        for density, electrons in zip(pair, targets):
            assert_density(density, electrons)
        seed_pairs.append(pair)
        return pair

    monkeypatch.setattr(guess_module, 'patom_spin_density_step', observed_step)
    closed_gpw = route == 'gpw' and family in ('rhf', 'rks')
    j_densities, k_densities = [], []
    if closed_gpw:
        original_j = gpw.GpwJBuilder.build_J
        original_k = gpw._core.build_jk_gamma_molecular_limit

        def observed_j(builder, density):
            j_densities.append(np.asarray(density).copy())
            return original_j(builder, density)

        def observed_k(basis_arg, system_arg, options_arg, density):
            k_densities.append(np.asarray(density).copy())
            return original_k(basis_arg, system_arg, options_arg, density)

        monkeypatch.setattr(gpw.GpwJBuilder, 'build_J', observed_j)
        monkeypatch.setattr(gpw._core, 'build_jk_gamma_molecular_limit', observed_k)

    module = gapw if route == 'gapw' else (gpw if closed_gpw else gpw_open)
    kwargs = dict(grid=grid, max_iter=1, initial_guess=vq.InitialGuess.PATOM, quiet=True)
    if family.endswith('ks'):
        kwargs['functional'] = 'lda'
    if route == 'gapw':
        kwargs.update(one_centre='block', n_radial=24, lebedev_order=17)
    result = getattr(module, f'run_periodic_{family}_{route}')(system, basis, **kwargs)

    if closed_gpw:
        # PATOM adds a J build before iteration one and final-energy refresh.
        # Pure LDA still executes full-HF K for this physical seed.
        assert len(j_densities) >= 3
        assert k_densities
        np.testing.assert_allclose(k_densities[0], j_densities[0], atol=1e-13, rtol=0)
        seed = guess_module.initial_density_closed_shell(
            system.unit_cell_molecule(), basis, sum(targets) // 2,
            vq.InitialGuess.SAD, is_periodic=True, periodic_system=system,
            overlap=overlap,
        )
        np.testing.assert_allclose(j_densities[0], seed, atol=1e-12, rtol=0)
        assert_density(j_densities[1], sum(targets))
    else:
        assert len(seed_pairs) == 1

    densities = ([result.density] if family in ('rhf', 'rks')
                 else [result.density_alpha, result.density_beta])
    populations = [sum(targets)] if len(densities) == 1 else targets
    for density, electrons in zip(densities, populations):
        assert_density(density, electrons)
    assert result.n_iter == 1
    selection = result.guess_selection
    assert (selection.requested, selection.effective, selection.transport) == (
        vq.InitialGuess.PATOM, vq.InitialGuess.PATOM, vq.InitialGuess.PATOM
    )


@pytest.mark.parametrize('name', ['run_periodic_uhf_gpw','run_periodic_uks_gpw','run_periodic_uks_gpw_multi_k','run_periodic_uhf_gapw','run_periodic_uks_gapw'])
def test_grid_routes_execute_atomic_spin_seed(monkeypatch,name):
    import importlib
    import vibeqc as vq
    system=vq.PeriodicSystem(3,np.eye(3)*8,[vq.Atom(1,[0,0,0]),vq.Atom(1,[0,0,2.])])
    basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g')
    import vibeqc.guess as guess
    original=guess.initial_densities_open_shell;seen=[]
    def capture(*args,**kwargs):
        ds=original(*args,**kwargs);seen.append((kwargs.get('atomic_spins'),ds,kwargs));return ds
    monkeypatch.setattr(guess,'initial_densities_open_shell',capture)
    mod=importlib.import_module('vibeqc.periodic_gapw_augment' if name.endswith('gapw') else 'vibeqc.periodic_gapw_open_shell')
    kw=dict(initial_guess='SAD',atomic_spins=[1,-1],cutoff_ha=5,max_iter=1,quiet=True)
    if 'uks' in name:kw['functional']='lda'
    if name.endswith('gapw'):kw.update(lmax=0,n_radial=8,lebedev_order=5)
    if name=='run_periodic_uhf_gapw':kw['molecular_limit']=True
    args=[system,basis]
    if name.endswith('multi_k'):args.append(vq.monkhorst_pack(system,[3,1,1]))
    r=getattr(mod,name)(*args,**kw)
    assert len(seen)==1 and seen[0][0]==[1,-1]
    da,db=seen[0][1]; assert (da-db)[0,0]>0 and (da-db)[1,1]<0
    assert r.guess_selection.effective==vq.InitialGuess.SAD
    with pytest.raises(ValueError,match='requires SAD'):
        getattr(mod,name)(*args,**dict(kw,initial_guess='HCORE'))
