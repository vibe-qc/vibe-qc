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


@pytest.mark.parametrize("route,family,use_davidson", [
    (route, family, False)
    for route in ("gpw", "gapw")
    for family in ("rhf", "rks", "uhf", "uks")
] + [("gpw", family, True) for family in ("rhf", "rks")]
  + [("gpw", family, False) for family in ("rohf", "roks")])
def test_gamma_grid_shared_fock_solve_matches_legacy(
    monkeypatch, route, family, use_davidson,
):
    """Physical SCF parity plus an independent generalized solve at every step."""
    import scipy.linalg as sla
    from vibeqc import periodic_gapw_augment as gapw
    from vibeqc import periodic_gapw_j as gpw
    from vibeqc import periodic_gapw_open_shell as gpw_open
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid
    from vibeqc.solvers._scf import FockDiagonalizer

    restricted_open = family.startswith("ro")
    open_shell = family.startswith("u") or restricted_open
    if open_shell:
        # The asymmetric H3 doublet also exercises unequal occupied subspaces.
        system = vq.PeriodicSystem(
            3, np.eye(3) * 12.,
            [vq.Atom(1, xyz) for xyz in ([5.1, 6., 6.], [6.9, 6., 6.], [6., 7.4, 6.])],
            multiplicity=2,
        )
    else:
        system = _h2_system(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(system.lattice, 18, 18, 18)
    module = gapw if route == "gapw" else (gpw_open if open_shell else gpw)
    driver = getattr(module, f"run_periodic_{family}_{route}")
    kwargs = dict(grid=grid, quiet=True, max_iter=60)
    if family.endswith("ks"):
        kwargs["functional"] = "lda"
    if route == "gapw":
        kwargs.update(one_centre="block", n_radial=24, lebedev_order=17)
    elif not open_shell:
        kwargs.update(use_davidson=use_davidson, davidson_min_dim=0)

    original_call = FockDiagonalizer.__call__
    channels = {}

    def checked_call(diagonalizer, fock):
        c, e = original_call(diagonalizer, fock)
        result = diagonalizer.last_result
        metadata = result.operator_metadata
        assert metadata.kind == ("ks" if family.endswith("ks") else "fock")
        assert metadata.kpoint == (0., 0., 0.)
        assert diagonalizer.iterative == use_davidson
        generations = channels.setdefault(metadata.spin, [])
        assert metadata.generation == len(generations)
        generations.append(metadata.generation)
        # LAPACK's direct generalized problem is independent of the driver's X.
        reference = sla.eigh(getattr(fock, "matrix", fock), diagonalizer.overlap, eigvals_only=True)
        np.testing.assert_allclose(e, reference, atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            c.T @ diagonalizer.overlap @ c, np.eye(len(e)), atol=1e-10, rtol=0,
        )
        assert max(result.residuals) < 1e-7
        return c, e

    monkeypatch.setattr(FockDiagonalizer, "__call__", checked_call)
    result = driver(system, basis, **kwargs)
    assert result.converged
    expected_channels = ({"restricted-open"} if restricted_open else
                         ({"alpha", "beta"} if open_shell else {"restricted"}))
    assert set(channels) == expected_channels
    assert all(len(generations) >= 2 for generations in channels.values())

    # Re-run the physical loop with the historical symmetrized dense solve.
    def legacy_call(diagonalizer, fock):
        projected = diagonalizer.X.T @ getattr(fock, "matrix", fock) @ diagonalizer.X
        e, y = np.linalg.eigh(0.5 * (projected + projected.T))
        return diagonalizer.X @ y, e

    monkeypatch.setattr(FockDiagonalizer, "__call__", legacy_call)
    reference = driver(system, basis, **kwargs)
    assert reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-10)
    overlap = gpw._overlap_lattice_gamma(basis, system)
    fields = ("density_alpha", "density_beta") if open_shell else ("density",)
    populations = (2, 1) if open_shell else (2,)
    for field, electrons in zip(fields, populations):
        density = np.asarray(getattr(result, field))
        np.testing.assert_allclose(density, getattr(reference, field), atol=1e-9, rtol=0)
        assert np.trace(density @ overlap) == pytest.approx(electrons, abs=1e-10)


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


# Multi-k PATOM uses a full-HF seed even for pure DFT. Its supported
# GPW/GAPW routes are covered together, including normalized seed input
# and seed-only exchange, in test_periodic_patom.py.


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


def _uhf_fock_state_case():
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid
    lattice = np.eye(3) * 16.
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, xyz) for xyz in
        ([7.1, 8., 8.], [8.9, 8., 8.], [8., 9.4, 8.])], multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    return system, basis, PlaneWaveGrid(lattice, 48, 48, 48)


@pytest.mark.parametrize('iterations,damping,guess,convention', [
    (60, 0., 'HCORE', 'ewald'), (60, .2, 'PATOM', 'ewald'),
    (1, .35, 'HCORE', 'ewald'), (0, 0., 'HCORE', 'ewald'),
    (1, .2, 'HCORE', 'smeared_erfc'),
])
def test_gpw_uhf_evaluated_state(monkeypatch, iterations, damping, guess, convention):
    from vibeqc import periodic_gapw_open_shell as driver
    from vibeqc import periodic_gapw_j as gj
    from vibeqc import periodic_gapw_runner_adapter as adapter
    from vibeqc import periodic_gpw_fock as gf
    from vibeqc.solvers._scf import FockDiagonalizer
    system, basis, grid = _uhf_fock_state_case()
    states, solves = {}, []
    original_build, original_solve = gf._GpwUHFBuilder.__call__, FockDiagonalizer.__call__

    def build(self, *d):
        state = original_build(self, *d)
        states[state.identity] = state
        return state

    def solve(self, operator):
        meta = operator.metadata
        state = states[meta.build_id]
        assert meta.density_id == state.density_id
        assert meta.spin in ('alpha', 'beta')
        solves.append(meta)
        return original_solve(self, operator)

    monkeypatch.setattr(gf._GpwUHFBuilder, '__call__', build)
    monkeypatch.setattr(FockDiagonalizer, '__call__', solve)
    # Mixed restart also exercises capped and zero-iteration density orbitals.
    kw = {} if guess == 'PATOM' else dict(initial_density=(np.eye(basis.nbasis), np.eye(basis.nbasis)))
    r = driver.run_periodic_uhf_gpw(system, basis, grid=grid, max_iter=iterations,
        damping=damping, initial_guess=guess, v_ne_convention=convention,
        smearing_alpha=.7 if convention == 'smeared_erfc' else None, quiet=True, **kw)
    state = r.fock_build
    assert state.method == 'UHF'
    assert r.energy == state.energy.total == r.breakdown.e_total
    assert r.converged == (iterations == 60)
    assert state.metric_electron_counts == pytest.approx((2., 1.), abs=1e-11)
    assert r.run_fields['fock_build_id'] == state.identity
    assert r.evaluated_build_history[-1]['terminal']
    assert r.evaluated_build_history[-1]['build_id'] == state.identity
    if r.scf_trace:
        assert r.scf_trace[-1]['energy'] == r.energy
    assert {m.spin for m in solves} == {'alpha', 'beta'}
    if iterations == 60:
        assert any(m.role == 'solver' and m.modification == 'pulay-diis' for m in solves)
        assert state.commutator_norm < 1e-7
        assert r.density_fixed_point_residual < 1e-7
    if guess == 'PATOM':
        assert solves[0].role == solves[1].role == 'physical'
        assert solves[0].build_id == solves[1].build_id
        assert solves[0].generation == 0
        from vibeqc.guess import _canonical_orthogonalizer
        seed_frame = states[solves[0].build_id].channels[0].operator.frame
        expected_x = _canonical_orthogonalizer(r.overlap, 1e-7)
        np.testing.assert_allclose(seed_frame.X, expected_x, atol=2e-12, rtol=0)
    s = r.overlap
    for i, spin in enumerate(('alpha', 'beta')):
        d, f, c, eps, occ = (np.asarray(getattr(r, name+'_'+spin)) for name in
                             ('density', 'fock', 'mo_coeffs', 'mo_energies', 'occupations'))
        np.testing.assert_array_equal(d, state.channels[i].density)
        np.testing.assert_array_equal(f, state.channels[i].operator.matrix)
        np.testing.assert_allclose((c*occ)@c.T, d, atol=2e-12, rtol=0)
        np.testing.assert_allclose(c.T@s@c, np.eye(len(occ)), atol=2e-12, rtol=0)
        np.testing.assert_allclose(eps, np.diag(c.T@f@c), atol=2e-12, rtol=0)
        assert not state.channels[i].density.flags.writeable
        assert not state.channels[i].operator.matrix.flags.writeable
    t = gj._kinetic_lattice_gamma(basis, system)
    v = driver._build_v_ne(basis, system, convention, .7 if convention == 'smeared_erfc' else None, grid)
    lo = core.LatticeSumOptions(); lo.cutoff_bohr = 25.
    j = gj.GpwJBuilder(basis, grid).build_J(r.density_alpha+r.density_beta)
    ks = [driver._k_per_spin(basis, system, ch.density, lo) for ch in state.channels]
    expected = 0.
    for ch, k in zip(state.channels, ks):
        np.testing.assert_allclose(ch.operator.matrix, t+v+j-k, atol=2e-12, rtol=0)
        expected += np.sum(ch.density*(t+v+.5*j-.5*k))
    assert state.energy.electronic == pytest.approx(expected, abs=2e-12)

    # Output must carry the evaluated state without a second integral build.
    def no_rebuild(*args, **kwargs):
        raise AssertionError('adapter rebuilt evaluated integrals')
    monkeypatch.setattr(gj.GpwJBuilder, 'build_J', no_rebuild)
    adapted = adapter.gpw_uhf_result_to_runner_shape(r, system, basis)
    assert adapted.fock_build is state
    assert adapted.run_fields == r.run_fields
    assert adapted.orbital_kind == 'natural'
    for spin in ('alpha', 'beta'):
        for name in ('density', 'fock', 'occupations'):
            np.testing.assert_array_equal(getattr(adapted, name+'_'+spin), getattr(r, name+'_'+spin))


def test_gpw_uhf_derivative_and_source(record_property):
    from vibeqc import periodic_gapw_j as gj
    from vibeqc import periodic_gpw_fock as gf
    from vibeqc.solvers.operators import OperatorFrame
    system, basis, grid = _uhf_fock_state_case()
    s = gj._overlap_lattice_gamma(basis, system)
    w, u = np.linalg.eigh(s); x = (u / np.sqrt(w)) @ u.T
    frame = OperatorFrame(basis_id='test-gpw-uhf', n=basis.nbasis, overlap=s, X=x,
                          ordering='libint-standard', image_gauge='gamma-periodic-ao')
    lo = core.LatticeSumOptions(); lo.cutoff_bohr = 25.
    hf = gf._GpwRHFBuilder(basis, system, lo, gj.GpwJBuilder(basis, grid),
        gj._kinetic_lattice_gamma(basis, system), gj._ewald_v_ne_gamma(basis, system),
        frame, core.ewald_nuclear_repulsion(system, core.EwaldOptions()),
        nuclear_convention='ewald', smearing_alpha=None)
    builder = gf._GpwUHFBuilder(hf)
    da, db = .6*x@x.T, .3*x@x.T
    state = builder(da, db)
    rng = np.random.default_rng(472)
    perturbations = []
    for _ in range(2):
        p = rng.normal(size=da.shape); perturbations.append((p+p.T)/2)
    for active in (0, 1, None):
        direction = tuple(p if active is None or i == active else np.zeros_like(p)
                          for i, p in enumerate(perturbations))
        h = 2e-5
        plus = builder(*(d+h*p for d, p in zip((da, db), direction))).energy.total
        minus = builder(*(d-h*p for d, p in zip((da, db), direction))).energy.total
        expected = sum(np.sum(c.operator.matrix*p) for c, p in zip(state.channels, direction))
        record_property(f'derivative_error_{active}', abs((plus-minus)/(2*h)-expected))
        assert (plus-minus)/(2*h) == pytest.approx(expected, abs=2e-8)
    swapped = builder(db, da)
    assert swapped.energy.total == pytest.approx(state.energy.total, abs=2e-12)
    assert swapped.density_id != state.density_id
    assert swapped.source == state.source
    np.testing.assert_allclose(swapped.channels[1].operator.matrix, state.channels[0].operator.matrix, atol=2e-12)
    da[:] = 0.
    assert np.linalg.norm(state.channels[0].density) > 0
    with pytest.raises(NotImplementedError, match='real spin'):
        builder(db.astype(complex), db)
    with pytest.raises(ValueError, match='dimensions'):
        builder(np.eye(1), db)
    lo.cutoff_bohr = 24.
    with pytest.raises(ValueError, match='source changed'):
        builder(db, db)
    lo.cutoff_bohr = 25.
    chi = hf.gpw.chi(); original = chi.flat[0]; chi.flat[0] += .01
    with pytest.raises(ValueError, match='source changed'):
        builder(db, db)
    chi.flat[0] = original
    assert builder(db, db).source == state.source


@pytest.mark.parametrize('guess', ['HCORE', 'PATOM'])
def test_gpw_uhf_closed_and_empty_spin_limits(guess):
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid
    for system, populations in ((_h2_system(16.), (1, 1)), (_h_atom_system(16.), (1, 0))):
        basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
        grid = PlaneWaveGrid(system.lattice, 48, 48, 48)
        r = vq.run_periodic_uhf_gpw(system, basis, grid=grid, initial_guess=guess, quiet=True)
        assert r.converged
        assert r.fock_build.metric_electron_counts == pytest.approx(populations, abs=2e-12)
        if populations[1]:
            restricted = vq.run_periodic_rhf_gpw(system, basis, grid=grid, initial_guess=guess, quiet=True)
            assert r.energy == pytest.approx(restricted.energy, abs=2e-10)
            np.testing.assert_allclose(r.density_alpha+r.density_beta, restricted.density, atol=2e-9, rtol=0)
            np.testing.assert_allclose(r.fock_alpha, restricted.fock, atol=2e-10, rtol=0)
        else:
            np.testing.assert_allclose(r.density_beta, 0., atol=1e-14, rtol=0)
            np.testing.assert_allclose(r.occupations_beta, 0., atol=1e-14, rtol=0)


def test_gpw_uhf_patom_retains_overlap_cutoff(monkeypatch):
    from vibeqc import periodic_gapw_open_shell as driver
    from vibeqc import periodic_gapw_j as gj
    from vibeqc import periodic_gpw_fock as gf
    system, basis, grid = _uhf_fock_state_case()
    from vibeqc import guess
    # This positive unit-diagonal metric leaves the SCF inverse defined,
    # but only one PATOM direction survives its normalized 1e-7 cutoff.
    metric = (1.-5e-8)*np.ones((3, 3)) + 5e-8*np.eye(3)
    monkeypatch.setattr(driver, '_overlap_lattice_gamma', lambda *args: metric)
    monkeypatch.setattr(guess, 'initial_densities_open_shell',
                        lambda *args, **kwargs: (np.eye(3), np.eye(3)))
    def no_integrals(*args, **kwargs):
        raise AssertionError('PATOM built integrals before checking retained rank')
    monkeypatch.setattr(gf._GpwUHFBuilder, '__call__', no_integrals)
    with pytest.raises(ValueError, match='PATOM: insufficient independent orbitals'):
        driver.run_periodic_uhf_gpw(system, basis, grid=grid, initial_guess='PATOM',
            max_iter=0, quiet=True)
