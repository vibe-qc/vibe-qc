"""Tests for the GAPW (GPW Hartree-J) path's DFT+U wiring.

v0.12 R1 milestone: ``run_periodic_rhf_gpw`` and
``run_periodic_rks_gpw_multi_k`` accept ``dft_plus_u_sites=[
HubbardSite(...)]`` and add the Dudarev (1998) on-site correction
to the Fock matrix during the SCF. The +U energy is reported on
:class:`GpwScfResult` / :class:`GpwMultiKScfResult` as
``e_dft_plus_u`` and folded into ``e_total`` on the breakdown.

Pins:

* ``dft_plus_u_sites=None`` is backwards-compatible — the SCF runs
  exactly as it did before and ``result.e_dft_plus_u == 0.0``.
* A single ``HubbardSite`` on the H 1s of vacuum-padded H2 STO-3G
  gives a *non-zero* +U energy and shifts the total energy from
  the no-+U baseline.
* The same site supplied to the multi-k entry point (Γ-only
  Monkhorst-Pack mesh, LDA functional) also produces a non-zero
  +U energy and shifts the total energy.
* Gamma UHF/UKS GPW uses the open-shell per-spin convention
  ``E_U = E_U_alpha + E_U_beta`` and records the requested route
  through ``run_periodic_job``.

These tests don't pin a numerical value against a reference code
(no DFT+U on the GPW path has shipped before); they pin the
*wiring* — i.e. the +U surface is reachable, fires the kernel,
and bookkeeping ends up on the result.
"""

from __future__ import annotations

import tomllib
import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from vibeqc.periodic_gapw_augment import (
    run_periodic_uhf_gapw,
    run_periodic_uks_gapw,
)
from vibeqc.periodic_gapw_open_shell import (
    run_periodic_uhf_gpw,
    run_periodic_uks_gpw,
)
from vibeqc.periodic_gapw_j import (
    _overlap_lattice_gamma,
    run_periodic_rhf_gpw,
    run_periodic_rks_gpw_multi_k,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


def _h2_periodic_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def _h2_basis(L: float = 16.0):
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    return vq.BasisSet(mol, "sto-3g")


def _h3_doublet_fixture(L: float = 12.0):
    coords = [
        [L / 2 - 0.9, L / 2, L / 2],
        [L / 2 + 0.9, L / 2, L / 2],
        [L / 2, L / 2 + 1.4, L / 2],
    ]
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(1, xyz) for xyz in coords]
    system.multiplicity = 2
    mol = vq.Molecule(
        [vq.Atom(1, xyz) for xyz in coords],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    return system, basis, PlaneWaveGrid(np.eye(3) * L, 18, 18, 18)


def _open_shell_periodic_dftu_reference(system, basis, result, sites):
    S = _overlap_lattice_gamma(basis, system)
    ao_groups = vq.ao_group_indices(basis)
    n_alpha = vq.compute_occupation_matrices(
        sites, np.asarray(result.density_alpha), S, ao_groups,
    )
    n_beta = vq.compute_occupation_matrices(
        sites, np.asarray(result.density_beta), S, ao_groups,
    )
    return (
        vq.compute_dudarev_energy(sites, n_alpha)
        + vq.compute_dudarev_energy(sites, n_beta)
    )


def test_rhf_gpw_dft_plus_u_none_is_backwards_compatible():
    """Passing ``dft_plus_u_sites=None`` (the default) leaves the SCF
    unchanged: ``e_dft_plus_u == 0.0`` and the energy matches the
    no-+U baseline."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)

    baseline = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
    )
    none_run = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
        dft_plus_u_sites=None,
    )

    assert baseline.converged
    assert none_run.converged
    assert none_run.e_dft_plus_u == 0.0
    assert none_run.breakdown.e_dft_plus_u == 0.0
    assert none_run.energy == pytest.approx(baseline.energy, abs=1e-10)


def test_rhf_gpw_dft_plus_u_single_h1s_site_shifts_energy():
    """A single ``HubbardSite`` on the H 1s gives a non-zero +U
    energy and shifts the total energy from the no-+U baseline. The
    test uses a small U_eff (~0.1 Ha = 2.72 eV) so the SCF still
    converges with the default settings."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32)

    # 0.1 Ha ≈ 2.7211 eV.
    U_ev = 0.1 * 27.211386245988
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=U_ev)]

    baseline = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
    )
    with_u = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
        dft_plus_u_sites=sites,
    )

    assert baseline.converged
    assert with_u.converged
    # +U energy must be non-zero (the H 1s on a H2 closed-shell SCF
    # is partially occupied in the AO-projected occupation matrix
    # via the bond — n is not 0 or 1, so the Dudarev penalty fires).
    assert with_u.e_dft_plus_u != 0.0
    assert with_u.breakdown.e_dft_plus_u == pytest.approx(
        with_u.e_dft_plus_u, abs=1e-12,
    )
    # Total energy differs from baseline (it's not the same SCF).
    assert with_u.energy != pytest.approx(baseline.energy, abs=1e-8)


def test_multi_k_gpw_dft_plus_u_site_shifts_energy():
    """The same ``HubbardSite`` supplied to the multi-k entry point
    on a Γ-only Monkhorst-Pack mesh + LDA functional gives a
    non-zero +U energy and shifts the total energy. The Γ-only
    mesh keeps the SCF cheap; the multi-k +U code path is the
    same regardless of mesh size."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32, cutoff_ha=300.0)

    U_ev = 0.1 * 27.211386245988
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=U_ev)]
    kmesh = core.monkhorst_pack(system, [1, 1, 1])

    baseline = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh, functional="lda",
        grid=grid, quiet=True,
    )
    with_u = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh, functional="lda",
        grid=grid, quiet=True,
        dft_plus_u_sites=sites,
    )

    assert baseline.converged
    assert with_u.converged
    assert baseline.e_dft_plus_u == 0.0
    assert with_u.e_dft_plus_u != 0.0
    assert with_u.breakdown.e_dft_plus_u == pytest.approx(
        with_u.e_dft_plus_u, abs=1e-12,
    )
    assert with_u.energy != pytest.approx(baseline.energy, abs=1e-8)


@pytest.mark.parametrize(
    ("runner", "kwargs"),
    [
        (run_periodic_uhf_gpw, {}),
        (run_periodic_uks_gpw, {"functional": "lda"}),
    ],
)
def test_open_shell_gpw_dft_plus_u_matches_per_spin_reference(runner, kwargs):
    """Gamma open-shell GPW +U must add the per-spin Dudarev energy and
    Fock contribution instead of failing through the high-level route guard."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    system, basis, grid = _h3_doublet_fixture()
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]

    with_u = runner(
        system,
        basis,
        grid=grid,
        max_iter=60,
        conv_tol_energy=1e-8,
        conv_tol_density=1e-6,
        dft_plus_u_sites=sites,
        quiet=True,
        **kwargs,
    )

    assert with_u.converged
    assert with_u.e_dft_plus_u > 0.0
    assert with_u.breakdown.e_dft_plus_u == pytest.approx(
        with_u.e_dft_plus_u,
        abs=1e-12,
    )
    assert with_u.e_dft_plus_u == pytest.approx(
        _open_shell_periodic_dftu_reference(system, basis, with_u, sites),
        rel=1e-10,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("runner", "kwargs"),
    [
        (run_periodic_uhf_gapw, {"molecular_limit": True}),
        (run_periodic_uks_gapw, {"functional": "lda"}),
    ],
)
def test_open_shell_gapw_dft_plus_u_matches_per_spin_reference(runner, kwargs):
    """Gamma open-shell GAPW +U uses the same per-spin Dudarev convention as
    GPW instead of staying gated in the high-level dispatcher."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    system, basis, grid = _h3_doublet_fixture()
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]

    with_u = runner(
        system,
        basis,
        grid=grid,
        max_iter=60,
        conv_tol_energy=1e-8,
        conv_tol_density=1e-6,
        dft_plus_u_sites=sites,
        quiet=True,
        **kwargs,
    )

    assert with_u.converged
    assert with_u.e_dft_plus_u > 0.0
    assert with_u.breakdown.e_dft_plus_u == pytest.approx(
        with_u.e_dft_plus_u,
        abs=1e-12,
    )
    assert with_u.e_dft_plus_u == pytest.approx(
        _open_shell_periodic_dftu_reference(system, basis, with_u, sites),
        rel=1e-10,
        abs=1e-12,
    )


def test_run_periodic_job_gpw_multik_dft_plus_u_uses_requested_route(tmp_path):
    """High-level multi-k GPW +U must not be hijacked by the legacy direct
    +U interception block; raw output and .system metadata record the route."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]
    output_stem = tmp_path / "gpw_rks_plus_u"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gpw",
        kpoints=(1, 1, 1),
        output=output_stem,
        dft_plus_u=sites,
        cutoff_ha=10.0,
        max_iter=40,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )

    assert result.converged
    assert result.e_dft_plus_u > 0.0
    text = output_stem.with_suffix(".out").read_text()
    assert "dft_plus_u_route    = gpw_rks_multi_k" in text
    assert "DFT+U (Dudarev)" in text
    manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
    run = manifest["run"]
    assert run["jk_method_requested"] == "gpw"
    assert run["jk_method_resolved"] == "gpw"
    assert run["jk_method_executed"] == "gpw"
    assert run["dft_plus_u"] is True
    assert run["dft_plus_u_route"] == "gpw_rks_multi_k"


def test_run_periodic_job_gpw_open_shell_dft_plus_u_uses_requested_route(tmp_path):
    """High-level Gamma UKS/GPW +U must execute the requested GPW route and
    record it in the paper-audit metadata."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    system, basis, _grid = _h3_doublet_fixture()
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]
    output_stem = tmp_path / "gpw_uks_plus_u"

    result = vq.run_periodic_job(
        system,
        basis,
        method="UKS",
        functional="lda",
        jk_method="gpw",
        kpoints=(1, 1, 1),
        output=output_stem,
        dft_plus_u=sites,
        cutoff_ha=2.0,
        max_iter=40,
        conv_tol_energy=1e-8,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=False,
    )

    assert result.converged
    assert result.e_dft_plus_u > 0.0
    text = output_stem.with_suffix(".out").read_text()
    assert "dft_plus_u_route    = gpw_uks_gamma" in text
    assert "DFT+U (Dudarev)" in text
    manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
    run = manifest["run"]
    assert run["jk_method_requested"] == "gpw"
    assert run["jk_method_resolved"] == "gpw"
    assert run["jk_method_executed"] == "gpw"
    assert run["dft_plus_u"] is True
    assert run["dft_plus_u_route"] == "gpw_uks_gamma"


def test_run_periodic_job_gapw_open_shell_dft_plus_u_uses_requested_route(tmp_path):
    """High-level Gamma UKS/GAPW +U must execute GAPW and record it in the
    paper-audit metadata."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    system, basis, _grid = _h3_doublet_fixture()
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]
    output_stem = tmp_path / "gapw_uks_plus_u"

    result = vq.run_periodic_job(
        system,
        basis,
        method="UKS",
        functional="lda",
        jk_method="gapw",
        kpoints=(1, 1, 1),
        output=output_stem,
        dft_plus_u=sites,
        cutoff_ha=2.0,
        max_iter=40,
        conv_tol_energy=1e-8,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=False,
    )

    assert result.converged
    assert result.e_dft_plus_u > 0.0
    text = output_stem.with_suffix(".out").read_text()
    assert "dft_plus_u_route    = gapw_uks_gamma" in text
    assert "DFT+U (Dudarev)" in text
    manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
    run = manifest["run"]
    assert run["jk_method_requested"] == "gapw"
    assert run["jk_method_resolved"] == "gapw"
    assert run["jk_method_executed"] == "gapw"
    assert run["dft_plus_u"] is True
    assert run["dft_plus_u_route"] == "gapw_uks_gamma"
    assert run["gapw_one_centre_resolved"] == "block"
    assert run["gapw_molecular_limit_declared"] is False


def test_run_periodic_job_gdf_multik_dft_plus_u_uses_requested_route(tmp_path):
    """Explicit closed-shell multi-k GDF +U runs on the requested backend."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    output_stem = tmp_path / "gdf_rks_plus_u"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        kpoints=(1, 1, 2),
        output=output_stem,
        dft_plus_u=[vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)],
        max_iter=40,
        conv_tol_energy=1e-8,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )

    assert result.converged
    assert result.e_dft_plus_u > 0.0
    text = output_stem.with_suffix(".out").read_text()
    assert "dft_plus_u_route    = gdf_rks_multi_k" in text
    assert "DFT+U (Dudarev)" in text
    manifest = tomllib.loads(output_stem.with_suffix(".system").read_text())
    run = manifest["run"]
    assert run["jk_method_requested"] == "gdf"
    assert run["jk_method_resolved"] == "gdf"
    assert run["jk_method_executed"] == "gdf"
    assert run["dft_plus_u"] is True
    assert run["dft_plus_u_route"] == "gdf_rks_multi_k"


def test_run_periodic_job_explicit_gdf_gamma_dft_plus_u_fails_closed(tmp_path):
    """Γ-only GDF +U still fails closed instead of pretending to be multi-k."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    with pytest.raises(NotImplementedError, match="explicit jk_method='gdf'"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="gdf",
            output=tmp_path / "gdf_rks_gamma_plus_u",
            dft_plus_u=[vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)],
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
        )


def test_qvf_provenance_records_periodic_dft_plus_u_route():
    """The QVF manifest provenance can carry the same route audit fields
    as the .system manifest for release-paper native artifacts."""
    from vibeqc.output.formats.qvf import _build_provenance

    provenance = _build_provenance(
        {
            "method": "RKS",
            "basis": "sto-3g",
            "functional": "lda",
            "jk_method": "gpw",
            "jk_method_resolved": "gpw",
            "jk_method_executed": "gpw",
            "dft_plus_u": True,
            "dft_plus_u_route": "gpw_rks_multi_k",
        }
    )

    assert provenance["jk_method"] == "gpw"
    assert provenance["jk_method_resolved"] == "gpw"
    assert provenance["jk_method_executed"] == "gpw"
    assert provenance["dft_plus_u"] is True
    assert provenance["dft_plus_u_route"] == "gpw_rks_multi_k"
