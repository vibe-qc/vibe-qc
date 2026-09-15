"""Density mixers (Anderson / Broyden [+ Kerker]) on the multi-k GDF route
(prompt 22: "Mixers are implemented below run_periodic_job but not exposed").

The machinery is the same as the multi-k EWALD_3D RKS driver
(``periodic_density_mixing``): per-k density-MATRIX mixing with optional
Kerker preconditioning of the residual, replacing Fock-DIIS / damping /
fock mixing when selected. Pins:

* Anderson-mixed multi-k GDF RKS converges to the same fixed point as the
  default Fock-DIIS run on a well-gapped insulator (LiH primitive fcc --
  a single SCF solution, so accelerator-vs-accelerator energy equality is
  well-posed; see the gilat sharp-Fermi-surface note for why a metal is
  NOT used here);
* Kerker requires a mixer (standalone Kerker raises);
* fixed-point-equality tests share ONE (deliberately coarse, ke=60 Ha)
  reciprocal mesh between the DIIS and mixed runs -- mesh quality
  cancels exactly, and the production-mesh cost (minutes per SCF)
  stays out of the unit suite;
* ``run_periodic_job`` forwards the knobs on the closed-shell GDF route
  with explicit ``kpoints=`` and still fails closed elsewhere (GPW route,
  Gamma-default without kpoints).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_k_gdf import run_krks_periodic_gdf


def _lih_fcc(a: float = 3.86):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.array(
        [[0.0, a, a], [a, 0.0, a], [a, a, 0.0]], dtype=float
    )
    sysp.unit_cell = [core.Atom(3, [0, 0, 0]), core.Atom(1, [a, a, a])]
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    return sysp, vq.BasisSet(mol, "sto-3g")


def test_anderson_matches_diis_fixed_point():
    sysp, basis = _lih_fcc()
    ref = run_krks_periodic_gdf(
        sysp, basis, (1, 1, 2), functional="lda",
        aux_basis="def2-svp-jk", rsgdf_ke_cutoff=60.0, progress=False,
    )
    mixed = run_krks_periodic_gdf(
        sysp, basis, (1, 1, 2), functional="lda",
        aux_basis="def2-svp-jk", rsgdf_ke_cutoff=60.0,
        density_mixer="anderson", progress=False,
    )
    assert ref.converged and mixed.converged
    assert mixed.energy == pytest.approx(ref.energy, abs=5e-7)


def test_broyden_kerker_matches_diis_fixed_point():
    sysp, basis = _lih_fcc()
    ref = run_krks_periodic_gdf(
        sysp, basis, (1, 1, 2), functional="lda",
        aux_basis="def2-svp-jk", rsgdf_ke_cutoff=60.0, progress=False,
    )
    mixed = run_krks_periodic_gdf(
        sysp, basis, (1, 1, 2), functional="lda",
        aux_basis="def2-svp-jk", rsgdf_ke_cutoff=60.0,
        density_mixer="broyden", density_mixer_kerker=True, progress=False,
    )
    assert ref.converged and mixed.converged
    # Kerker preconditioning leaves the fixed point unchanged.
    assert mixed.energy == pytest.approx(ref.energy, abs=5e-7)


def test_kerker_without_mixer_raises():
    sysp, basis = _lih_fcc()
    with pytest.raises(ValueError, match="requires"):
        run_krks_periodic_gdf(
            sysp, basis, (1, 1, 2), functional="lda",
            aux_basis="def2-svp-jk", rsgdf_ke_cutoff=60.0,
            density_mixer_kerker=True, progress=False,
        )


def test_runner_forwards_mixer_on_gdf_route(tmp_path):
    """run_periodic_job(density_mixer='anderson', jk_method='gdf',
    kpoints=...) reaches the mixed GDF driver and converges to the DIIS
    fixed point."""
    sysp, basis = _lih_fcc()
    common = dict(
        system=sysp, basis=basis, method="RKS", functional="lda",
        jk_method="gdf", kpoints=(1, 1, 2), rsgdf_ke_cutoff=60.0,
        citations=False,
        progress=False, output_qvf=False, write_molden_file=False,
        write_xyz_file=False, write_poscar_file=False,
        write_xsf_structure_file=False, write_cif_file=False,
        write_population_file=False,
    )
    ref = vq.run_periodic_job(output=str(tmp_path / "ref"), **common)
    mixed = vq.run_periodic_job(
        output=str(tmp_path / "mix"), density_mixer="anderson", **common
    )
    assert ref.converged and mixed.converged
    assert float(mixed.energy) == pytest.approx(float(ref.energy), abs=5e-7)


def test_runner_mixer_still_fails_closed_elsewhere(tmp_path):
    sysp, basis = _lih_fcc()
    # GPW route: not wired.
    with pytest.raises(NotImplementedError, match="density_mixer"):
        vq.run_periodic_job(
            system=sysp, basis=basis, method="RKS", functional="lda",
            jk_method="gpw", kpoints=(1, 1, 2),
            density_mixer="anderson",
            output=str(tmp_path / "x"), progress=False,
        )
    # Gamma default (no kpoints): must ask for an explicit Nk=1 mesh.
    with pytest.raises(NotImplementedError, match="kpoints"):
        vq.run_periodic_job(
            system=sysp, basis=basis, method="RKS", functional="lda",
            jk_method="gdf", density_mixer="anderson",
            output=str(tmp_path / "y"), progress=False,
        )
