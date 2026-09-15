"""Runtime coverage for the default periodic SCF accelerator."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import vibeqc as vq


def _small_cell(open_shell: bool) -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    box = 12.0
    center = box / 2.0
    if open_shell:
        atoms = [vq.Atom(1, [center, center, center])]
    else:
        atoms = [
            vq.Atom(1, [center, center, center - 0.7]),
            vq.Atom(1, [center, center, center + 0.7]),
        ]
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        atoms,
        charge=0,
        multiplicity=2 if open_shell else 1,
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


@pytest.mark.parametrize(
    ("driver", "open_shell", "is_ks"),
    [
        (vq.run_rhf_periodic_gamma_ewald3d, False, False),
        (vq.run_rks_periodic_gamma_ewald3d, False, True),
        (vq.run_uhf_periodic_gamma_ewald3d, True, False),
        (vq.run_uks_periodic_gamma_ewald3d, True, True),
    ],
    ids=("rhf", "rks", "uhf", "uks"),
)
def test_default_accelerator_runs_three_periodic_scf_iterations(
    driver: Callable,
    open_shell: bool,
    is_ks: bool,
) -> None:
    """The Python hybrid and its bound EDIIS/ADIIS API stay in sync.

    This is deliberately a real SCF run, not an import or collection check.
    Zero convergence tolerances force iteration three, where the default
    EDIIS_DIIS hybrid has produced and discarded extrapolations.  A stale
    native extension lacking ``discard_last_extrapolation`` fails here.
    """
    system, basis = _small_cell(open_shell)
    options = vq.PeriodicKSOptions() if is_ks else vq.PeriodicRHFOptions()
    assert options.scf_accelerator == vq.SCFAccelerator.EDIIS_DIIS

    if is_ks:
        options.functional = "LDA"
    options.max_iter = 3
    options.conv_tol_energy = 0.0
    options.conv_tol_grad = 0.0
    options.lattice_opts.cutoff_bohr = 8.0
    options.lattice_opts.nuclear_cutoff_bohr = 8.0

    result = driver(
        system,
        basis,
        options,
        omega=0.5,
        spacing_bohr=0.6,
    )

    assert result.n_iter == 3
    assert np.isfinite(result.energy)
